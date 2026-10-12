"""End-of-turn (semantic endpointing) evaluation on pause points with known labels.

A pause point is a VAD silence of >= --pause-ms inside a scenario. Label:
  turn_end   the pause starts within +-0.3 s of a reference turn end (speaker change, or the end
             of a scripted utterance in the synthetic set)                      -> should be "complete"
  mid_sent   inside a reference utterance and the last word before the pause has no sentence
             punctuation (Whisper word timestamps)                              -> should be "incomplete"
  sent_end   inside a reference utterance but after a complete sentence: ambiguous (the speaker
             happened to continue); reported separately
For each pause point a detector sees the audio of the current turn up to the pause (<= 8 s).

  python eot_eval.py                       # smart-turn v3.2 (cpu/gpu onnx) + punctuation baseline
  python eot_eval.py --detectors smartturn_cpu

Detectors live in eot_models/<name>.py:  load() -> handle ;  complete(handle, audio16k) -> prob in [0,1]
"""
import argparse
import glob
import importlib
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from vad import StreamVAD  # noqa: E402

SR = 16000
SETS = [("data", None), ("data_yt", "yt_mix_real_5spk"), ("data_yt2", "yt_mix_real_5spk")]


def pause_points(audio, pause_ms):
    v = StreamVAD()
    v.feed(audio)
    p = np.array(v.probs) > 0.5
    pts, i = [], 0
    while i < len(p):
        if p[i]:
            i += 1
            continue
        j = i
        while j < len(p) and not p[j]:
            j += 1
        if (j - i) * 512 >= SR * pause_ms / 1000 and i > 0:
            pts.append((i * 512 / SR, j * 512 / SR))   # silence start, silence end
        i = j
    return pts


def words_with_times(audio):
    from faster_whisper import WhisperModel
    global _WM
    if "_WM" not in globals():
        _WM = WhisperModel("/home/stt-model/models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/"
                           "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf", device="cuda", compute_type="float16")
    segs, _ = _WM.transcribe(audio, beam_size=1, temperature=0.0, word_timestamps=True, vad_filter=True,
                             condition_on_previous_text=False)
    return [w for s in segs for w in (s.words or [])]


def build_cases(pause_ms):
    cases = []
    for d, only in SETS:
        for f in sorted(glob.glob(os.path.join(HERE, d, "*.json"))):
            ref = json.load(open(f))
            if only and ref["name"] != only:
                continue
            if d == "data_yt" or d == "data_yt2":
                pass
            audio = sf.read(f[:-5] + ".wav", dtype="float32")[0]
            utts = ref["utterances"]
            # turn ends: speaker changes (or every scripted utterance end in the synthetic set)
            turn_ends = []
            for k, u in enumerate(utts):
                nxt = utts[k + 1] if k + 1 < len(utts) else None
                # real mixes were cut by duration: count a turn end only where the text really ends a sentence
                if (d == "data" or nxt is None or nxt["speaker"] != u["speaker"]) and \
                        (d == "data" or u["text"].rstrip()[-1:] in ".?!。？！"):
                    turn_ends.append(u["end"])
            turn_starts = [utts[0]["start"]] + [utts[k + 1]["start"] for k in range(len(utts) - 1)
                                                if d == "data" or utts[k + 1]["speaker"] != utts[k]["speaker"]]
            words = words_with_times(audio)
            for s0, s1 in pause_points(audio, pause_ms):
                if any(abs(s0 - e) <= 0.3 for e in turn_ends):
                    label = "turn_end"
                elif any(u["start"] + 0.3 <= s0 <= u["end"] - 0.3 for u in utts):
                    prev = [w for w in words if w.end <= s0 + 0.15]
                    if not prev or s0 - prev[-1].end > 0.6:
                        continue
                    label = "sent_end" if prev[-1].word.strip()[-1:] in ".?!。？！" else "mid_sent"
                else:
                    continue
                t_start = max([t for t in turn_starts if t <= s0], default=0.0)
                seg = audio[int(max(t_start, s0 - 8.0) * SR):int(min(len(audio) / SR, s0 + 0.2) * SR)]
                cases.append({"set": d, "name": ref["name"], "t": s0, "label": label, "audio": seg, "pause": s1 - s0})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default="smartturn_cpu,smartturn_gpu,punct_whisper")
    ap.add_argument("--pause-ms", type=int, default=200)
    ap.add_argument("--dump", help="write per-case scores as JSON")
    a = ap.parse_args()
    cases = build_cases(a.pause_ms)
    n = {k: sum(c["label"] == k for c in cases) for k in ("turn_end", "mid_sent", "sent_end")}
    print(f"pause points >= {a.pause_ms} ms: {len(cases)}  turn ends {n['turn_end']}, mid-sentence pauses {n['mid_sent']}, "
          f"sentence-end pauses inside a turn {n['sent_end']} (ambiguous)")
    out = {}
    for name in a.detectors.split(","):
        mod = importlib.import_module(f"eot_models.{name}")
        h = mod.load()
        mod.complete(h, cases[0]["audio"])
        t0 = time.time()
        scores = [mod.complete(h, c["audio"]) for c in cases]
        lat = (time.time() - t0) / len(cases) * 1000
        def rates(thr):
            te = [s >= thr for s, c in zip(scores, cases) if c["label"] == "turn_end"]
            ms = [s >= thr for s, c in zip(scores, cases) if c["label"] == "mid_sent"]
            se = [s >= thr for s, c in zip(scores, cases) if c["label"] == "sent_end"]
            return (sum(te) / max(1, len(te)), sum(ms) / max(1, len(ms)), sum(se) / max(1, len(se)))
        r, m, e = rates(0.5)
        best = max(np.arange(0.1, 0.95, 0.05), key=lambda t: rates(t)[0] - rates(t)[1])
        rb, mb, eb = rates(best)
        print(f"{name:16s} @0.5: turn-end recall {100*r:5.1f}%  mid-sentence false-end {100*m:5.1f}%  (sentence-end pauses called complete {100*e:4.1f}%) "
              f"| best thr {best:.2f}: recall {100*rb:.1f}% false-end {100*mb:.1f}% | {lat:.1f} ms/call")
        out[name] = scores
        # policy simulation on these pause points: current rule = <end> once silence >= 0.5 s;
        # two-stage = at 0.2 s ask the detector, <end> now if complete, else at `hold` s of silence
        te = [(sc, c) for sc, c in zip(scores, cases) if c["label"] == "turn_end"]
        ms = [(sc, c) for sc, c in zip(scores, cases) if c["label"] == "mid_sent"]
        for hold in (0.5, 0.8, 1.2):
            lag = np.mean([0.25 if sc >= 0.5 else hold for sc, c in te])
            fe = np.mean([(sc >= 0.5) or c["pause"] >= hold for sc, c in ms]) if ms else float("nan")
            print(f"      policy 0.2s+model, hold {hold:.1f}s: turn-end lag mean {lag:.2f}s | mid-sentence false-end {100*fe:.0f}%"
                  + ("" if hold != 0.5 else f"   [current silence rule: lag 0.50s, false-end {100*np.mean([c['pause'] >= 0.5 for _, c in ms]):.0f}%]"))
        for d in ("data", "data_yt", "data_yt2"):
            te = [s >= 0.5 for s, c in zip(scores, cases) if c["set"] == d and c["label"] == "turn_end"]
            ms = [s >= 0.5 for s, c in zip(scores, cases) if c["set"] == d and c["label"] == "mid_sent"]
            print(f"      {d:9s} turn-end recall {100*sum(te)/max(1,len(te)):5.1f}% (n={len(te)})  mid-sentence false-end {100*sum(ms)/max(1,len(ms)):5.1f}% (n={len(ms)})")
    if a.dump:
        json.dump({"cases": [{k: v for k, v in c.items() if k != "audio"} for c in cases], "scores": out}, open(a.dump, "w"))


if __name__ == "__main__":
    main()
