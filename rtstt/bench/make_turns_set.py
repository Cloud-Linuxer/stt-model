"""Rapid speaker alternation test set -> bench/data_turns/.

Synthetic (edge-tts voices of make_dataset.DIALOG_KO / DIALOG_MIX) with a fixed gap between turns:
0, 100, 200, 400 ms and an overlap of 300 ms (next speaker starts before the previous one ends).
Real voices: 5 YouTube speakers (set 2) interleaved with 50-150 ms gaps.

  python make_turns_set.py --edge-tts-path <dir with edge_tts>  --raw <yt2 raw dir>
"""
import argparse
import json
import os
import random
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_dataset as md  # noqa: E402
import make_youtube_set as my  # noqa: E402

SR = 16000


def build(dialog, voices, gap, out, name, cache, edge_path, tmp):
    parts, utts, t = [np.zeros(int(SR * 0.5), np.float32)], [], 0.5
    audio = np.zeros(int(SR * 0.5), np.float32)
    for spk, text, ref in dialog:
        key = (voices[spk], text)
        if key not in cache:
            cache[key] = md.tts(text, voices[spk], edge_path, tmp)
        a = cache[key]
        lang = "en" if voices[spk].startswith("en") else "ko"
        start = max(0.0, t + gap)                      # gap < 0 -> overlap with the previous turn
        s0 = int(start * SR)
        if s0 + len(a) > len(audio):
            audio = np.concatenate([audio, np.zeros(s0 + len(a) - len(audio), np.float32)])
        audio[s0:s0 + len(a)] += a
        utts.append({"speaker": spk, "lang": lang, "text": text, "ref_translation": ref,
                     "start": round(start, 3), "end": round(start + len(a) / SR, 3)})
        t = start + len(a) / SR
    audio = np.concatenate([audio, np.zeros(SR, np.float32)])
    audio = np.clip(audio, -1, 1)
    sf.write(os.path.join(out, f"{name}.wav"), (audio * 32767).astype(np.int16), SR, subtype="PCM_16")
    json.dump({"name": name, "duration": len(audio) / SR, "num_speakers": len(voices), "gap_s": gap,
               "utterances": utts}, open(os.path.join(out, f"{name}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{name}: {len(audio)/SR:.1f}s, {len(utts)} turns, gap {gap:+.2f}s")


def real_mix(raw, out, gap_lo, gap_hi, name):
    # D is an interview: the host's cues (marked "- " in the subtitles) are another voice -> excluded
    host_d = [[15, 31.58], [65.71, 68.0], [78.38, 82.71], [123.12, 131.83]]
    sources = [("A", "0PZkxmOhvu0", "ko", 15, 140, []), ("B", "Khb2_U0_jvM", "ko", 20, 140, []), ("C", "TJeAhO3cYCs", "ko", 15, 140, []),
               ("D", "Pb3hXSrU4xQ", "en", 20, 140, host_d), ("E", "DkHQcXLVSUU", "en", 5, 120, [])]
    rng = random.Random(1)
    pools, cache = {}, {}
    for spk, vid, lang, s0, e0, skip in sources:
        cues = [c for c in my.parse_vtt(my.sub_path(raw, vid, lang)) if c[0] >= s0 and c[1] <= e0
                and not any(c[0] < b and c[1] > a for a, b in skip)]
        turns, cur = [], None
        for s, e, txt in cues:
            if cur and s - cur[1] < 0.6 and e - cur[0] < 6:
                cur = (cur[0], e, cur[2] + " " + txt)
            else:
                if cur:
                    turns.append(cur)
                cur = (s, e, txt)
        if cur:
            turns.append(cur)
        pools[spk] = (vid, lang, [x for x in turns if 1.5 <= x[1] - x[0] and x[2].rstrip()[-1:] in ".?!"])
    parts, utts, t, prev = [np.zeros(int(SR * 0.5), np.float32)], [], 0.5, None
    for _ in range(30):
        cands = [k for k in pools if k != prev and pools[k][2]]
        if not cands:
            break
        spk = rng.choice(cands)
        vid, lang, turns = pools[spk]
        s, e, text = turns.pop(0)
        if vid not in cache:
            cache[vid] = my.decode_audio(my.find_audio(raw, vid), sampling_rate=SR)
        seg = cache[vid][int(s * SR):int(e * SR)]
        utts.append({"speaker": spk, "lang": lang, "text": text, "ref_translation": "", "start": round(t, 3),
                     "end": round(t + len(seg) / SR, 3)})
        gap = rng.uniform(gap_lo, gap_hi)
        parts += [seg, np.zeros(int(SR * gap), np.float32)]
        t += len(seg) / SR + gap
        prev = spk
    a = np.concatenate(parts + [np.zeros(SR, np.float32)])
    sf.write(os.path.join(out, f"{name}.wav"), (np.clip(a, -1, 1) * 32767).astype(np.int16), SR, subtype="PCM_16")
    json.dump({"name": name, "duration": len(a) / SR, "num_speakers": len({u["speaker"] for u in utts}),
               "translation_target": None, "utterances": utts}, open(os.path.join(out, f"{name}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"{name}: {len(a)/SR:.0f}s, {len(utts)} turns, gaps {gap_lo}-{gap_hi}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-tts-path")
    ap.add_argument("--raw", help="yt2 raw dir (audio + subtitles) for the real-voice mix")
    ap.add_argument("--out", default=os.path.join(HERE, "data_turns"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = {}
    with tempfile.TemporaryDirectory() as tmp:
        ko_voices = {"A": "ko-KR-SunHiNeural", "B": "ko-KR-InJoonNeural"}
        for gap in (0.0, 0.1, 0.2, 0.4, -0.3):
            tag = f"ov{int(-gap*1000):03d}" if gap < 0 else f"gap{int(gap*1000):03d}"
            build(md.DIALOG_KO, ko_voices, gap, a.out, f"turns_ko_{tag}", cache, a.edge_tts_path, tmp)
        build(md.DIALOG_MIX, {"A": "ko-KR-SunHiNeural", "B": "ko-KR-InJoonNeural", "C": "en-US-GuyNeural"}, 0.1,
              a.out, "turns_mix_gap100", cache, a.edge_tts_path, tmp)
    if a.raw:
        real_mix(a.raw, a.out, 0.05, 0.15, "turns_real5_gap100")


if __name__ == "__main__":
    main()
