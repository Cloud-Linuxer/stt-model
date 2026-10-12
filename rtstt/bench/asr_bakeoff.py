"""Offline ASR bake-off: same VAD chunks for every model, CER/WER + speed.

  python asr_bakeoff.py run --model whisper_turbo --device cuda     # -> results/<model>_<device>.json
  python asr_bakeoff.py report                                       # table of all results

Each model is a module in asr_models/ with:
  load(device) -> handle ;  transcribe(handle, chunks: list[np.float32 16 kHz], lang: str) -> list[str]
Models with conflicting deps run from their own venv (created with --system-site-packages).
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
SR = 16000
DATA_DIRS = [os.path.join(HERE, "data_yt"), os.path.join(HERE, "data")]


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


def edit(a, b):
    d = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        prev, d[0] = d.copy(), i
        for j, y in enumerate(b, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (x != y))
    return int(d[len(b)])


def vad_chunks(audio, max_s=20.0):
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    speech = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=250, speech_pad_ms=100))
    chunks, cur = [], None
    for s in speech:
        if cur and s["end"] - cur[0] <= max_s * SR:
            cur = (cur[0], s["end"])
        else:
            if cur:
                chunks.append(cur)
            cur = (s["start"], s["end"])
    if cur:
        chunks.append(cur)
    out = []
    for a, b in chunks:  # hard-split anything still too long
        while b - a > max_s * SR:
            out.append((a, a + int(max_s * SR)))
            a += int(max_s * SR)
        out.append((a, b))
    return [audio[a:b] for a, b in out]


def clips():
    for d in DATA_DIRS:
        for j in sorted(glob.glob(os.path.join(d, "*.json"))):
            ref = json.load(open(j))
            yield ref, j[:-5] + ".wav"


def run(model, device, out, only):
    mod = importlib.import_module(f"asr_models.{model}")
    t = time.time()
    h = mod.load(device)
    load_s = time.time() - t
    res = {"model": model, "device": device, "load_s": load_s, "clips": {}}
    for ref, wav in clips():
        if only and not any(o in ref["name"] for o in only):
            continue
        audio = sf.read(wav, dtype="float32")[0]
        chunks = vad_chunks(audio, getattr(mod, "MAX_S", 20.0))
        langs = sorted({u["lang"] for u in ref["utterances"]})
        lang = langs[0] if len(langs) == 1 else "auto"
        mod.transcribe(h, chunks[:1], lang)  # warm-up
        t = time.time()
        texts = mod.transcribe(h, chunks, lang)
        dt = time.time() - t
        res["clips"][ref["name"]] = {"text": " ".join(texts), "seconds": dt, "audio_s": len(audio) / SR,
                                     "chunks": len(chunks)}
        print(f"{ref['name']:22s} {dt:7.1f}s  RTF {dt / (len(audio) / SR):.3f}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)


def score_clip(ref, text):
    langs = {u["lang"] for u in ref["utterances"]}
    r_txt = " ".join(u["text"] for u in ref["utterances"])
    if langs == {"en"}:
        r, h = [norm(w) for w in r_txt.split() if norm(w)], [norm(w) for w in text.split() if norm(w)]
        return "wer", edit(r, h) / max(1, len(r))
    # ko or mixed: character error rate over alnum chars
    r, h = norm(r_txt), norm(text)
    return "cer", edit(r, h) / max(1, len(r))


def report():
    refs = {ref["name"]: ref for ref, _ in clips()}
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        res = json.load(open(f))
        sc = {}
        for name, c in res["clips"].items():
            sc[name] = score_clip(refs[name], c["text"])[1]
        tot_s = sum(c["seconds"] for c in res["clips"].values())
        tot_a = sum(c["audio_s"] for c in res["clips"].values())
        rows.append((f"{res['model']}@{res['device']}", sc, tot_s / tot_a))
    names = [n for n in refs if any(n in r[1] for r in rows)]
    groups = {"ko_real": [n for n in names if n.startswith("yt_ko")],
              "en_real": [n for n in names if n.startswith("yt_en")],
              "mix_real": [n for n in names if n.startswith("yt_mix")],
              "synthetic": [n for n in names if not n.startswith("yt_")]}
    print(f"{'model':38s} " + " ".join(f"{g:>10s}" for g in groups) + f" {'RTF':>7s}  {'x realtime':>10s}")
    for model, sc, rtf in sorted(rows, key=lambda r: np.mean([r[1][n] for n in groups['ko_real'] if n in r[1]] or [9])):
        cells = []
        for g, ns in groups.items():
            v = [sc[n] for n in ns if n in sc]
            cells.append(f"{100 * np.mean(v):9.2f}%" if v else f"{'-':>10s}")
        print(f"{model:38s} " + " ".join(cells) + f" {rtf:7.3f}  {1 / rtf:9.0f}x")
    print("\nko_real/mix/synthetic = CER, en_real = WER (lower is better); RTF = processing time / audio time")
    if "--detail" in sys.argv:
        for model, sc, _ in rows:
            print(model, {k: round(100 * v, 2) for k, v in sc.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.model, a.device, os.path.join(HERE, "results", f"{a.model}_{a.device}.json"), a.only)
    else:
        report()
