"""Streaming scenarios from bench/data_fleurs: per language, 8 clips joined with 0.8 s gaps.

-> bench/data_fleurs_stream/<lang>.wav + .json (run_bench.py format; translation to English,
   or to Korean for English audio).
"""
import glob
import json
import os

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
SR, N, GAP = 16000, 8, 0.8
out = os.path.join(HERE, "data_fleurs_stream")
os.makedirs(out, exist_ok=True)
for meta in sorted(glob.glob(os.path.join(HERE, "data_fleurs", "*", "meta.json"))):
    m = json.load(open(meta))
    lang, d = m["lang"], os.path.dirname(meta)
    tgt = "ko" if lang == "en" else "en"
    parts, utts, t = [np.zeros(int(0.5 * SR), np.float32)], [], 0.5
    for u in m["utts"][:N]:
        a, sr = sf.read(os.path.join(d, u["file"]), dtype="float32")
        a = a if a.ndim == 1 else a.mean(1)
        assert sr == SR
        utts.append({"speaker": "?", "lang": lang, "text": u["text"], "ref_translation": u[tgt],
                     "start": round(t, 3), "end": round(t + len(a) / SR, 3)})
        parts += [a, np.zeros(int(GAP * SR), np.float32)]
        t += len(a) / SR + GAP
    audio = np.concatenate(parts + [np.zeros(SR, np.float32)])
    sf.write(os.path.join(out, f"{lang}.wav"), (np.clip(audio, -1, 1) * 32767).astype(np.int16), SR, subtype="PCM_16")
    json.dump({"name": lang, "duration": len(audio) / SR, "num_speakers": None, "translation_target": tgt,
               "utterances": utts}, open(os.path.join(out, f"{lang}.json"), "w"), ensure_ascii=False, indent=1)
print(len(glob.glob(os.path.join(out, "*.json"))), "scenarios")
