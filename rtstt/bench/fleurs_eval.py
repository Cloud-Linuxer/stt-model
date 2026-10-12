"""Per-language ASR and MT evaluation on bench/data_fleurs (58 of the 60 supported languages).

  python fleurs_eval.py asr --model whisper_turbo --device cuda     # -> results_fleurs/asr_<model>.json
  python fleurs_eval.py mt --engine hymt --url http://localhost:8095/v1
  python fleurs_eval.py mt --engine madlad --path ../models/mt/madlad400-3b-ct2-int8
  python fleurs_eval.py report

ASR: CER for zh/ja/th/ko, WER otherwise, after Unicode punctuation/symbol stripping.
MT: the human source transcript is translated to English and Korean; chrF against FLEURS's
parallel English / Korean sentences (same FLoRes sentence id).
"""
import argparse
import asyncio
import glob
import importlib
import json
import os
import sys
import time
import unicodedata

import numpy as np
import sacrebleu
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
DATA = os.path.join(HERE, "data_fleurs")
OUT = os.path.join(HERE, "results_fleurs")
CER_LANGS = {"zh", "ja", "th", "ko"}
HYMT = set("zh en fr pt es ja tr ru ar ko th it de vi ms id tl hi pl cs nl fa gu ur te mr he bn ta uk kk".split())
NEMOTRON = {"en": "en-US", "es": "es-ES", "fr": "fr-FR", "it": "it-IT", "pt": "pt-BR", "nl": "nl-NL", "de": "de-DE",
            "tr": "tr-TR", "ru": "ru-RU", "ar": "ar-AR", "hi": "hi-IN", "ja": "ja-JP", "ko": "ko-KR", "vi": "vi-VN",
            "uk": "uk-UA", "pl": "pl-PL", "sv": "sv-SE", "cs": "cs-CZ", "no": "nb-NO", "da": "da-DK", "bg": "bg-BG",
            "fi": "fi-FI", "hr": "hr-HR", "sk": "sk-SK", "zh": "zh-CN", "hu": "hu-HU", "ro": "ro-RO", "et": "et-EE"}


def norm_text(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return " ".join(s.split())


def edit(a, b):
    d = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        prev, d[0] = d.copy(), i
        for j, y in enumerate(b, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (x != y))
    return int(d[len(b)])


def error_rate(lang, refs, hyps):
    e = n = 0
    for r, h in zip(refs, hyps):
        r, h = norm_text(r), norm_text(h)
        if lang in CER_LANGS:
            r, h = r.replace(" ", ""), h.replace(" ", "")
        else:
            r, h = r.split(), h.split()
        e += edit(r, h)
        n += len(r)
    return e / max(1, n)


def langs():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(DATA, "*", "meta.json")))


def load(lang):
    m = json.load(open(os.path.join(DATA, lang, "meta.json")))
    return m["utts"], [os.path.join(DATA, lang, u["file"]) for u in m["utts"]]


def run_asr(model, device, only):
    mod = importlib.import_module(f"asr_models.{model}")
    h = mod.load(device)
    res = {"model": model, "device": device, "langs": {}}
    for lang in langs():
        if only and lang not in only:
            continue
        utts, files = load(lang)
        audio = [sf.read(f, dtype="float32")[0] for f in files]
        audio = [a if a.ndim == 1 else a.mean(1) for a in audio]
        code = NEMOTRON.get(lang, lang) if model.startswith("sherpa_nemotron") else lang
        mod.transcribe(h, audio[:1], code)
        t = time.time()
        hyps = mod.transcribe(h, audio, code)
        dt = time.time() - t
        er = error_rate(lang, [u["text"] for u in utts], hyps)
        res["langs"][lang] = {"err": er, "metric": "cer" if lang in CER_LANGS else "wer", "hyps": hyps,
                              "seconds": dt, "audio_s": sum(len(a) for a in audio) / 16000}
        print(f"{lang}: {'CER' if lang in CER_LANGS else 'WER'} {100 * er:6.2f}%  ({dt:.1f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"asr_{model}_{device}.json"), "w"), ensure_ascii=False, indent=1)


async def run_mt(engine, url, path, only):
    if engine == "madlad":
        import ctranslate2
        import sentencepiece as spm
        tr = ctranslate2.Translator(path, device="cuda", compute_type="int8_float16")
        sp = spm.SentencePieceProcessor(os.path.join(path, "spiece.model"))

        async def translate(texts, src, tgt):
            toks = [sp.encode(f"<2{tgt}> {t}", out_type=str) for t in texts]
            out = tr.translate_batch(toks, beam_size=1, max_decoding_length=256)
            return [sp.decode(o.hypotheses[0]) for o in out]
    else:
        from translate import Translator
        t_ = Translator(url, "mt", style="hymt" if engine == "hymt" else "chat")

        async def one(text, src, tgt):
            return "".join([c async for c in t_.stream(text, src, tgt)]).strip()

        async def translate(texts, src, tgt):
            return list(await asyncio.gather(*[one(t, src, tgt) for t in texts]))
    res = {"engine": engine, "langs": {}}
    for lang in langs():
        if only and lang not in only:
            continue
        utts, _ = load(lang)
        row = {}
        for tgt in ("en", "ko"):
            if tgt == lang or (engine == "hymt" and (lang not in HYMT or tgt not in HYMT)):
                continue
            hyps = await translate([u["text"] for u in utts], lang, tgt)
            row[tgt] = {"chrf": sacrebleu.corpus_chrf(hyps, [[u[tgt] for u in utts]]).score, "hyps": hyps}
        res["langs"][lang] = row
        print(lang, {k: round(v["chrf"], 1) for k, v in row.items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"mt_{engine}.json"), "w"), ensure_ascii=False, indent=1)


def report():
    asr = {os.path.basename(f)[4:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "asr_*.json")))}
    mt = {os.path.basename(f)[3:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "mt_*.json")))}
    cols = list(asr) + [f"{e}>{t}" for e in mt for t in ("en", "ko")]
    print(f"{'lang':5s} " + " ".join(f"{c[:16]:>16s}" for c in cols))
    for lang in langs():
        cells = []
        for name, r in asr.items():
            x = r["langs"].get(lang)
            cells.append(f"{100 * x['err']:15.1f}%" if x else f"{'-':>16s}")
        for e, r in mt.items():
            for t in ("en", "ko"):
                x = r["langs"].get(lang, {}).get(t)
                cells.append(f"{x['chrf']:16.1f}" if x else f"{'-':>16s}")
        print(f"{lang:5s} " + " ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["asr", "mt", "report"])
    ap.add_argument("--model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--engine", choices=["hymt", "madlad", "chat"])
    ap.add_argument("--url")
    ap.add_argument("--path")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd == "asr":
        run_asr(a.model, a.device, a.only)
    elif a.cmd == "mt":
        asyncio.run(run_mt(a.engine, a.url, a.path, a.only))
    else:
        report()
