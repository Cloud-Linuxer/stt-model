"""Offline translation bake-off on human transcripts (isolates MT quality from ASR errors).

  python mt_bakeoff.py run --name hymt2-1.8b-q8 --engine openai --url http://localhost:8091/v1 --style hymt
  python mt_bakeoff.py run --name madlad-3b --engine madlad --path ../models/mt/madlad400-3b-ct2-int8
  python mt_bakeoff.py report

Source cues are merged into sentences, translated in order with the previous 3 sentences
as context (as the live server does), and scored with document-level chrF against the
human translation. Latency = wall time per sentence (first token and full).
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
import time

import sacrebleu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from translate import Translator  # noqa: E402

DATA_DIRS = [os.path.join(HERE, "data_yt"), os.path.join(HERE, "data")]
END = (".", "?", "!", "。", "？", "！")


def documents():
    """Yield (name, [(src_sentence, src_lang, tgt_lang)], reference_text)."""
    for d in DATA_DIRS:
        for j in sorted(glob.glob(os.path.join(d, "*.json"))):
            ref = json.load(open(j))
            utts = [u for u in ref["utterances"] if u.get("ref_translation")]
            if not utts:
                continue
            sents, cur = [], None
            for u in utts:
                tgt = ref.get("translation_target") or ("ko" if u["lang"] == "en" else "en")
                if cur and cur[1] == u["lang"] and not cur[0].rstrip().endswith(END):
                    cur = (cur[0] + " " + u["text"], u["lang"], tgt)
                else:
                    if cur:
                        sents.append(cur)
                    cur = (u["text"], u["lang"], tgt)
            if cur:
                sents.append(cur)
            # one reference per target language; mixed docs are scored per direction
            for tgt in sorted({s[2] for s in sents}):
                refs = " ".join(u["ref_translation"] for u in utts
                                if (ref.get("translation_target") or ("ko" if u["lang"] == "en" else "en")) == tgt)
                yield f"{ref['name']}>{tgt}", [s for s in sents if s[2] == tgt], refs


class MadladEngine:
    def __init__(self, path, threads):
        import ctranslate2
        import sentencepiece as spm
        self.tr = ctranslate2.Translator(path, device="cpu", intra_threads=threads)
        self.sp = spm.SentencePieceProcessor(os.path.join(path, "spiece.model"))

    async def stream(self, text, src, tgt, history=(), glossary=None, general=None):
        toks = self.sp.encode(f"<2{tgt}> {text}", out_type=str)
        out = self.tr.translate_batch([toks], beam_size=1, max_decoding_length=256)[0].hypotheses[0]
        yield self.sp.decode(out)


async def run(args):
    if args.engine == "madlad":
        eng = MadladEngine(args.path, args.threads)
    else:
        eng = Translator(args.url, args.model, style=args.style)
    res = {"name": args.name, "docs": {}}
    for name, sents, refs in documents():
        hyp, firsts, totals, hist = [], [], [], []
        for text, src, tgt in sents:
            t0 = time.time()
            first, out = None, []
            async for chunk in eng.stream(text, src, tgt, hist[-3:]):
                first = first or time.time() - t0
                out.append(chunk)
            totals.append(time.time() - t0)
            firsts.append(first or totals[-1])
            hyp.append("".join(out).strip())
            hist.append((text, hyp[-1]))
        res["docs"][name] = {"hyp": hyp, "src": [s[0] for s in sents], "ref": refs, "first": firsts,
                             "total": totals}
        chrf = sacrebleu.corpus_chrf([" ".join(hyp)], [[refs]]).score
        print(f"{name:28s} chrF {chrf:5.1f}  first p50 {st.median(firsts):.2f}s  total p95 "
              f"{sorted(totals)[int(0.95 * (len(totals) - 1))]:.2f}s  n={len(sents)}", flush=True)
    os.makedirs(os.path.join(HERE, "results_mt"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results_mt", f"{args.name}.json"), "w"), ensure_ascii=False, indent=1)


def report():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "results_mt", "*.json"))):
        r = json.load(open(f))
        g = {"ko>en real": [], "en>ko real": [], "synthetic": []}
        first, total = [], []
        for name, d in r["docs"].items():
            chrf = sacrebleu.corpus_chrf([" ".join(d["hyp"])], [[d["ref"]]]).score
            key = "synthetic" if not name.startswith("yt_") else ("ko>en real" if name.endswith(">en") else "en>ko real")
            g[key].append(chrf)
            first += d["first"]
            total += d["total"]
        rows.append((r["name"], {k: st.mean(v) if v else float("nan") for k, v in g.items()},
                     st.median(first), sorted(total)[int(0.95 * (len(total) - 1))], st.median(total)))
    print(f"{'model':30s} {'ko>en real':>11s} {'en>ko real':>11s} {'synthetic':>10s} {'first p50':>10s} "
          f"{'full p50':>9s} {'full p95':>9s}")
    for name, g, f50, t95, t50 in sorted(rows, key=lambda r: -(r[1]["ko>en real"] + r[1]["en>ko real"])):
        print(f"{name:30s} {g['ko>en real']:11.1f} {g['en>ko real']:11.1f} {g['synthetic']:10.1f} "
              f"{f50:9.2f}s {t50:8.2f}s {t95:8.2f}s")
    print("\nchrF (higher is better), latency per sentence")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--name")
    ap.add_argument("--engine", choices=["openai", "madlad"], default="openai")
    ap.add_argument("--url")
    ap.add_argument("--model", default="default")
    ap.add_argument("--style", choices=["chat", "hymt"], default="chat")
    ap.add_argument("--path")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "run":
        import asyncio
        asyncio.run(run(a))
    else:
        report()
