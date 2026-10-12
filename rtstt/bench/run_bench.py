"""Benchmark a real-time transcription endpoint (our wire protocol) against ground-truth scenarios.

  python run_bench.py                                  # local compat server, all scenarios
  python run_bench.py --url wss://<host>/transcribe-websocket --api-key $API_KEY --model <model>   # another compatible endpoint
  python run_bench.py --concurrency 4 --scenarios ko_2spk

Streams audio in real time (--chunk-ms, default 100 ms) with diarization, endpoint detection
and translation enabled, then scores against bench/data/<scenario>.json.

Result-arrival metrics (audio time == wall time, the client paces audio in real time):
  tok_nonfinal_*  first result covering a word's audio (any non-final/final token) - word end
  tok_final_*     final token arrival - word end (word times as reported by the model)
  utt_final_*     last final token of a reference utterance - reference utterance end
"""
import argparse
import asyncio
import glob
import itertools
import json
import os
import statistics as st
import time

import numpy as np
import sacrebleu
import soundfile as sf
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))

# Acceptance criteria (see CRITERIA.md). (metric, comparator, threshold)
CRITERIA = [
    ("cer_ko", "<=", 0.05), ("wer_en", "<=", 0.08),
    ("first_token_p50_s", "<=", 1.0), ("end_lag_p50_s", "<=", 1.0), ("end_lag_p95_s", "<=", 1.5),
    ("speaker_acc", ">=", 0.90), ("speaker_count_ok", "==", 1),
    ("chrf_to_en", ">=", 55.0), ("chrf_to_ko", ">=", 35.0),
    ("tr_first_lag_p50_s", "<=", 0.5), ("tr_done_lag_p95_s", "<=", 2.0),
]


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


def edit(a, b):
    d = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        prev, d[0] = d.copy(), i
        for j, y in enumerate(b, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (x != y))
    return int(d[len(b)])


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


async def stream(path, url, api_key, model, cfg_extra, chunk_ms=100):
    audio, sr = sf.read(path, dtype="int16")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    cfg = {"model": model, "audio_format": "pcm_s16le", "sample_rate": sr, "num_channels": 1,
           "enable_endpoint_detection": True, "enable_speaker_diarization": True, **cfg_extra}
    events = []  # (wall_s, token)
    chunk = int(sr * chunk_ms / 1000)
    async with websockets.connect(url, additional_headers=headers, max_size=2**22) as ws:
        await ws.send(json.dumps(cfg))
        t0 = time.time()

        async def sender():
            for i in range(0, len(audio), chunk):
                await ws.send(audio[i:i + chunk].tobytes())
                await asyncio.sleep(max(0, t0 + (i + chunk) / sr - time.time()))
            await ws.send("")

        task = asyncio.create_task(sender())
        async for msg in ws:
            r = json.loads(msg)
            if r.get("error_code"):
                raise RuntimeError(r)
            now = time.time() - t0
            events += [(now, t) for t in r.get("tokens", [])]
            if r.get("finished"):
                break
        await task
    return events


def is_marker(t):
    return t["text"].startswith("<") and t["text"].endswith(">")


def score(events, ref):
    utts = ref["utterances"]
    spoken_final = [t for _, t in events if t["is_final"] and "start_ms" in t
                    and t.get("translation_status") != "translation" and not is_marker(t)]
    m = {}

    # --- map each final spoken token to a reference utterance (by time midpoint) ---
    def utt_of(t):
        mid = (t["start_ms"] + t["end_ms"]) / 2000
        return min(range(len(utts)), key=lambda i: 0 if utts[i]["start"] <= mid <= utts[i]["end"]
                   else min(abs(mid - utts[i]["start"]), abs(mid - utts[i]["end"])))
    per_utt = [[] for _ in utts]
    for t in spoken_final:
        per_utt[utt_of(t)].append(t)

    # --- transcription accuracy ---
    # single-language clip: whole-document comparison (robust to subtitle timing offsets);
    # mixed-language clip: per-utterance mapping by time
    single = len({u["lang"] for u in utts}) == 1
    for lang, unit in (("ko", "cer"), ("en", "wer")):
        errs = n = 0
        groups = ([(" ".join(u["text"] for u in utts), spoken_final)] if single
                  else [(u["text"], toks) for u, toks in zip(utts, per_utt) if u["lang"] == lang])
        for text, toks in groups:
            if single and utts[0]["lang"] != lang:
                continue
            u = {"text": text}
            hyp = "".join(t["text"] for t in toks)
            if unit == "cer":
                r, h = norm(u["text"]), norm(hyp)
            else:
                r, h = [norm(w) for w in u["text"].split()], [norm(w) for w in hyp.split() if norm(w)]
            errs += edit(r, h)
            n += len(r)
        if n:
            m[f"{unit}_{lang}"] = errs / n
    if single and utts[0]["lang"] not in ("ko", "en"):
        # other languages: CER for zh/ja/th, WER otherwise (Unicode punctuation stripped)
        import unicodedata
        def nt(x):
            x = unicodedata.normalize("NFKC", x).lower()
            return " ".join("".join(" " if unicodedata.category(c)[0] in "PS" else c for c in x).split())
        lang = utts[0]["lang"]
        r, h = nt(" ".join(u["text"] for u in utts)), nt("".join(t["text"] for t in spoken_final))
        r, h = (r.replace(" ", ""), h.replace(" ", "")) if lang in ("zh", "ja", "th") else (r.split(), h.split())
        m["asr_err"] = edit(r, h) / max(1, len(r))

    # --- latency ---
    first = []
    for u in utts:
        ws = [w for w, t in events if not t["is_final"] and "start_ms" in t
              and u["start"] - 0.3 <= t["start_ms"] / 1000 <= u["end"]]
        ws += [w for w, t in events if t["is_final"] and "start_ms" in t
               and u["start"] - 0.3 <= t["start_ms"] / 1000 <= u["end"]]
        if ws:
            first.append(min(ws) - u["start"])
    m["first_token_p50_s"] = pct(first, 50)
    lags, last_end = [], 0.0
    for w, t in events:
        if "end_ms" in t and t["is_final"] and not is_marker(t):
            last_end = max(last_end, t["end_ms"] / 1000)
        if t["text"] == "<end>":
            lags.append(w - last_end)
    lags = lags[:-1] if len(lags) > 1 else lags  # last <end> is forced by end-of-stream
    m["end_lag_p50_s"], m["end_lag_p95_s"] = pct(lags, 50), pct(lags, 95)

    # --- result arrival per word ---
    first_seen = {}  # 100 ms audio bin -> earliest wall time any spoken token covered it
    for w, t in events:
        if "start_ms" in t and t.get("translation_status") != "translation" and not is_marker(t):
            for b in range(t["start_ms"] // 100, t["end_ms"] // 100 + 1):
                if w < first_seen.get(b, float("inf")):
                    first_seen[b] = w
    nf_lat, f_lat = [], []
    final_events = [(w, t) for w, t in events if t["is_final"] and "start_ms" in t
                    and t.get("translation_status") != "translation" and not is_marker(t) and norm(t["text"])]
    for w, t in final_events:
        e = t["end_ms"] / 1000
        f_lat.append(w - e)
        seen = min((first_seen.get(b, w) for b in range(t["start_ms"] // 100, t["end_ms"] // 100 + 1)), default=w)
        nf_lat.append(min(seen, w) - e)
    m["tok_nonfinal_p50_s"], m["tok_nonfinal_p90_s"] = pct(nf_lat, 50), pct(nf_lat, 90)
    m["tok_final_p50_s"], m["tok_final_p90_s"] = pct(f_lat, 50), pct(f_lat, 90)
    arrival = {id(t): w for w, t in final_events}
    utt_lat = [max(arrival[id(t)] for t in toks if id(t) in arrival) - u["end"]
               for u, toks in zip(utts, per_utt) if any(id(t) in arrival for t in toks)]
    utt_lat = utt_lat[:-1] if len(utt_lat) > 1 else utt_lat   # last one is flushed by end-of-stream
    m["utt_final_p50_s"], m["utt_final_p95_s"] = pct(utt_lat, 50), pct(utt_lat, 95)

    if ref.get("num_speakers") is None:
        return score_translation(events, utts, m, ref.get("translation_target"))

    # --- diarization: char-weighted speaker accuracy under best label mapping ---
    pairs = [(u["speaker"], t.get("speaker"), len(norm(t["text"])))
             for u, toks in zip(utts, per_utt) for t in toks]
    refs = sorted({p[0] for p in pairs})
    hyps = sorted({p[1] for p in pairs if p[1] is not None})
    total = sum(p[2] for p in pairs) or 1
    best = 0
    if hyps:
        for perm in itertools.permutations(hyps, min(len(hyps), len(refs))):
            mp = dict(zip(perm, refs))
            best = max(best, sum(w for r, h, w in pairs if mp.get(h) == r))
    m["speaker_acc"] = best / total
    m["speakers_found"] = len(hyps)
    m["speaker_count_ok"] = int(len(hyps) == ref["num_speakers"])

    return score_translation(events, utts, m, ref.get("translation_target"))


def score_translation(events, utts, m, target=None):
    tr = [(w, t) for w, t in events if t.get("translation_status") == "translation" and t["is_final"]]
    for tgt, key in (("en", "chrf_to_en"), ("ko", "chrf_to_ko")):
        if target and tgt != target:
            continue
        refs_t = [u["ref_translation"] for u in utts if u["lang"] != tgt and u["ref_translation"]]
        hyp_t = "".join(t["text"] for _, t in tr if t["language"] == tgt)
        if refs_t and hyp_t:
            m[key] = sacrebleu.corpus_chrf([hyp_t], [[" ".join(refs_t)]]).score
        elif refs_t:
            m[key] = 0.0
    # Translation latency per burst: a burst is a run of translation tokens with no spoken
    # final token in between; lag is measured from the last spoken final token before it
    # (the moment its source text became final) to the burst's first / last token.
    first_l, done_l = [], []
    last_src, burst = None, []
    for w, t in events + [(float("inf"), {"text": "", "is_final": True, "start_ms": 0})]:
        is_tr = t.get("translation_status") == "translation"
        if is_tr and not t["is_final"]:
            continue
        if is_tr:
            burst.append(w)
            continue
        if t["is_final"] and "start_ms" in t and not is_marker(t):
            if burst and last_src is not None:
                first_l.append(burst[0] - last_src)
                done_l.append(burst[-1] - last_src)
            burst = []
            last_src = w
    if tr:
        m["tr_first_lag_p50_s"], m["tr_done_lag_p95_s"] = pct(first_l, 50), pct(done_l, 95)
    return m


def check(m):
    out = {}
    for k, op, thr in CRITERIA:
        if k not in m or m[k] != m[k]:
            continue
        v = m[k]
        out[k] = v <= thr if op == "<=" else v >= thr if op == ">=" else v == thr
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765/transcribe-websocket")
    ap.add_argument("--api-key")
    ap.add_argument("--model", default="rtstt-v1")
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--no-translation", action="store_true")
    ap.add_argument("--dump", help="dir to write raw token events per scenario")
    ap.add_argument("--parallel", action="store_true", help="stream all scenarios at the same time")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--chunk-ms", type=int, default=100, help="audio sent per WebSocket frame")
    ap.add_argument("--extra-config", default="{}", help="JSON merged into the start request")
    ap.add_argument("--latency-only", action="store_true", help="print only accuracy + latency rows")
    args = ap.parse_args()

    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(args.data_dir, "*.json")))
    if args.scenarios != "all":
        names = args.scenarios.split(",")
    async def run_one(name):
        ref = json.load(open(os.path.join(args.data_dir, f"{name}.json")))
        langs = sorted({u["lang"] for u in ref["utterances"]})
        extra = {"language_hints": langs, **json.loads(args.extra_config)}
        if not args.no_translation and ref.get("translation_target"):
            extra["translation"] = {"type": "one_way", "target_language": ref["translation_target"]}
        elif not args.no_translation:
            extra["translation"] = ({"type": "two_way", "language_a": "ko", "language_b": "en"}
                                    if len(langs) > 1 else {"type": "one_way", "target_language": "en"})
        runs = await asyncio.gather(*[stream(os.path.join(args.data_dir, f"{name}.wav"), args.url, args.api_key,
                                             args.model, extra, args.chunk_ms) for _ in range(args.concurrency)])
        if args.dump:
            os.makedirs(args.dump, exist_ok=True)
            json.dump(runs[0], open(os.path.join(args.dump, f"{name}.json"), "w"), ensure_ascii=False)
        ms = [score(ev, ref) for ev in runs]
        agg = {k: st.mean(m[k] for m in ms if k in m) for k in ms[0]}
        return name, agg, check(agg)

    if args.parallel:
        rows = list(await asyncio.gather(*[run_one(n) for n in names]))
    else:
        rows = [await run_one(n) for n in names]

    keys = [k for k, _, _ in CRITERIA] + ["asr_err", "speakers_found", "tok_nonfinal_p50_s", "tok_nonfinal_p90_s",
                                          "tok_final_p50_s", "tok_final_p90_s", "utt_final_p50_s", "utt_final_p95_s"]
    if args.latency_only:
        keys = ["cer_ko", "wer_en", "asr_err", "speaker_acc", "chrf_to_en", "chrf_to_ko", "tok_nonfinal_p50_s",
                "tok_nonfinal_p90_s", "tok_final_p50_s", "tok_final_p90_s", "utt_final_p50_s", "utt_final_p95_s",
                "end_lag_p50_s", "end_lag_p95_s", "tr_first_lag_p50_s", "tr_done_lag_p95_s"]
    print(f"\nconcurrency={args.concurrency} url={args.url}")
    print(f"{'metric':22s} {'target':>10s} " + " ".join(f"{n[:18]:>18s}" for n, _, _ in rows))
    for k in keys:
        tgt = next((f"{op}{thr}" for kk, op, thr in CRITERIA if kk == k), "")
        cells = []
        for _, m, ok in rows:
            if k not in m:
                cells.append(f"{'-':>18s}")
                continue
            mark = "" if k not in ok else (" ✓" if ok[k] else " ✗")
            cells.append(f"{m[k]:>16.3f}{mark:2s}")
        print(f"{k:22s} {tgt:>10s} " + " ".join(cells))
    passed = sum(v for _, _, ok in rows for v in ok.values())
    total = sum(len(ok) for _, _, ok in rows)
    print(f"\nPASS {passed}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
