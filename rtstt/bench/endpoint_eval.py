"""Endpoint quality from run_bench dumps: <end> lag at true turn ends, false <end> inside turns, merged turns.

  python endpoint_eval.py NAME=DUMP_DIR[,DUMP_DIR...] ...
Turn ends: synthetic set = every scripted utterance end; real mixes = speaker changes.
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = {"ko_2spk": "data", "ko_2spk_noisy": "data", "ko_2spk_samegender": "data", "mix_3spk": "data"}


def ref_for(name, dump_dir):
    for d in ("data", "data_yt", "data_yt2"):
        f = os.path.join(HERE, d, f"{name}.json")
        if os.path.exists(f) and (d != "data_yt" or "yt" in dump_dir or True):
            return json.load(open(f)), d
    return None, None


def evaluate(dirs):
    lags, false_in, merged, turns = [], 0, 0, 0
    for dd in dirs:
        for f in sorted(glob.glob(os.path.join(dd, "*.json"))):
            name = os.path.basename(f)[:-5]
            ref, d = (json.load(open(os.path.join(HERE, "data", f"{name}.json"))), "data") if os.path.exists(os.path.join(HERE, "data", f"{name}.json")) else (None, None)
            if ref is None:
                for cand in ("data_yt2", "data_yt"):
                    p = os.path.join(HERE, cand, f"{name}.json")
                    if os.path.exists(p) and os.path.basename(dd).endswith(cand.replace("data_", "")):
                        ref, d = json.load(open(p)), cand
            if ref is None:
                continue
            utts = ref["utterances"]
            ends = [w for w, t in json.load(open(f)) if t["text"] == "<end>"]
            # turn boundaries
            t_ends = []
            for k, u in enumerate(utts):
                nxt = utts[k + 1] if k + 1 < len(utts) else None
                if d == "data" or nxt is None or nxt["speaker"] != u["speaker"]:
                    t_ends.append((u["end"], nxt["start"] if nxt else u["end"] + 3))
            turns += len(t_ends)
            for e, nxt_start in t_ends:
                c = [w for w in ends if e - 0.3 <= w <= nxt_start + 0.3]
                if c:
                    lags.append(c[0] - e)
                else:
                    merged += 1
            # false ends: inside a turn, >0.3 s from any turn boundary
            for w in ends:
                if any(u["start"] + 0.3 < w < u["end"] - 0.3 for u in utts):
                    false_in += 1
    return lags, false_in, merged, turns


if __name__ == "__main__":
    print(f"{'system':14s} {'turns':>5s} {'<end> lag p50/p95':>18s} {'merged (missed) turns':>22s} {'false <end> inside turns':>24s}")
    for arg in sys.argv[1:]:
        name, dirs = arg.split("=", 1)
        lags, fi, mg, n = evaluate(dirs.split(","))
        print(f"{name:14s} {n:5d} {st.median(lags):8.2f}/{sorted(lags)[int(0.95*(len(lags)-1))]:.2f}s {mg:14d} ({100*mg/max(1,n):.0f}%) {fi:24d}")
