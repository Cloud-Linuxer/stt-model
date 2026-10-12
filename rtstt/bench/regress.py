"""Regression gate for a running server: protocol + edge cases + accuracy/latency thresholds.

  python bench/regress.py [--url ws://localhost:8765/transcribe-websocket] [--quick] [--baseline bench/baseline.json]

Runs, in order: test_protocol.py, edge_tests.py (skipped with --quick), run_bench on the synthetic
set (CER / speaker / latency thresholds, 2 repeats to catch run-to-run variance), the rapid-turn
set, and the Pipecat-style time-to-final. Compares against bench/baseline.json when present and
fails on any threshold or a regression beyond tolerance. Exit code 0 = pass.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TOL = {"cer_ko": 0.01, "wer_en": 0.02, "speaker_acc": -0.03, "end_lag_p50_s": 0.15, "end_lag_p95_s": 0.3,
       "tr_done_lag_p95_s": 0.5, "utt_final_p50_s": 0.2, "first_token_p50_s": 0.15}   # allowed worsening vs baseline (sign = direction of "worse")


def run(cmd, timeout=1800):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=os.path.dirname(HERE))
    return r.returncode, r.stdout + r.stderr


def parse_table(out):
    """metric rows of run_bench --latency-only -> {metric: [values...]}"""
    m = {}
    for line in out.splitlines():
        t = line.split()
        if not t or t[0] in ("metric", "PASS") or t[0].startswith("concurrency"):
            continue
        vals = []
        for x in t[1:]:
            if x[:2] in ("<=", ">=", "==") or x in ("✓", "✗", "-"):
                continue
            try:
                vals.append(float(x))
            except ValueError:
                pass
        if vals:
            m[t[0]] = vals
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765/transcribe-websocket")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--baseline", default=os.path.join(HERE, "baseline.json"))
    ap.add_argument("--save-baseline", action="store_true", help="save results as the new baseline (only if nothing failed)")
    ap.add_argument("--force-baseline", action="store_true", help="with --save-baseline: save even when the gate failed")
    a = ap.parse_args()
    fails, results = [], {}

    code, out = run([PY, os.path.join(HERE, "test_protocol.py"), "--url", a.url])
    n_pass = out.count("\nPASS") + out.startswith("PASS")
    print(f"[protocol] {n_pass} passed, exit {code}")
    if code != 0:
        fails.append("protocol"); print(out[-1500:])

    if not a.quick:
        code, out = run([PY, os.path.join(HERE, "edge_tests.py"), "--url", a.url], timeout=2400)
        print("[edge] " + (re.search(r"\d+/\d+ passed.*", out).group(0) if re.search(r"\d+/\d+ passed", out) else out[-300:]))
        if code != 0:
            fails.append("edge"); print("\n".join(l for l in out.splitlines() if l.startswith("FAIL")))

    agg = {}
    for rep in range(2):
        code, out = run([PY, os.path.join(HERE, "run_bench.py"), "--url", a.url])   # full table: every criterion row is visible
        m = parse_table(out)
        for k, v in m.items():
            agg.setdefault(k, []).append(v)
        if "PASS" in out:
            print(f"[synthetic run {rep + 1}] " + re.search(r"PASS \d+/\d+", out).group(0))
            for line in out.splitlines():
                if "✗" in line:
                    print("    " + " ".join(line.split()))   # the row(s) that missed run_bench's own criteria
    for k, runs in agg.items():
        per_run = [sum(v) / len(v) for v in runs]
        results[k] = sum(per_run) / len(per_run)
        spread = max(per_run) - min(per_run)
        if k in ("cer_ko", "wer_en") and spread > 0.01 or k == "speaker_acc" and spread > 0.03:
            fails.append(f"variance:{k}:{spread:.3f}")
    print("[synthetic] " + ", ".join(f"{k}={v:.3f}" for k, v in results.items()
                                      if k in ("cer_ko", "wer_en", "speaker_acc", "first_token_p50_s", "end_lag_p50_s", "end_lag_p95_s",
                                               "tr_done_lag_p95_s", "utt_final_p50_s")))
    thresholds = {"cer_ko": ("<=", 0.03), "wer_en": ("<=", 0.05), "speaker_acc": (">=", 0.95), "end_lag_p50_s": ("<=", 1.0),
                  "end_lag_p95_s": ("<=", 1.5), "tr_done_lag_p95_s": ("<=", 2.0)}
    for k, (op, thr) in thresholds.items():
        if k in results and not (results[k] <= thr if op == "<=" else results[k] >= thr):
            fails.append(f"threshold:{k}={results[k]:.3f} {op} {thr}")

    code, out = run([PY, os.path.join(HERE, "run_bench.py"), "--url", a.url, "--data-dir", os.path.join(HERE, "data_turns"),
                     "--parallel", "--no-translation", "--latency-only"])
    m = parse_table(out)
    if "speaker_acc" in m:
        results["turns_speaker_acc_min"] = min(m["speaker_acc"])
        results["turns_speaker_acc_mean"] = sum(m["speaker_acc"]) / len(m["speaker_acc"])
        print(f"[turns] speaker acc mean {results['turns_speaker_acc_mean']:.3f} min {results['turns_speaker_acc_min']:.3f}")
        if results["turns_speaker_acc_mean"] < 0.85:
            fails.append(f"threshold:turns_speaker_acc_mean={results['turns_speaker_acc_mean']:.3f} >= 0.85")

    code, out = run([PY, os.path.join(HERE, "finalize_latency.py"), "--url", a.url])
    mm = re.search(r"median (\d+) ms, P95 (\d+) ms", out)
    if mm:
        results["ttf_median_ms"], results["ttf_p95_ms"] = int(mm.group(1)), int(mm.group(2))
        print(f"[time-to-final] median {results['ttf_median_ms']} ms, p95 {results['ttf_p95_ms']} ms")
        if results["ttf_median_ms"] > 400:
            fails.append(f"threshold:ttf_median_ms={results['ttf_median_ms']} <= 400")

    if os.path.exists(a.baseline):   # also before overwriting with --save-baseline: a regression must be seen, not saved over
        base = json.load(open(a.baseline))
        for k, tol in TOL.items():
            if k in base and k in results:
                worse = (results[k] - base[k]) if tol > 0 else (base[k] - results[k])
                if worse > abs(tol):
                    fails.append(f"regression:{k} {base[k]:.3f} -> {results[k]:.3f}")
        for k in ("turns_speaker_acc_mean",):
            if k in base and k in results and base[k] - results[k] > 0.05:
                fails.append(f"regression:{k} {base[k]:.3f} -> {results[k]:.3f}")
        if "ttf_median_ms" in base and results.get("ttf_median_ms", 0) - base["ttf_median_ms"] > 60:
            fails.append(f"regression:ttf_median_ms {base['ttf_median_ms']} -> {results['ttf_median_ms']}")
    if a.save_baseline and (not fails or a.force_baseline):
        json.dump(results, open(a.baseline, "w"), indent=1)
        print(f"baseline saved to {a.baseline}")
    elif a.save_baseline:
        print("baseline NOT saved: fix or accept the failures (--force-baseline) first")
    print("\nRESULT: " + ("PASS" if not fails else "FAIL " + "; ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
