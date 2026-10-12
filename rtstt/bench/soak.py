"""Soak test: N concurrent long streams against a running server, watching memory and errors.

  python bench/soak.py --streams 16 --minutes 10 [--pid <server pid>] [--log /tmp/rtstt-gpu-multi.log]

Each stream loops the synthetic dialogs (ko_2spk, noisy, samegender, mix_3spk) at real time with
diarization + endpointing + one-way translation (half the streams) until `--minutes` of audio were
sent, then sends the end-of-audio frame and waits for `finished`. Every 30 s the server RSS, GPU
memory and token counts are sampled. Fails if any stream errors/disconnects, a stream does not
finish within 60 s of its audio end, the server log gained Tracebacks, RSS grew by more than
`--max-rss-growth-gb` after the first 90 s (session state allocation), or a stream produced fewer
than 80% of the spoken finals of another stream with the same audio + translation config.
Run it twice on the same server: the second run's start->end growth is the leak check.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000


def rss_gb(pid):
    try:
        return int(open(f"/proc/{pid}/status").read().split("VmRSS:")[1].split()[0]) / 1048576
    except (OSError, IndexError):
        return float("nan")


def gpu_mb():
    return gpu_stat()[0]


def cpu_s(pid):
    """CPU seconds consumed by the server process (user + system)."""
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return float("nan")


def gpu_stat():
    """(memory used MB, utilization %)"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split(",")
        return int(out[0]), int(out[1])
    except Exception:
        return -1, -1


def server_pid():
    out = subprocess.run(["pgrep", "-f", r"^\.venvs/.*python server\.py"], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


async def stream(i, url, audio, minutes, translate, stats):
    n = int(minutes * 60 * SR)
    x = np.tile(audio, int(np.ceil(n / len(audio))))[:n]
    cfg = {"model": "rtstt-v1", "audio_format": "pcm_s16le", "sample_rate": SR, "num_channels": 1,
           "language_hints": ["ko", "en"], "enable_endpoint_detection": True, "enable_speaker_diarization": True}
    if translate:
        cfg["translation"] = {"type": "one_way", "target_language": "en"}
    st = stats[i] = {"finals": 0, "nonfinals": 0, "ends": 0, "errors": 0, "finished": False, "end_lag": None,
                     "final_lags_ms": [], "nonfinal_lags_ms": [], "speakers": set(), "group": (len(audio), translate)}
    try:
        async with websockets.connect(url, max_size=2**24, ping_interval=20, ping_timeout=60) as ws:
            await ws.send(json.dumps(cfg))
            t0 = time.time()
            step = SR // 10
            audio_end = None

            async def sender():
                nonlocal audio_end
                for k in range(0, len(x), step):
                    await ws.send(x[k:k + step].tobytes())
                    await asyncio.sleep(max(0.0, t0 + (k + step) / SR - time.time()))
                audio_end = time.time()
                await ws.send("")

            task = asyncio.create_task(sender())
            async for m in ws:
                r = json.loads(m)
                if "error_code" in r:
                    st["errors"] += 1
                    st["last_error"] = r
                    break
                for t in r.get("tokens", []):
                    if t["text"] == "<end>":
                        st["ends"] += 1
                    elif t["text"] == "<fin>":
                        pass
                    elif t.get("is_final"):
                        if t.get("translation_status", "original") in ("original", "none"):
                            st["finals"] += 1   # spoken finals only (translation tokens vary per config)
                            st["final_lags_ms"].append((time.time() - t0) * 1000 - t["end_ms"])
                        if t.get("speaker"):
                            st["speakers"].add(t["speaker"])
                    else:
                        st["nonfinals"] += 1
                        if "end_ms" in t:
                            st["nonfinal_lags_ms"].append((time.time() - t0) * 1000 - t["end_ms"])
                if r.get("finished"):
                    st["finished"] = True
                    st["end_lag"] = time.time() - audio_end if audio_end else None
                    break
            if not task.done():
                task.cancel()
    except Exception as e:  # noqa: BLE001
        st["errors"] += 1
        st["last_error"] = repr(e)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765/transcribe-websocket")
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--pid", type=int)
    ap.add_argument("--log", default="/tmp/rtstt-gpu-multi.log")
    ap.add_argument("--max-rss-growth-gb", type=float, default=0.5, help="allowed RSS growth after the 90 s ramp")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    pid = a.pid or server_pid()
    names = ["ko_2spk", "ko_2spk_noisy", "ko_2spk_samegender", "mix_3spk"]
    audios = [sf.read(os.path.join(a.data_dir, n + ".wav"), dtype="int16")[0] for n in names]
    log_tb0 = open(a.log).read().count("Traceback") if a.log and os.path.exists(a.log) else 0
    rss0 = rss_gb(pid) if pid else float("nan")
    print(f"streams={a.streams} minutes={a.minutes} server pid={pid} RSS0={rss0:.2f} GB GPU0={gpu_mb()} MB")
    stats = {}
    tasks = [asyncio.create_task(stream(i, a.url, audios[i % len(audios)], a.minutes, i % 2 == 0, stats))
             for i in range(a.streams)]
    t0, samples, utils = time.time(), [], []
    cpu0 = cpu_s(pid) if pid else float("nan")
    while not all(t.done() for t in tasks):
        for _ in range(6):   # GPU utilization every 5 s, the rest every 30 s
            await asyncio.sleep(5)
            utils.append(gpu_stat()[1])
            if all(t.done() for t in tasks):
                break
        el = time.time() - t0
        fin = sum(s["finals"] for s in stats.values())
        mb, util = gpu_stat()
        samples.append((round(el), round(rss_gb(pid), 2) if pid else None, mb, fin))
        print(f"  t={el:5.0f}s RSS={samples[-1][1]} GB GPU={mb} MB util={np.mean(utils[-6:]):.0f}% finals={fin} "
              f"done={sum(t.done() for t in tasks)}/{len(tasks)}", flush=True)
        if el > a.minutes * 60 + 120:
            print("  timeout waiting for streams to finish")
            for t in tasks:
                t.cancel()
            break
    await asyncio.gather(*tasks, return_exceptions=True)
    fails = []
    # streams with the same audio + translation config must produce about the same number of spoken finals
    best = {}
    for s in stats.values():
        best[s["group"]] = max(best.get(s["group"], 0), s["finals"])
    all_lags = []
    for i, s in sorted(stats.items()):
        flag = ""
        ref = best[s["group"]]
        lags = np.array(s["final_lags_ms"]) if s["final_lags_ms"] else np.zeros(1)
        all_lags += s["final_lags_ms"]
        if s["errors"]:
            flag = f"ERROR {s.get('last_error')}"
        elif not s["finished"]:
            flag = "NOT FINISHED"
        elif s["end_lag"] is not None and s["end_lag"] > 60:
            flag = f"finished {s['end_lag']:.0f}s after audio end"
        elif ref and s["finals"] < 0.8 * ref:
            flag = f"only {s['finals']} finals vs {ref} for the same config"
        print(f"  stream {i:2d} finals={s['finals']:5d} nonfinals={s['nonfinals']:6d} ends={s['ends']:4d} "
              f"speakers={len(s['speakers'])} end_lag={s['end_lag'] if s['end_lag'] is None else round(s['end_lag'], 2)} "
              f"final_lag p50/p95/max={np.median(lags)/1000:.2f}/{np.percentile(lags, 95)/1000:.2f}/{lags.max()/1000:.1f}s {flag}")
        if flag:
            fails.append(f"stream{i}:{flag}")
    rss1 = rss_gb(pid) if pid else float("nan")
    tb = (open(a.log).read().count("Traceback") if a.log and os.path.exists(a.log) else 0) - log_tb0
    ramp = next((r for t, r, _, _ in samples if t >= 90 and r is not None), rss1)   # after sessions allocated their state
    if all_lags:
        al = np.array(all_lags)
        print(f"final lag over all streams: p50 {np.median(al)/1000:.2f}s p95 {np.percentile(al, 95)/1000:.2f}s max {al.max()/1000:.1f}s"
              f"  (single-stream reference: tok_final_p50 in bench/baseline.json)")
    nl = np.array([x for s in stats.values() for x in s["nonfinal_lags_ms"]])
    if len(nl):
        print(f"non-final (first-pass) lag: p50 {np.median(nl)/1000:.2f}s p90 {np.percentile(nl, 90)/1000:.2f}s")
    busy = [u for u in utils if u >= 0]
    if busy:
        print(f"GPU utilization: mean {np.mean(busy):.0f}% p90 {np.percentile(busy, 90):.0f}%")
    if pid:
        wall = time.time() - t0
        print(f"server CPU: {(cpu_s(pid) - cpu0) / wall * 100:.0f}% of one core on average ({(cpu_s(pid) - cpu0) / wall:.1f} cores)")
    print(f"RSS {rss0:.2f} -> {rss1:.2f} GB (start->end {rss1 - rss0:+.2f}; after 90 s ramp {rss1 - ramp:+.2f}), new Tracebacks in log: {tb}")
    print("samples (t, RSS GB, GPU MB, finals):", samples)
    if pid and rss1 - ramp > a.max_rss_growth_gb:
        fails.append(f"rss_growth_after_ramp {rss1 - ramp:.2f} GB")
    if tb:
        fails.append(f"tracebacks {tb}")
    print("\nRESULT: " + ("PASS" if not fails else "FAIL " + "; ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    asyncio.run(main())
