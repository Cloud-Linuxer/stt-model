"""Run a benchmark command and measure the resources the stack used meanwhile.

  python resource_probe.py --pids 123 456 -- python3 run_bench.py --concurrency 8 ...

Reports CPU cores used (process CPU time / wall time, all threads of the given PIDs), peak RSS,
average / peak GPU utilization and peak VRAM, then the command's own output.
"""
import argparse
import os
import subprocess
import threading
import time


def cpu_seconds(pid):
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")   # utime + stime


def rss_gb(pid):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", nargs="+", type=int, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    gpu, rss, stop = [], {p: 0.0 for p in a.pids}, threading.Event()

    def sample():
        while not stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip().split(",")
            if len(out) == 2:
                gpu.append((float(out[0]), float(out[1])))
            for p in a.pids:
                rss[p] = max(rss[p], rss_gb(p))
            stop.wait(0.5)
    th = threading.Thread(target=sample, daemon=True)
    c0, t0 = {p: cpu_seconds(p) for p in a.pids}, time.time()
    th.start()
    res = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    stop.set()
    th.join()
    cores = {p: (cpu_seconds(p) - c0[p]) / wall for p in a.pids}
    print(f"RESOURCES wall {wall:.0f}s | CPU cores used: " + ", ".join(f"pid{p} {c:.1f}" for p, c in cores.items())
          + f" (total {sum(cores.values()):.1f}) | peak RSS {sum(rss.values()):.1f} GB"
          + (f" | GPU util avg {sum(g for g, _ in gpu) / len(gpu):.0f}% peak {max(g for g, _ in gpu):.0f}%"
             f" | VRAM peak {max(m for _, m in gpu) / 1024:.1f} GB (whole GPU)" if gpu else ""))
    print(res.stdout[-6000:])


if __name__ == "__main__":
    main()
