"""Time-to-final-segment benchmark, same method as pipecat-ai/stt-benchmark.

The client streams 16 kHz PCM in 20 ms frames in real time and, 200 ms after each reference
utterance ends (an ideal client-side VAD "stop"), sends {"type": "finalize"}. Latency is measured
from the true end of speech to the arrival of the <fin> marker (all final tokens for the utterance
arrive before it). Server-side endpoint detection is off, as in the Pipecat setup.

  python finalize_latency.py [--url ws://...] [--scenarios ko_2spk,mix_3spk] [--concurrency N]
"""
import argparse
import asyncio
import json
import os
import time

import numpy as np
import soundfile as sf
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


async def run(path, ref, url, api_key, chunk_ms, vad_stop_ms):
    audio, sr = sf.read(path, dtype="int16")
    langs = sorted({u["lang"] for u in ref["utterances"]})
    cfg = {"model": "rtstt-v1", "audio_format": "pcm_s16le", "sample_rate": sr, "num_channels": 1,
           "language_hints": langs, "enable_endpoint_detection": False}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    fins, finals = [], []
    async with websockets.connect(url, additional_headers=headers, max_size=2**22) as ws:
        await ws.send(json.dumps(cfg))
        t0 = time.time()
        chunk = int(sr * chunk_ms / 1000)
        marks = [u["end"] + vad_stop_ms / 1000 for u in ref["utterances"]]

        async def sender():
            k = 0
            for i in range(0, len(audio), chunk):
                await ws.send(audio[i:i + chunk].tobytes())
                pos = (i + chunk) / sr
                while k < len(marks) and pos >= marks[k]:
                    await ws.send(json.dumps({"type": "finalize"}))
                    k += 1
                await asyncio.sleep(max(0, t0 + pos - time.time()))
            await ws.send("")

        task = asyncio.create_task(sender())
        async for msg in ws:
            r = json.loads(msg)
            now = time.time() - t0
            for t in r.get("tokens", []):
                if t["text"] == "<fin>":
                    fins.append(now)
                elif t.get("is_final") and t.get("translation_status") != "translation" and not t["text"].startswith("<"):
                    finals.append((now, t["text"]))
            if r.get("finished"):
                break
        await task
    # pair each finalize with the next <fin> (one <fin> per finalize request)
    lat = [f - u["end"] for f, u in zip(fins, ref["utterances"])]
    text = "".join(t for _, t in finals)
    r_txt = norm(" ".join(u["text"] for u in ref["utterances"]))
    return lat, text, r_txt


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765/transcribe-websocket")
    ap.add_argument("--api-key")
    ap.add_argument("--scenarios", default="ko_2spk,ko_2spk_noisy,ko_2spk_samegender,mix_3spk")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--chunk-ms", type=int, default=20)
    ap.add_argument("--vad-stop-ms", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()
    jobs = []
    for name in a.scenarios.split(","):
        ref = json.load(open(os.path.join(a.data_dir, f"{name}.json")))
        for _ in range(a.concurrency):
            jobs.append(run(os.path.join(a.data_dir, f"{name}.wav"), ref, a.url, a.api_key, a.chunk_ms, a.vad_stop_ms))
    res = await asyncio.gather(*jobs)
    lat = [x for r in res for x in r[0]]
    lat_ms = np.array(lat) * 1000
    print(f"utterances={len(lat)} concurrency={a.concurrency} chunk={a.chunk_ms}ms vad_stop={a.vad_stop_ms}ms")
    print(f"time to final segment (true end of speech -> <fin>): median {np.median(lat_ms):.0f} ms, "
          f"P95 {np.percentile(lat_ms, 95):.0f} ms, P99 {np.percentile(lat_ms, 99):.0f} ms, max {lat_ms.max():.0f} ms")
    print(f"  server part (minus the {a.vad_stop_ms} ms client VAD wait): median {np.median(lat_ms) - a.vad_stop_ms:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
