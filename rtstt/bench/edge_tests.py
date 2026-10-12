"""Edge-case / robustness tests against a running server (default ws://localhost:8765).

Each test opens its own connection; after each test the server must still answer a trivial
session (liveness). Prints PASS/FAIL per test and exits non-zero on any failure.

  python bench/edge_tests.py [--url ws://host:port/transcribe-websocket] [--only name,name]
"""
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
WAV = os.path.join(HERE, "data", "ko_2spk.wav")
BASE = {"model": "rtstt-v1", "audio_format": "pcm_s16le", "sample_rate": 16000, "num_channels": 1,
        "language_hints": ["ko"], "enable_endpoint_detection": True, "enable_speaker_diarization": True}
results = []


def record(name, ok, info=""):
    results.append((name, ok, info))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  [{info}]" if info else ""), flush=True)


async def collect(ws, timeout=60):
    out, t0 = [], time.time()
    try:
        while time.time() - t0 < timeout:
            m = await asyncio.wait_for(ws.recv(), timeout=max(0.1, timeout - (time.time() - t0)))
            r = json.loads(m)
            out.append(r)
            if r.get("finished") or r.get("error_code"):
                break
    except (asyncio.TimeoutError, websockets.ConnectionClosed):
        pass
    return out


def tokens(res):
    return [t for r in res for t in r.get("tokens", [])]


def spoken_final_text(res):
    return "".join(t["text"] for t in tokens(res) if t["is_final"] and "start_ms" in t and not t["text"].startswith("<"))


async def send_realtime(ws, audio, sr, chunk_ms=100, speed=1.0, sink=None):
    """Paced sending; concurrently drains incoming messages into `sink` (a list) like a real client."""
    chunk = int(sr * chunk_ms / 1000)
    t0 = time.time()
    stop = asyncio.Event()

    async def reader():
        while not stop.is_set():
            try:
                m = await asyncio.wait_for(ws.recv(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                return
            if sink is not None:
                sink.append(json.loads(m))
    rt = asyncio.create_task(reader())
    for i in range(0, len(audio), chunk):
        await ws.send(audio[i:i + chunk].tobytes())
        await asyncio.sleep(max(0, t0 + (i + chunk) / sr / speed - time.time()))
    stop.set()
    await rt


async def liveness(url):
    try:
        async with websockets.connect(url, open_timeout=5) as ws:
            await ws.send(json.dumps(BASE))
            await ws.send(np.zeros(1600, np.int16).tobytes())
            await ws.send("")
            res = await collect(ws, timeout=15)
            return bool(res) and res[-1].get("finished") is True
    except Exception as e:
        print("   liveness error:", repr(e)[:120])
        return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765/transcribe-websocket")
    ap.add_argument("--only", default="")
    ap.add_argument("--metrics-url", default="http://localhost:8766", help="server --metrics-port base URL ('' skips)")
    ap.add_argument("--limits", action="store_true",
                    help="only the limit tests; needs a server started with --max-sessions 2 --idle-timeout-s 3 --max-stream-min 0.05")
    a = ap.parse_args()
    url = a.url
    if a.limits:
        return await limit_tests(url, sf.read(WAV, dtype="int16")[0])
    audio, sr = sf.read(WAV, dtype="int16")
    only = set(a.only.split(",")) if a.only else None

    async def run(name, coro):
        if only and name not in only:
            return
        try:
            await asyncio.wait_for(coro, timeout=240)
        except Exception as e:
            record(name, False, f"exception {type(e).__name__}: {str(e)[:120]}")
        alive = await liveness(url)
        if not alive:
            record(name + " (server alive after)", False)

    # 1. silence only
    async def t_silence():
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(BASE))
            pre = []
            await send_realtime(ws, np.zeros(sr * 4, np.int16), sr, sink=pre)
            await ws.send("")
            res = pre + await collect(ws)
            toks = [t for t in tokens(res) if "start_ms" in t]
            record("silence-only stream", res and res[-1].get("finished") is True and not toks,
                   f"finished={res[-1].get('finished') if res else None} spoken tokens={len(toks)}")
    await run("silence-only stream", t_silence())

    # 2. very short audio (200 ms) then end
    async def t_short():
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(BASE))
            await ws.send(audio[: sr // 5].tobytes())
            await ws.send("")
            res = await collect(ws)
            record("200 ms stream", bool(res) and res[-1].get("finished") is True)
    await run("200 ms stream", t_short())

    # 3. immediate end (no audio)
    async def t_noaudio():
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(BASE))
            await ws.send("")
            res = await collect(ws)
            record("end without audio", bool(res) and res[-1].get("finished") is True)
    await run("end without audio", t_noaudio())

    # 4. burst: whole file at once (faster than real time)
    async def t_burst():
        async with websockets.connect(url, max_size=2**24) as ws:
            await ws.send(json.dumps(BASE))
            for i in range(0, len(audio), sr):
                await ws.send(audio[i:i + sr].tobytes())
            await ws.send("")
            t0 = time.time()
            res = await collect(ws, timeout=120)
            text = spoken_final_text(res)
            record("burst send (50 s at once)", bool(res) and res[-1].get("finished") is True and len(text) > 200,
                   f"{len(text)} chars, drained in {time.time()-t0:.1f}s")
    await run("burst send (50 s at once)", t_burst())

    # 5. odd byte count frames (int16 misaligned)
    async def t_odd():
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(BASE))
            raw = audio[: sr * 3].tobytes()
            for i in range(0, len(raw), 3001):          # odd-sized frames
                await ws.send(raw[i:i + 3001])
                await asyncio.sleep(0.01)
            await ws.send("")
            res = await collect(ws)
            text = spoken_final_text(res)
            record("odd-sized binary frames", bool(res) and res[-1].get("finished") is True and len(text) > 5,
                   f"text={text[:40]!r}")
    await run("odd-sized binary frames", t_odd())

    # 6. finalize spam every 200 ms
    async def t_finspam():
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(BASE))
            chunk = sr // 10
            for i in range(0, sr * 12, chunk):
                await ws.send(audio[i:i + chunk].tobytes())
                if (i // chunk) % 2 == 1:
                    await ws.send(json.dumps({"type": "finalize"}))
                await asyncio.sleep(0.1)
            await ws.send("")
            res = await collect(ws)
            fins = [t for t in tokens(res) if t["text"] == "<fin>"]
            fin = [t for t in tokens(res) if t["is_final"] and "start_ms" in t]
            mono = all(fin[i]["start_ms"] <= fin[i + 1]["start_ms"] for i in range(len(fin) - 1))
            record("finalize every 200 ms", bool(res) and res[-1].get("finished") is True and mono,
                   f"<fin> {len(fins)}, finals {len(fin)}, monotonic={mono}")
    await run("finalize every 200 ms", t_finspam())

    # 7. 30 s pause mid-stream with keepalives
    async def t_pause():
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(BASE))
            pre = []
            await send_realtime(ws, audio[: sr * 5], sr, sink=pre)
            for _ in range(6):
                await ws.send(json.dumps({"type": "keepalive"}))
                await asyncio.sleep(5)
            await send_realtime(ws, audio[sr * 5: sr * 10], sr, sink=pre)
            await ws.send("")
            res = pre + await collect(ws)
            text = spoken_final_text(res)
            record("30 s idle gap with keepalive", bool(res) and res[-1].get("finished") is True and len(text) > 20, f"{len(text)} chars")
    await run("30 s idle gap with keepalive", t_pause())

    # 8. client disconnects mid-stream (no end frame)
    async def t_disconnect():
        ws = await websockets.connect(url)
        await ws.send(json.dumps(BASE))
        await send_realtime(ws, audio[: sr * 3], sr, sink=[])
        await ws.close()
        await asyncio.sleep(1)
        record("client disconnect mid-stream", True, "server must stay alive (checked next)")
    await run("client disconnect mid-stream", t_disconnect())

    # 9. invalid field types / unknown fields / unknown language code
    async def t_badcfg():
        outcomes = []
        for cfg, label in (({**BASE, "sample_rate": "16000"}, "sample_rate as string"),
                           ({**BASE, "language_hints": ["xx"]}, "unknown language code"),
                           ({**BASE, "some_unknown_field": 1, "client_reference_id": "abc"}, "unknown fields"),
                           ({**BASE, "language_hints": []}, "empty language_hints"),
                           ({**BASE, "translation": {"type": "two_way", "language_a": "ko", "language_b": "ko"}}, "two_way same language"),
                           ([1, 2, 3], "config is a JSON array"),
                           ({**BASE, "translation": "one_way"}, "translation as string"),
                           ({**BASE, "context": "just text"}, "context as string"),
                           ({**BASE, "num_channels": "two"}, "num_channels as word"),
                           ({**BASE, "max_endpoint_delay_ms": "abc", "endpoint_sensitivity": None}, "endpoint knobs wrong type"),
                           ({**BASE, "language_hints": [None, {"a": 1}]}, "language_hints with non-strings")):
            try:
                async with websockets.connect(url) as ws:
                    await ws.send(json.dumps(cfg))
                    await ws.send(audio[: sr * 2].tobytes())
                    await ws.send("")
                    res = await collect(ws, timeout=30)
                    ok = bool(res) and (res[-1].get("finished") is True or res[-1].get("error_code") == 400)
                    outcomes.append((label, ok, "finished" if res and res[-1].get("finished") else (res[-1].get("error_message", "?")[:40] if res else "no response")))
            except Exception as e:
                outcomes.append((label, False, repr(e)[:60]))
        for label, ok, info in outcomes:
            record(f"config: {label}", ok, info)
    await run("config: sample_rate as string", t_badcfg())

    # 10. sample rates / formats
    async def t_formats():
        for cfg, conv, label in (({**BASE, "sample_rate": 8000}, lambda x: x[::2], "8 kHz"),
                                 ({**BASE, "sample_rate": 44100}, lambda x: np.interp(np.arange(0, len(x), 16000 / 44100), np.arange(len(x)), x).astype(np.int16), "44.1 kHz"),
                                 ({**BASE, "audio_format": "pcm_f32le"}, lambda x: (x.astype(np.float32) / 32768), "pcm_f32le")):
            x = conv(audio[: sr * 6])
            async with websockets.connect(url, max_size=2**24) as ws:
                await ws.send(json.dumps(cfg))
                await ws.send(x.tobytes())
                await ws.send("")
                res = await collect(ws, timeout=60)
                text = spoken_final_text(res)
                record(f"format: {label}", bool(res) and res[-1].get("finished") is True and "안녕" in text, f"{text[:30]!r}")
        # stereo (L = speech, R = quiet copy) sent in frames that split a 4-byte stereo sample
        mono = audio[: sr * 6]
        stereo = np.stack([mono, (mono * 0.3).astype(np.int16)], axis=1).reshape(-1)
        raw = stereo.tobytes()
        async with websockets.connect(url, max_size=2**24) as ws:
            await ws.send(json.dumps({**BASE, "num_channels": 2}))
            for i in range(0, len(raw), 6002):
                await ws.send(raw[i:i + 6002])
            await ws.send("")
            res = await collect(ws, timeout=60)
            text = spoken_final_text(res)
            record("format: stereo, frames splitting samples", bool(res) and res[-1].get("finished") is True and "안녕" in text, f"{text[:30]!r}")
    await run("format: 8 kHz", t_formats())

    # 10b. garbage text frames mid-stream -> 400, not a silent close
    async def t_garbage():
        outcomes = []
        for frame, label in (("not json", "non-JSON text frame"), ("[1,2]", "JSON array control"), ('{"type":"bogus"}', "unknown control type")):
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps(BASE))
                await ws.send(audio[: sr].tobytes())
                await ws.send(frame)
                res = await collect(ws, timeout=20)
                outcomes.append((label, bool(res) and res[-1].get("error_code") == 400, res[-1].get("error_message", "?")[:50] if res else "no response"))
        for label, ok, info in outcomes:
            record(f"control: {label}", ok, info)
    await run("control: non-JSON text frame", t_garbage())

    # 11. continuous speech, no pauses, 60 s (concatenate speech-only parts)
    async def t_continuous():
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        af = audio.astype(np.float32) / 32768
        segs = get_speech_timestamps(af, VadOptions(min_silence_duration_ms=150))
        speech = np.concatenate([af[s["start"]:s["end"]] for s in segs])
        unit_s = len(speech) / sr
        ref = json.load(open(WAV[:-4] + ".json"))
        norm_len = lambda t: sum(c.isalnum() for c in t)
        expected = norm_len(" ".join(u["text"] for u in ref["utterances"])) * 60 / unit_s
        speech = np.tile(speech, int(np.ceil(60 * sr / len(speech))))[: 60 * sr]
        x = (speech * 32767).astype(np.int16)
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(BASE))
            t0 = time.time()
            pre = []
            await send_realtime(ws, x, sr, sink=pre)
            await ws.send("")
            res = pre + await collect(ws, timeout=120)
            fin = [(r, t) for r in res for t in r.get("tokens", []) if t["is_final"] and "start_ms" in t and not t["text"].startswith("<")]
            text = "".join(t["text"] for _, t in fin)
            got = norm_len(text)
            record("60 s continuous speech", bool(res) and res[-1].get("finished") is True and got >= 0.75 * expected,
                   f"{got} chars of ~{expected:.0f} expected, finals {len(fin)}, drain {time.time()-t0-60:.1f}s after audio end")
    await run("60 s continuous speech", t_continuous())

    # 12. 40 sequential connections (leak check is done by the caller via RSS)
    async def t_many():
        ok = 0
        for _ in range(40):
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps(BASE))
                await ws.send(audio[: sr].tobytes())
                await ws.send("")
                res = await collect(ws, timeout=30)
                ok += bool(res) and res[-1].get("finished") is True
        record("40 sequential sessions", ok == 40, f"{ok}/40 finished")
    await run("40 sequential sessions", t_many())

    # 13. context edge cases
    async def t_ctx():
        for ctx, label in (({"general": [], "terms": [], "translation_terms": []}, "empty lists"),
                           ({"general": "문자열", "terms": ["", "  "], "text": ""}, "string general, blank terms"),
                           ({"terms": ["😀 이모지", "A" * 500]}, "emoji and a 500-char term"),
                           ({"text": "x" * 9990}, "text at the limit")):
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({**BASE, "context": ctx, "translation": {"type": "one_way", "target_language": "en"}}))
                await ws.send(audio[: sr * 3].tobytes())
                await ws.send("")
                res = await collect(ws, timeout=60)
                record(f"context: {label}", bool(res) and res[-1].get("finished") is True,
                       "" if res and res[-1].get("finished") else (res[-1].get("error_message", "?")[:50] if res else "no response"))
    await run("context: empty lists", t_ctx())

    # 14. /health and /metrics (plain HTTP on --metrics-port)
    async def t_metrics():
        import urllib.request
        h = json.loads(urllib.request.urlopen(a.metrics_url + "/health", timeout=5).read())
        m = urllib.request.urlopen(a.metrics_url + "/metrics", timeout=5).read().decode()
        want = ("rtstt_sessions_active", "rtstt_sessions_total", "rtstt_audio_seconds_total")
        record("metrics: /health and /metrics", h.get("status") == "ok" and all(w in m for w in want),
               f"health={h} metrics_lines={len(m.splitlines())}")
    if a.metrics_url:
        await run("metrics: /health and /metrics", t_metrics())

    finish()


def finish():
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


async def limit_tests(url, audio):
    """Server-side limits (server started with --max-sessions 2 --idle-timeout-s 3 --max-stream-min 0.05)."""
    sr = 16000

    async def expect_error(ws, code, etype, timeout=30):
        res = await collect(ws, timeout=timeout)
        err = next((r for r in res if r.get("error_code")), {})
        return err.get("error_code") == code and err.get("error_type") == etype, err.get("error_message", "no error")[:70]

    # 429 limit_exceeded: third concurrent session
    ws1 = await websockets.connect(url)
    ws2 = await websockets.connect(url)
    await ws1.send(json.dumps(BASE))
    await ws2.send(json.dumps(BASE))
    await ws1.send(audio[:sr].tobytes())
    await ws2.send(audio[:sr].tobytes())
    await asyncio.sleep(0.5)
    async with websockets.connect(url) as ws3:
        await ws3.send(json.dumps(BASE))
        ok, info = await expect_error(ws3, 429, "limit_exceeded", timeout=10)
        record("limits: 429 limit_exceeded on the 3rd concurrent stream", ok, info)
    await ws1.send(""), await ws2.send("")
    await collect(ws1), await collect(ws2)
    await ws1.close(), await ws2.close()

    # 408 request_timeout: no audio or keepalive for > 3 s
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(BASE))
        await ws.send(audio[:sr].tobytes())
        t0 = time.time()
        ok, info = await expect_error(ws, 408, "request_timeout", timeout=15)
        record("limits: 408 request_timeout after idle", ok, f"{info} after {time.time() - t0:.1f}s")

    # keepalive prevents the idle timeout
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(BASE))
        await ws.send(audio[:sr].tobytes())
        for _ in range(4):
            await asyncio.sleep(1.5)
            await ws.send(json.dumps({"type": "keepalive"}))
        await ws.send("")
        res = await collect(ws, timeout=15)
        record("limits: keepalive keeps the idle stream open", bool(res) and res[-1].get("finished") is True)

    # 413 max_duration_reached: 0.05 min = 3 s cap, finals delivered before the error
    async with websockets.connect(url, max_size=2**24) as ws:
        await ws.send(json.dumps(BASE))
        pre = []
        try:
            await send_realtime(ws, audio[: sr * 5], sr, sink=pre)
        except websockets.ConnectionClosed:
            pass   # the server closes right after the 413 error
        res = pre + await collect(ws, timeout=20)
        err = next((r for r in res if r.get("error_code")), {})
        text = spoken_final_text(res)
        record("limits: 413 max_duration_reached with finals first", err.get("error_type") == "max_duration_reached" and len(text) > 3,
               f"{err.get('error_message', 'no error')[:50]} text={text[:30]!r}")
    finish()


if __name__ == "__main__":
    asyncio.run(main())
