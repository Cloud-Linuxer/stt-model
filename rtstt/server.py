"""rtstt: real-time WebSocket STT server on open models (transcription, speaker diarization, translation).

ASR backends (asr_backends.py):
  --asr nemotron  CPU only: Nemotron-3.5 streaming (sherpa-onnx int8), optional Qwen3-ASR
                  second pass for final tokens (--final qwen3-0.6b | llamacpp)
  --asr whisper   faster-whisper (GPU), non-final tokens by re-decoding the current piece

Wire protocol (the `model` value in the start config is accepted as-is):
  - first text frame: JSON config (model, audio_format, sample_rate, num_channels,
    language_hints, context, enable_endpoint_detection, max_endpoint_delay_ms)
  - binary frames: raw audio
  - {"type": "finalize"} -> finalize pending audio, emit "<fin>"
  - {"type": "keepalive"} -> ignored
  - empty text frame -> finalize, send {"finished": true}, close
Responses: {"tokens": [...], "final_audio_proc_ms", "total_audio_proc_ms"}.

Also supported:
  - enable_speaker_diarization -> "speaker": "1".. on every spoken token (online clustering)
  - translation one_way / two_way -> translation tokens streamed after each finalized
    segment (translation_status "original"/"translation"/"none", source_language)
Not supported (ignored): endpoint_sensitivity, endpoint_latency_adjustment_level.
"""
import argparse
import asyncio
import collections
import glob
import json
import logging
import os
import signal
import time
import uuid

import numpy as np
import websockets
from faster_whisper.vad import VadOptions

from asr_backends import LlamaCppQwenAsr, NemotronBackend, SherpaQwenAsr, WhisperBackend, WhisperSecondPass
from diarize import DEFAULT_MODEL as DIAR_MODEL, Embedder, OnlineDiarizer
from eot import DEFAULT_MODEL as EOT_MODEL, NAMO_DIR, CombinedDetector, NamoTextDetector, SmartTurnDetector
from routing import LANGS, MadladTranslator, MultiBackend, OmniEngine, RoutedSecondPass, RoutedTranslator
from translate import Translator
from agc import AGC
from vad import StreamVAD

HERE = os.path.dirname(os.path.abspath(__file__))

SR = 16000
log = logging.getLogger("rtstt")

SUPPORTED_FORMATS = {"pcm_s16le": np.int16, "pcm_f32le": np.float32}
METRICS = collections.Counter()   # Prometheus counters (sessions_total, errors_total{code}, audio_seconds_total)
ACTIVE = set()                    # request ids of open sessions


class AudioBuffer:
    """Stream audio addressed by ABSOLUTE sample index; samples older than trim() are dropped.
    Behaves like the full array for a[start:end] and len(a), so a 300-minute stream costs ~1 MB
    per retained minute instead of 1.1 GB."""

    def __init__(self, capacity=SR * 90):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self.offset = 0   # absolute index of _buf[0]
        self.n = 0        # absolute length

    def __len__(self):
        return self.n

    def append(self, x):
        used = self.n - self.offset
        if used + len(x) > len(self._buf):
            grow = np.zeros(max(len(self._buf), used + len(x)), dtype=np.float32)
            grow[:used] = self._buf[:used]
            self._buf = grow
        self._buf[used:used + len(x)] = x
        self.n += len(x)

    def __getitem__(self, sl):
        if not isinstance(sl, slice):
            raise TypeError("AudioBuffer supports slices only")
        start, stop, step = sl.indices(self.n)
        if step != 1:
            raise ValueError("step not supported")
        start = max(start, self.offset)
        stop = max(stop, start)
        return self._buf[start - self.offset:stop - self.offset]

    def trim(self, keep_from):
        """Forget samples before absolute index keep_from."""
        keep_from = min(max(keep_from, self.offset), self.n)
        k = keep_from - self.offset
        if k <= 0:
            return
        used = self.n - self.offset
        self._buf[:used - k] = self._buf[k:used]
        self.offset = keep_from
MAX_CONTEXT_CHARS = 10000   # about 8,000 tokens


def context_chars(ctx):
    """Characters of text content in a context object (keys/values/terms), ignoring JSON syntax."""
    if isinstance(ctx, dict):
        return sum(context_chars(v) for v in ctx.values())
    if isinstance(ctx, list):
        return sum(context_chars(v) for v in ctx)
    return len(str(ctx)) if ctx is not None else 0


def degenerate(src, tgt):
    """Translation that is empty or implausibly short for its source."""
    n_src = sum(c.isalnum() for c in src)
    n_tgt = sum(c.isalnum() for c in tgt)
    return n_src >= 12 and n_tgt < 0.25 * n_src


def parse_context(ctx):
    """Start-request `context` -> (Whisper initial prompt, hotwords, translation glossary, general summary).
    general: [{key, value}] (or a plain string), text: str, terms: [str], translation_terms: [{source, target}]."""
    ctx = ctx or {}
    general = ctx.get("general") or []
    pairs = [general] if isinstance(general, str) else [
        f"{str(g.get('key', '')).strip()}: {str(g.get('value', '')).strip()}".strip(": ")
        for g in general if isinstance(g, dict)]
    general_s = "; ".join(x for x in pairs if x) or None
    terms = [str(t).strip() for t in (ctx.get("terms") or []) if str(t).strip()]
    text = str(ctx.get("text") or "").strip()
    # Whisper prompt = short domain hints only. `text` is deliberately NOT fed to the ASR prompt: on
    # real videos a description paragraph as prompt made CER 4-5 pp worse (style leakage / hallucination)
    prompt = (general_s + ".") if general_s else None
    _ = text   # kept for the translation side / future use
    hotwords = ", ".join(terms) or None
    glossary = [t for t in (ctx.get("translation_terms") or [])
                if isinstance(t, dict) and t.get("source") and t.get("target")] or None
    return prompt, hotwords, glossary, general_s


class Session:
    def __init__(self, ws, engine, cfg, args, embedder, translator, eot=None):
        self.ws, self.engine, self.args = ws, engine, args
        self.eot = eot                 # semantic end-of-turn detector (None = silence-based only)
        self.turn_start = 0            # sample where the current turn began (after the last <end>)
        self.turn_text = []            # final token texts of the current turn (for the text detector)
        self.eot_key, self.eot_prob = None, None
        self.cfg = cfg
        self.dtype = SUPPORTED_FORMATS[cfg["audio_format"]]
        self.in_rate = int(cfg["sample_rate"])
        self.channels = int(cfg["num_channels"])
        hints = cfg.get("language_hints") or []
        self.language = hints[0] if len(hints) == 1 else None
        self.ctx_prompt, self.ctx_hotwords, self.tr_glossary, self.tr_general = parse_context(cfg.get("context"))
        self.asr = engine.open(self.language, hints if len(hints) > 1 else None, self.ctx_prompt, self.ctx_hotwords)
        self.step_s = args.step_s or getattr(self.asr, "step_s", None) or engine.step_s
        self.endpointing = bool(cfg.get("enable_endpoint_detection"))
        # longest a word may stay non-final in continuous speech (client option
        # max_non_final_tokens_duration_ms, 360-6000): the open piece is cut at a pause-like point
        self.max_open = SR * max(0.36, min(6.0, cfg.get("max_non_final_tokens_duration_ms",
                                                          args.max_nonfinal_ms) / 1000))
        if "max_non_final_tokens_duration_ms" not in cfg:
            if getattr(self.asr, "max_open_s", None):
                self.max_open = int(SR * self.asr.max_open_s)
            elif self.asr.progressive:
                self.max_open = SR * 30   # progressive finalization keeps the full context instead of cutting
        # client endpoint options: endpoint_latency_adjustment_level 0..3 (lower latency), endpoint_sensitivity
        # -1..1 (higher = earlier endpoints); max_endpoint_delay_ms caps the silence we wait for.
        level = max(0, min(3, int(cfg.get("endpoint_latency_adjustment_level", 0))))
        sens = max(-1.0, min(1.0, float(cfg.get("endpoint_sensitivity", 0.0))))
        base = args.endpoint_silence_ms * (1.0, 0.8, 0.6, 0.4)[level] * (1 - 0.3 * sens)
        cap = int(cfg.get("max_endpoint_delay_ms", 2000))
        self.endpoint_ms = min(base, cap)
        # "semantic" shortcut: the open hypothesis already ends a sentence -> wait less
        self.endpoint_ms_sentence = min(base * args.endpoint_sentence_factor, cap)
        # semantic detector tiers: confident-complete -> end at eot_fast_ms of silence,
        # confident-incomplete -> hold up to eot_hold_ms (scaled by the same knobs, capped)
        self.eot_hold_ms = min(args.eot_hold_ms * (1.0, 0.8, 0.6, 0.4)[level] * (1 - 0.3 * sens), cap)

        self.pending_bytes = b""   # partial sample left over from the previous binary frame
        self._audio = AudioBuffer()   # stream audio, 16 kHz mono, absolute indices, old audio trimmed
        self.last_trim = 0
        # the VAD sees level-normalized audio (very quiet speakers); ASR models get the raw audio
        self.vad = StreamVAD(AGC(decay_s=0.25) if args.vad_agc else None)
        self.committed = 0      # samples already finalized
        self.processed = 0      # samples seen by last decode
        self.vad_checked = 0
        self.started = False    # first token of the stream sent (no leading space)
        self.last_sent_nonfinal = False
        split_pause = getattr(self.asr, "split_pause_ms", None) or args.split_pause_ms
        if self.asr.progressive and len(hints) == 1:
            # one language and words are finalized while speaking: close pieces only at longer
            # pauses, so the final pass sees whole sentences (short pieces cost accuracy)
            split_pause = max(split_pause, args.split_pause_progressive_ms)
        self.vad_opts = VadOptions(min_silence_duration_ms=min(split_pause, self.endpoint_ms),
                                   speech_pad_ms=100)
        # the coarse VAD cannot see pauses shorter than its min silence: refine the tail with this
        self.fine_vad = VadOptions(min_silence_duration_ms=60, speech_pad_ms=60, min_speech_duration_ms=0)
        # diarization labels per speech segment: a 120 ms pause separates two speakers' turns
        self.diar_vad = VadOptions(min_silence_duration_ms=200, speech_pad_ms=80, min_speech_duration_ms=150)
        self.unended = False  # final tokens sent since last <end>/<fin>
        self.lock = asyncio.Lock()  # serializes step() with client finalize / end-of-stream
        # closed pieces waiting for their final words (second ASR pass), emitted in order
        self.pieces = collections.deque()
        self.partial = []       # non-final tokens of the open piece
        self.spec = None        # (cut, task): second pass started early during a pause
        self.final_end = 0      # samples covered by emitted final tokens
        self.emit, self.drained = asyncio.Event(), asyncio.Event()
        self.emitter = asyncio.create_task(self._emitter())
        # progressive finalization of the open piece (LocalAgreement-2 over second-pass re-reads)
        self.piece_id = 0
        self.prog_t = None        # words of the open piece ending before this time are final
        self.prog_last = []       # last committed words (boundary de-duplication)
        self.prog_tail = []       # previous re-read after prog_t, for agreement
        self.prog_task = None
        self.prog_at = 0

        self.diarizer = (OnlineDiarizer(embedder, threshold=args.diar_threshold)
                         if cfg.get("enable_speaker_diarization") else None)
        self.diar_cache = None  # (piece start, audio length, (token times, labels)) for non-final tokens
        tr = cfg.get("translation") or None
        if tr and tr.get("type") == "one_way":
            self.tr_map = {"*": tr["target_language"]}
        elif tr and tr.get("type") == "two_way":
            self.tr_map = {tr["language_a"]: tr["language_b"], tr["language_b"]: tr["language_a"]}
        else:
            self.tr_map = None
        self.translator = translator if self.tr_map else None
        self.tr_history = []
        self.tr_buf = []
        self.tr_queue = asyncio.Queue()
        self.tr_task = asyncio.create_task(self._translate_worker()) if self.translator else None

    def set_context(self, ctx):
        """Extension: replace the context mid-stream (applies to pieces finalized from now on)."""
        self.ctx_prompt, self.ctx_hotwords, self.tr_glossary, self.tr_general = parse_context(ctx)
        for attr, val in (("prompt", self.ctx_prompt), ("hotwords", self.ctx_hotwords)):
            if hasattr(self.asr, attr):
                setattr(self.asr, attr, val)

    # ---- audio ----
    @property
    def audio(self):
        return self._audio

    def _trim_audio(self):
        """Drop audio nothing can still reference: 10 s before the oldest live index."""
        live = [self.committed, self.processed]
        live += [p["start"] for p in self.pieces]
        if self.spec:
            live.append(self.spec[0])
        if self.eot is not None:
            live.append(self.turn_start)
        keep_from = max(0, min(live) - 10 * SR)
        self._audio.trim(keep_from)
        self.vad.trim(keep_from)

    def add_audio(self, data: bytes):
        if self.pending_bytes:
            data = self.pending_bytes + data
        frame = np.dtype(self.dtype).itemsize * self.channels   # one sample of every channel
        rem = len(data) % frame
        self.pending_bytes, data = (data[len(data) - rem:], data[:len(data) - rem]) if rem else (b"", data)
        if not data:
            return
        x = np.frombuffer(data, dtype=self.dtype).astype(np.float32)
        if self.dtype == np.int16:
            x /= 32768.0
        if self.channels > 1:
            x = x[: len(x) // self.channels * self.channels].reshape(-1, self.channels).mean(axis=1)
        if self.in_rate != SR and len(x):
            n = int(round(len(x) * SR / self.in_rate))
            x = np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x).astype(np.float32)
        self._audio.append(x)
        if len(self._audio) - self.last_trim > SR * 10:
            self.last_trim = len(self._audio)
            self._trim_audio()

    # ---- decoding ----
    def _tokens(self, words, is_final):
        toks = []
        for w in words:
            text = w.text
            if not self.started and not toks:
                text = text.lstrip()
            toks.append({
                "text": text,
                "start_ms": int(w.start * 1000),
                "end_ms": int(w.end * 1000),
                "confidence": round(w.prob, 3),
                "is_final": is_final,
                "language": w.lang or self.language,
            })
        if is_final and toks:
            self.started = True
        return toks

    async def _label_speakers(self, toks, update, start):
        if not self.diarizer or not toks:
            return
        if not update:
            # non-final tokens: re-embed the open piece at most once per second of new audio
            c = self.diar_cache
            if c and c[0] == start and len(self.audio) - c[1] < SR:
                times, labels = c[2]
                for t in toks:
                    i = int(np.argmin(np.abs(times - t["start_ms"]))) if t["start_ms"] <= times[-1] else -1
                    t["speaker"] = labels[i]
                return
        base = start / SR
        spans = [(t["start_ms"] / 1000 - base, t["end_ms"] / 1000 - base) for t in toks]
        end = min(len(self.audio), int(toks[-1]["end_ms"] / 1000 * SR) + int(0.2 * SR))
        seg = self.audio[start:end]
        segments = self.vad.segments(start, end, self.diar_vad) if self.args.diar_mode == "segment" else None
        labels = await asyncio.get_running_loop().run_in_executor(
            None, self.diarizer.label_words, seg, start, spans, update, segments)
        for t, spk in zip(toks, labels):
            t["speaker"] = spk
        if not update:
            self.diar_cache = (start, len(self.audio), (np.array([t["start_ms"] for t in toks]), labels))

    def _tag_nonfinal(self, toks):
        if self.tr_map:
            for t in toks:
                t["translation_status"] = "original" if self._target(t["language"]) else "none"
        return toks

    # ---- translation ----
    def _target(self, lang):
        if not self.tr_map:
            return None
        tgt = self.tr_map.get(lang, self.tr_map.get("*"))
        return tgt if tgt and tgt != lang else None

    def _mark_and_enqueue(self, final, flush):
        """Tag spoken tokens with translation_status and queue complete units for translation.
        A unit is a run of same speaker+language, cut at sentence end, endpoint or change."""
        if not self.tr_map:
            return
        for t in final:
            t["translation_status"] = "original" if self._target(t["language"]) else "none"
        self.tr_buf += final
        runs = []
        for t in self.tr_buf:
            if runs and (t.get("speaker"), t["language"]) == (runs[-1][0].get("speaker"), runs[-1][0]["language"]):
                runs[-1].append(t)
            else:
                runs.append([t])
        keep = []
        if runs and not flush:
            # keep the trailing incomplete sentence of the last run
            last = runs.pop()
            cut = max((i + 1 for i, t in enumerate(last) if t["text"].rstrip()[-1:] in ".?!。？！"), default=0)
            if cut:
                runs.append(last[:cut])
            keep = last[cut:]
            if sum(len(t["text"]) for t in keep) > self.args.tr_max_chars:
                # no sentence punctuation for a long stretch: translate up to this pause anyway
                runs.append(keep)
                keep = []
        for run in runs:
            tgt = self._target(run[0]["language"])
            if tgt:
                self.tr_queue.put_nowait(("".join(x["text"] for x in run).strip(), run[0]["language"], tgt,
                                          run[0].get("speaker")))
        self.tr_buf = keep

    async def _translate_worker(self):
        failures, disabled_until = 0, 0.0
        while True:
            text, src, tgt, spk = await self.tr_queue.get()
            if time.time() < disabled_until:
                self.tr_queue.task_done()      # translation server down: skip, transcription goes on
                continue
            try:
                # history = recent good pairs only; a degenerate pair would be imitated by the LLM
                hist = [(a, b) for a, b, l in self.tr_history[-3:] if l == src and not degenerate(a, b)]
                out, t0, ttft = [], time.time(), None
                async for chunk in self.translator.stream(text, src, tgt, hist, self.tr_glossary, self.tr_general):
                    out.append(chunk)
                    ttft = ttft or time.time() - t0
                    tok = {"text": chunk, "is_final": True, "translation_status": "translation",
                           "language": tgt, "source_language": src}
                    if spk:
                        tok["speaker"] = spk
                    await self._send([tok])
                if degenerate(text, "".join(out)):
                    # empty / truncated answer (seen with glossary prompts): retry once, plain prompt
                    log.info("degenerate translation %r -> %r; retrying without glossary/history", text[:40], "".join(out)[:40])
                    out = []
                    async for chunk in self.translator.stream(text, src, tgt, (), None, None):
                        out.append(chunk)
                        tok = {"text": chunk, "is_final": True, "translation_status": "translation",
                               "language": tgt, "source_language": src}
                        if spk:
                            tok["speaker"] = spk
                        await self._send([tok])
                self.tr_history.append((text, "".join(out), src))
                del self.tr_history[:-20]   # only the last few pairs are ever used as history
                log.info("translation %s->%s ttft=%.3fs total=%.3fs queue=%d", src, tgt, ttft or -1,
                         time.time() - t0, self.tr_queue.qsize())
                failures = 0
            except Exception as e:  # translation failure must not kill transcription
                failures += 1
                log.warning("translation failed (%d in a row): %s", failures, e)
                if failures >= 3:
                    disabled_until = time.time() + 30
                    log.warning("translation disabled for 30 s after repeated failures")
            finally:
                self.tr_queue.task_done()

    async def _send(self, tokens):
        await self.ws.send(json.dumps({
            "tokens": tokens,
            "final_audio_proc_ms": int(self.final_end / SR * 1000),
            "total_audio_proc_ms": int(self.processed / SR * 1000),
        }, ensure_ascii=False))

    async def _send_update(self, final=()):
        """New final tokens plus the complete current set of non-final tokens (each response
        replaces the client's non-final tokens)."""
        nonfinal = [t for p in self.pieces for t in p["nonfinal"]] + self.partial
        if final or nonfinal or self.last_sent_nonfinal:
            self.last_sent_nonfinal = bool(nonfinal)
            await self._send(list(final) + nonfinal)

    @staticmethod
    def _drop_committed(words, t, last):
        """Words not yet finalized progressively: after time t, minus a repeated boundary n-gram."""
        if t is None:
            return list(words)
        out = [w for w in words if (w.start + w.end) / 2 >= t - 0.02]
        key = lambda w: "".join(c for c in w.text.lower() if c.isalnum())
        tail = [key(w) for w in last]
        for k in range(min(3, len(tail), len(out)), 0, -1):
            if [key(w) for w in out[:k]] == tail[-k:]:
                return out[k:]
        return out

    async def _progress(self, task, pid, read_end):
        try:
            words, _ = await task
        except (asyncio.CancelledError, Exception):
            words = None
        if words is None:          # re-read failed or the language's final engine is the first pass
            async with self.lock:
                if pid == self.piece_id:
                    self.prog_task = None
            return
        async with self.lock:
            if pid != self.piece_id:      # piece closed meanwhile: its close result covers it
                return
            self.prog_task = None
            key = lambda w: "".join(c for c in w.text.lower() if c.isalnum())
            tail = self._drop_committed(words, self.prog_t, self.prog_last)
            n = 0
            while n < min(len(tail), len(self.prog_tail)) and key(tail[n]) == key(self.prog_tail[n]):
                n += 1
            # words near the end of the re-read audio are the least reliable (cut mid-phrase):
            # the hold is measured from the end of the audio that was re-read, not from now
            # never commit a repetition run (a looping decode agrees with itself)
            key2 = lambda w: "".join(c for c in w.text.lower() if c.isalnum())
            while n >= 3 and key2(tail[n - 1]) == key2(tail[n - 2]) == key2(tail[n - 3]):
                n -= 1
            horizon = read_end / SR - self.args.prog_hold_s
            k = 0
            while k < n and tail[k].end <= horizon:
                k += 1
            commit, self.prog_tail = tail[:k], tail[k:]
            if commit:
                seg_start = int(self.prog_t * SR) if self.prog_t else self.committed
                self.prog_t = commit[-1].end
                self.prog_last = (self.prog_last + commit)[-3:]
                # speakers are labelled on the committed slice itself. Labelling with the whole open
                # piece as context was tried (2026-10-10) and was worse: the still-growing last unit
                # re-shaped the centroids at every commit (rapid-turn accuracy 0.94 -> 0.73).
                self.pieces.append({"start": seg_start, "end": int(commit[-1].end * SR), "words": commit,
                                    "second": None, "marker": None, "flush": False, "nonfinal": []})
                self.emit.set()

    def _soft_cut(self, start, end):
        """Least speech-like frame in the later part of the open piece (usually between words)."""
        f0 = max((start + int(0.5 * (end - start))) // 512 + 1, self.vad.probs_off)
        f1 = (end - int(0.3 * SR)) // 512
        probs = self.vad.probs[f0 - self.vad.probs_off:f1 - self.vad.probs_off]   # probs start at frame probs_off
        if not probs:
            return end
        i = min(range(len(probs)), key=lambda k: (round(probs[k], 2), -k))   # lowest, prefer later
        return (f0 + i) * 512 + 256

    def _speech_end(self, start, end, speech):
        """Absolute end of the last speech in audio[start:end] given its coarse VAD segments."""
        last = start + speech[-1]["end"]
        if end - last < SR * self.args.split_pause_ms / 1000:
            t0 = max(start, end - int(0.6 * SR))
            fine = self.vad.segments(t0, end, self.fine_vad)
            if fine:
                last = min(last, t0 + fine[-1]["end"])
        return last

    async def close(self, end, marker=None, flush=False):
        """Close the open piece audio[committed:end] (trailing non-speech trimmed). Its final
        tokens are emitted in order by _emitter, after the second ASR pass if there is one.
        marker: "<end>"/"<fin>" appended if anything was finalized since the last marker."""
        self.vad.feed(self.audio)
        speech = self.vad.segments(self.committed, end, self.vad_opts)
        cut = self._speech_end(self.committed, end, speech) if speech else self.committed
        flush = flush or marker is not None
        spec, self.spec = self.spec, None
        if spec and (spec[1] is None or abs(spec[0] - cut) > 0.15 * SR):
            if spec[1] is not None:
                spec[1].cancel()
            spec = None
        if not speech and not self.asr.has_open():
            words, second = [], None   # nothing to transcribe: answer a finalize / endpoint right away
        else:
            words, second = await self.asr.close(self.audio, self.committed, cut, flush, len(self.audio),
                                                 spec[1] if spec else None)
        skip_t, skip_last = self.prog_t, list(self.prog_last)
        words = self._drop_committed(words, skip_t, skip_last)
        if second is not None:
            async def without_committed(coro=second):
                return self._drop_committed(await coro, skip_t, skip_last)
            second = without_committed()
        # the open piece is gone: reset progressive finalization
        self.piece_id += 1
        if self.prog_task:
            self.prog_task.cancel()
        self.prog_t, self.prog_last, self.prog_tail, self.prog_task, self.prog_at = None, [], [], None, 0
        nonfinal = self._tokens(words, False) if second else []
        await self._label_speakers(nonfinal, False, self.committed)
        self.pieces.append({"start": self.committed, "end": end, "words": words, "second": second,
                            "marker": marker, "flush": flush, "nonfinal": self._tag_nonfinal(nonfinal)})
        self.committed = self.processed = max(self.committed, end)
        self.partial = []
        self.emit.set()
        if second:
            await self._send_update()   # first-pass words stay visible as non-final meanwhile

    async def _emitter(self):
        while True:
            try:
                await self._emit_pending()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("emitter failed")
                if self.pieces:
                    self.pieces.popleft()   # drop the piece that failed, keep the session alive

    async def _emit_pending(self):
        await self.emit.wait()
        self.emit.clear()
        while self.pieces:
            p = self.pieces[0]
            words = p["words"]
            if p["second"] is not None:
                try:
                    words = await p["second"]
                except Exception as e:  # second pass failed: keep the first-pass words
                    log.warning("second pass failed: %r", e)
            final = self._tokens(words, True)
            self.turn_text += [t["text"] for t in final]
            del self.turn_text[:-300]   # the text detector reads the tail only; without <end> this would grow forever
            await self._label_speakers(final, p.get("update", True), p["start"])
            self._mark_and_enqueue(final, p["flush"])
            self.pieces.popleft()
            self.final_end = max(self.final_end, p["end"])
            self.unended |= bool(final)
            if p["marker"] and (self.unended or p["marker"] == "<fin>"):
                final.append({"text": p["marker"], "is_final": True})
                self.unended = False
                if p["marker"] == "<end>":
                    self.turn_start = p["end"]
                    self.turn_text = []
            await self._send_update(final)
        self.drained.set()

    async def drain(self):
        """Wait until every closed piece has been emitted."""
        while self.pieces:
            self.drained.clear()
            self.emit.set()
            await self.drained.wait()

    async def step(self):
        """Called every tick: close pieces at pauses / endpoints, else emit non-final tokens."""
        end = len(self.audio)
        if end - self.vad_checked < SR * self.args.vad_interval_s:
            return
        self.vad_checked = end
        pending = self.audio[self.committed:end]
        self.vad.feed(self.audio)
        speech = self.vad.segments(self.committed, end, self.vad_opts)
        if not speech:
            # pure silence: drop it (keep a short tail as left context)
            self.committed = self.processed = max(self.committed, end - int(SR * 0.3))
            return
        speech_end = self._speech_end(self.committed, end, speech)
        trailing_sil = (end - speech_end) / SR * 1000
        if self.spec and speech_end > self.spec[0] + int(0.1 * SR):
            # speech resumed after the pause the final result was prepared for
            if self.spec[1] is not None:
                self.spec[1].cancel()
            self.spec = None
        ends_sentence = bool(self.partial) and self.partial[-1]["text"].rstrip()[-1:] in ".?!。？！"
        wait_ms = self.endpoint_ms_sentence if ends_sentence else self.endpoint_ms
        if self.eot is not None and self.endpointing and trailing_sil >= self.args.eot_check_ms:
            # hypothesis of the current turn (finals + open partial); it lags the audio a little,
            # so the pause is re-scored whenever the text changes
            # finals of the turn + words of pieces still waiting for their second pass + the open partial
            pending = [w for p_ in self.pieces for w in p_["words"]]
            text = "".join(self.turn_text) + "".join(w.text for w in pending) + "".join(t["text"] for t in self.partial)
            last_tok = self.partial[-1] if self.partial else None
            lang = (last_tok.get("language") if last_tok else None) or self.language or "ko"
            covered = (last_tok["end_ms"] / 1000 if last_tok else (pending[-1].end if pending else 0.0))
            # the text detector is only meaningful once the hypothesis reaches the pause; before that
            # (streaming lag) the audio detector decides
            use_text = covered >= speech_end / SR - 0.35
            key = (speech_end // (SR // 10), len(text), use_text)
            if self.eot_key != key:
                self.eot_key = key
                a0 = max(self.turn_start, speech_end - 8 * SR, self._audio.offset)
                self.eot_prob = await self.eot.score(self.audio[a0:min(end, speech_end + int(0.2 * SR))],
                                                     text if use_text else "", lang)
            if self.eot_prob >= self.args.eot_hi:
                wait_ms = min(wait_ms, self.args.eot_fast_ms)        # complete: end early
            else:
                wait_ms = max(wait_ms, self.eot_hold_ms)             # incomplete: wait (capped by max delay)
        if trailing_sil >= wait_ms:
            await self.close(end, "<end>" if self.endpointing else None)
            return
        if len(speech) > 1:
            # internal pause (>= split_pause_ms): close the completed part now, no <end>.
            # Each part is decoded separately, so language switches between speakers survive.
            await self.close(self.committed + (speech[-2]["end"] + speech[-1]["start"]) // 2)
            return
        if len(pending) >= self.max_open:
            # continuous speech without a usable pause: finalize the older part now
            await self.close(self._soft_cut(self.committed, end))
            return
        if len(pending) / SR >= self.args.max_segment_s:
            await self.close(end)
            return
        prefinal = False
        if trailing_sil >= self.args.spec_silence_ms and self.spec is None:
            # speech just stopped: prepare the final result now, so an endpoint or a client
            # "finalize" can be answered immediately (second pass task, or a fresh decode that
            # covers all speech and is promoted to final without re-decoding)
            task = self.asr.second_pass(self.audio, self.committed, speech_end)
            self.spec = (speech_end, task)
            prefinal = task is None
        if self.asr.progressive and self.prog_task is None and end - self.committed >= SR * self.args.prog_min_s:
            # the re-read interval stretches with the second-pass queue: fast while the decoder is idle,
            # graceful under load instead of skipping (a skipped word waits for the piece close)
            queued = self.asr.reread_queued()
            interval = self.args.prog_interval_s * (1 + queued / self.args.prog_load_queue)
            if end - self.prog_at >= SR * interval and queued <= self.args.prog_max_queued:
                self.prog_at = end
                # re-read from a few seconds before the last committed word, not the whole open piece
                # (up to 30 s): the encoder cost is fixed, the decoder cost grows with the text length.
                # The window still contains the boundary n-gram that _drop_committed aligns on.
                start = self.committed if self.prog_t is None else max(self.committed, int(self.prog_t * SR) - int(self.args.prog_window_s * SR))
                self.prog_task = asyncio.create_task(
                    self._progress(self.asr.reread(self.audio, start, end), self.piece_id, end))
        if end - self.processed < SR * self.step_s and not prefinal:
            return
        self.processed = end
        words = await self.asr.partial(self.audio, self.committed, end, urgent=prefinal)
        toks = self._tokens(self._drop_committed(words, self.prog_t, self.prog_last), False)
        await self._label_speakers(toks, False, self.committed)
        self.partial = self._tag_nonfinal(toks)
        await self._send_update()


async def handler(ws, engine, args, embedder, translator, eot=None):
    rid = str(uuid.uuid4())
    path = ws.request.path
    if path.rstrip("/") != "/transcribe-websocket":
        await ws.close(1008, "unknown path")
        return

    async def error(code, etype, msg):
        METRICS[f'errors_total{{code="{code}",type="{etype}"}}'] += 1
        try:
            await ws.send(json.dumps({"tokens": [], "error_code": code, "error_type": etype,
                                      "error_message": msg, "request_id": rid}))
            await ws.close()
        except websockets.ConnectionClosed:
            pass

    try:
        first = await asyncio.wait_for(ws.recv(), timeout=40)
    except (asyncio.TimeoutError, websockets.ConnectionClosed):
        return
    if isinstance(first, bytes):
        return await error(400, "invalid_request", "The first frame must be a text frame holding the JSON start config.")
    try:
        cfg = json.loads(first)
    except (TypeError, json.JSONDecodeError):
        cfg = None
    if not isinstance(cfg, dict):
        return await error(400, "invalid_request", "The start config must be a JSON object.")
    if args.api_key:
        auth = ws.request.headers.get("Authorization", "")
        offered = ws.request.headers.get("Sec-WebSocket-Protocol", "")
        if args.api_key not in (cfg.get("api_key"), auth.removeprefix("Bearer ").strip()) \
                and args.api_key not in [p.strip() for p in offered.split(",")]:
            return await error(401, "unauthenticated", "API key is not valid.")
    fmt = cfg.get("audio_format")
    if fmt not in SUPPORTED_FORMATS:
        return await error(400, "invalid_request",
                           f"audio_format '{fmt}' not supported; use one of {list(SUPPORTED_FORMATS)}.")
    if not cfg.get("sample_rate") or not cfg.get("num_channels"):
        return await error(400, "invalid_request", "sample_rate and num_channels are required for raw audio.")

    hints = cfg.get("language_hints")
    if hints is not None and (not isinstance(hints, list) or any(not isinstance(h, str) or h not in LANGS for h in hints)):
        bad = [h for h in hints if h not in LANGS] if isinstance(hints, list) else hints
        return await error(400, "invalid_request", f"Unsupported language in language_hints: {bad}.")
    if hints and len(set(hints)) != len(hints):
        return await error(400, "invalid_request", "language_hints contains duplicate codes.")
    # out-of-range endpoint options are rejected rather than clamped
    for field, lo, hi in (("max_endpoint_delay_ms", 500, 3000), ("endpoint_latency_adjustment_level", 0, 3)):
        v = cfg.get(field)
        if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi):
            return await error(400, "invalid_request", f"{field} is out of range (allowed {lo}..{hi}).")
    sens = cfg.get("endpoint_sensitivity")
    if sens is not None and (not isinstance(sens, (int, float)) or isinstance(sens, bool) or not -1 <= sens <= 1):
        return await error(400, "invalid_request", "endpoint_sensitivity is out of range (allowed -1..1).")
    ref = cfg.get("client_reference_id")
    if ref is not None and len(str(ref)) > 256:
        return await error(400, "invalid_request",
                           f"client_reference_id is {len(str(ref))} characters long; the limit is 256.")
    tr = cfg.get("translation")
    if tr is not None and not isinstance(tr, dict):
        return await error(400, "invalid_request", "translation must be an object.")
    if tr and (tr.get("target_language", "en") not in LANGS or tr.get("language_a", "en") not in LANGS
               or tr.get("language_b", "en") not in LANGS):
        return await error(400, "invalid_request", "Unsupported language in translation config.")
    if tr and not (tr.get("type") == "one_way" and tr.get("target_language")
                   or tr.get("type") == "two_way" and tr.get("language_a") and tr.get("language_b")):
        return await error(400, "invalid_request", "Invalid translation config.")
    if tr and translator is None:
        return await error(400, "invalid_request", "Translation is not enabled on this server (--llm-url).")
    if cfg.get("context") is not None and not isinstance(cfg["context"], dict):
        return await error(400, "invalid_request", "context must be an object.")
    if cfg.get("context") is not None and context_chars(cfg["context"]) > MAX_CONTEXT_CHARS:
        return await error(400, "invalid_request", "Context is too long.")
    if args.max_sessions and len(ACTIVE) >= args.max_sessions:
        return await error(429, "limit_exceeded",
                           f"Too many concurrent streams on this server (limit {args.max_sessions}).")
    try:
        s = Session(ws, engine, cfg, args, embedder, translator, eot)
    except (TypeError, ValueError) as e:   # wrong type in a numeric option (sample_rate, endpoint knobs, ...)
        return await error(400, "invalid_request", f"Invalid config value: {e}")
    ACTIVE.add(rid)
    METRICS["sessions_total"] += 1
    log.info("session %s start cfg=%s", rid, {k: v for k, v in cfg.items() if k != "api_key"})
    t0 = time.time()
    max_samples = int(args.max_stream_min * 60 * SR) if args.max_stream_min else None

    async def ticker():
        while True:
            await asyncio.sleep(0.05)
            try:
                async with s.lock:
                    await s.step()
            except asyncio.CancelledError:
                raise
            except Exception:   # one bad step must not silence the session
                log.exception("session %s step failed", rid)

    tick = asyncio.create_task(ticker())
    try:
        while True:
            try:   # idle stream: no audio or keepalive for too long
                msg = await asyncio.wait_for(ws.recv(), timeout=args.idle_timeout_s or None)
            except asyncio.TimeoutError:
                tick.cancel()
                return await error(408, "request_timeout",
                                   f"No audio or keepalive received for {args.idle_timeout_s:g} s; closing the idle stream.")
            if isinstance(msg, bytes):
                s.add_audio(msg)
                if max_samples and len(s.audio) >= max_samples:   # per-stream duration cap
                    tick.cancel()
                    async with s.lock:
                        await s.close(len(s.audio), "<fin>", flush=True)
                    await s.drain()
                    return await error(413, "max_duration_reached",
                                       f"Stream reached the per-connection limit of {args.max_stream_min:g} minutes; final results were "
                                       "sent before closing. Open a new connection to continue.")
                continue
            if msg == "":
                tick.cancel()
                async with s.lock:
                    await s.close(len(s.audio), "<end>" if s.endpointing else None, flush=True)
                await s.drain()
                if s.tr_task:
                    await s.tr_queue.join()
                await ws.send(json.dumps({"tokens": [], "finished": True,
                                          "final_audio_proc_ms": int(len(s.audio) / SR * 1000),
                                          "total_audio_proc_ms": int(len(s.audio) / SR * 1000)}))
                await ws.close()
                break
            try:
                ctrl = json.loads(msg)
            except (TypeError, json.JSONDecodeError):
                ctrl = None
            if not isinstance(ctrl, dict):
                tick.cancel()
                return await error(400, "invalid_request", "Text frames must be JSON control messages.")
            if ctrl.get("type") == "finalize":
                async with s.lock:
                    await s.close(len(s.audio), "<fin>")
            elif ctrl.get("type") == "context":   # extension: update context mid-stream
                if not isinstance(ctrl.get("context") or {}, dict) or context_chars(ctrl.get("context") or {}) > MAX_CONTEXT_CHARS:
                    tick.cancel()
                    return await error(400, "invalid_request", "Context must be an object of at most 10,000 characters.")
                async with s.lock:
                    s.set_context(ctrl.get("context"))
            elif ctrl.get("type") != "keepalive":
                tick.cancel()
                return await error(400, "invalid_request",
                                   'Unknown control message type; expected "finalize", "keepalive" or "context".')
    except websockets.ConnectionClosed:
        pass
    finally:
        tick.cancel()
        s.asr.release()
        s.emitter.cancel()
        if s.spec:
            s.spec[1].cancel()
        if s.tr_task:
            s.tr_task.cancel()
        ACTIVE.discard(rid)
        METRICS["audio_seconds_total"] += len(s.audio) / SR
        log.info("session %s end audio=%.1fs wall=%.1fs", rid, len(s.audio) / SR, time.time() - t0)


async def metrics_server(host, port):
    """GET /metrics (Prometheus text format) and GET /health on a plain HTTP port; stdlib only."""
    from asr_backends import WHISPER_BACKENDS

    async def handle(reader, writer):
        try:
            line = await asyncio.wait_for(reader.readline(), 5)
            path = line.split()[1].decode() if len(line.split()) > 1 else "/"
            while (await asyncio.wait_for(reader.readline(), 5)).strip():
                pass
            if path.startswith("/health"):
                body, ctype = json.dumps({"status": "ok", "active_sessions": len(ACTIVE)}), "application/json"
            else:
                rows = [("sessions_active", len(ACTIVE))] + sorted(METRICS.items())
                for i, b in enumerate(WHISPER_BACKENDS):
                    rows += [(f'whisper_{k}{{backend="{i}"}}', v) for k, v in sorted(b.total.items())]
                    rows.append((f'whisper_queue_depth{{backend="{i}"}}', b.busy()))
                body = "".join(f"rtstt_{k} {v}\n" for k, v in rows)
                ctype = "text/plain; version=0.0.4"
            data = body.encode()
            writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {len(data)}\r\n"
                         f"Connection: close\r\n\r\n".encode() + data)
            await writer.drain()
        except Exception:   # noqa: BLE001 - a bad scrape must not matter
            pass
        finally:
            writer.close()
    return await asyncio.start_server(handle, host, port)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--metrics-port", type=int, default=8766, help="HTTP port for /metrics and /health (0 disables)")
    p.add_argument("--max-sessions", type=int, default=0, help="concurrent streams; more get 429 limit_exceeded (0 = unlimited)")
    p.add_argument("--max-stream-min", type=float, default=300, help="per-stream audio cap in minutes -> 413 max_duration_reached")
    p.add_argument("--idle-timeout-s", type=float, default=60, help="no audio/keepalive for this long -> 408 request_timeout (0 disables)")
    p.add_argument("--drain-s", type=float, default=30, help="on SIGTERM/SIGINT: stop accepting and wait this long for open sessions")
    p.add_argument("--asr", choices=["nemotron", "whisper", "multi"], default="nemotron",
                   help="multi: per-language routing over Nemotron / Whisper turbo + large-v3 / Omnilingual (60 languages)")
    p.add_argument("--large-model", default="/home/stt-model/models/models--Systran--faster-whisper-large-v3/snapshots/"
                   "edaa852ec7e145841d8ffdb056a99866b5f0a478", help="--asr multi: Whisper large-v3 (CT2)")
    p.add_argument("--omni-dir", default=os.path.join(
        HERE, "models/asr/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05.dl"))
    p.add_argument("--omni-threads", type=int, default=4)
    p.add_argument("--madlad-dir", default="", help="MADLAD-400 CT2 dir: route non ko/en/zh/de pairs to it")
    p.add_argument("--nemotron-dir", default=os.path.join(
        HERE, "models/asr/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.dl"),
        help="sherpa-onnx Nemotron-3.5 streaming model dir (160ms / 560ms / 1120ms chunk variants)")
    p.add_argument("--asr-threads", type=int, default=2, help="onnxruntime threads per streaming decode worker")
    p.add_argument("--asr-workers", type=int, default=4, help="streaming decode workers (model copies)")
    p.add_argument("--final", choices=["none", "qwen3-0.6b", "llamacpp", "whisper"], default="none",
                   help="second pass that re-transcribes each finalized piece (whisper: --model on --device)")
    p.add_argument("--final-url", default="http://localhost:8092/v1", help="llama.cpp server with Qwen3-ASR")
    p.add_argument("--final-dir", default=os.path.join(HERE, "models/asr/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.dl"))
    p.add_argument("--final-threads", type=int, default=4)
    p.add_argument("--final-budget-s", type=float, default=1.5,
                   help="max wait for the second pass after a piece closes; then first-pass words are final")
    p.add_argument("--final-slots", type=int, default=4, help="concurrent second-pass requests (server slots)")
    p.add_argument("--model", default=os.environ.get(
        "WHISPER_MODEL", "/home/stt-model/models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/"
        "snapshots/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
        help="--asr whisper: CTranslate2 Whisper dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--workers", type=int, default=4, help="parallel Whisper decodes")
    p.add_argument("--whisper-batch", type=int, default=1, help="max final requests decoded in one batch")
    p.add_argument("--step-s", type=float, default=None, help="non-final update interval (backend default)")
    p.add_argument("--endpoint-silence-ms", type=int, default=500)
    p.add_argument("--eot", action="store_true", help="semantic end-of-turn detector (smart-turn v3) for endpointing")
    p.add_argument("--eot-model", default=EOT_MODEL)
    p.add_argument("--eot-check-ms", type=int, default=200, help="silence before the detector is asked")
    p.add_argument("--eot-fast-ms", type=int, default=250, help="<end> this soon when detector + punctuation say complete")
    p.add_argument("--eot-hold-ms", type=int, default=800, help="wait this long when the detector says incomplete")
    p.add_argument("--eot-hi", type=float, default=0.5, help="P(complete) needed to end early")
    p.add_argument("--eot-text-model", default=NAMO_DIR, help="Korean text detector dir ('' = audio only)")
    p.add_argument("--endpoint-sentence-factor", type=float, default=0.7,
                   help="endpoint silence multiplier when the hypothesis ends with . ? !")
    p.add_argument("--split-pause-progressive-ms", type=int, default=450,
                   help="split pause for single-language sessions with progressive finalization")
    p.add_argument("--split-pause-ms", type=int, default=250,
                   help="pause that soft-finalizes a phrase (final tokens without <end>)")
    p.add_argument("--vad-interval-s", type=float, default=0.1, help="how often the VAD looks at new audio")
    p.add_argument("--no-vad-agc", dest="vad_agc", action="store_false",
                   help="disable gain normalization of the VAD input")
    p.add_argument("--spec-silence-ms", type=int, default=250,
                   help="silence after which the final result is prepared ahead of endpoint/finalize")
    p.add_argument("--max-segment-s", type=float, default=25.0)
    p.add_argument("--prog-interval-s", type=float, default=1.0, help="progressive finalization re-read interval")
    p.add_argument("--prog-min-s", type=float, default=3.0, help="open piece length before re-reads start")
    p.add_argument("--prog-max-queued", type=int, default=32,
                   help="skip progressive re-reads while this many GPU requests are already queued")
    p.add_argument("--prog-load-queue", type=float, default=8.0,
                   help="re-read interval grows by one base interval per this many queued GPU requests")
    p.add_argument("--prog-window-s", type=float, default=3.0,
                   help="re-reads start this long before the last progressively committed word")
    p.add_argument("--prog-hold-s", type=float, default=0.6, help="agreed words must end this long ago")
    p.add_argument("--max-nonfinal-ms", type=int, default=5000,
                   help="open piece longer than this is cut at its quietest point (word stays non-final at most ~this)")
    p.add_argument("--api-key", default=os.environ.get("RTSTT_API_KEY"))
    p.add_argument("--diar-model", default=DIAR_MODEL)
    p.add_argument("--diar-threshold", type=float, default=0.45, help="cosine sim to join existing speaker")
    p.add_argument("--diar-mode", choices=["piece", "segment"], default="segment",
                   help="label speakers per ASR piece (split at detected changes) or per VAD speech segment")
    p.add_argument("--llm-url", default=os.environ.get("LLM_URL", "http://localhost:8000/v1"),
                   help="OpenAI-compatible endpoint for translation ('' disables)")
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "q36.gguf"))
    p.add_argument("--tr-max-chars", type=int, default=300,
                   help="translate an unpunctuated run at the next pause once it exceeds this many characters")
    p.add_argument("--llm-style", choices=["chat", "hymt"], default=os.environ.get("LLM_STYLE", "chat"),
                   help="hymt = Tencent Hy-MT2 translation prompt format")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.asr == "whisper":
        engine = WhisperBackend(args.model, args.device, args.compute_type, args.workers)
    elif args.asr == "multi":
        sub = lambda d: d if os.path.exists(os.path.join(d, "tokens.txt")) else glob.glob(os.path.join(d, "*"))[0]
        turbo = WhisperBackend(args.model, args.device, args.compute_type, args.workers, args.whisper_batch)
        large = WhisperBackend(args.large_model, args.device, args.compute_type, args.workers, args.whisper_batch)
        omni = OmniEngine(sub(args.omni_dir), args.omni_threads)
        routed = RoutedSecondPass(turbo, large, omni)
        nemo = NemotronBackend(sub(args.nemotron_dir), args.asr_threads, routed, args.asr_workers, args.final_budget_s)
        engine = MultiBackend(nemo, turbo, routed, omni, args.final_budget_s)
    else:
        second = None
        if args.final == "llamacpp":
            second = LlamaCppQwenAsr(args.final_url, args.final_slots)
        elif args.final == "whisper":
            second = WhisperSecondPass(WhisperBackend(args.model, args.device, args.compute_type, args.workers,
                                                      args.whisper_batch))
        elif args.final == "qwen3-0.6b":
            second = SherpaQwenAsr(glob.glob(os.path.join(args.final_dir, "*"))[0]
                                   if not os.path.exists(os.path.join(args.final_dir, "tokenizer")) else args.final_dir,
                                   args.final_threads)
        d = args.nemotron_dir
        d = d if os.path.exists(os.path.join(d, "tokens.txt")) else glob.glob(os.path.join(d, "*"))[0]
        engine = NemotronBackend(d, args.asr_threads, second, args.asr_workers, args.final_budget_s)
    embedder = Embedder(args.diar_model)
    translator = Translator(args.llm_url, args.llm_model, style=args.llm_style) if args.llm_url else None
    eot = None
    if args.eot:
        text_det = NamoTextDetector(args.eot_text_model) if args.eot_text_model and os.path.exists(args.eot_text_model) else None
        eot = CombinedDetector(SmartTurnDetector(args.eot_model), text_det)
    if args.madlad_dir:
        translator = RoutedTranslator(translator, MadladTranslator(args.madlad_dir, args.device))

    def select_subprotocol(conn, offered):
        return offered[0] if offered else None

    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    metrics = await metrics_server(args.host, args.metrics_port) if args.metrics_port else None
    async with websockets.serve(lambda ws: handler(ws, engine, args, embedder, translator, eot), args.host, args.port,
                                select_subprotocol=select_subprotocol, max_size=2**22) as server:
        log.info("READY ws://%s:%d/transcribe-websocket%s", args.host, args.port,
                 f" (metrics http://{args.host}:{args.metrics_port}/metrics)" if metrics else "")
        await stop.wait()
        # graceful drain: stop accepting, let open sessions finish for up to --drain-s
        server.close(close_connections=False)
        t_end = time.time() + args.drain_s
        log.info("shutting down: %d open sessions, draining up to %ds", len(ACTIVE), args.drain_s)
        while ACTIVE and time.time() < t_end:
            await asyncio.sleep(0.2)
        if ACTIVE:
            log.warning("drain timeout: closing %d sessions", len(ACTIVE))
    if metrics:
        metrics.close()


if __name__ == "__main__":
    asyncio.run(main())
