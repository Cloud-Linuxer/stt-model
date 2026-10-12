"""ASR backends for the rtstt server.

The session owns audio buffering, VAD segmentation, diarization and translation. A backend
stream only turns audio into words (absolute times, seconds from stream start):

  await st.partial(audio, start, end)    -> non-final words of the open piece
  await st.close(audio, start, cut, flush, end, spec)
        -> (first-pass words, awaitable of final words or None)
     Closes the piece audio[start:cut]. With a second pass the final words arrive later, so
     the session can keep streaming non-final tokens meanwhile. `spec` is an already running
     second-pass task for the same audio (started speculatively during the pause).
  st.second_pass(audio, start, cut) -> task  (None when the backend has no second pass)

WhisperBackend   faster-whisper; non-final = re-decode the current piece (GPU-friendly).
NemotronBackend  Nemotron-3.5 streaming transducer via sherpa-onnx (CPU, int8). Audio is
                 fed once; all sessions are decoded together in one worker thread.
                 Optional second pass (Qwen3-ASR) re-transcribes each finalized piece.
"""
import asyncio
import base64
import concurrent.futures
import difflib
import heapq
import io
import itertools
import logging
import math
import os
import threading
import time
import zlib
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

SR = 16000
log = logging.getLogger("rtstt")


@dataclass
class Word:
    text: str          # leading space marks a word boundary
    start: float
    end: float
    prob: float
    lang: str | None


def script_lang(text, latin="en"):
    """Language of a word from its script (ko/ja/zh/latin); None for digits/punctuation."""
    for c in text:
        o = ord(c)
        if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            return "ko"
        if 0x3040 <= o <= 0x30FF:
            return "ja"
        if 0x4E00 <= o <= 0x9FFF:
            return "zh"
        if c.isascii() and c.isalpha():
            return latin
    return None


def fill_langs(words, fallback):
    prev = fallback
    for w in words:
        w.lang = w.lang or prev
        prev = w.lang
    return words


# --------------------------------------------------------------------------- Whisper
WHISPER_BACKENDS = []   # every WhisperBackend instance (metrics)


class WhisperBackend:
    """One model shared by all sessions; up to `workers` decodes run in parallel.
    Requests are prioritized: finals (piece close, pre-final at a pause) before non-final
    updates, and a queued non-final request is dropped when the same stream asks again,
    so under load the non-final refresh rate degrades instead of the final latency."""

    step_s = 0.5

    def __init__(self, model_path, device, compute_type, workers, batch=1):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type, num_workers=workers)
        self.workers = workers
        self.batch = batch   # >1: queued final requests are decoded together (one encoder batch)
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)
        self.queue, self.seq, self.wake = [], itertools.count(), None
        self.queued_partial = {}   # stream key -> future of its queued non-final request
        self.stats = {"jobs": 0, "batches": 0, "busy_s": 0.0, "audio_s": 0.0, "qmax": 0}   # since the last report
        self.total = {"decodes_total": 0, "batches_total": 0, "decoder_busy_seconds_total": 0.0, "audio_seconds_total": 0.0}
        WHISPER_BACKENDS.append(self)   # for the /metrics endpoint
        if batch > 1:
            from faster_whisper.transcribe import BatchedInferencePipeline
            self.pipe = BatchedInferencePipeline(self.model)
            self.pipe_lock = threading.Lock()
        self.model.transcribe(np.zeros(SR, dtype=np.float32), language="ko")  # warm-up

    def open(self, language, candidates, prompt, hotwords):
        if self.wake is None:
            self.wake = asyncio.Event()
            # one worker only takes finals, so a final never waits behind a re-read / refresh batch
            for i in range(self.workers):
                asyncio.get_running_loop().create_task(self._worker(finals_only=self.workers > 1 and i == 0))
            asyncio.get_running_loop().create_task(self._report())
        return WhisperStream(self, language, candidates, prompt, hotwords)

    async def _report(self, every=30):
        """Load line every `every` s while there is work: decodes/s, batch size, decoder-seconds per second."""
        while True:
            await asyncio.sleep(every)
            s, self.stats = self.stats, {"jobs": 0, "batches": 0, "busy_s": 0.0, "audio_s": 0.0, "qmax": 0}
            if s["jobs"]:
                log.info("whisper load: %.1f decodes/s, %.1f audio s/s, batch avg %.1f, decoder busy %.2f/%d workers, queue max %d",
                         s["jobs"] / every, s["audio_s"] / every, s["jobs"] / max(1, s["batches"]), s["busy_s"] / every,
                         self.workers, s["qmax"])

    def busy(self):
        """Queued requests that are not superseded."""
        return sum(1 for _, _, f, _ in self.queue if not f.done())

    def fit_context(self, prompt, hotwords, max_hot=64, max_prompt=96):
        """Keep the context within a safe share of Whisper's 448-token decoder window."""
        tk = self.model.hf_tokenizer
        if hotwords:
            ids = tk.encode(" " + hotwords.strip(), add_special_tokens=False).ids
            if len(ids) > max_hot:
                hotwords = tk.decode(ids[:max_hot]).strip()
            # Whisper imitates the prompt's style: an unpunctuated term list makes it drop
            # sentence punctuation (which the translation segmentation relies on)
            if hotwords and hotwords[-1] not in ".。?!":
                hotwords += "."
        if prompt:
            ids = tk.encode(prompt, add_special_tokens=False).ids
            if len(ids) > max_prompt:
                prompt = tk.decode(ids[:max_prompt]).strip()
        return prompt or None, hotwords or None

    async def transcribe(self, audio, language, prompt, hotwords, candidates=None, final=True, key=None, prio=None):
        """Returns (words, language), or None if a newer non-final request superseded this one.
        Priority: 0 = final, 1 = progressive finalization, 2 = non-final refresh."""
        prompt, hotwords = self.fit_context(prompt, hotwords)
        fut = asyncio.get_running_loop().create_future()
        prio = prio if prio is not None else (0 if final else 2)
        if prio == 2:
            old = self.queued_partial.get(key)
            if old is not None and not old.done():
                old.set_result(None)
            self.queued_partial[key] = fut
        heapq.heappush(self.queue, (prio, next(self.seq), fut,
                                    (audio, language, prompt, hotwords, candidates)))
        self.wake.set()
        return await fut

    @staticmethod
    def _batchable(prio, job):
        return prio <= 1   # (progressive) finals; jobs are grouped by (language, prompt, hotwords)

    async def _worker(self, finals_only=False):
        loop = asyncio.get_running_loop()
        while True:
            while not self.queue or (finals_only and self.queue[0][0] > 0):
                self.wake.clear()
                await self.wake.wait()
            prio, _, fut, job = heapq.heappop(self.queue)
            if fut.done():   # superseded
                continue
            group = [(fut, job)]
            if self.batch > 1 and self._batchable(prio, job):
                while (self.queue and len(group) < self.batch and self.queue[0][0] == prio
                       and self._batchable(self.queue[0][0], self.queue[0][3])):
                    _, _, f2, j2 = heapq.heappop(self.queue)
                    if not f2.done():
                        group.append((f2, j2))
            t0 = time.monotonic()
            self.stats["qmax"] = max(self.stats["qmax"], len(self.queue))
            try:
                if len(group) == 1:
                    res = [await loop.run_in_executor(self.pool, self._transcribe, *job)]
                else:
                    res = await loop.run_in_executor(self.pool, self._transcribe_batch, [j for _, j in group])
                for (f, _), r in zip(group, res):
                    f.done() or f.set_result(r)
                dt, audio_s = time.monotonic() - t0, sum(len(j[0]) for _, j in group) / SR
                self.stats["jobs"] += len(group)
                self.stats["batches"] += 1
                self.stats["busy_s"] += dt
                self.stats["audio_s"] += audio_s
                self.total["decodes_total"] += len(group)
                self.total["batches_total"] += 1
                self.total["decoder_busy_seconds_total"] += dt
                self.total["audio_seconds_total"] += audio_s
            except Exception as e:
                for f, _ in group:
                    f.done() or f.set_exception(e)
            self.wake.set()   # let a waiting finals-only worker re-check the queue head

    def _transcribe_batch(self, jobs):
        """Final requests without prompt: one batched encoder/decoder pass per language."""
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import TranscriptionOptions, get_suppressed_tokens
        m = self.model
        feats, langs = [], []
        for audio, language, _, _, candidates in jobs:
            f = m.feature_extractor(audio)[..., :-1]
            if language is None:
                _, _, probs = m.detect_language(features=f)
                probs = dict(probs)
                language = max(candidates or probs, key=lambda c: probs.get(c, 0.0))
            feats.append(pad_or_trim(f))
            langs.append(language)
        out = [None] * len(jobs)
        keys = [(lang, j[2], j[3]) for lang, j in zip(langs, jobs)]
        for lang, prompt_, hot_ in dict.fromkeys(keys):
            idx = [i for i, k in enumerate(keys) if k == (lang, prompt_, hot_)]
            tok = Tokenizer(m.hf_tokenizer, m.model.is_multilingual, task="transcribe", language=lang)
            opts = TranscriptionOptions(
                beam_size=1, best_of=1, patience=1, length_penalty=1, repetition_penalty=1,
                no_repeat_ngram_size=0, log_prob_threshold=-1.0, no_speech_threshold=0.6,
                compression_ratio_threshold=2.4, condition_on_previous_text=False, prompt_reset_on_temperature=0.5,
                temperatures=[0.0], initial_prompt=prompt_, prefix=None, suppress_blank=True,
                suppress_tokens=get_suppressed_tokens(tok, [-1]), without_timestamps=True, max_initial_timestamp=0.0,
                word_timestamps=True, prepend_punctuations="\"'“¿([{-", append_punctuations="\"'.。,，!！?？:：”)]}、",
                multilingual=False, max_new_tokens=None, clip_timestamps=None, hallucination_silence_threshold=None,
                hotwords=hot_)
            res = []
            for k in range(0, len(idx), self.batch):   # bounded GPU batch
                part = idx[k:k + self.batch]
                metas = [{"offset": 0.0, "duration": len(jobs[i][0]) / SR, "segments": []} for i in part]
                with self.pipe_lock:   # forward() keeps per-call state on the pipeline object
                    self.pipe.last_speech_timestamp = 0.0
                    res += self.pipe.forward(np.stack([feats[i] for i in part]), tok, metas, opts)
            for i, segs in zip(idx, res):
                words = [SimpleNamespace(word=w["word"], start=w["start"], end=w["end"], probability=w["probability"])
                         for sg in segs for w in sg.get("words", [])]
                if looping(words):   # rare: redo this one piece on the single-item path (penalized retry)
                    out[i] = self._transcribe(jobs[i][0], lang, jobs[i][2], jobs[i][3], None)
                else:
                    out[i] = (words, lang)
        return out

    def _transcribe(self, audio, language, prompt, hotwords, candidates):
        if language is None and candidates:
            # restrict language ID to language_hints
            _, _, probs = self.model.detect_language(audio)
            probs = dict(probs)
            language = max(candidates, key=lambda c: probs.get(c, 0.0))
        kw = dict(language=language, beam_size=1, word_timestamps=True, temperature=0.0,
                  condition_on_previous_text=False, initial_prompt=prompt, hotwords=hotwords,
                  vad_filter=False, without_timestamps=False)
        segments, info = self.model.transcribe(audio, **kw)
        words = [w for s in segments for w in (s.words or [])]
        if looping(words):
            # repetition hallucination: decode again with a repetition penalty (greedy, no temperature)
            segments, info = self.model.transcribe(audio, repetition_penalty=1.2, no_repeat_ngram_size=3, **kw)
            words = collapse_repeats([w for s in segments for w in (s.words or [])])
        return words, info.language


# Whisper's well-known outro hallucinations (YouTube-style), matched at the end of a segment
OUTRO = ["시청해주셔서감사합니다", "다음영상에서만나요", "구독과좋아요", "thanksforwatching", "thankyouforwatching",
         "subtitlesby", "pleasesubscribe", "camoncacbandatheodoi", "cảmơncácbạnđãtheodõi", "ご視聴ありがとうございました",
         "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目", "字幕志愿者", "amaraorg", "untertitelimauftragdeszdf",
         "sous-titragestfr", "graciasporver", "obrigadoporassistir", "продолжениеследует"]


def _key(text):
    return "".join(c for c in text.lower() if c.isalnum())


def looping(words, max_run=4, ratio=2.4):
    """Whisper repetition hallucination: a word repeated >= max_run times in a row, or highly
    compressible text (zlib ratio, as in Whisper's compression_ratio_threshold)."""
    toks = [_key(w.word) for w in words]
    run = 0
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a and a == b else 0
        if run + 1 >= max_run:
            return True
    text = "".join(w.word for w in words).encode()
    return len(text) > 60 and len(text) / len(zlib.compress(text)) > ratio


def collapse_repeats(words, keep=2):
    """Keep at most `keep` consecutive identical words."""
    out = []
    for w in words:
        if len(out) >= keep and all(_key(x.word) == _key(w.word) for x in out[-keep:]):
            continue
        out.append(w)
    return out


def drop_tail_hallucinations(words):
    """Whisper tail hallucinations ("감사합니다", "다음 영상에서 만나요") come out as
    near-zero-duration or low-probability words, or as a known outro phrase, at the end of a segment."""
    words = list(words)
    while words and (words[-1].probability < 0.3
                     or (words[-1].end - words[-1].start) / max(1, len(words[-1].word.strip())) < 0.03):
        words.pop()
    for n in range(min(12, len(words)), 0, -1):   # longest outro phrase that ends the segment
        tail = _key("".join(w.word for w in words[-n:]))
        if any(tail.startswith(o) or (len(tail) >= 8 and o.startswith(tail)) for o in OUTRO):
            return words[:-n]
    return words


class WhisperStream:
    step_s = 0.5

    def __init__(self, backend, language, candidates, prompt, hotwords):
        self.b, self.language, self.candidates = backend, language, candidates
        self.prompt, self.hotwords = prompt, hotwords
        self.piece_lang = None   # language detected for the current piece
        self.last, self.last_end = [], 0

    async def _decode(self, audio, start, end, final=True):
        seg = audio[start:end]
        if len(seg) < SR * 0.3:
            return []
        # language ID costs an extra encoder pass: do it once per piece (>= 1 s of audio)
        lang = self.language or self.piece_lang
        res = await self.b.transcribe(seg.copy(), lang, self.prompt, self.hotwords,
                                      None if lang else self.candidates, final, id(self))
        if res is None:
            return None
        words, lang_out = res
        if lang is None and len(seg) >= SR:
            self.piece_lang = lang_out
        words = drop_tail_hallucinations(words)
        base = start / SR
        return [Word(w.word, base + w.start, base + w.end, float(w.probability), lang_out) for w in words]

    async def partial(self, audio, start, end, urgent=False):
        words = await self._decode(audio, start, end, final=urgent)
        if words is None:          # superseded by a newer request: keep showing the last result
            return self.last
        self.last, self.last_end = words, end
        return self.last

    async def close(self, audio, start, cut, flush, end, spec=None):
        if self.last and self.last_end >= cut:
            # last hypothesis already covers all speech: no re-decode (words after the cut
            # belong to the next piece)
            words = [w for w in self.last if (w.start + w.end) / 2 < cut / SR]
        else:
            words = await self._decode(audio, start, cut)
        self.piece_lang, self.last, self.last_end = None, [], 0
        return words, None

    def second_pass(self, audio, start, cut):
        return None

    def release(self):
        pass

    def has_open(self):
        return False

    progressive = False


# --------------------------------------------------------------------------- second pass
LANG_CODES = {"korean": "ko", "english": "en", "japanese": "ja", "chinese": "zh", "spanish": "es",
              "french": "fr", "german": "de"}


class LlamaCppQwenAsr:
    """Qwen3-ASR GGUF on a llama.cpp server (OpenAI chat API with input_audio)."""

    def __init__(self, url, max_inflight=4):
        import httpx
        self.url = url.rstrip("/") + "/chat/completions"
        self.client = httpx.AsyncClient(timeout=60)
        self.sem = asyncio.Semaphore(max_inflight)  # = server slots; the rest waits here, cheap to cancel

    async def transcribe(self, audio):
        async with self.sem:
            return await self._transcribe(audio)

    async def _transcribe(self, audio):
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
        r = await self.client.post(self.url, json={"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(),
                                                    "format": "wav"}}]}], "temperature": 0, "max_tokens": 512})
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"]
        head, _, text = out.partition("<asr_text>")
        lang = LANG_CODES.get(head.replace("language", "").strip().lower()) if text else None
        return (text if text else out).strip(), lang


class SherpaQwenAsr:
    """Qwen3-ASR-0.6B int8 ONNX in-process (sherpa-onnx)."""

    def __init__(self, model_dir, threads, workers=2):
        import sherpa_onnx
        d = model_dir
        self.rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=f"{d}/conv_frontend.onnx", encoder=f"{d}/encoder.int8.onnx",
            decoder=f"{d}/decoder.int8.onnx", tokenizer=f"{d}/tokenizer", num_threads=threads, max_new_tokens=512)
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)

    def _run(self, audio):
        s = self.rec.create_stream()
        s.accept_waveform(SR, audio)
        self.rec.decode_stream(s)
        return s.result.text.strip()

    async def transcribe(self, audio):
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._run, audio), None


class WhisperSecondPass:
    """faster-whisper (GPU) as the final pass: one decode per closed piece, with word timestamps."""

    timed = True

    def __init__(self, backend):
        self.b = backend

    async def transcribe_words(self, audio, offset_s, language, candidates, prio=0, prompt=None, hotwords=None):
        words, lang = await self.b.transcribe(audio, language, prompt, hotwords, None if language else candidates, prio=prio)
        return [Word(w.word, offset_s + w.start, offset_s + w.end, float(w.probability), lang)
                for w in drop_tail_hallucinations(words)], lang


def align_words(text, ref, t0, t1, lang_hint, latin):
    """Give second-pass words times by aligning their characters to first-pass words."""
    toks = text.split()
    if not toks:
        return []
    ref_chars, ref_times = [], []
    for w in ref:
        cs = [c.lower() for c in w.text if c.isalnum()]
        for k, c in enumerate(cs):
            ref_chars.append(c)
            ref_times.append(w.start + (w.end - w.start) * k / max(1, len(cs)))
    hyp_chars, owner = [], []
    for i, w in enumerate(toks):
        for c in w:
            if c.isalnum():
                hyp_chars.append(c.lower())
                owner.append(i)
    starts = [None] * len(toks)
    for a, b, n in difflib.SequenceMatcher(None, hyp_chars, ref_chars, autojunk=False).get_matching_blocks():
        for k in range(n):
            i = owner[a + k]
            if starts[i] is None:
                starts[i] = ref_times[b + k]
    # interpolate unanchored words between neighbours (or the piece bounds)
    known = [(-1, t0)] + [(i, s) for i, s in enumerate(starts) if s is not None] + [(len(toks), t1)]
    for (i0, s0), (i1, s1) in zip(known, known[1:]):
        for i in range(i0 + 1, i1):
            starts[i] = s0 + (s1 - s0) * (i - i0) / (i1 - i0)
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1])
    ref_end = max((w.end for w in ref), default=t1)
    ends = starts[1:] + [min(t1, max(starts[-1] + 0.2, ref_end))]
    return [Word(" " + w, s, max(e, s), 1.0, script_lang(w, latin) or lang_hint) for w, s, e in zip(toks, starts, ends)]


# --------------------------------------------------------------------------- Nemotron
class NemotronBackend:
    """`workers` decode threads, each with its own recognizer; a session stays on one worker,
    and each worker batch-decodes all of its ready streams at once.
    The second pass must finish within `budget_s` of a piece closing; otherwise the piece is
    finalized with its first-pass words (overload degrades accuracy, not latency)."""

    step_s = 0.2

    def __init__(self, model_dir, threads, second_pass=None, workers=1, budget_s=1.5):
        import sherpa_onnx
        d = model_dir
        self.recs = [sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{d}/tokens.txt", encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx",
            joiner=f"{d}/joiner.int8.onnx", num_threads=threads) for _ in range(workers)]
        self.second = second_pass
        self.budget_s = budget_s
        self.missed = 0
        self.pools = [concurrent.futures.ThreadPoolExecutor(1) for _ in range(workers)]
        self.jobs = [[] for _ in range(workers)]
        self.wakes = None
        self.load = [0] * workers   # open streams per worker

    def open(self, language, candidates, prompt, hotwords):
        if self.wakes is None:
            self.wakes = [asyncio.Event() for _ in self.recs]
            for w in range(len(self.recs)):
                asyncio.get_running_loop().create_task(self._loop(w))
            if isinstance(self.second, WhisperSecondPass) and self.second.b.wake is None:
                self.second.b.open(None, None, None, None)   # start its worker tasks (routed: MultiBackend)
        latin = next((c for c in candidates or [language] if c and c not in ("ko", "ja", "zh")), "en")
        w = self.load.index(min(self.load))
        self.load[w] += 1
        return NemotronStream(self, w, language, latin, candidates, prompt, hotwords)

    def release(self, st):
        self.load[st.w] -= 1

    async def submit(self, st):
        fut = asyncio.get_running_loop().create_future()
        self.jobs[st.w].append((st, fut))
        self.wakes[st.w].set()
        await fut

    async def _loop(self, w):
        loop = asyncio.get_running_loop()
        while True:
            await self.wakes[w].wait()
            self.wakes[w].clear()
            jobs, self.jobs[w] = self.jobs[w], []
            if not jobs:
                continue
            try:
                await loop.run_in_executor(self.pools[w], self._work, w,
                                           list(dict.fromkeys(st for st, _ in jobs)))
                for _, f in jobs:
                    f.done() or f.set_result(None)
            except Exception as e:  # surface to the waiting sessions
                for _, f in jobs:
                    f.done() or f.set_exception(e)

    MAX_CHUNKS_PER_CALL = 6   # bound one call (~3 s of audio per stream); a burst backlog is drained over later calls

    def _work(self, w, streams):
        rec = self.recs[w]
        for st in streams:
            st._apply()
        for _ in range(self.MAX_CHUNKS_PER_CALL):  # batch-decode every stream that has a full chunk ready
            ready = [st.s for st in streams if rec.is_ready(st.s)]
            if not ready:
                break
            rec.decode_streams(ready)
        for st in streams:
            st._collect()
            st.backlog = rec.is_ready(st.s)


class NemotronStream:
    def __init__(self, backend, w, language, latin, candidates=None, prompt=None, hotwords=None):
        self.b, self.w, self.rec = backend, w, backend.recs[w]
        self.lang_opt, self.latin, self.candidates = language or "auto", latin, candidates
        self.prompt, self.hotwords = prompt, hotwords   # context for the second pass (sherpa NeMo: greedy only)
        self.second_enabled = True   # False: this first pass is final for the session's language
        self.language = language
        self.s = None
        self.base = 0            # absolute sample where the current sherpa stream starts
        self.fed = 0             # absolute samples handed to the stream
        self.pending = []        # audio waiting to be fed (applied on the worker thread)
        self.finish = False      # flush request (input_finished)
        self.restart = False     # start a fresh stream at self.fed on next apply
        self.toks = []           # (text, abs time, logprob) of the current stream
        self.done = 0            # tokens[:done] are finalized
        self.min_t = 0.0         # second pass: tokens before this time were re-transcribed
        # first-pass tokens decoded after their piece was handed to the second pass; kept so a
        # piece that falls back to first-pass words (second pass over budget) loses nothing
        self.limbo, self.limbo_lock = {}, threading.Lock()
        self.backlog = False     # audio fed but not yet decoded (input faster than real time)
        self.piece_langs = set()   # languages the re-reads detected for the open piece

    # worker thread only
    def _apply(self):
        if self.s is None or self.restart:
            self.s = self.rec.create_stream()
            self.s.set_option("language", self.lang_opt)
            self.base, self.toks, self.done, self.restart = self.fed - sum(map(len, self.pending)), [], 0, False
        for a in self.pending:
            self.s.accept_waveform(SR, a)
        self.pending = []
        if self.finish:
            # tail padding so the last chunk gets decoded, then flush
            self.s.accept_waveform(SR, np.zeros(int(0.6 * SR), np.float32))
            self.s.input_finished()
            self.finish = False

    def _collect(self):
        toks = self.rec.tokens(self.s)
        ts = self.rec.timestamps(self.s)
        lp = self.rec.ys_probs(self.s)
        lp = lp if len(lp) == len(toks) else [0.0] * len(toks)
        base = self.base / SR
        self.toks = [(t, base + s, p) for t, s, p in zip(toks, ts, lp)]
        if self.min_t:
            with self.limbo_lock:
                for tk in self.toks[self.done:]:
                    if tk[1] < self.min_t:
                        self.limbo[(tk[1], tk[0])] = tk

    def _feed(self, audio, end):
        if end > self.fed:
            self.pending.append(audio[self.fed:end].copy())
            self.fed = end

    def _words(self, toks):
        words, closed = [], True
        for i, (t, s, p) in enumerate(toks):
            if not t.strip():
                closed = True
                continue
            if closed or t.startswith(" ") or not words:
                words.append([" " + t.strip(), s, s, [p]])
            else:
                words[-1][0] += t
                words[-1][3].append(p)
            closed = False
            nxt = toks[i + 1][1] if i + 1 < len(toks) else s + 0.12
            words[-1][2] = max(nxt, s + 0.04)
        out = [Word(w, s, e, float(min(1.0, math.exp(sum(ps) / len(ps)))), script_lang(w, self.latin))
               for w, s, e, ps in words]
        return fill_langs(out, self.language)

    def _open_tokens(self):
        return [t for t in self.toks[self.done:] if t[1] >= self.min_t]

    async def partial(self, audio, start, end, urgent=False):
        self._feed(audio, end)
        await self.b.submit(self)
        return self._words(self._open_tokens())

    def release(self):
        self.b.release(self)

    def has_open(self):
        """Decoded first-pass tokens that are not final yet."""
        return bool(self._open_tokens())

    @property
    def progressive(self):
        return self.second_enabled and getattr(self.b.second, "timed", False)

    def reread_queued(self):
        """Second-pass requests waiting for a GPU worker (load signal for the re-read interval)."""
        return self.b.second.b.busy()

    def _piece_lang(self):
        """Language for the piece's final decode: reused when every re-read agreed (saves the
        detection encoder pass), detected afresh when the piece mixed languages."""
        return self.language or (next(iter(self.piece_langs)) if len(self.piece_langs) == 1 else None)

    def reread(self, audio, start, end):
        """Timed second-pass words for the open piece (progressive finalization). Each re-read
        detects its own window's language so a speaker/language switch inside a piece is followed."""
        seg = audio[start:end].copy()

        async def go():
            words, lang = await self.b.second.transcribe_words(seg, start / SR, self.language, self.candidates,
                                                               prio=1, prompt=self.prompt, hotwords=self.hotwords)
            if lang:
                self.piece_langs.add(lang)
            return words, lang
        return asyncio.get_running_loop().create_task(go())

    def second_pass(self, audio, start, cut):
        if not self.b.second or not self.second_enabled or cut <= start + int(0.3 * SR):
            return None
        seg = audio[start:cut].copy()
        if hasattr(self.b.second, "transcribe_words"):
            coro = self.b.second.transcribe_words(seg, start / SR, self._piece_lang(), self.candidates,
                                                  prompt=self.prompt, hotwords=self.hotwords)
        else:
            coro = self.b.second.transcribe(seg)
        return asyncio.get_running_loop().create_task(coro)

    async def close(self, audio, start, cut, flush, end, spec=None):
        self._feed(audio, end)
        cut_t = cut / SR
        task = spec or self.second_pass(audio, start, cut)
        self.piece_langs = set()   # the next piece detects its language afresh
        if task is None:
            # first pass is final: decode (and on flush, drain) everything up to the cut
            if flush:
                self.finish = True
            await self.b.submit(self)
        # With a second pass the final words come from it, so closing never waits for the
        # streaming decoder (it may be behind under load); its words only serve as the
        # non-final display meanwhile and as the fallback.
        open_toks = self._open_tokens()
        if flush:
            mine = open_toks
        else:
            # tokens are emitted with a chunk of look-ahead; anything not decoded yet
            # (time < cut) is finalized with the next piece instead of being lost
            mine = [t for t in open_toks if t[1] < cut_t]
        words = self._words(mine)
        if mine:
            self.done = self.toks.index(mine[-1]) + 1
        if flush:
            self.restart = True   # endpoint: next audio starts a fresh stream (bounded state)
        if task is None:
            self.min_t = 0.0
            return words, None
        self.min_t = max(self.min_t, cut_t)   # first-pass tokens before the cut are re-transcribed
        deadline = time.monotonic() + self.b.budget_s

        start_t = start / SR

        def late_tokens():
            with self.limbo_lock:
                late = [tk for k, tk in self.limbo.items() if start_t <= k[0] < cut_t]
                for k in [k for k in self.limbo if k[0] < cut_t]:   # taken now, or stale (before this piece)
                    self.limbo.pop(k, None)
            have = set(mine)
            return [tk for tk in late if tk not in have]

        async def finish():
            try:
                text, lang = await asyncio.wait_for(task, max(0.0, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                self.b.missed += 1
                log.info("second pass over budget (%d so far): keeping first-pass words", self.b.missed)
                return self._words(sorted(mine + late_tokens(), key=lambda tk: tk[1]))
            late = late_tokens()   # re-transcribed by the second pass: discarded, unless...
            if text is None:       # ...the language's final engine is this first pass itself
                return self._words(sorted(mine + late, key=lambda tk: tk[1]))
            if isinstance(text, list):   # timed words (Whisper second pass)
                return fill_langs(text, lang or self.language)
            out = align_words(text, words, start_t, cut_t, lang or self.language, self.latin)
            return fill_langs(out, lang or self.language)
        return words, finish()
