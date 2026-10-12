"""Language routing for the 60 supported languages.

Which engine serves each language was chosen on FLEURS (bench/fleurs_eval.py, 20 clips per
language): the final pass uses the most accurate engine for the language, the non-final
(streaming) pass uses Nemotron where it supports the language, Omnilingual-ASR where that is
the final engine (Indic and some low-resource languages), and Whisper re-decoding otherwise.
Translation: Hy-MT2 between ko/en/zh/de (where it won on real conversation and FLEURS),
MADLAD-400 for every other pair (any-to-any, 400+ languages).
"""
import asyncio
import collections
import concurrent.futures
import os
import unicodedata

import numpy as np

from asr_backends import SR, Word, WhisperStream, drop_tail_hallucinations, fill_langs, script_lang

LANGS = ("af sq ar az eu be bn bs bg ca zh hr cs da nl en et fi fr gl de el gu he hi hu id it ja kn kk ko "
                "lv lt mk ms ml mr no fa pl pt pa ro ru sr sk sl es sw sv tl ta te th tr uk ur vi cy").split()
# Nemotron-3.5 streaming: transcription-ready + broad-coverage locales
NEMOTRON = {"en": "en-US", "es": "es-ES", "fr": "fr-FR", "it": "it-IT", "pt": "pt-BR", "nl": "nl-NL", "de": "de-DE",
            "tr": "tr-TR", "ru": "ru-RU", "ar": "ar-AR", "hi": "hi-IN", "ja": "ja-JP", "ko": "ko-KR", "vi": "vi-VN",
            "uk": "uk-UA", "pl": "pl-PL", "sv": "sv-SE", "cs": "cs-CZ", "no": "nb-NO", "da": "da-DK", "bg": "bg-BG",
            "fi": "fi-FI", "hr": "hr-HR", "sk": "sk-SK", "zh": "zh-CN", "hu": "hu-HU", "ro": "ro-RO", "et": "et-EE"}
# most accurate final engine per language on FLEURS (sq, eu: not in FLEURS, Whisper large-v3)
LARGE = "az cy da el en et fi he id mk ro ru th tr uk ur zh sq eu".split()
OMNI = "af be bn fa gu kk kn ml mr pa sl sw ta te".split()
NEMO_FINAL = "ar hi".split()
FINAL = {**{l: "whisper_turbo" for l in LANGS}, **{l: "whisper_large_v3" for l in LARGE},
         **{l: "omnilingual" for l in OMNI}, **{l: "nemotron" for l in NEMO_FINAL}}
HYMT_PAIRS = {"ko", "en", "zh", "de"}
MADLAD_TAG = {"tl": "fil"}   # MADLAD target tags that differ from our language codes


def _script(word):
    for c in word:
        if c.isalpha():
            try:
                return unicodedata.name(c).split()[0]
            except ValueError:
                return None
    return None


class OmniEngine:
    """Omnilingual-ASR 1B CTC (sherpa-onnx, CPU) with character timestamps."""

    def __init__(self, model_dir, threads=4, workers=2):
        import sherpa_onnx
        self.rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=f"{model_dir}/model.int8.onnx", tokens=f"{model_dir}/tokens.txt", num_threads=threads)
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)

    def _run(self, audio):
        s = self.rec.create_stream()
        s.accept_waveform(SR, audio)
        self.rec.decode_stream(s)
        return list(s.result.tokens), list(s.result.timestamps)

    async def words(self, audio, offset_s, lang):
        toks, ts = await asyncio.get_running_loop().run_in_executor(self.pool, self._run, audio)
        words, cur, start = [], "", None
        for t, s in zip(toks + [" "], ts + [ts[-1] + 0.1 if ts else 0.0]):
            if t.isspace() or t == "▁":
                if cur:
                    words.append([" " + cur, start, s])
                cur, start = "", None
                continue
            if start is None:
                start = s
            cur += t
        # CTC sometimes emits stray words in another script (Arabic / CJK inside Tamil): keep
        # the piece's dominant script (plus digits / Latin)
        scripts = [_script(w) for w, _, _ in words]
        main = collections.Counter(x for x in scripts if x not in (None, "LATIN")).most_common(1)
        main = main[0][0] if main else None
        out = []
        for i, ((w, s, e), sc) in enumerate(zip(words, scripts)):
            if main and sc not in (None, "LATIN", main):
                continue
            nxt = words[i + 1][1] if i + 1 < len(words) else e + 0.1
            out.append(Word(w, offset_s + s, offset_s + max(s + 0.05, min(nxt, e + 0.3)), 1.0, lang))
        return out


class RoutedSecondPass:
    """Final pass that picks the engine by language (detected with Whisper when not given)."""

    timed = True

    def __init__(self, turbo, large=None, omni=None):
        self.turbo, self.large, self.omni = turbo, large, omni
        self.b = turbo   # for load checks (busy())

    def engine(self, lang):
        e = FINAL.get(lang, "whisper_turbo")
        if (e == "whisper_large_v3" and not self.large) or (e == "omnilingual" and not self.omni):
            return "whisper_turbo"
        return e

    async def detect(self, audio, candidates):
        def run():
            _, _, probs = self.turbo.model.detect_language(audio)
            probs = dict(probs)
            return max(candidates or probs, key=lambda c: probs.get(c, 0.0))
        return await asyncio.get_running_loop().run_in_executor(self.turbo.pool, run)

    async def transcribe_words(self, audio, offset_s, language, candidates, prio=0, prompt=None, hotwords=None):
        lang = language or await self.detect(audio, candidates)
        e = self.engine(lang)
        if e == "nemotron":
            return None, lang     # the streaming first pass is the final result for this language
        if e == "omnilingual":
            return await self.omni.words(audio, offset_s, lang), lang
        backend = self.large if e == "whisper_large_v3" else self.turbo
        words, _ = await backend.transcribe(audio, lang, prompt, hotwords, None, prio=prio)
        return [Word(w.word, offset_s + w.start, offset_s + w.end, float(w.probability), lang)
                for w in drop_tail_hallucinations(words)], lang


class OmniStream:
    """First (and final) pass for Omnilingual languages: re-decode the open piece on the CPU.
    Pieces close only at longer pauses (the CTC model needs context), so finals come per
    sentence. (LocalAgreement did not help here: CTC re-reads rarely agree word for word.)"""

    progressive = False
    min_gap_s = 0.8    # re-decode the open piece at most this often
    max_open_s = 12.0  # CTC is stable up to ~15 s; shorter forced cuts split words
    split_pause_ms = 450   # longer pieces: the CTC model needs context more than early finals

    def __init__(self, engine, language):
        self.e, self.language = engine, language
        self.last, self.last_end = [], 0

    async def partial(self, audio, start, end, urgent=False):
        if (end - self.last_end < SR * self.min_gap_s and not urgent) or end - start < SR * 0.3:
            return self.last
        self.last, self.last_end = await self.e.words(audio[start:end].copy(), start / SR, self.language), end
        return self.last

    async def close(self, audio, start, cut, flush, end, spec=None):
        if self.last and self.last_end >= cut:
            words = [w for w in self.last if (w.start + w.end) / 2 < cut / SR]
        else:
            words = await self.e.words(audio[start:cut].copy(), start / SR, self.language) if cut > start else []
        self.last, self.last_end = [], 0
        return words, None

    def second_pass(self, audio, start, cut):
        return None

    def has_open(self):
        return False

    def release(self):
        pass


class RoutedWhisperStream(WhisperStream):
    """Whisper turbo re-decoding for non-final tokens; finals from the routed engine when it is
    not turbo (large-v3 / Omnilingual), with the same budget fallback as the Nemotron path."""

    def __init__(self, backend, language, candidates, prompt, hotwords, second, budget_s):
        super().__init__(backend, language, candidates, prompt, hotwords)
        self.second, self.budget_s = second, budget_s

    progressive = True

    def reread_queued(self):
        return self.b.busy()

    def reread(self, audio, start, end):
        return asyncio.get_running_loop().create_task(self.second.transcribe_words(
            audio[start:end].copy(), start / SR, self.language or self.piece_lang, self.candidates, prio=1,
            prompt=self.prompt, hotwords=self.hotwords))

    def second_pass(self, audio, start, cut):
        lang = self.language or self.piece_lang
        if cut <= start + int(0.3 * SR) or (lang and self.second.engine(lang) == "whisper_turbo"):
            return None   # turbo is the final engine: the re-decode is promoted as is
        return asyncio.get_running_loop().create_task(self.second.transcribe_words(
            audio[start:cut].copy(), start / SR, lang, self.candidates, prompt=self.prompt, hotwords=self.hotwords))

    async def close(self, audio, start, cut, flush, end, spec=None):
        task = spec or self.second_pass(audio, start, cut)
        words, _ = await super().close(audio, start, cut, flush, end)
        if task is None:
            return words, None
        deadline = asyncio.get_running_loop().time() + self.budget_s

        async def finish():
            try:
                routed, lang = await asyncio.wait_for(task, max(0.0, deadline - asyncio.get_running_loop().time()))
            except asyncio.TimeoutError:
                return words
            return words if routed is None else fill_langs(routed, lang)
        return words, finish()


class MultiBackend:
    """Picks the streaming path per session from its language hints."""

    step_s = 0.2

    def __init__(self, nemotron, turbo, routed, omni, budget_s):
        self.nemotron, self.turbo, self.routed, self.omni, self.budget_s = nemotron, turbo, routed, omni, budget_s

    def open(self, language, candidates, prompt, hotwords):
        langs = [language] if language else list(candidates or [])
        if self.turbo.wake is None:
            self.turbo.open(None, None, None, None)
        if self.routed.large is not None and self.routed.large.wake is None:
            self.routed.large.open(None, None, None, None)
        if len(langs) == 1 and FINAL.get(langs[0]) == "omnilingual" and self.omni:
            return OmniStream(self.omni, langs[0])
        if langs and all(l in NEMOTRON for l in langs):
            st = self.nemotron.open(language, candidates, prompt, hotwords)
            st.lang_opt = NEMOTRON[language] if language else "auto"
            st.second_enabled = not (language and FINAL.get(language) == "nemotron")
            return st
        return RoutedWhisperStream(self.turbo, language, candidates, prompt, hotwords, self.routed, self.budget_s)


class MadladTranslator:
    """MADLAD-400-3B (CTranslate2): any-to-any, 400+ languages; whole sentence at once."""

    def __init__(self, path, device="cuda"):
        import ctranslate2
        import sentencepiece as spm
        self.tr = ctranslate2.Translator(path, device=device,
                                         compute_type="int8_float16" if device == "cuda" else "int8")
        self.sp = spm.SentencePieceProcessor(os.path.join(path, "spiece.model"))
        self.pool = concurrent.futures.ThreadPoolExecutor(2)

    def _run(self, text, tgt):
        toks = self.sp.encode(f"<2{MADLAD_TAG.get(tgt, tgt)}> {text}", out_type=str)
        return self.sp.decode(self.tr.translate_batch([toks], beam_size=1, max_decoding_length=256)[0].hypotheses[0])

    async def stream(self, text, src, tgt, history=(), glossary=None, general=None):
        yield await asyncio.get_running_loop().run_in_executor(self.pool, self._run, text, tgt)


class RoutedTranslator:
    def __init__(self, hymt, madlad):
        self.hymt, self.madlad = hymt, madlad

    def stream(self, text, src, tgt, history=(), glossary=None, general=None):
        use_hymt = self.hymt is not None and src in HYMT_PAIRS and tgt in HYMT_PAIRS
        return (self.hymt if use_hymt else self.madlad).stream(text, src, tgt, history, glossary, general)
