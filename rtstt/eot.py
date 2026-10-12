"""Semantic end-of-turn detection (pluggable), used by the session's endpointing.

The session asks `score(turn_audio, hypothesis_text)` once a pause of `check_ms` has formed and
gets P(turn complete). Detectors:
  SmartTurnDetector  pipecat-ai/smart-turn v3 (Whisper-tiny encoder + linear head, ~8 MB int8 ONNX,
                     23 languages incl. ko/en, BSD-2). Audio of the current turn, last 8 s,
                     zero-padded at the BEGINNING. ~12 ms per call on one CPU core.
  NamoTextDetector   videosdk-live/Namo-Turn-Detector-v1-Korean (DistilBERT, quantized ONNX, Apache-2.0):
                     P(turn complete) from the Korean hypothesis text of the current turn. <14 ms CPU.
  CombinedDetector   Korean hypothesis -> Namo (text), anything else -> smart-turn (audio).
Measured on our pause points (bench/eot_eval.py): Namo keeps mid-sentence false ends at the
silence rule's level (~19%) while ending complete turns at ~0.3 s instead of 0.5 s.
"""
import concurrent.futures
import os

import numpy as np
import onnxruntime as ort

SR = 16000
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "eot", "smart-turn-v3.2-cpu.onnx")
SENT_END = ".?!。？！"
NAMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "eot", "namo_ko")


class SmartTurnDetector:
    def __init__(self, path=DEFAULT_MODEL, threads=2, workers=2):
        from transformers import WhisperFeatureExtractor
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.fe = WhisperFeatureExtractor(chunk_length=8)
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)

    def _score(self, audio):
        audio = np.asarray(audio[-8 * SR:], dtype=np.float32)
        if len(audio) < 8 * SR:
            audio = np.concatenate([np.zeros(8 * SR - len(audio), np.float32), audio])
        feats = self.fe(audio, sampling_rate=SR, return_tensors="np", do_normalize=True)["input_features"]
        return float(self.sess.run(None, {"input_features": feats.astype(np.float32)})[0][0][0])

    async def score(self, audio, text=None):
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._score, audio)


class NamoTextDetector:
    def __init__(self, path=NAMO_DIR, threads=2, workers=2):
        from transformers import AutoTokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(os.path.join(path, "model_quant.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(path)
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)

    def _score(self, text):
        enc = self.tok(text[-600:], truncation=True, max_length=512, return_tensors="np")
        logits = self.sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                      "attention_mask": enc["attention_mask"].astype(np.int64)})[0][0]
        p = np.exp(logits - logits.max())
        return float(p[1] / p.sum())

    async def score(self, text):
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._score, text)


class CombinedDetector:
    def __init__(self, audio, text=None, text_langs=("ko",)):
        self.audio, self.text, self.text_langs = audio, text, set(text_langs)

    async def score(self, audio, text, lang):
        if self.text is not None and lang in self.text_langs and text and text.strip():
            return await self.text.score(text)
        return await self.audio.score(audio)
