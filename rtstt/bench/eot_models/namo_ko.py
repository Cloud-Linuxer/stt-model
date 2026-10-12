"""videosdk-live/Namo-Turn-Detector-v1-Korean: text-only DistilBERT (quantized ONNX).
The transcript of the turn comes from Whisper turbo (GPU) here; in the server it would be the live hypothesis."""
import os

import numpy as np
import onnxruntime as ort
from faster_whisper import WhisperModel
from transformers import AutoTokenizer

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "eot", "namo_ko")
WHISPER = "/home/stt-model/models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"


class Namo:
    def __init__(self):
        self.sess = ort.InferenceSession(os.path.join(D, "model_quant.onnx"), providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(D)
        self.asr = WhisperModel(WHISPER, device="cuda", compute_type="float16")
        self.last_text = ""

    def text_of(self, audio):
        segs, _ = self.asr.transcribe(audio, beam_size=1, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return "".join(s.text for s in segs).strip()

    def __call__(self, audio):
        text = self.text_of(audio)
        self.last_text = text
        if not text:
            return 0.0
        enc = self.tok(text, truncation=True, max_length=512, padding=True, return_tensors="np")
        feed = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}
        logits = self.sess.run(None, feed)[0][0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        return float(p[1])   # label 1 = end of turn


def load():
    return Namo()


def complete(h, audio):
    return h(audio)
