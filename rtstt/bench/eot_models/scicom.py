"""Scicom-intl/semantic-vad-eot-whisper-small: Whisper-small encoder + head, int8 ONNX (CPU).
Input: log-mel of the last 8 s (left-padded, do_normalize=False); output already a probability."""
import os

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "eot", "scicom", "model.int8.onnx")
SR, WINDOW = 16000, 8 * 16000


class Scicom:
    def __init__(self, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
        self.fe = WhisperFeatureExtractor(feature_size=80, sampling_rate=SR, chunk_length=8)

    def __call__(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32)
        pcm = pcm[-WINDOW:] if len(pcm) >= WINDOW else np.pad(pcm, (WINDOW - len(pcm), 0))
        feats = self.fe([pcm], sampling_rate=SR, return_tensors="np", padding="max_length", max_length=WINDOW,
                        truncation=True, do_normalize=False)["input_features"].astype(np.float32)
        return float(self.sess.run(None, {"input_features": feats})[0].reshape(-1)[0])


def load():
    return Scicom()


def complete(h, audio):
    return h(audio)
