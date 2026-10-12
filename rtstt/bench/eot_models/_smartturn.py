"""pipecat-ai/smart-turn v3: Whisper-tiny log-mel of the last 8 s of the turn -> P(turn complete)."""
import os

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "eot")


class SmartTurn:
    def __init__(self, path, providers):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(path, so, providers=providers)
        self.fe = WhisperFeatureExtractor(chunk_length=8)

    def __call__(self, audio):
        """audio: the current turn (16 kHz). Last 8 s are used; shorter turns are zero-padded at
        the BEGINNING (right-padding makes the model answer 'complete' for everything)."""
        audio = audio[-8 * 16000:].astype(np.float32)
        if len(audio) < 8 * 16000:
            audio = np.concatenate([np.zeros(8 * 16000 - len(audio), np.float32), audio])
        feats = self.fe(audio, sampling_rate=16000, return_tensors="np", do_normalize=True)["input_features"]
        return float(self.sess.run(None, {"input_features": feats.astype(np.float32)})[0][0][0])   # already a probability
