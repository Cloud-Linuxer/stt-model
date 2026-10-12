"""anyreach-ai/dualturn-endpointing (offline ONNX): frozen Mimi features + causal head, 80 ms frames.
Input: 2-channel 24 kHz waveform [user, agent]; agent channel is silent here. Output: EOT prob per frame;
we read the last frame (the pause the policy is deciding on)."""
import os

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "eot", "dualturn", "model.onnx")


class DualTurn:
    def __init__(self, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])

    def __call__(self, audio16k):
        x = torchaudio.functional.resample(torch.from_numpy(np.asarray(audio16k, dtype=np.float32)), 16000, 24000).numpy()
        inp = np.stack([x, np.zeros_like(x)]).astype(np.float32)
        eot = self.sess.run(["eot_probs"], {"audio_24k": inp})[0]
        return float(eot[0, -1, 0])


def load():
    return DualTurn()


def complete(h, audio):
    return h(audio)
