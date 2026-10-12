"""Shared helpers for sherpa-onnx (CPU, int8) adapters."""
import glob
import os

import numpy as np

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "asr")
THREADS = int(os.environ.get("THREADS", "4"))
SR = 16000


def mdir(prefix):
    return glob.glob(os.path.join(MODELS, prefix + "*.dl", "*"))[0]


def offline(rec, chunks, lang=None):
    out = []
    for c in chunks:
        s = rec.create_stream()
        if lang is not None:
            s.set_option("language", lang)
        s.accept_waveform(SR, c)
        rec.decode_stream(s)
        out.append(s.result.text.strip())
    return out


def online(rec, chunks, lang=None):
    """Feed each chunk as a live stream in 100 ms pushes (how the server would use it)."""
    out = []
    tail = np.zeros(int(0.8 * SR), np.float32)
    for c in chunks:
        s = rec.create_stream()
        if lang is not None:
            s.set_option("language", lang)
        a = np.concatenate([c, tail])
        for i in range(0, len(a), 1600):
            s.accept_waveform(SR, a[i:i + 1600])
            while rec.is_ready(s):
                rec.decode_stream(s)
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        out.append(rec.get_result(s).strip())
    return out
