"""Slow automatic gain control for quiet input (boost only, capped).

Very quiet recordings (e.g. -60 dBFS) are invisible to the VAD although ASR models can still
read them. The gain follows a decaying peak of the chunk RMS, so it settles within a few
seconds, never attenuates, and never boosts digital silence / the noise floor beyond the cap.
"""
import numpy as np


class AGC:
    def __init__(self, sr=16000, target_rms=0.05, max_gain=30.0, floor_rms=1e-4, decay_s=4.0):
        self.sr, self.target, self.max_gain, self.floor = sr, target_rms, max_gain, floor_rms
        self.decay_s = decay_s
        self.level = 0.0     # decaying peak of chunk RMS
        self.gain = 1.0

    def __call__(self, x):
        if len(x) == 0:
            return x
        rms = float(np.sqrt(np.mean(x ** 2)))
        self.level = max(rms, self.level * 0.5 ** (len(x) / self.sr / self.decay_s))
        want = 1.0 if self.level < self.floor else min(self.max_gain, max(1.0, self.target / self.level))
        a = min(1.0, len(x) / self.sr / 0.5)          # ~0.5 s smoothing of gain changes
        self.gain += a * (want - self.gain)
        return np.clip(x * self.gain, -1.0, 1.0) if self.gain != 1.0 else x
