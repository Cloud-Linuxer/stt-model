"""Incremental Silero VAD: each new 32 ms frame is scored once per stream.

faster-whisper's get_speech_timestamps() re-scores the whole pending audio on every call;
with many streams that dominated the event loop. Here the frame probabilities are kept per
stream (the decoder LSTM state is carried across calls) and segmentation runs on the stored
probabilities, so several segmentations (coarse / fine tail) cost no extra model runs.
Segmentation is the same algorithm as faster_whisper.vad.get_speech_timestamps.
"""
import numpy as np
from faster_whisper.vad import get_vad_model

FRAME = 512
CTX = 64


class StreamVAD:
    def __init__(self, agc=None):
        """agc: optional gain control applied to the VAD input only (quiet speakers)."""
        self.agc = agc
        self.model = get_vad_model()
        self.state = np.zeros((2, 1, 128), dtype="float32")
        self.ctx = np.zeros(CTX, dtype="float32")
        self.probs = []                              # one per frame, from absolute frame probs_off on
        self.probs_off = 0                          # frames dropped by trim()
        self.pos = 0                                # samples scored

    def feed(self, audio):
        """Score all complete new frames of `audio` (the full stream buffer)."""
        n = (len(audio) - self.pos) // FRAME
        if n <= 0:
            return
        new = audio[self.pos:self.pos + n * FRAME]
        if self.agc is not None:
            new = self.agc(new)
        frames = new.reshape(n, FRAME)
        ctx = np.concatenate([self.ctx[None], frames[:-1, -CTX:]], 0)
        enc = self.model.encoder_session.run(None, {"input": np.concatenate([ctx, frames], 1)})[0].reshape(n, 128)
        out = np.empty(n, dtype="float32")
        for i in range(n):
            p, self.state = self.model.decoder_session.run(None, {"input": enc[i:i + 1], "state": self.state})
            out[i] = p.reshape(-1)[0]
        self.ctx = frames[-1, -CTX:].copy()
        self.probs.extend(out.tolist())
        self.pos += n * FRAME

    def trim(self, keep_from):
        """Forget frame scores before absolute sample keep_from (nothing will ask for them again)."""
        k = keep_from // FRAME - self.probs_off
        if k > 0:
            del self.probs[:k]
            self.probs_off += k

    def segments(self, start, end, opts):
        """Speech segments of audio[start:end] as [{"start", "end"}] relative to `start`."""
        f0 = max(start // FRAME, self.probs_off)
        f1 = min(len(self.probs) + self.probs_off, end // FRAME)
        off = start - f0 * FRAME                    # frame grid offset relative to start
        probs = self.probs[f0 - self.probs_off:f1 - self.probs_off] if f1 > f0 else []
        length = end - start
        threshold = opts.threshold
        neg = opts.neg_threshold if opts.neg_threshold is not None else max(threshold - 0.15, 0.01)
        min_speech = 16000 * opts.min_speech_duration_ms / 1000
        pad = 16000 * opts.speech_pad_ms / 1000
        min_sil = 16000 * opts.min_silence_duration_ms / 1000
        speeches, cur, triggered, temp_end = [], {}, False, 0
        for i, p in enumerate(probs):
            pos = max(0, FRAME * i - off)
            if p >= threshold and temp_end:
                temp_end = 0
            if p >= threshold and not triggered:
                triggered = True
                cur["start"] = pos
                continue
            if p < neg and triggered:
                if not temp_end:
                    temp_end = pos
                if pos - temp_end < min_sil:
                    continue
                cur["end"] = temp_end
                if cur["end"] - cur["start"] > min_speech:
                    speeches.append(cur)
                cur, triggered, temp_end = {}, False, 0
        if cur and length - cur["start"] > min_speech:
            cur["end"] = length
            speeches.append(cur)
        for i, sp in enumerate(speeches):
            if i == 0:
                sp["start"] = int(max(0, sp["start"] - pad))
            if i != len(speeches) - 1:
                gap = speeches[i + 1]["start"] - sp["end"]
                if gap < 2 * pad:
                    sp["end"] += int(gap // 2)
                    speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - gap // 2))
                else:
                    sp["end"] = int(min(length, sp["end"] + pad))
                    speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - pad))
            else:
                sp["end"] = int(min(length, sp["end"] + pad))
        return speeches
