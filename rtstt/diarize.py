"""Streaming speaker diarization: sliding-window speaker embeddings + online clustering."""
import os

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

SR = 16000
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                             "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")


class Embedder:
    def __init__(self, path=DEFAULT_MODEL, threads=2):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.scale = 1.0 if meta.get("normalize_samples", "1") == "1" else 32768.0
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, windows):
        """windows: list of equal-length float32 arrays -> (N, D) L2-normalized."""
        feats = []
        for w in windows:
            f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(w * self.scale)[None], num_mel_bins=80,
                                                  dither=0.0, sample_frequency=SR)
            feats.append((f - f.mean(0)).numpy())
        e = self.sess.run(None, {self.inp: np.stack(feats)})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class OnlineDiarizer:
    """Per-session state. Speakers are "1", "2", ... in order of first appearance.

    The server soft-splits audio at pauses, so a segment is usually one speaker:
    segments up to `single_s` get one whole-segment embedding (short windows are too
    noisy); longer ones use `win_s` windows. Segments shorter than `min_new_s` can only
    join an existing speaker and never update centroids."""

    def __init__(self, embedder, threshold=0.45, win_s=3.0, hop_s=1.0, single_s=6.0,
                 min_new_s=1.5, max_speakers=15):
        self.emb, self.thr = embedder, threshold
        self.win, self.hop = int(win_s * SR), int(hop_s * SR)
        self.single, self.min_new = int(single_s * SR), int(min_new_s * SR)
        self.max_speakers = max_speakers
        self.centroids = []  # running sums (unnormalized)
        self._cache = {}     # (abs start, abs end) -> embedding; partial re-labels of a growing piece reuse old units
        self._updated = set()   # unit keys that already shaped a centroid (a piece is re-labelled several times)

    def _windows(self, audio):
        if len(audio) <= self.single:
            starts, wins = [0], [audio]
        else:
            starts = list(range(0, len(audio) - self.win + 1, self.hop))
            if starts[-1] + self.win < len(audio):
                starts.append(len(audio) - self.win)
            wins = [audio[s:s + self.win] for s in starts]
        # model input must share a length: tile short ones up to the longest
        n = max(len(w) for w in wins)
        return starts, [np.tile(w, int(np.ceil(n / len(w))))[:n] for w in wins], [len(w) for w in wins]

    def _assign(self, e, allow_new):
        """Nearest centroid index, or len(centroids) meaning "new speaker"."""
        if self.centroids:
            C = np.array([c / np.linalg.norm(c) for c in self.centroids])
            sims = C @ e
            k = int(np.argmax(sims))
            if sims[k] >= self.thr or not allow_new or len(self.centroids) >= self.max_speakers:
                return k
        return len(self.centroids)

    # ---- speaker changes inside a piece ----
    CTX = int(2.0 * SR)      # left/right block compared at each candidate boundary
    HOP = int(0.5 * SR)
    CHANGE_SIM = 0.30        # cos(left block, right block) below this -> a different voice
    MIN_SEG = int(1.0 * SR)

    def _change_points(self, audio, words):
        """Speaker-change times (seconds, relative) inside `audio`, snapped to word boundaries.
        Windows of CTX at HOP are embedded in one batch; a boundary at t compares the window
        ending at t with the window starting at t (both 2 s, no overlap)."""
        L = len(audio)
        if L < 2 * self.CTX + self.HOP or len(words) < 2:
            return []
        starts = list(range(0, L - self.CTX + 1, self.HOP))
        E = self.emb([audio[s:s + self.CTX] for s in starts])
        n_ahead = self.CTX // self.HOP   # window index offset for "the window starting at t"
        sims = {}
        for j in range(n_ahead, len(starts)):
            t = starts[j]                # boundary: left = window j-n_ahead ends at t, right = window j
            sims[t] = float(E[j - n_ahead] @ E[j])
        gaps = [((a.end + b.start) / 2) for a, b in zip(words, words[1:])]   # word boundaries (s)
        cps = []
        for t, sim in sims.items():
            if sim >= self.CHANGE_SIM:
                continue
            # local minimum among neighbours
            if any(sims.get(t + d, 1.0) < sim for d in (-self.HOP, self.HOP)):
                continue
            # snap to the nearest word boundary within 0.35 s
            cand = [g for g in gaps if abs(g - t / SR) <= 0.35]
            if not cand:
                continue
            g = min(cand, key=lambda x: abs(x - t / SR))
            if cps and (g - cps[-1]) * SR < self.MIN_SEG:
                continue
            if g * SR < self.MIN_SEG or L - g * SR < self.MIN_SEG:
                continue
            cps.append(g)
        return cps

    def _embed_seg(self, seg):
        tile = np.tile(seg, int(np.ceil(self.win / max(1, len(seg)))))[:self.win] if len(seg) < self.win else seg
        return self.emb([tile])[0]

    def _label_seg(self, seg, e, update, prev):
        """Label one speech segment. prev = (embedding, label) of the previous segment or None."""
        n = len(seg)
        if self.centroids:
            C = np.array([c / np.linalg.norm(c) for c in self.centroids])
            sims = C @ e
            k = int(np.argmax(sims))
            best = float(sims[k])
        else:
            k, best = -1, -1.0
        if n >= self.min_new:
            lab = k if best >= self.thr or len(self.centroids) >= self.max_speakers else len(self.centroids)
        elif best >= self.thr - 0.1:
            lab = k                                              # short but a known voice
        elif prev is not None and float(prev[0] @ e) >= self.thr - 0.1:
            lab = prev[1]                                        # short: same voice as the previous segment
        else:
            lab = k if k >= 0 else len(self.centroids)           # short, unclear: nearest (never a new speaker)
        if update:
            if lab >= len(self.centroids):
                self.centroids.append(e.copy())
                lab = len(self.centroids) - 1
            elif n >= self.min_new and best >= self.thr:
                self.centroids[lab] = self.centroids[lab] + e    # only clean, long segments shape a centroid
        return min(lab, max(0, len(self.centroids) - 1)) if self.centroids else 0

    def label_words(self, audio, offset, words, update, segments=None):
        """audio: piece samples; words: [(start_s, end_s)] relative to the piece;
        segments: VAD speech segments [{"start","end"}] (samples, relative) - the labelling unit.
        Long segments are split at detected speaker changes. Returns speaker strings ("1", "2", ...)."""
        if not words:
            return []
        if len(audio) < SR * 0.4:
            return [self._last()] * len(words)
        from collections import namedtuple
        W = namedtuple("W", "start end")
        segs = [(int(x["start"]), int(min(len(audio), x["end"]))) for x in (segments or [{"start": 0, "end": len(audio)}])]
        segs = [x for x in segs if x[1] - x[0] >= int(0.3 * SR)] or [(0, len(audio))]
        # merge fragments shorter than 1 s into the previous segment (a short burst is not a turn)
        merged = []
        for s0, s1 in segs:
            if merged and (s1 - s0 < SR or s0 - merged[-1][1] < int(0.15 * SR)) and s0 - merged[-1][1] < int(0.4 * SR):
                merged[-1] = (merged[-1][0], s1)
            else:
                merged.append((s0, s1))
        segs = merged
        # split long segments at speaker changes (word-boundary snapped)
        units = []
        for s0, s1 in segs:
            wl = [W(a - s0 / SR, b - s0 / SR) for a, b in words if s0 / SR <= (a + b) / 2 < s1 / SR]
            cps = self._change_points(audio[s0:s1], wl) if s1 - s0 >= 2 * self.CTX + self.HOP else []
            bounds = [s0] + [s0 + int(c * SR) for c in cps] + [s1]
            units += list(zip(bounds, bounds[1:]))
        labels, prev = {}, None
        for u0, u1 in units:
            seg = audio[u0:u1]
            if len(seg) < int(0.2 * SR):
                continue
            key = (offset + u0, offset + u1)
            e = self._cache.get(key)
            if e is None and not update:
                # provisional labels of a still-growing unit: re-embed only once it grew by >= 1 s
                prev_key = next((k for k in reversed(self._cache) if k[0] == key[0] and 0 <= key[1] - k[1] < SR), None)
                e = self._cache.get(prev_key) if prev_key else None
            if e is None:
                e = self._cache[key] = self._embed_seg(seg)
                while len(self._cache) > 128:
                    del self._cache[next(iter(self._cache))]
            upd = update and key not in self._updated
            if upd:
                self._updated.add(key)
                while len(self._updated) > 256:
                    self._updated.discard(next(iter(self._updated)))
            lab = self._label_seg(seg, e, upd, prev)
            prev = (e, lab)
            labels[(u0, u1)] = lab
        if not labels:
            return [self._last()] * len(words)
        keys = list(labels)
        out = []
        for a, b in words:
            mid = (a + b) / 2 * SR
            inside = [k for k in keys if k[0] <= mid < k[1]]
            k = inside[0] if inside else min(keys, key=lambda kk: min(abs(kk[0] - mid), abs(kk[1] - mid)))
            out.append(str(labels[k] + 1))
        return out

    def _last(self):
        return str(max(1, len(self.centroids)))

    @staticmethod
    def _smooth(labels):
        # median-of-3 style: drop single-window flips
        out = list(labels)
        for i in range(1, len(labels) - 1):
            if labels[i - 1] == labels[i + 1] != labels[i]:
                out[i] = labels[i - 1]
        return out
