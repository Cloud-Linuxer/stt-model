"""Shared faster-whisper adapter."""
import os

from faster_whisper import WhisperModel


def load_fw(path, device):
    ct = "float16" if device == "cuda" else "int8"
    return WhisperModel(path, device=device, compute_type=ct, cpu_threads=int(os.environ.get("THREADS", "4")))


def transcribe_fw(m, chunks, lang):
    out = []
    for c in chunks:
        segs, _ = m.transcribe(c, language=None if lang == "auto" else lang, beam_size=1, temperature=0.0,
                               condition_on_previous_text=False, vad_filter=False)
        out.append("".join(s.text for s in segs).strip())
    return out
