"""Baseline: the current hypothesis ends with sentence punctuation (Whisper turbo decode of the turn)."""
from faster_whisper import WhisperModel

PATH = "/home/stt-model/models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"


def load():
    return WhisperModel(PATH, device="cuda", compute_type="float16")


def complete(m, audio):
    segs, _ = m.transcribe(audio, beam_size=1, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
    text = "".join(s.text for s in segs).strip()
    return 1.0 if text[-1:] in ".?!。？！" else 0.0
