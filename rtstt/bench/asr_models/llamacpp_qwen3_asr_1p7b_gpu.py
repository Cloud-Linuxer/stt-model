"""Qwen3-ASR-1.7B (Q8_0 GGUF) on the GPU llama.cpp server (rtstt-asr2, gpu-min profile)."""
import base64
import io

import requests
import soundfile as sf

URL = "http://localhost:8094/v1/chat/completions"


def load(device):
    return requests.Session()


def transcribe(sess, chunks, lang):
    out = []
    for c in chunks:
        buf = io.BytesIO()
        sf.write(buf, c, 16000, format="WAV", subtype="PCM_16")
        r = sess.post(URL, json={"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {
            "data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}]}],
            "temperature": 0, "max_tokens": 512}, timeout=300)
        text = r.json()["choices"][0]["message"]["content"]
        out.append(text.split("<asr_text>", 1)[-1].strip())
    return out
