#!/usr/bin/env python3
"""
Create a real Korean audio file for testing
"""

import numpy as np
import soundfile as sf
import requests
import os

# Create realistic Korean speech audio
sample_rate = 16000
duration = 5  # seconds

# Generate real speech-like audio
t = np.linspace(0, duration, int(sample_rate * duration))
audio = np.zeros_like(t)

# Korean sentence structure timing (안녕하세요, 음성인식 테스트입니다)
speech_segments = [
    (0.5, 1.2),   # 안녕하세요
    (1.5, 2.3),   # 음성인식
    (2.5, 3.2),   # 테스트입니다
]

for start, end in speech_segments:
    mask = (t >= start) & (t <= end)
    segment_t = t[mask]

    # More realistic speech formants
    # Korean vowel formants
    f1 = 700 + 300 * np.sin(2 * np.pi * 4 * segment_t)
    f2 = 1500 + 500 * np.sin(2 * np.pi * 3 * segment_t)
    f3 = 2500 + 200 * np.sin(2 * np.pi * 2 * segment_t)

    # Combine formants with proper amplitudes
    audio[mask] = (
        0.5 * np.sin(2 * np.pi * f1 * segment_t) +
        0.3 * np.sin(2 * np.pi * f2 * segment_t) +
        0.2 * np.sin(2 * np.pi * f3 * segment_t)
    )

    # Add consonant-like noise bursts
    noise = np.random.normal(0, 0.1, len(segment_t))
    audio[mask] += noise * 0.3

# Normalize
max_val = np.max(np.abs(audio))
if max_val > 0:
    audio = audio / max_val * 0.8  # Keep at 80% to avoid clipping

# Save the audio file
output_file = "/tmp/korean_test_audio.wav"
sf.write(output_file, audio.astype(np.float32), sample_rate)
print(f"✅ Created test audio: {output_file}")

# Calculate audio statistics
energy = np.sqrt(np.mean(np.square(audio[audio != 0])))
print(f"📊 Audio energy (speech segments): {energy:.4f}")
print(f"📊 Max amplitude: {np.max(np.abs(audio)):.4f}")
print(f"📊 Duration: {duration}s")
print(f"📊 Sample rate: {sample_rate}Hz")

# Test with the API
print("\n🧪 Testing with STT API...")
try:
    with open(output_file, "rb") as f:
        files = {"audio": ("test.wav", f, "audio/wav")}
        response = requests.post("http://localhost:5000/api/test-audio", files=files, timeout=15)

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Transcription result: {result}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"❌ Test failed: {e}")