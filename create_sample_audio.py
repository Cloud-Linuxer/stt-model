#!/usr/bin/env python3
"""
한국어+영어 테스트용 샘플 오디오 생성
TTS를 사용하여 실제 음성 샘플 생성
"""

import os
import numpy as np
import soundfile as sf

def create_test_samples():
    """테스트용 오디오 샘플 생성"""

    print("🎤 테스트용 오디오 샘플 생성 중...")

    # data 디렉토리 생성
    os.makedirs("/app/data", exist_ok=True)

    # 다양한 길이의 테스트 오디오 생성
    sample_rate = 16000

    # 1. 짧은 한국어 시뮬레이션 (10초)
    print("  - 한국어 테스트 (10초)")
    korean_audio = np.sin(2 * np.pi * 440 * np.arange(sample_rate * 10) / sample_rate) * 0.1
    korean_audio += np.random.randn(len(korean_audio)) * 0.01
    sf.write("/app/data/korean_test.wav", korean_audio, sample_rate)

    # 2. 짧은 영어 시뮬레이션 (10초)
    print("  - 영어 테스트 (10초)")
    english_audio = np.sin(2 * np.pi * 523 * np.arange(sample_rate * 10) / sample_rate) * 0.1
    english_audio += np.random.randn(len(english_audio)) * 0.01
    sf.write("/app/data/english_test.wav", english_audio, sample_rate)

    # 3. 혼합 언어 시뮬레이션 (20초)
    print("  - 한국어+영어 혼합 (20초)")
    mixed_audio = np.concatenate([korean_audio, english_audio])
    sf.write("/app/data/mixed_korean_english.wav", mixed_audio, sample_rate)

    # 4. 긴 오디오 (60초)
    print("  - 긴 오디오 테스트 (60초)")
    long_audio = np.sin(2 * np.pi * 600 * np.arange(sample_rate * 60) / sample_rate) * 0.08
    long_audio += np.random.randn(len(long_audio)) * 0.005
    sf.write("/app/data/long_test.wav", long_audio, sample_rate)

    # 5. 침묵 구간이 있는 오디오 (30초)
    print("  - 침묵 구간 포함 (30초)")
    silence_audio = np.zeros(sample_rate * 30)
    # 5초 음성, 5초 침묵 반복
    for i in range(3):
        start = i * 10 * sample_rate
        end = start + 5 * sample_rate
        silence_audio[start:end] = np.sin(2 * np.pi * 500 * np.arange(5 * sample_rate) / sample_rate) * 0.1
    sf.write("/app/data/with_silence.wav", silence_audio, sample_rate)

    print("✅ 5개 테스트 오디오 파일 생성 완료!")
    print("\n📂 생성된 파일:")
    for file in os.listdir("/app/data"):
        if file.endswith('.wav'):
            file_path = f"/app/data/{file}"
            info = sf.info(file_path)
            print(f"  - {file}: {info.duration:.1f}초, {info.samplerate}Hz")

if __name__ == "__main__":
    create_test_samples()