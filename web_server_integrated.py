#!/usr/bin/env python3
"""
통합 실시간 STT 웹 서버 (Flask + WebSocket 통합)
"""

import os
import json
import numpy as np
import time
from pathlib import Path
from flask import Flask, render_template
from flask_cors import CORS
from flask_sock import Sock
from faster_whisper import WhisperModel
import base64
import io
import soundfile as sf
import threading

# Flask 앱 생성
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})
sock = Sock(app)

# 전역 변수
model = None

class WhisperProcessor:
    """Whisper 처리 클래스"""

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.language = "auto"
        self.sample_rate = 16000

    def load_model(self):
        """모델 로드"""
        print("🔄 Faster-Whisper 모델 로딩 중 (CPU 모드)...")

        self.model = WhisperModel(
            "large-v3",
            device="cpu",
            compute_type="int8",
            download_root="/app/models",
            num_workers=4,
            cpu_threads=8
        )
        print("✅ CPU 모델 로드 완료!")
        return True

    def transcribe_audio(self, audio_data, sample_rate=16000):
        """오디오 데이터를 텍스트로 변환"""
        if self.model is None:
            return None

        # 임시 파일 저장
        temp_file = f"/tmp/audio_{time.time()}.wav"
        sf.write(temp_file, audio_data, sample_rate)

        try:
            # 언어 설정
            language = None if self.language == "auto" else self.language

            # Transcribe (한국어 최적화)
            segments, info = self.model.transcribe(
                temp_file,
                language="ko" if language == "auto" else language,  # 기본값을 한국어로
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,  # 더 일관된 결과
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.4,  # 더 민감하게
                    min_speech_duration_ms=200,  # 짧은 발화도 인식
                    min_silence_duration_ms=300,
                    speech_pad_ms=300
                ),
                initial_prompt="이것은 한국어 음성입니다.",  # 한국어 힌트
            )

            # 결과 수집
            text = ""
            for segment in segments:
                text += segment.text

            result = {
                "text": text.strip(),
                "language": info.language,
                "confidence": float(info.language_probability)
            }

            return result

        except Exception as e:
            print(f"❌ Transcribe 오류: {e}")
            return None
        finally:
            # 임시 파일 삭제
            Path(temp_file).unlink(missing_ok=True)

# Whisper 프로세서 인스턴스
processor = WhisperProcessor()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index_fixed.html')

@app.route('/api/config')
def get_config():
    """설정 정보 제공"""
    return {
        'status': 'ready',
        'websocket': 'integrated'
    }

@sock.route('/ws')
def websocket(ws):
    """WebSocket 연결 처리"""
    print(f"✅ 클라이언트 연결")

    language_setting = "auto"

    while True:
        try:
            message = ws.receive()
            if message is None:
                break

            # JSON 메시지 처리
            try:
                data = json.loads(message)

                if data.get("type") == "audio":
                    # Base64 오디오 데이터 디코드
                    audio_base64 = data.get("data", "")
                    audio_bytes = base64.b64decode(audio_base64)

                    # Float32 배열로 변환
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

                    # Whisper로 처리
                    processor.language = language_setting
                    result = processor.transcribe_audio(audio_array, data.get("sampleRate", 16000))

                    if result:
                        # 결과 전송
                        response = {
                            "type": "transcription",
                            "text": result["text"],
                            "language": result["language"],
                            "confidence": result["confidence"],
                            "timestamp": time.time()
                        }
                        ws.send(json.dumps(response, ensure_ascii=False))
                        print(f"📝 [{result['language']}] {result['text'][:50]}...")

                elif data.get("type") == "config":
                    # 설정 업데이트
                    if "language" in data:
                        language_setting = data["language"]
                        print(f"⚙️ 언어 설정: {language_setting}")

                    ws.send(json.dumps({
                        "type": "config_updated",
                        "language": language_setting
                    }))

            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")

        except Exception as e:
            print(f"❌ WebSocket 오류: {e}")
            break

    print(f"❌ 클라이언트 연결 종료")

def main():
    """메인 함수"""
    print("=" * 60)
    print("🎤 통합 실시간 STT 웹 애플리케이션")
    print("Integrated Real-time Speech-to-Text Web Application")
    print("=" * 60)

    # 템플릿 디렉토리 생성
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # 모델 로드
    processor.load_model()

    # Flask + WebSocket 서버 시작
    print("\n🌐 통합 서버 시작: http://localhost:5000")
    print("WebSocket은 /ws 경로에서 처리됩니다.\n")

    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()