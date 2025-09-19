#!/usr/bin/env python3
"""
GPU 최적화 실시간 STT 웹 서버
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
import torch

# Flask 앱 생성
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})
sock = Sock(app)

class WhisperProcessor:
    """GPU 최적화 Whisper 처리 클래스"""

    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.sample_rate = 16000

    def load_model(self):
        """모델 로드"""
        print("🔄 Faster-Whisper 모델 로딩 중 (GPU 모드)...")

        # GPU 사용 가능 확인
        if torch.cuda.is_available():
            print(f"✅ GPU 감지됨: {torch.cuda.get_device_name(0)}")
            print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            device = "cuda"
            compute_type = "float16"
        else:
            print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 전환합니다.")
            device = "cpu"
            compute_type = "int8"

        try:
            self.model = WhisperModel(
                "large-v3",
                device=device,
                device_index=0,
                compute_type=compute_type,
                download_root="/app/models",
                num_workers=4 if device == "cpu" else 1,
                cpu_threads=8 if device == "cpu" else 0
            )
            print(f"✅ {device.upper()} 모델 로드 완료!")
            self.device = device
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False

    def transcribe_audio(self, audio_data, sample_rate=16000):
        """오디오 데이터를 텍스트로 변환 (GPU 최적화)"""
        if self.model is None:
            return None

        # 오디오 정규화
        audio_data = audio_data / np.max(np.abs(audio_data) + 1e-10)

        # 임시 파일 저장
        temp_file = f"/tmp/audio_{time.time()}.wav"
        sf.write(temp_file, audio_data, sample_rate)

        try:
            # GPU 최적화 설정으로 Transcribe
            segments, info = self.model.transcribe(
                temp_file,
                language="ko",  # 한국어 기본
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200
                },
                condition_on_previous_text=False,
                initial_prompt=None,
            )

            # 결과 수집
            text = ""
            for segment in segments:
                text += segment.text

            # 후처리
            text = text.strip().replace("  ", " ")

            result = {
                "text": text,
                "language": info.language if info.language else "ko",
                "confidence": float(info.language_probability) if info.language_probability else 0.95,
                "device": self.device
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
        'websocket': 'integrated',
        'device': processor.device,
        'gpu': torch.cuda.is_available()
    }

@sock.route('/ws')
def websocket(ws):
    """WebSocket 연결 처리"""
    print(f"✅ 클라이언트 연결")

    audio_buffer = []
    buffer_duration = 0
    max_buffer_duration = 2.0  # 2초마다 처리

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
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                    audio_buffer.extend(audio_chunk)

                    # 버퍼 시간 계산
                    buffer_duration = len(audio_buffer) / 16000

                    # 2초 이상 쌓이면 처리
                    if buffer_duration >= max_buffer_duration:
                        audio_array = np.array(audio_buffer)

                        # Whisper로 처리
                        result = processor.transcribe_audio(audio_array, 16000)

                        if result and result["text"]:
                            # 결과 전송
                            response = {
                                "type": "transcription",
                                "text": result["text"],
                                "language": result["language"],
                                "confidence": result["confidence"],
                                "device": result["device"],
                                "timestamp": time.time()
                            }
                            ws.send(json.dumps(response, ensure_ascii=False))
                            print(f"📝 [{result['device'].upper()}] {result['language']}: {result['text']}")

                        # 버퍼 초기화
                        audio_buffer = []
                        buffer_duration = 0

                elif data.get("type") == "config":
                    # 설정 업데이트
                    ws.send(json.dumps({
                        "type": "config_updated",
                        "language": "ko",
                        "device": processor.device
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
    print("🎤 GPU 최적화 실시간 STT 웹 애플리케이션")
    print("GPU-Optimized Real-time Speech-to-Text")
    print("=" * 60)

    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ GPU를 사용할 수 없습니다.")

    # 템플릿 디렉토리 생성
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # 모델 로드
    processor.load_model()

    # Flask + WebSocket 서버 시작
    print(f"\n🌐 STT 서버 시작: http://localhost:5000")
    print(f"🖥️ Device: {processor.device.upper()}")
    print("WebSocket은 /ws 경로에서 처리됩니다.\n")

    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()