#!/usr/bin/env python3
"""
실시간 STT 웹 서버
Real-time Speech-to-Text Web Application
"""

import os
import json
import asyncio
import numpy as np
import time
from pathlib import Path
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
import websockets
from websockets.server import serve
from threading import Thread
from faster_whisper import WhisperModel
import base64
import io
import soundfile as sf

# Flask 앱 생성
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# 전역 변수
model = None
clients = set()

class WhisperProcessor:
    """Whisper 처리 클래스"""

    def __init__(self):
        self.model = None
        self.device = "cpu"  # CPU 모드로 강제
        self.language = "auto"
        self.sample_rate = 16000

    def load_model(self):
        """모델 로드"""
        print("🔄 Faster-Whisper 모델 로딩 중 (CPU 모드)...")

        # CPU 모드로만 실행
        self.model = WhisperModel(
            "large-v3",
            device="cpu",
            compute_type="int8",
            download_root="/app/models",
            num_workers=4,  # CPU 코어 활용
            cpu_threads=8   # 스레드 수 지정
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

            # Transcribe
            segments, info = self.model.transcribe(
                temp_file,
                language=language,
                task="transcribe",
                beam_size=5,
                best_of=5,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                ),
                initial_prompt="한국어와 영어를 정확하게 인식하세요." if language is None else None,
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
    # WebSocket URL을 환경변수나 설정에서 가져오기
    ws_url = os.environ.get('WEBSOCKET_URL', 'wss://operation-numbers-floating-opt.trycloudflare.com')
    return render_template('index.html', ws_url=ws_url)

@app.route('/test')
def test():
    """WebSocket 테스트 페이지"""
    return render_template('test_websocket.html')

@app.route('/api/config')
def get_config():
    """WebSocket URL 등 설정 정보 제공"""
    # config.json 파일에서 읽기 시도
    try:
        with open('/app/config.json', 'r') as f:
            config = json.load(f)
            ws_url = config.get('ws_url')
    except:
        ws_url = None

    # 환경변수 또는 기본값 사용
    if not ws_url:
        ws_url = os.environ.get('WEBSOCKET_URL', 'wss://operation-numbers-floating-opt.trycloudflare.com')

    return {
        'ws_url': ws_url,
        'status': 'ready'
    }

@app.route('/static/<path:path>')
def send_static(path):
    """정적 파일 서빙"""
    return send_from_directory('static', path)

async def websocket_handler(websocket, path):
    """WebSocket 연결 처리"""

    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    print(f"✅ 클라이언트 연결: {client_id}")

    clients.add(websocket)

    try:
        async for message in websocket:
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
                        await websocket.send(json.dumps(response, ensure_ascii=False))
                        print(f"📝 [{result['language']}] {result['text'][:50]}...")

                elif data.get("type") == "config":
                    # 설정 업데이트
                    if "language" in data:
                        processor.language = data["language"]
                        print(f"⚙️ 언어 설정: {processor.language}")

                    await websocket.send(json.dumps({
                        "type": "config_updated",
                        "language": processor.language
                    }))

            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")

    except websockets.exceptions.ConnectionClosed:
        print(f"❌ 클라이언트 연결 종료: {client_id}")
    except Exception as e:
        print(f"❌ 오류: {e}")
    finally:
        clients.remove(websocket)

async def start_websocket_server():
    """WebSocket 서버 시작"""
    print("🚀 WebSocket 서버 시작 (포트 8766)")
    async with serve(websocket_handler, "0.0.0.0", 8766):
        await asyncio.Future()

def run_websocket_server():
    """WebSocket 서버를 별도 스레드에서 실행"""
    asyncio.set_event_loop(asyncio.new_event_loop())
    asyncio.get_event_loop().run_until_complete(start_websocket_server())

def main():
    """메인 함수"""
    print("=" * 60)
    print("🎤 실시간 STT 웹 애플리케이션")
    print("Real-time Speech-to-Text Web Application")
    print("=" * 60)

    # 템플릿 디렉토리 생성
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # 모델 로드
    processor.load_model()

    # WebSocket 서버를 별도 스레드에서 시작
    ws_thread = Thread(target=run_websocket_server)
    ws_thread.daemon = True
    ws_thread.start()

    # Flask 서버 시작
    print("\n🌐 웹 서버 시작: http://localhost:5000")
    print("📡 WebSocket 서버: ws://localhost:8766")
    print("\n웹 브라우저에서 http://localhost:5000 접속하세요!\n")

    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()