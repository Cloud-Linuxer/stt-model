#!/usr/bin/env python3
"""
초저지연 실시간 STT 웹 서버 - 최적화 버전
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
import threading
from queue import Queue
import asyncio

# Flask 앱 생성
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})
sock = Sock(app)

class RealtimeWhisperProcessor:
    """초저지연 실시간 Whisper 처리 클래스"""

    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.sample_rate = 16000
        self.processing_queue = Queue()
        self.worker_thread = None

    def load_model(self):
        """모델 로드"""
        print("🔄 초저지연 Faster-Whisper 모델 로딩 중 (GPU 모드)...")

        # GPU 사용 가능 확인
        if torch.cuda.is_available():
            print(f"✅ GPU 감지됨: {torch.cuda.get_device_name(0)}")
            print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            device = "cuda"
            compute_type = "float16"
        else:
            print("⚠️ GPU를 사용할 수 없습니다.")
            device = "cpu"
            compute_type = "int8"

        try:
            self.model = WhisperModel(
                "large-v3",
                device=device,
                device_index=0,
                compute_type=compute_type,
                download_root="/app/models",
                num_workers=1,
                cpu_threads=0
            )
            print(f"✅ {device.upper()} 모델 로드 완료!")
            self.device = device

            # 워커 스레드 시작
            self.start_worker()
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False

    def start_worker(self):
        """처리 워커 스레드 시작"""
        self.worker_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
        self.worker_thread.start()
        print("🔧 백그라운드 처리 워커 시작")

    def _process_audio_queue(self):
        """큐에서 오디오 처리 (백그라운드)"""
        while True:
            try:
                audio_data, callback = self.processing_queue.get()
                if audio_data is not None:
                    result = self._transcribe_chunk(audio_data)
                    if callback:
                        callback(result)
            except Exception as e:
                print(f"❌ 워커 오류: {e}")

    def _transcribe_chunk(self, audio_data):
        """실제 전사 처리"""
        if self.model is None:
            return None

        # 오디오 정규화
        audio_data = audio_data / np.max(np.abs(audio_data) + 1e-10)

        # 임시 파일 저장
        temp_file = f"/tmp/audio_{time.time()}.wav"
        sf.write(temp_file, audio_data, self.sample_rate)

        try:
            # 초저지연 설정으로 Transcribe
            segments, info = self.model.transcribe(
                temp_file,
                language="ko",
                task="transcribe",
                beam_size=2,  # 균형점 (속도와 정확도)
                best_of=2,    # 적절한 후보 탐색
                temperature=0.0,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.4,  # 적절한 민감도
                    "min_speech_duration_ms": 150,  # 최소 발화 길이
                    "min_silence_duration_ms": 200,  # 자연스러운 침묵 구간
                    "speech_pad_ms": 100  # 적절한 패딩
                },
                condition_on_previous_text=False,
                initial_prompt=None,
                word_timestamps=False  # 타임스탬프 비활성화 (속도 향상)
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
                "device": self.device,
                "processing_time": time.time()
            }

            return result

        except Exception as e:
            print(f"❌ Transcribe 오류: {e}")
            return None
        finally:
            # 임시 파일 삭제
            Path(temp_file).unlink(missing_ok=True)

    def process_audio_async(self, audio_data, callback):
        """비동기 오디오 처리"""
        self.processing_queue.put((audio_data, callback))

# Whisper 프로세서 인스턴스
processor = RealtimeWhisperProcessor()

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
        'gpu': torch.cuda.is_available(),
        'mode': 'ultra_realtime'
    }

@sock.route('/ws')
def websocket(ws):
    """WebSocket 연결 처리"""
    print(f"✅ 클라이언트 연결")
    print(f"📡 초저지연 모드 활성화: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    audio_buffer = []
    buffer_duration = 0
    max_buffer_duration = 1.2  # 1.2초마다 처리 (균형점)
    min_buffer_duration = 0.8  # 최소 0.8초 버퍼 (문맥 확보)
    last_result_text = ""  # 중복 제거용

    def send_result(result):
        """결과 전송 콜백"""
        nonlocal last_result_text
        if result and result["text"] and result["text"] != last_result_text:
            response = {
                "type": "transcription",
                "text": result["text"],
                "language": result["language"],
                "confidence": result["confidence"],
                "device": result["device"],
                "timestamp": time.time(),
                "latency": time.time() - result.get("processing_time", time.time())
            }
            try:
                ws.send(json.dumps(response, ensure_ascii=False))
                print(f"📝 [{result['device'].upper()}] {result['text'][:50]}... (지연: {response['latency']:.2f}초)")
                last_result_text = result["text"]
            except:
                pass

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

                    # 동적 버퍼 처리
                    # VAD가 음성을 감지하면 더 짧은 간격으로 처리
                    if buffer_duration >= min_buffer_duration:
                        # 음성 에너지 체크 (간단한 VAD)
                        energy = np.sqrt(np.mean(np.square(audio_chunk)))

                        if energy > 0.01 or buffer_duration >= max_buffer_duration:
                            # 음성이 감지되거나 최대 버퍼 도달시 처리
                            audio_array = np.array(audio_buffer)

                            # 비동기 처리 큐에 추가
                            processor.process_audio_async(audio_array, send_result)

                            # 버퍼 초기화 (일부 오버랩 유지)
                            overlap_samples = int(0.1 * 16000)  # 0.1초 오버랩
                            audio_buffer = audio_buffer[-overlap_samples:] if len(audio_buffer) > overlap_samples else []
                            buffer_duration = len(audio_buffer) / 16000

                elif data.get("type") == "config":
                    # 설정 업데이트
                    ws.send(json.dumps({
                        "type": "config_updated",
                        "language": "ko",
                        "device": processor.device,
                        "mode": "ultra_realtime"
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
    print("🚀 초저지연 실시간 STT 웹 애플리케이션")
    print("Ultra-Low Latency Real-time Speech-to-Text")
    print("=" * 60)

    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("⚡ 초저지연 모드 활성화")
    else:
        print("⚠️ GPU를 사용할 수 없습니다.")

    # 템플릿 디렉토리 생성
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # 모델 로드
    processor.load_model()

    # Flask + WebSocket 서버 시작
    print(f"\n🌐 최적화된 실시간 STT 서버 시작: http://localhost:5000")
    print(f"🖥️ Device: {processor.device.upper()}")
    print(f"⚡ 지연시간: ~1초 (정확도와 속도의 균형)")
    print("WebSocket은 /ws 경로에서 처리됩니다.\n")

    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()