#!/usr/bin/env python3
"""
실시간 음성 스트리밍 처리 서버
Real-time Audio Streaming Server with Faster-Whisper
"""

import asyncio
import json
import numpy as np
import soundfile as sf
import io
import time
from pathlib import Path
from threading import Thread, Lock
from queue import Queue
from faster_whisper import WhisperModel
import websockets
from websockets.server import serve

class RealtimeWhisperProcessor:
    """실시간 Whisper 처리기"""

    def __init__(self, model_name="large-v3", device="cuda", language="auto"):
        """
        Args:
            model_name: Whisper 모델 크기
            device: cuda 또는 cpu
            language: 언어 설정 (auto, ko, en, 등)
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.model = None
        self.audio_buffer = []
        self.buffer_lock = Lock()
        self.processing_queue = Queue()
        self.results_queue = Queue()

        # 스트리밍 설정
        self.sample_rate = 16000  # Whisper는 16kHz 필요
        self.chunk_duration = 5  # 5초 단위로 처리
        self.min_chunk_duration = 1  # 최소 1초
        self.overlap_duration = 0.5  # 0.5초 오버랩

        # VAD 설정
        self.vad_enabled = True
        self.silence_threshold = 0.01
        self.silence_duration = 1.0  # 1초 침묵시 처리

        print(f"🎤 실시간 Whisper 프로세서 초기화")
        print(f"  - 모델: {model_name}")
        print(f"  - 디바이스: {device}")
        print(f"  - 언어: {language}")

    def load_model(self):
        """모델 로드"""
        print("🔄 Faster-Whisper 모델 로딩 중...")

        compute_type = "float16" if self.device == "cuda" else "int8"

        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=compute_type,
                download_root="/app/models",
                device_index=0,
                num_workers=1
            )
            print("✅ 모델 로드 완료!")

            # 처리 스레드 시작
            self.processing_thread = Thread(target=self._process_audio_chunks)
            self.processing_thread.daemon = True
            self.processing_thread.start()

        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise

    def _process_audio_chunks(self):
        """오디오 청크 처리 스레드"""
        print("🎯 오디오 처리 스레드 시작")

        while True:
            if not self.processing_queue.empty():
                audio_chunk, timestamp = self.processing_queue.get()

                try:
                    # Whisper로 처리
                    result = self._transcribe_chunk(audio_chunk, timestamp)
                    if result:
                        self.results_queue.put(result)
                except Exception as e:
                    print(f"❌ 청크 처리 오류: {e}")
            else:
                time.sleep(0.1)

    def _transcribe_chunk(self, audio_chunk, timestamp):
        """오디오 청크를 텍스트로 변환"""

        # 임시 파일로 저장 (Faster-Whisper는 파일 필요)
        temp_file = f"/tmp/chunk_{timestamp}.wav"
        sf.write(temp_file, audio_chunk, self.sample_rate)

        try:
            # 언어 설정
            if self.language == "auto":
                language = None
            else:
                language = self.language

            # Transcribe
            segments, info = self.model.transcribe(
                temp_file,
                language=language,
                task="transcribe",
                beam_size=3,  # 실시간을 위해 줄임
                best_of=3,
                patience=0.5,
                temperature=0.0,
                vad_filter=self.vad_enabled,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                ),
                initial_prompt="한국어와 영어를 정확하게 인식하세요." if language is None else None,
                condition_on_previous_text=False  # 실시간에서는 False
            )

            # 결과 수집
            text = ""
            for segment in segments:
                text += segment.text

            # 임시 파일 삭제
            Path(temp_file).unlink(missing_ok=True)

            if text.strip():
                return {
                    "timestamp": timestamp,
                    "text": text.strip(),
                    "language": info.language,
                    "confidence": info.language_probability
                }

        except Exception as e:
            print(f"❌ Transcribe 오류: {e}")
            Path(temp_file).unlink(missing_ok=True)

        return None

    def add_audio_data(self, audio_data, sample_rate=None):
        """오디오 데이터 추가"""

        # 샘플레이트 변환
        if sample_rate and sample_rate != self.sample_rate:
            # Resample to 16kHz
            import librosa
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )

        # 버퍼에 추가
        with self.buffer_lock:
            self.audio_buffer.extend(audio_data)

            # 충분한 데이터가 쌓이면 처리
            buffer_duration = len(self.audio_buffer) / self.sample_rate

            if buffer_duration >= self.chunk_duration:
                # 청크 추출
                chunk_samples = int(self.chunk_duration * self.sample_rate)
                chunk = np.array(self.audio_buffer[:chunk_samples])

                # 큐에 추가
                timestamp = time.time()
                self.processing_queue.put((chunk, timestamp))

                # 오버랩을 고려한 버퍼 업데이트
                overlap_samples = int(self.overlap_duration * self.sample_rate)
                self.audio_buffer = self.audio_buffer[chunk_samples - overlap_samples:]

                print(f"📦 청크 처리 큐에 추가 (버퍼: {len(self.audio_buffer)/self.sample_rate:.1f}초)")

    def get_results(self):
        """처리 결과 가져오기"""
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())
        return results

    def flush_buffer(self):
        """버퍼 강제 처리"""
        with self.buffer_lock:
            if len(self.audio_buffer) > self.min_chunk_duration * self.sample_rate:
                chunk = np.array(self.audio_buffer)
                timestamp = time.time()
                self.processing_queue.put((chunk, timestamp))
                self.audio_buffer = []
                print("🔄 버퍼 플러시 완료")


class WebSocketServer:
    """WebSocket 스트리밍 서버"""

    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.processor = RealtimeWhisperProcessor()
        self.clients = set()

        print(f"\n🌐 WebSocket 서버 설정")
        print(f"  - 주소: ws://{host}:{port}")

    async def handler(self, websocket, path):
        """WebSocket 연결 처리"""

        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"\n✅ 클라이언트 연결: {client_id}")

        self.clients.add(websocket)

        try:
            async for message in websocket:
                # 바이너리 오디오 데이터 처리
                if isinstance(message, bytes):
                    # 오디오 데이터 디코드
                    audio_array = np.frombuffer(message, dtype=np.float32)

                    # 프로세서에 추가
                    self.processor.add_audio_data(audio_array)

                    # 결과 확인 및 전송
                    results = self.processor.get_results()
                    for result in results:
                        await websocket.send(json.dumps(result, ensure_ascii=False))
                        print(f"📝 [{result['language']}] {result['text']}")

                # JSON 명령 처리
                elif isinstance(message, str):
                    try:
                        data = json.loads(message)

                        if data.get("command") == "flush":
                            # 버퍼 플러시
                            self.processor.flush_buffer()
                            await websocket.send(json.dumps({"status": "flushed"}))

                        elif data.get("command") == "config":
                            # 설정 업데이트
                            if "language" in data:
                                self.processor.language = data["language"]
                            if "vad" in data:
                                self.processor.vad_enabled = data["vad"]
                            await websocket.send(json.dumps({"status": "configured"}))

                    except json.JSONDecodeError:
                        pass

        except websockets.exceptions.ConnectionClosed:
            print(f"❌ 클라이언트 연결 종료: {client_id}")

        finally:
            self.clients.remove(websocket)

    async def start(self):
        """서버 시작"""

        # 모델 로드
        self.processor.load_model()

        # WebSocket 서버 시작
        print(f"\n🚀 WebSocket 서버 시작: ws://{self.host}:{self.port}")
        print("📡 실시간 음성 스트리밍 대기 중...\n")

        async with serve(self.handler, self.host, self.port):
            await asyncio.Future()  # 무한 대기


def create_test_client():
    """테스트용 클라이언트 코드 생성"""

    client_code = '''
# 테스트 클라이언트 (Python)
import asyncio
import websockets
import numpy as np
import sounddevice as sd
import json

async def stream_microphone():
    """마이크 음성을 서버로 스트리밍"""

    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        print("✅ 서버 연결됨")

        # 설정
        await websocket.send(json.dumps({
            "command": "config",
            "language": "auto",  # 자동 감지
            "vad": True
        }))

        def audio_callback(indata, frames, time, status):
            """오디오 콜백"""
            if status:
                print(status)

            # Float32로 변환
            audio = indata[:, 0].astype(np.float32)

            # 서버로 전송
            asyncio.create_task(websocket.send(audio.tobytes()))

        # 마이크 스트림 시작
        with sd.InputStream(
            samplerate=16000,
            channels=1,
            callback=audio_callback,
            blocksize=1600  # 0.1초
        ):
            print("🎤 마이크 스트리밍 시작 (Ctrl+C로 종료)")

            # 결과 수신
            while True:
                result = await websocket.recv()
                data = json.loads(result)

                if "text" in data:
                    print(f"[{data['language']}] {data['text']}")

if __name__ == "__main__":
    asyncio.run(stream_microphone())
'''

    with open("/home/stt-model/client_example.py", "w") as f:
        f.write(client_code)

    print("📝 client_example.py 생성 완료")


def main():
    """메인 함수"""

    print("=" * 60)
    print("🎤 실시간 음성 인식 서버")
    print("Faster-Whisper WebSocket Streaming Server")
    print("=" * 60)

    # 테스트 클라이언트 생성
    create_test_client()

    # 서버 시작
    server = WebSocketServer()

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n\n🛑 서버 종료")

if __name__ == "__main__":
    main()