#!/usr/bin/env python3
"""
STT + LLM 키워드 추출 통합 서버 (스트리밍 버전)
- VAD 기반 음성 세그먼트 감지
- 침묵 감지 시 처리
- 할루시네이션 최소화
"""

import json
import time
import torch
import numpy as np
import soundfile as sf
import threading
import queue
import os
import requests
from collections import deque
from flask import Flask, render_template, request, jsonify
from flask_sock import Sock
from faster_whisper import WhisperModel
import warnings
warnings.filterwarnings("ignore")

# 환경 변수
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/completions")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
sock = Sock(app)

# 전역 변수
stt_processor = None
keyword_extractor = None

class VoiceActivityDetector:
    """간단한 VAD 구현"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.energy_threshold = 0.02  # 에너지 임계값
        self.silence_duration = 1.0  # 침묵 감지 시간 (초)

    def is_speech(self, audio_chunk):
        """음성인지 판단"""
        energy = np.sqrt(np.mean(np.square(audio_chunk)))
        return energy > self.energy_threshold

class StreamingSTT:
    """스트리밍 STT 처리"""

    def __init__(self):
        self.model = None
        self.device = None
        self.vad = VoiceActivityDetector()
        self.audio_buffer = []
        self.silence_frames = 0
        self.silence_threshold = 16  # 약 1초 (16000Hz / 1000ms)
        self.min_speech_length = 8000  # 최소 0.5초
        self.max_buffer_size = 160000  # 최대 10초

        # 할루시네이션 패턴
        self.hallucination_patterns = [
            "감사합니다", "시청해주셔서", "구독", "좋아요",
            "MBC 뉴스", "KBS 뉴스", "다음 영상", "영상편집",
            "이 시각 세계", "뉴스", "기자입니다", "앵커입니다"
        ]

    def load_model(self):
        """모델 로드"""
        print("🔄 Whisper 모델 로딩 중 (스트리밍 모드)...")

        if torch.cuda.is_available():
            print(f"✅ GPU 감지됨: {torch.cuda.get_device_name(0)}")
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"

        self.model = WhisperModel(
            "medium",
            device=device,
            device_index=0 if device == "cuda" else None,
            compute_type=compute_type,
            download_root="/app/models",
            num_workers=1,
            cpu_threads=0
        )
        print(f"✅ {device.upper()} 모델 로드 완료!")
        self.device = device
        return True

    def process_stream(self, audio_chunk):
        """오디오 스트림 처리"""
        if self.model is None:
            return None

        # VAD 체크
        is_speech = self.vad.is_speech(audio_chunk)

        if is_speech:
            # 음성이 감지되면 버퍼에 추가
            self.audio_buffer.extend(audio_chunk)
            self.silence_frames = 0

            # 버퍼 크기 제한
            if len(self.audio_buffer) > self.max_buffer_size:
                # 너무 길면 처리
                return self._process_buffer()

        else:
            # 침묵 카운트
            self.silence_frames += 1

            # 충분한 침묵이 감지되고 버퍼에 데이터가 있으면
            if self.silence_frames > self.silence_threshold and len(self.audio_buffer) > self.min_speech_length:
                return self._process_buffer()

        return None

    def _process_buffer(self):
        """버퍼 처리"""
        if not self.audio_buffer:
            return None

        # 버퍼를 numpy 배열로 변환
        audio_data = np.array(self.audio_buffer, dtype=np.float32)

        # 정규화
        audio_data = audio_data / np.max(np.abs(audio_data) + 1e-10)

        # 임시 파일 저장
        temp_file = f"/tmp/audio_{time.time()}.wav"
        sf.write(temp_file, audio_data, 16000)

        try:
            # Whisper 처리
            segments, info = self.model.transcribe(
                temp_file,
                language="ko",
                beam_size=5,
                best_of=5,
                patience=1,
                length_penalty=1,
                temperature=[0.0],  # 낮은 temperature로 할루시네이션 감소
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                initial_prompt=None,  # 프롬프트 없이
                prefix=None,
                suppress_blank=True,
                suppress_tokens=[-1],
                without_timestamps=True,
                word_timestamps=False,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    max_speech_duration_s=30,
                    min_silence_duration_ms=500,
                    speech_pad_ms=100
                )
            )

            text = "".join([segment.text for segment in segments]).strip()

            # 할루시네이션 체크
            if text and not self._is_hallucination(text):
                print(f"📝 인식된 텍스트: {text}")

                # 버퍼 초기화
                self.audio_buffer = []

                return {
                    "text": text,
                    "language": info.language,
                    "probability": info.language_probability
                }

        except Exception as e:
            print(f"❌ 처리 오류: {e}")
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # 버퍼 초기화
        self.audio_buffer = []
        return None

    def _is_hallucination(self, text):
        """할루시네이션 체크"""
        # 너무 짧은 텍스트
        if len(text.strip()) < 2:
            return True

        # 패턴 체크
        for pattern in self.hallucination_patterns:
            if pattern in text:
                print(f"⚠️ 할루시네이션 감지: {text}")
                return True

        return False

class KeywordExtractor:
    """LLM 키워드 추출"""

    def __init__(self):
        self.api_url = VLLM_API_URL
        self.min_importance = 0.8

    def extract_keywords(self, text):
        """키워드 추출"""
        if not text or len(text.strip()) < 10:
            return []

        prompt = f"""다음 텍스트에서 핵심 명사만 추출하세요.

텍스트: "{text}"

규칙:
1. 명사만 추출 (사람, 장소, 회사, 제품, 기술 이름)
2. 최대 5개까지
3. JSON 형식으로 응답

응답 형식:
{{"keywords": [{{"word": "단어", "category": "카테고리", "importance": 0.9}}]}}"""

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.1,
                    "stop": ["}}", "\n\n"]
                },
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                text_response = result['choices'][0]['text']

                # JSON 파싱
                import re
                json_pattern = r'\{[^{}]*"keywords"[^{}]*\[[^\]]*\][^{}]*\}'
                json_match = re.search(json_pattern, text_response, re.DOTALL)

                if json_match:
                    data = json.loads(json_match.group(0))
                    keywords = data.get('keywords', [])
                    filtered = [k for k in keywords if k.get('importance', 0) >= self.min_importance]
                    return filtered

        except Exception as e:
            print(f"LLM 키워드 추출 오류: {e}")

        return []

# WebSocket 핸들러
@sock.route('/ws')
def websocket(ws):
    """WebSocket 연결 처리"""
    print(f"🔌 새 WebSocket 연결: {request.remote_addr}")

    # 전역 변수로 미리 로드된 모델 사용
    global stt_processor, keyword_extractor

    if stt_processor is None or keyword_extractor is None:
        # 만약 없으면 로드 (fallback)
        streaming_stt = StreamingSTT()
        streaming_stt.load_model()
        keyword_extractor = KeywordExtractor()
    else:
        streaming_stt = stt_processor

    audio_buffer = []
    process_chunk_size = 1600  # 100ms 단위로 처리

    try:
        while True:
            message = ws.receive()
            if message:
                data = json.loads(message)

                if data.get('type') == 'audio':
                    # Base64 오디오 데이터 디코딩
                    import base64
                    audio_bytes = base64.b64decode(data['data'])
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

                    # 버퍼에 추가
                    audio_buffer.extend(audio_array)

                    # 충분한 데이터가 모이면 처리
                    while len(audio_buffer) >= process_chunk_size:
                        chunk = audio_buffer[:process_chunk_size]
                        audio_buffer = audio_buffer[process_chunk_size:]

                        # 스트리밍 처리
                        result = streaming_stt.process_stream(chunk)

                        if result:
                            # 키워드 추출
                            keywords = keyword_extractor.extract_keywords(result['text'])
                            result['keywords'] = keywords

                            # 결과 전송
                            ws.send(json.dumps({
                                "type": "transcription",
                                **result
                            }))

    except Exception as e:
        print(f"WebSocket 오류: {e}")
    finally:
        print(f"🔌 WebSocket 연결 종료: {request.remote_addr}")

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index_keywords_scroll.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """설정 정보"""
    return jsonify({
        "sample_rate": 16000,
        "language": "ko",
        "model": "medium",  # Changed from large-v3 to medium
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.is_available(),  # Added for frontend compatibility
        "llm_enabled": True,  # LLM is enabled via vLLM server
        "mode": "streaming",
        "buffer_mode": True  # VAD buffer mode is active
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎤 STT 스트리밍 서버 (VAD 기반)")
    print("Streaming STT with Voice Activity Detection")
    print("="*60 + "\n")

    if torch.cuda.is_available():
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Whisper 모델을 서버 시작 시 미리 로드
    print("🔄 Whisper 모델 미리 로딩 중...")
    stt_processor = StreamingSTT()
    stt_processor.load_model()
    print("✅ Whisper 모델 로드 완료!")

    # 키워드 추출기도 미리 초기화
    keyword_extractor = KeywordExtractor()
    print("✅ 키워드 추출기 초기화 완료!")

    print("\n🌐 서버 시작: http://localhost:5000")
    print("📡 스트리밍 모드 활성화")
    print("🔇 VAD 기반 음성 감지\n")

    app.run(host='0.0.0.0', port=5000, debug=False)