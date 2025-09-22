#!/usr/bin/env python3
"""
STT + LLM 키워드 추출 통합 서버 (버퍼링 개선 버전)
- STT는 실시간 청크 처리
- LLM은 버퍼에 모인 텍스트를 주기적으로 처리
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

class KeywordExtractor:
    """LLM을 사용한 키워드 추출 클래스 (버퍼링)"""

    def __init__(self):
        self.api_url = VLLM_API_URL
        self.min_importance = 0.8

        # 텍스트 버퍼 관리
        self.text_buffer = deque(maxlen=10)  # 최근 10개 문장 저장
        self.buffer_lock = threading.Lock()
        self.last_extraction_time = time.time()
        self.extraction_interval = 5.0  # 5초마다 키워드 추출

    def add_to_buffer(self, text):
        """텍스트를 버퍼에 추가 (할루시네이션 텍스트는 제외)"""
        if not text or len(text.strip()) <= 5:
            return

        # 할루시네이션 패턴이 포함된 텍스트는 버퍼에 추가하지 않음
        hallucination_patterns = [
            "영상편집 박진주", "감사합니다", "시청해주셔서",
            "이 시각 세계였습니다", "MBC 뉴스", "KBS 뉴스"
        ]

        for pattern in hallucination_patterns:
            if pattern in text:
                print(f"⚠️ 버퍼 추가 차단 (할루시네이션): {text}")
                return

        with self.buffer_lock:
            self.text_buffer.append(text)

    def should_extract(self):
        """키워드 추출 시점 확인"""
        current_time = time.time()
        # 5초 경과 또는 버퍼가 충분히 찬 경우
        return (current_time - self.last_extraction_time >= self.extraction_interval
                or len(self.text_buffer) >= 5)

    def extract_keywords_from_buffer(self):
        """버퍼의 전체 텍스트에서 키워드 추출"""
        with self.buffer_lock:
            if not self.text_buffer:
                return []

            # 버퍼의 텍스트를 결합 (최대 5개 문장)
            recent_texts = list(self.text_buffer)[-5:]
            combined_text = " ".join(recent_texts)
            self.last_extraction_time = time.time()

        # 충분한 텍스트가 모였을 때만 추출
        if len(combined_text) < 30:
            return []

        return self.extract_keywords(combined_text)

    def extract_keywords(self, text):
        """텍스트에서 키워드 추출"""
        if not text or len(text.strip()) < 10:
            return []

        # 키워드 추출 프롬프트
        prompt = f"""다음 텍스트에서 전체 문맥을 고려하여 가장 중요한 명사만 추출하세요.

텍스트: "{text}"

추출 규칙:
1. 반드시 명사만 추출 (사람, 장소, 회사, 제품, 기술, 장비, 서비스 이름)
2. 동사는 절대 추출하지 마세요 (가다, 듣다, 있다, 하다, 되다, 미치다 등 금지)
3. 동사의 어간이나 어미도 추출하지 마세요 (듣고, 가고, 있는, 되는 등 금지)
4. 구체적인 대상을 지칭하는 명사만 추출
5. 최대 3개까지만, 가장 핵심적인 명사만
6. 전체 문맥에서 반복되거나 중요한 주제를 우선 추출

절대 추출하면 안 되는 것들:
- 동사와 그 활용형: 듣다, 듣고, 가다, 가고, 있다, 있는, 되다, 되는, 하다, 하는, 미치다, 미치는
- 조사: 은, 는, 이, 가, 을, 를, 에, 에서, 로, 와, 과, 도, 만, 까지
- 대명사: 나, 너, 우리, 그것, 이것, 저것
- 감정표현: ㅋ, ㅎ, ㅠ

JSON 응답:
{{
  "keywords": [
    {{"word": "명사1", "importance": 0.9, "category": "카테고리"}},
    {{"word": "명사2", "importance": 0.8, "category": "카테고리"}}
  ]
}}

카테고리: 인물, 장소, 회사, 기술, 제품, 장비, 서비스, 기타
중요도: 0.8 이상만 (매우 중요한 명사만)

텍스트가 짧거나 의미있는 키워드가 없으면 빈 배열 반환.
응답:"""

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
                try:
                    # JSON 부분만 추출
                    import re

                    # JSON 패턴 찾기
                    json_pattern = r'\{[^{}]*"keywords"[^{}]*\[[^\]]*\][^{}]*\}'
                    json_match = re.search(json_pattern, text_response, re.DOTALL)

                    if json_match:
                        json_text = json_match.group(0)
                    else:
                        # 기본 방법 시도
                        if '{' in text_response:
                            json_start = text_response.index('{')
                            json_end = text_response.rfind('}')
                            if json_end > json_start:
                                json_text = text_response[json_start:json_end+1]
                            else:
                                json_text = text_response[json_start:] + '}'
                        else:
                            raise ValueError("No JSON found")

                    data = json.loads(json_text)
                    keywords = data.get('keywords', [])

                    # 중요도 필터링
                    filtered = [k for k in keywords if k.get('importance', 0) >= self.min_importance]

                    # 키워드 로깅
                    if filtered:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print(f"\n{'='*60}")
                        print(f"📝 [{timestamp}] 추출된 키워드 (버퍼 기반):")
                        print(f"   원본 텍스트: {text[:150]}...")
                        print(f"   키워드 목록:")
                        for kw in filtered:
                            print(f"      - {kw['word']} ({kw.get('category', '기타')}, 중요도: {kw['importance']:.1f})")
                        print(f"{'='*60}\n")

                        # 파일로도 로깅
                        with open("/tmp/keywords_log.txt", "a", encoding="utf-8") as f:
                            f.write(f"\n[{timestamp}]\n")
                            f.write(f"텍스트: {text}\n")
                            f.write(f"키워드: {', '.join([k['word'] for k in filtered])}\n")
                            f.write("-" * 50 + "\n")

                    return filtered
                except json.JSONDecodeError:
                    print(f"JSON 파싱 실패: {text_response}")
                    return []

        except Exception as e:
            print(f"LLM 키워드 추출 오류: {e}")
            return []

        return []

class STTProcessor:
    """STT 처리 클래스 (실시간 청크 처리)"""

    def __init__(self, keyword_extractor):
        self.model = None
        self.device = None
        self.sample_rate = 16000
        self.processing_queue = queue.Queue()
        self.worker_thread = None
        self.keyword_extractor = keyword_extractor

        # Hallucination 패턴 (로그 분석 기반 강화)
        self.hallucination_patterns = [
            "시청해주셔서 감사합니다",
            "구독과 좋아요",
            "알림 설정",
            "다음 영상에서",
            "MBC 뉴스",
            "KBS 뉴스",
            "SBS 뉴스",
            "YTN 뉴스",
            "JTBC 뉴스",
            "지금까지",
            "날씨입니다",
            "뉴스 특보",
            "속보입니다",
            "기자입니다",
            "앵커입니다",
            "영상편집 박진주",  # 로그에서 반복 확인
            "이 시각 세계였습니다",  # 로그에서 반복 확인
            "한글자막 by",  # 자막 관련
            "기상캐스터",  # 날씨 방송 관련
            "김성현입니다",  # 뉴스 앵커 이름
            "배혜지",  # 기상캐스터 이름
            "자막 제공",  # 자막 관련
            "자막을 사용",  # 자막 관련
            "네 감사합니다",  # 방송 종료 멘트
            "수고하셨습니다",  # 방송 종료 멘트
            "고생하셨습니다",  # 방송 종료 멘트
            "다음 시간에",  # 방송 예고
            "만나요",  # 방송 종료 멘트
            "뵙겠습니다",  # 방송 종료 멘트
            "여기까지",  # 방송 종료 멘트
            "마치겠습니다",  # 방송 종료 멘트
            "마무리",  # 방송 종료 멘트
            "끝으로",  # 방송 종료 멘트
            "정리하겠습니다"  # 방송 종료 멘트
        ]

        self.recent_texts = []  # 최근 인식 텍스트 저장

    def load_model(self):
        """모델 로드"""
        print("🔄 Whisper 모델 로딩 중 (GPU 모드)...")

        if torch.cuda.is_available():
            print(f"✅ GPU 감지됨: {torch.cuda.get_device_name(0)}")
            device = "cuda"
            compute_type = "float16"
        else:
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

            # 키워드 추출 타이머 시작
            self.start_keyword_timer()

            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False

    def start_worker(self):
        """처리 워커 스레드 시작"""
        self.worker_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
        self.worker_thread.start()
        print("🔧 백그라운드 처리 워커 시작")

    def start_keyword_timer(self):
        """주기적 키워드 추출 타이머"""
        def extract_periodically():
            while True:
                time.sleep(5)  # 5초마다 체크
                if self.keyword_extractor.should_extract():
                    keywords = self.keyword_extractor.extract_keywords_from_buffer()
                    if keywords:
                        # 최신 콜백에 키워드 전송
                        if hasattr(self, 'last_callback') and self.last_callback:
                            try:
                                self.last_callback({
                                    "keywords_update": True,
                                    "keywords": keywords
                                })
                            except:
                                pass

        timer_thread = threading.Thread(target=extract_periodically, daemon=True)
        timer_thread.start()
        print("⏱️ 키워드 추출 타이머 시작")

    def _process_audio_queue(self):
        """큐에서 오디오 처리 (백그라운드)"""
        while True:
            try:
                audio_data, callback = self.processing_queue.get()
                if audio_data is not None:
                    # STT 처리 (실시간)
                    text_result = self._transcribe_chunk(audio_data)

                    if text_result and text_result.get("text"):
                        # 텍스트를 버퍼에 추가
                        self.keyword_extractor.add_to_buffer(text_result["text"])

                        # 콜백 저장 (나중에 키워드 업데이트용)
                        self.last_callback = callback

                        # 즉시 텍스트만 전송 (키워드 없이)
                        text_result["keywords"] = []  # 일단 빈 배열

                    if callback:
                        callback(text_result)

            except Exception as e:
                print(f"❌ 워커 오류: {e}")

    def is_hallucination(self, text):
        """Hallucination 텍스트인지 확인"""
        if not text:
            return False

        # 단어 정규화 (공백 제거, 소문자 변환)
        normalized = text.strip().lower()

        # 단일 인사말/감사말 필터링
        single_greetings = [
            "감사합니다", "고맙습니다", "감사드립니다",
            "고마워요", "감사해요", "수고하셨습니다",
            "네", "예", "아니요", "아니오", "응", "어",
            "안녕하세요", "안녕", "안녕히", "네 감사합니다"
        ]

        # 정확히 단일 인사말만 있는 경우
        if normalized in [g.lower() for g in single_greetings]:
            if len(normalized) < 10:  # 짧은 단일 인사말
                print(f"⚠️ 단일 인사말 필터링: {text}")
                return True

        # 패턴 매칭으로 hallucination 확인
        for pattern in self.hallucination_patterns:
            if pattern in text:
                print(f"⚠️ Hallucination 감지: '{pattern}' in '{text}'")
                return True

        # 최근 텍스트와 동일한지 확인 (반복 감지)
        if len(self.recent_texts) >= 3:
            if self.recent_texts.count(text) >= 2:
                print(f"⚠️ 반복 감지: '{text}'")
                return True

        return False

    def _transcribe_chunk(self, audio_data):
        """실제 전사 처리"""
        if self.model is None:
            return None

        # 오디오 정규화
        audio_data = audio_data / np.max(np.abs(audio_data) + 1e-10)

        # 오디오 에너지 계산 (더 엄격한 임계값)
        energy = np.sqrt(np.mean(np.square(audio_data)))
        if energy < 0.01:  # 임계값을 0.005에서 0.01로 상향
            print(f"⏭️ 너무 조용한 오디오 스킵 (energy: {energy:.4f})")
            return None

        # 임시 파일 저장
        temp_file = f"/tmp/audio_{time.time()}.wav"
        sf.write(temp_file, audio_data, self.sample_rate)

        try:
            segments, info = self.model.transcribe(
                temp_file,
                language="ko",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.6,  # 더 높은 음성 감지 임계값
                    min_speech_duration_ms=500,  # 최소 음성 지속 시간 증가
                    max_speech_duration_s=30,  # 최대 음성 길이 제한
                    min_silence_duration_ms=800,  # 더 긴 침묵 감지
                    speech_pad_ms=100  # 패딩 감소
                ),
                without_timestamps=True,
                # 할루시네이션 억제 파라미터 추가
                suppress_blank=True,
                suppress_tokens=[-1],  # 특수 토큰 억제
                condition_on_previous_text=False  # 이전 텍스트에 조건화 하지 않음
            )

            text = "".join([segment.text for segment in segments]).strip()

            if text:
                # Hallucination 체크
                if self.is_hallucination(text):
                    print(f"⚠️ Hallucination 필터링: {text}")
                    return None

                # 최근 텍스트 업데이트
                self.recent_texts.append(text)
                if len(self.recent_texts) > 5:
                    self.recent_texts.pop(0)

                print(f"📝 텍스트: {text}")

                return {
                    "text": text,
                    "language": info.language,
                    "device": self.device
                }

        except Exception as e:
            print(f"❌ 전사 오류: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return None

    def process_audio_stream(self, audio_data, callback=None):
        """오디오 스트림 처리 (비동기)"""
        self.processing_queue.put((audio_data, callback))

# WebSocket 핸들러
@sock.route('/ws')
def websocket(ws):
    """WebSocket 연결 처리"""
    print(f"🔌 새 WebSocket 연결: {request.remote_addr}")

    audio_buffer = []
    chunk_size = 48000  # 3초 분량으로 증가 (더 긴 컨텍스트)

    def send_result(result):
        """처리 결과 전송"""
        if result:
            try:
                ws.send(json.dumps({
                    "type": "transcription",
                    **result
                }))
            except:
                pass

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
                    if len(audio_buffer) >= chunk_size:
                        chunk = np.array(audio_buffer[:chunk_size])
                        audio_buffer = audio_buffer[chunk_size:]

                        # 비동기 처리
                        stt_processor.process_audio_stream(chunk, send_result)

                elif data.get('type') == 'config':
                    # 설정 업데이트
                    ws.send(json.dumps({
                        "type": "config_updated",
                        "language": data.get('language', 'ko')
                    }))

    except Exception as e:
        print(f"WebSocket 오류: {e}")
    finally:
        print(f"🔌 WebSocket 연결 종료: {request.remote_addr}")

# 라우트
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index_keywords_scroll.html')

@app.route('/api/config')
def get_config():
    """설정 정보 반환"""
    return jsonify({
        "gpu": torch.cuda.is_available(),
        "device": stt_processor.device if stt_processor else None,
        "llm_enabled": True,
        "buffer_mode": True  # 버퍼 모드 활성화 표시
    })

# 메인 실행
if __name__ == '__main__':
    # GPU 설정
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 키워드 추출기 초기화
    keyword_extractor = KeywordExtractor()

    # STT 프로세서 초기화
    stt_processor = STTProcessor(keyword_extractor)
    if not stt_processor.load_model():
        print("❌ 모델 로드 실패. 서버를 종료합니다.")
        exit(1)

    print("\n" + "="*60)
    print("🎤 STT + 🤖 LLM 키워드 추출 시스템 (버퍼링 모드)")
    print("Real-time STT with Buffered AI Keyword Extraction")
    print("="*60)

    # 서버 시작
    print(f"\n🌐 통합 서버 시작: http://localhost:5000")
    print("🔑 키워드 추출 활성화 (버퍼링: 5초/5문장)")
    print("WebSocket은 /ws 경로에서 처리됩니다.\n")

    app.run(host='0.0.0.0', port=5000, debug=False)