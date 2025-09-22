#!/usr/bin/env python3
"""
STT + LLM 키워드 추출 통합 서버
"""

import os
import json
import numpy as np
import time
import requests
from pathlib import Path
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_sock import Sock
from faster_whisper import WhisperModel
import base64
import soundfile as sf
import torch
import threading
from queue import Queue
import asyncio

# Flask 앱 생성
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})
sock = Sock(app)

# vLLM 서버 설정
VLLM_API_URL = "http://localhost:8000/v1/completions"

class KeywordExtractor:
    """LLM을 사용한 키워드 추출 클래스"""

    def __init__(self):
        self.api_url = VLLM_API_URL
        self.min_importance = 0.8  # 중요도 임계값 (매우 중요한 명사만)

    def extract_keywords(self, text):
        """텍스트에서 키워드 추출"""
        if not text or len(text.strip()) < 10:
            return []

        prompt = f"""다음 텍스트에서 핵심 개념과 주제를 나타내는 의미있는 키워드만 추출하세요.

텍스트: "{text}"

추출 규칙:
1. 반드시 명사만 추출 (사람, 장소, 회사, 제품, 기술, 장비, 서비스 이름)
2. 동사는 절대 추출하지 마세요 (가다, 듣다, 있다, 하다, 되다, 미치다 등 금지)
3. 동사의 어간이나 어미도 추출하지 마세요 (듣고, 가고, 있는, 되는 등 금지)
4. 구체적인 대상을 지칭하는 명사만 추출
5. 최대 2개까지만, 가장 핵심적인 명사만

절대 추출하면 안 되는 것들:
- 동사와 그 활용형: 듣다, 듣고, 가다, 가고, 있다, 있는, 되다, 되는, 하다, 하는, 미치다, 미치는
- 조사: 은, 는, 이, 가, 을, 를, 에, 에서, 로, 와, 과, 도, 만, 까지
- 대명사: 나, 너, 우리, 그것, 이것, 저것
- 감정표현: ㅋ, ㅎ, ㅠ

JSON 응답:
{{
  "keywords": [
    {{"word": "키워드", "importance": 0.9, "category": "카테고리"}}
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
                    # JSON 부분만 추출 (더 강력한 처리)
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
                            # 마지막 } 찾기
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
                            print(f"📝 [{timestamp}] 추출된 키워드:")
                            print(f"   원본 텍스트: {text[:100]}...")
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
                    return self._fallback_extraction(text)

        except Exception as e:
            print(f"LLM 키워드 추출 오류: {e}")
            return self._fallback_extraction(text)

        return []

    def _fallback_extraction(self, text):
        """LLM 실패시 간단한 키워드 추출"""
        import re

        # 조사, 어미 제거 패턴
        text_clean = re.sub(r'[은는이가을를에서도의로부터까지만]+\s', ' ', text)
        text_clean = re.sub(r'[습니다니까세요어요아요죠네요]+[\s\.\,\!]', ' ', text_clean)

        # 단어 추출
        words = text_clean.split()
        keywords = []
        seen = set()

        for word in words:
            # 특수문자 제거
            word = re.sub(r'[^\w가-힣a-zA-Z0-9]', '', word)

            # 2글자 이상, 중복 제거
            if len(word) >= 2 and word not in seen:
                seen.add(word)
                keywords.append({
                    "word": word,
                    "importance": 0.6,
                    "category": "기타"
                })

        return keywords[:5]  # 상위 5개만

class WhisperProcessor:
    """Whisper STT 처리 클래스"""

    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.sample_rate = 16000
        self.processing_queue = Queue()
        self.worker_thread = None
        self.keyword_extractor = KeywordExtractor()
        # Hallucination 필터링을 위한 패턴
        self.hallucination_patterns = [
            "시청해주셔서 감사합니다",
            "시청해 주셔서 감사합니다",
            "MBC 뉴스",
            "KBS 뉴스",
            "SBS 뉴스",
            "YTN 뉴스",
            "입니다",
            "감사합니다",
            "구독과 좋아요",
            "알림 설정"
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
                    # STT 처리
                    text_result = self._transcribe_chunk(audio_data)

                    if text_result and text_result.get("text"):
                        # 키워드 추출
                        keywords = self.keyword_extractor.extract_keywords(text_result["text"])
                        text_result["keywords"] = keywords

                    if callback:
                        callback(text_result)

            except Exception as e:
                print(f"❌ 워커 오류: {e}")

    def is_hallucination(self, text):
        """Hallucination 텍스트인지 확인"""
        if not text:
            return False

        # 패턴 매칭으로 hallucination 확인
        for pattern in self.hallucination_patterns:
            if pattern in text:
                print(f"⚠️ Hallucination 감지: '{pattern}' in '{text}'")
                return True

        # 최근 텍스트와 동일한지 확인 (반복 감지)
        if len(self.recent_texts) >= 3:
            # 최근 3개 텍스트 중 2개 이상 동일하면 hallucination
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

        # 오디오 에너지 계산
        energy = np.sqrt(np.mean(np.square(audio_data)))
        if energy < 0.005:  # 너무 조용한 오디오는 스킵
            print(f"⏭️ 너무 조용한 오디오 스킵 (energy: {energy:.4f})")
            return None

        # 임시 파일 저장
        temp_file = f"/tmp/audio_{time.time()}.wav"
        sf.write(temp_file, audio_data, self.sample_rate)

        try:
            segments, info = self.model.transcribe(
                temp_file,
                language="ko",
                task="transcribe",
                beam_size=2,
                best_of=2,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.6,  # 더 엄격하게
                    "min_speech_duration_ms": 400,  # 더 길게
                    "min_silence_duration_ms": 500,  # 더 길게
                    "speech_pad_ms": 30  # 더 짧게
                },
                condition_on_previous_text=False,
                initial_prompt="실시간 대화 음성 인식",  # 프롬프트 추가
                word_timestamps=False,
                no_speech_threshold=0.7,  # 무음 임계값 추가
                compression_ratio_threshold=2.0  # 압축률 임계값
            )

            text = ""
            for segment in segments:
                # no_speech_prob 확인
                if hasattr(segment, 'no_speech_prob') and segment.no_speech_prob > 0.7:
                    continue
                text += segment.text

            text = text.strip().replace("  ", " ")

            # Hallucination 체크
            if self.is_hallucination(text):
                print(f"🚫 Hallucination 필터링: {text}")
                return None

            # 최근 텍스트 리스트 업데이트
            self.recent_texts.append(text)
            if len(self.recent_texts) > 5:
                self.recent_texts.pop(0)

            # 너무 짧은 텍스트 필터링
            if len(text) < 2:
                return None

            result = {
                "text": text,
                "language": info.language if info.language else "ko",
                "confidence": float(info.language_probability) if info.language_probability else 0.95,
                "device": self.device,
                "timestamp": time.time()
            }

            return result

        except Exception as e:
            print(f"❌ Transcribe 오류: {e}")
            return None
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def process_audio_async(self, audio_data, callback):
        """비동기 오디오 처리"""
        self.processing_queue.put((audio_data, callback))

# Whisper 프로세서 인스턴스
processor = WhisperProcessor()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index_keywords.html')

@app.route('/api/config')
def get_config():
    """설정 정보 제공"""
    return {
        'status': 'ready',
        'websocket': 'integrated',
        'device': processor.device,
        'gpu': torch.cuda.is_available(),
        'mode': 'keyword_extraction',
        'llm_enabled': True
    }

@sock.route('/ws')
def websocket(ws):
    """WebSocket 연결 처리"""
    print(f"✅ 클라이언트 연결")
    print(f"📡 키워드 추출 모드 활성화: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    audio_buffer = []
    buffer_duration = 0
    max_buffer_duration = 1.5
    min_buffer_duration = 1.0
    last_result_text = ""

    def send_result(result):
        """결과 전송 콜백"""
        nonlocal last_result_text
        if result and result.get("text") and result["text"] != last_result_text:
            response = {
                "type": "transcription",
                "text": result["text"],
                "keywords": result.get("keywords", []),
                "language": result.get("language", "ko"),
                "confidence": result.get("confidence", 0.95),
                "device": result.get("device", "cuda"),
                "timestamp": time.time()
            }
            try:
                ws.send(json.dumps(response, ensure_ascii=False))
                print(f"📝 텍스트: {result['text'][:50]}...")
                print(f"🔑 키워드: {[k['word'] for k in result.get('keywords', [])]}")
                last_result_text = result["text"]
            except:
                pass

    while True:
        try:
            message = ws.receive()
            if message is None:
                break

            try:
                data = json.loads(message)

                if data.get("type") == "audio":
                    audio_base64 = data.get("data", "")
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                    audio_buffer.extend(audio_chunk)

                    buffer_duration = len(audio_buffer) / 16000

                    if buffer_duration >= min_buffer_duration:
                        energy = np.sqrt(np.mean(np.square(audio_chunk)))

                        if energy > 0.01 or buffer_duration >= max_buffer_duration:
                            audio_array = np.array(audio_buffer)
                            processor.process_audio_async(audio_array, send_result)

                            overlap_samples = int(0.1 * 16000)
                            audio_buffer = audio_buffer[-overlap_samples:] if len(audio_buffer) > overlap_samples else []
                            buffer_duration = len(audio_buffer) / 16000

                elif data.get("type") == "config":
                    ws.send(json.dumps({
                        "type": "config_updated",
                        "language": "ko",
                        "device": processor.device,
                        "mode": "keyword_extraction"
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
    print("🎤 STT + 🤖 LLM 키워드 추출 시스템")
    print("Real-time STT with AI Keyword Extraction")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    processor.load_model()

    print(f"\n🌐 통합 서버 시작: http://localhost:5000")
    print(f"🔑 키워드 추출 활성화")
    print("WebSocket은 /ws 경로에서 처리됩니다.\n")

    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()