#!/usr/bin/env python3
"""
STT + LLM 키워드 추출 통합 서버 - Stable Version
- Error handling for CUDNN issues
- Fallback to CPU if GPU fails
- Graceful error recovery
"""

import json
import time
import torch
import numpy as np
import soundfile as sf
import os
import requests
import base64
from flask import Flask, render_template, request, jsonify
from flask_sock import Sock
from faster_whisper import WhisperModel
import warnings
import re
import traceback
warnings.filterwarnings("ignore")

# 환경 변수
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/completions")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
sock = Sock(app)

# 전역 변수
stt_processor = None
keyword_extractor = None

class StreamingSTT:
    """스트리밍 STT 처리 - Stable Version"""

    def __init__(self):
        self.model = None
        self.device = None
        self.audio_buffer = []
        self.process_interval = 16000  # 1초마다 처리 (16kHz)
        self.use_gpu = False

        # 할루시네이션 패턴
        self.hallucination_patterns = [
            "감사합니다", "시청해주셔서", "구독", "좋아요",
            "MBC 뉴스", "KBS 뉴스", "다음 영상", "영상편집"
        ]

    def load_model(self):
        """모델 로드 - GPU 실패시 CPU 폴백"""
        print("🔄 Whisper 모델 로딩 중...")

        # Try GPU first with error handling
        try:
            if torch.cuda.is_available():
                print(f"✅ GPU 감지됨: {torch.cuda.get_device_name(0)}")
                print("🔄 GPU 모델 로드 시도...")

                # Try to load with GPU
                self.model = WhisperModel(
                    "medium",
                    device="cuda",
                    device_index=0,
                    compute_type="float16",
                    download_root="/app/models",
                    num_workers=1,
                    cpu_threads=0
                )

                # Test the model with a small audio
                test_audio = np.zeros(16000, dtype=np.float32)
                test_file = "/tmp/test_model.wav"
                sf.write(test_file, test_audio, 16000)

                # Try a test transcription
                segments, _ = self.model.transcribe(
                    test_file,
                    language="ko",
                    vad_filter=False
                )
                # Consume the generator to test
                for _ in segments:
                    break

                os.remove(test_file)
                self.device = "cuda"
                self.use_gpu = True
                print("✅ GPU 모델 로드 성공!")
                return True

        except Exception as e:
            print(f"⚠️ GPU 로드 실패: {e}")
            print("🔄 CPU 모드로 폴백...")

        # Fallback to CPU
        try:
            self.model = WhisperModel(
                "medium",
                device="cpu",
                compute_type="int8",
                download_root="/app/models",
                num_workers=1,
                cpu_threads=4
            )
            self.device = "cpu"
            self.use_gpu = False
            print("✅ CPU 모델 로드 완료!")
            return True

        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False

    def process_stream(self, audio_chunk):
        """오디오 스트림 처리 - 무조건 버퍼에 추가"""
        if self.model is None:
            return None

        # 버퍼에 추가
        self.audio_buffer.extend(audio_chunk)

        # 디버그 로그 (매 10번째 청크만)
        if len(self.audio_buffer) % 16000 == 0:
            print(f"📊 Buffer: {len(self.audio_buffer)} samples")

        # 1초 이상 모이면 처리
        if len(self.audio_buffer) >= self.process_interval:
            return self._process_buffer()

        return None

    def _process_buffer(self):
        """버퍼 처리 - 에러 처리 강화"""
        if not self.audio_buffer or len(self.audio_buffer) < 4800:  # 최소 0.3초
            self.audio_buffer = []
            return None

        print(f"🎯 Processing {len(self.audio_buffer)} samples...")

        # 버퍼를 numpy 배열로 변환
        audio_data = np.array(self.audio_buffer, dtype=np.float32)

        # 정규화
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # 임시 파일 저장
        temp_file = f"/tmp/audio_{time.time()}.wav"

        try:
            sf.write(temp_file, audio_data, 16000)

            # Whisper 처리 with error handling
            print(f"🔊 Running Whisper transcription ({self.device})...")

            segments, info = self.model.transcribe(
                temp_file,
                language="ko",
                beam_size=3 if self.device == "cpu" else 5,
                best_of=3 if self.device == "cpu" else 5,
                patience=1,
                length_penalty=1,
                temperature=[0.0],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                word_timestamps=False,
                vad_filter=False
            )

            # 전사 결과 수집
            full_text = []
            for segment in segments:
                text = segment.text.strip()
                if text and not self._is_hallucination(text):
                    full_text.append(text)
                    print(f"📝 Segment: {text}")

            # 버퍼 초기화
            self.audio_buffer = []

            if full_text:
                result_text = " ".join(full_text)
                print(f"✅ Transcription: {result_text}")
                return {
                    "text": result_text,
                    "language": info.language if info else "ko",
                    "timestamp": time.time(),
                    "device": self.device
                }
            else:
                print("⚠️ No valid text from Whisper")

        except Exception as e:
            print(f"❌ Whisper 처리 오류: {e}")
            print(f"📋 Error details: {traceback.format_exc()}")

            # GPU 에러시 CPU로 재시도
            if self.use_gpu and "cudnn" in str(e).lower():
                print("🔄 GPU 에러 감지, CPU로 재시도...")
                self.use_gpu = False
                self.device = "cpu"

                # 모델 재로드 (CPU)
                try:
                    self.model = WhisperModel(
                        "medium",
                        device="cpu",
                        compute_type="int8",
                        download_root="/app/models",
                        num_workers=1,
                        cpu_threads=4
                    )
                    print("✅ CPU 모델로 전환 완료")

                    # CPU로 재시도
                    return self._process_buffer()

                except Exception as cpu_error:
                    print(f"❌ CPU 폴백 실패: {cpu_error}")

            # 버퍼 초기화
            self.audio_buffer = []

        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

        return None

    def _is_hallucination(self, text):
        """할루시네이션 체크"""
        for pattern in self.hallucination_patterns:
            if pattern in text:
                return True
        return False

class KeywordExtractor:
    """키워드 추출 (LLM 사용)"""

    def __init__(self):
        self.min_importance = 0.3
        self.vllm_url = VLLM_API_URL

    def extract_keywords(self, text):
        """텍스트에서 키워드 추출"""
        if not text or len(text) < 5:
            return []

        # 프롬프트 구성
        prompt = f"""다음 텍스트에서 중요한 키워드를 추출하세요.
JSON 형식으로만 응답하세요.

텍스트: "{text}"

응답 형식:
{{
    "keywords": [
        {{"keyword": "단어1", "importance": 0.9}},
        {{"keyword": "단어2", "importance": 0.7}}
    ]
}}

키워드:"""

        try:
            response = requests.post(
                self.vllm_url,
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.1,
                    "stop": ["}", "}}"]
                },
                timeout=5
            )

            if response.status_code == 200:
                text_response = response.json()['choices'][0]['text']

                # JSON 파싱 시도
                json_pattern = r'\{[^{}]*"keywords"[^{}]*\[[^\]]*\][^{}]*\}'
                json_match = re.search(json_pattern, text_response, re.DOTALL)

                if json_match:
                    data = json.loads(json_match.group(0))
                    keywords = data.get('keywords', [])
                    return [k for k in keywords if k.get('importance', 0) >= self.min_importance]

        except Exception as e:
            print(f"⚠️ LLM 키워드 추출 오류: {e}")

        return []

# WebSocket 핸들러
@sock.route('/ws')
def websocket(ws):
    """WebSocket 연결 처리"""
    print(f"🔌 새 WebSocket 연결: {request.remote_addr}")

    # 전역 변수로 미리 로드된 모델 사용
    global stt_processor, keyword_extractor

    if stt_processor is None:
        streaming_stt = StreamingSTT()
        if not streaming_stt.load_model():
            ws.send(json.dumps({"type": "error", "message": "Model load failed"}))
            return
        stt_processor = streaming_stt
    else:
        streaming_stt = stt_processor

    if keyword_extractor is None:
        keyword_extractor = KeywordExtractor()

    audio_buffer = []
    process_chunk_size = 1600  # 100ms 단위로 처리
    connection_active = True
    packet_count = 0

    try:
        while connection_active:
            try:
                message = ws.receive(timeout=1)
                if message is None:
                    continue

                data = json.loads(message)

                if data.get('type') == 'audio':
                    # Base64 오디오 데이터 디코딩
                    try:
                        audio_bytes = base64.b64decode(data['data'])

                        # Float32 배열로 변환
                        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

                        packet_count += 1
                        if packet_count % 100 == 1:
                            print(f"🎵 Audio packet #{packet_count}: size={len(audio_array)}")

                        # 버퍼에 추가
                        audio_buffer.extend(audio_array)

                        # 충분한 데이터가 모이면 처리
                        while len(audio_buffer) >= process_chunk_size:
                            chunk = audio_buffer[:process_chunk_size]
                            audio_buffer = audio_buffer[process_chunk_size:]

                            # 스트리밍 처리
                            result = streaming_stt.process_stream(chunk)

                            if result:
                                print(f"✅ 전사 완료: {result['text']}")

                                # 키워드 추출
                                keywords = keyword_extractor.extract_keywords(result['text'])
                                result['keywords'] = keywords

                                # 결과 전송
                                ws.send(json.dumps({
                                    "type": "transcription",
                                    **result
                                }))

                    except Exception as e:
                        print(f"⚠️ 오디오 처리 오류: {e}")
                        # 에러를 클라이언트에 전달하지 않고 계속 처리

                elif data.get('type') == 'config':
                    # 설정 메시지 처리
                    print(f"📋 설정 수신: {data}")

            except json.JSONDecodeError as e:
                print(f"⚠️ JSON 파싱 오류: {e}")
            except TimeoutError:
                # 타임아웃은 정상 - 계속 진행
                pass
            except Exception as e:
                if "Connection closed" in str(e):
                    connection_active = False
                else:
                    print(f"⚠️ 메시지 처리 오류: {e}")

    except Exception as e:
        print(f"❌ WebSocket 오류: {e}")
    finally:
        print(f"🔌 WebSocket 연결 종료: {request.remote_addr}")

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index_keywords_scroll_fixed.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """설정 정보"""
    global stt_processor

    device = "unknown"
    gpu_available = torch.cuda.is_available()

    if stt_processor:
        device = stt_processor.device
    elif gpu_available:
        device = "cuda"
    else:
        device = "cpu"

    return jsonify({
        "sample_rate": 16000,
        "language": "ko",
        "model": "medium",
        "device": device,
        "gpu": gpu_available,
        "llm_enabled": True,
        "mode": "streaming",
        "buffer_mode": True,
        "stable_version": True
    })

@app.route('/api/test-audio', methods=['POST'])
def test_audio():
    """오디오 파일 테스트 엔드포인트"""
    global stt_processor, keyword_extractor

    if stt_processor is None:
        stt_processor = StreamingSTT()
        if not stt_processor.load_model():
            return jsonify({"error": "Model load failed"}), 500

    if keyword_extractor is None:
        keyword_extractor = KeywordExtractor()

    # 파일 받기
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files['audio']

    # 임시 파일로 저장
    temp_path = f"/tmp/test_{time.time()}.wav"
    audio_file.save(temp_path)

    try:
        # 오디오 로드
        audio_data, sr = sf.read(temp_path)

        # 16kHz로 리샘플링 (필요한 경우)
        if sr != 16000:
            from scipy import signal
            audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sr))

        # 프로세싱
        stt_processor.audio_buffer = list(audio_data)
        result = stt_processor._process_buffer()

        if result:
            keywords = keyword_extractor.extract_keywords(result['text'])
            result['keywords'] = keywords
            return jsonify(result)
        else:
            return jsonify({"error": "No transcription"}), 500

    except Exception as e:
        print(f"❌ Test audio error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.route('/api/health', methods=['GET'])
def health():
    """헬스 체크 엔드포인트"""
    global stt_processor

    status = {
        "status": "healthy",
        "model_loaded": stt_processor is not None,
        "gpu_available": torch.cuda.is_available()
    }

    if stt_processor:
        status["device"] = stt_processor.device
        status["use_gpu"] = stt_processor.use_gpu

    return jsonify(status)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎤 STT 스트리밍 서버 (Stable Version)")
    print("="*60 + "\n")

    if torch.cuda.is_available():
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("💻 CPU 모드로 실행")

    # 모델 미리 로드
    print("\n🚀 모델 초기화 중...")
    stt_processor = StreamingSTT()
    if stt_processor.load_model():
        keyword_extractor = KeywordExtractor()
        print("✅ 모델 준비 완료!\n")
    else:
        print("⚠️ 모델 로드 실패 - 런타임에 재시도합니다.\n")
        stt_processor = None

    # 서버 시작
    print(f"🌐 서버 시작: http://localhost:5000")
    print(f"💪 Stable Version - Error Recovery Enabled")
    print(f"🔌 WebSocket 엔드포인트: ws://localhost:5000/ws")
    print(f"🧪 테스트 엔드포인트: POST /api/test-audio")
    print(f"❤️ 헬스 체크: GET /api/health")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)