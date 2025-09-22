# STT + LLM 실시간 음성인식 및 키워드 추출 시스템

실시간 음성 인식(STT)과 LLM 기반 키워드 추출을 통합한 고성능 AI 시스템입니다.

## 🎯 주요 기능

### 실시간 음성 인식
- **Faster-Whisper**: OpenAI Whisper보다 4배 빠른 속도, 50% 메모리 절약
- **VAD(Voice Activity Detection)**: 음성 구간 자동 감지
- **스트리밍 처리**: WebSocket 기반 실시간 스트리밍
- **할루시네이션 방지**: 37개 패턴 필터링

### LLM 키워드 추출
- **vLLM 서버**: Qwen2.5-7B-Instruct 모델로 고속 추론
- **실시간 키워드 추출**: 인식된 텍스트에서 핵심 명사 자동 추출
- **카테고리 분류**: 추출된 키워드를 자동 분류

### 웹 인터페이스
- **실시간 모니터링**: 음성 인식 결과와 키워드를 실시간 표시
- **상태 표시**: GPU/LLM 상태 실시간 모니터링
- **자동 스크롤**: 새로운 내용 자동 스크롤 표시

## 🚀 Quick Start

### 1. 시스템 시작
```bash
# 전체 시스템 시작 (vLLM + STT)
./start_system.sh

# 개별 서비스 시작
./start_keywords.sh  # 키워드 추출 버전
```

### 2. 웹 접속
- 로컬: http://localhost:5000
- 외부 접속: Cloudflare Tunnel URL 확인

## 🏗️ 시스템 구성

### 컴포넌트
- **STT 서버**: Whisper Medium 모델 (GPU 가속)
- **vLLM 서버**: Qwen2.5-7B (50% GPU 메모리)
- **웹 서버**: Flask + WebSocket

### GPU 메모리 사용 (RTX 5090 32GB 기준)
- Whisper Medium: ~5GB
- vLLM (Qwen2.5-7B): ~13GB
- **총 사용량**: ~18GB

## 📁 디렉토리 구조
```
.
├── Dockerfile.vllm           # vLLM 서버 Docker 이미지
├── web_server_streaming.py   # 메인 STT 스트리밍 서버
├── web_server_keywords.py    # 키워드 추출 서버
├── start_system.sh          # 통합 시작 스크립트
├── templates/               # 웹 UI 템플릿
│   ├── index_keywords_scroll.html
│   └── index_keywords.html
├── data/                    # 테스트 오디오 파일
├── outputs/                 # 변환 결과 저장
└── models/                  # 모델 캐시
```

## 🔧 설정 및 최적화

### GPU 메모리 조정
`Dockerfile.vllm`에서 메모리 사용량 조정:
```python
--gpu-memory-utilization 0.5  # GPU 메모리 50% 사용
--max-model-len 224           # 최대 시퀀스 길이
```

### Whisper 모델 변경
`web_server_streaming.py`에서 모델 크기 변경:
```python
self.model = WhisperModel(
    "medium",  # tiny, base, small, medium, large-v3
    device="cuda",
    compute_type="float16"
)
```

### VAD 파라미터 조정
```python
self.energy_threshold = 0.02  # 에너지 임계값
self.silence_duration = 1.0   # 침묵 감지 시간
```

## 🐳 Docker 명령어

### 이미지 빌드
```bash
# vLLM 서버 빌드
docker build -t vllm-server -f Dockerfile.vllm .

# STT 서버는 기존 이미지 사용
docker-compose build
```

### 컨테이너 관리
```bash
# 상태 확인
docker ps

# 로그 확인
docker logs vllm-server -f
docker logs stt-streaming -f

# 재시작
docker restart stt-streaming
docker restart vllm-server
```

## 📊 모니터링

### GPU 상태
```bash
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv
```

### 시스템 로그
```bash
# vLLM 로그
docker logs vllm-server --tail 50

# STT 로그
docker logs stt-streaming --tail 50
```

## 🛠️ 트러블슈팅

### GPU 메모리 부족
1. Whisper 모델 크기 줄이기 (large → medium → small)
2. vLLM gpu-memory-utilization 값 낮추기
3. max-model-len 값 줄이기

### WebSocket 연결 실패
1. 방화벽 설정 확인
2. 포트 5000이 열려있는지 확인
3. Docker 네트워크 설정 확인

### 할루시네이션 발생
1. temperature 값 낮추기 (0.0 권장)
2. 할루시네이션 패턴 추가
3. VAD 파라미터 조정

## 📋 API 엔드포인트

### REST API
- `GET /`: 웹 인터페이스
- `GET /api/config`: 시스템 설정 정보

### WebSocket
- `/ws`: 실시간 오디오 스트리밍
  - 입력: Base64 인코딩된 오디오 데이터
  - 출력: 인식된 텍스트 + 키워드

### vLLM API
- `POST http://localhost:8000/v1/completions`: LLM 추론
- `GET http://localhost:8000/v1/models`: 모델 정보

## 🔑 주요 기술

- **Faster-Whisper**: CTranslate2 기반 최적화된 Whisper
- **vLLM**: PagedAttention으로 고속 LLM 서빙
- **WebSocket**: 실시간 양방향 통신
- **VAD**: 실시간 음성 구간 감지
- **Docker**: 컨테이너 기반 배포

## 📄 라이센스

MIT License

## 🤝 기여

Issues와 Pull Requests를 환영합니다!

## 📞 문의

- GitHub: [Cloud-Linuxer/stt-model](https://github.com/Cloud-Linuxer/stt-model)