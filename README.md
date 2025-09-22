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
- **임베딩 서비스**: BGE-M3 다국어 임베딩 모델
- **벡터 DB**: Qdrant (1024차원 벡터 저장)
- **RAG 오케스트레이터**: FastAPI 기반 RAG 파이프라인
- **캐시**: Redis (TTL 1시간)

### GPU 메모리 사용 (RTX 5090 32GB 기준)
- Whisper Medium: ~5GB
- vLLM (Qwen2.5-7B): ~13GB
- BGE-M3 임베딩: ~3GB
- **총 사용량**: ~21GB

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

### 🔧 일반적인 문제 해결

#### GPU 메모리 부족
1. Whisper 모델 크기 줄이기 (large → medium → small)
2. vLLM gpu-memory-utilization 값 낮추기
3. max-model-len 값 줄이기

#### WebSocket 연결 실패
1. 방화벽 설정 확인
2. 포트 5000이 열려있는지 확인
3. Docker 네트워크 설정 확인

#### 할루시네이션 발생
1. temperature 값 낮추기 (0.0 권장)
2. 할루시네이션 패턴 추가
3. VAD 파라미터 조정

### 🚨 CUDNN 경고 및 서비스 복원

#### 문제 상황
- CUDNN 라이브러리 경고 메시지 발생
- STT 서비스 중단 또는 불안정
- WebSocket 연결 끊김 현상

#### 해결 방법

##### 1. 기존 작동 이미지로 복원
```bash
# 현재 STT 컨테이너 정리
docker stop $(docker ps -q --filter "name=stt") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=stt") 2>/dev/null || true

# 네트워크 생성 (없는 경우)
docker network create stt-network 2>/dev/null || true

# 원래 작동했던 STT 이미지로 복원
docker run -d \
    --name stt-streaming \
    --gpus all \
    -p 5000:5000 \
    -v /app/models:/app/models \
    -e VLLM_API_URL=http://vllm-server:8000/v1/completions \
    --network stt-network \
    stt-model-stt-streaming:latest
```

##### 2. 시스템 상태 확인
```bash
# 컨테이너 상태 확인
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(vllm|stt|embedding)"

# STT 서비스 로그 확인
docker logs stt-streaming --tail 20

# GPU 상태 확인
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
```

##### 3. 서비스 동작 확인
```bash
# 설정 정보 확인
curl -s http://localhost:5000/api/config | jq

# 웹 인터페이스 접속 테스트
curl -s -I http://localhost:5000/
```

### 🐳 Docker 이미지 관리

#### 사용 가능한 이미지 확인
```bash
# STT 관련 이미지 목록
docker images | grep -E "(stt|gpu)" | head -10

# 권장 이미지: stt-model-stt-streaming:latest
```

#### 네트워크 설정
```bash
# 필요한 네트워크가 없는 경우
docker network create stt-network

# 네트워크 상태 확인
docker network ls | grep stt
```

### ⚠️ 주의사항

#### 작동하는 시스템 유지
- **CUDNN 경고가 있어도 기능이 정상 작동하면 굳이 수정하지 말 것**
- GPU 메모리와 STT 기능이 정상이면 경고 메시지는 무시 가능
- 불필요한 시행착오로 작동하는 시스템을 망가뜨리지 말 것

#### 올바른 문제 진단
- WebSocket 끊김은 주로 클라이언트 측 재연결 로직
- CUDNN 경고는 기능적 문제가 아닐 수 있음
- 로그에서 실제 오류와 단순 경고를 구분할 것

### 📊 정상 상태 확인 지표

#### 시스템 헬스체크
```bash
# 모든 서비스가 healthy 상태인지 확인
docker ps | grep healthy

# GPU 메모리 사용량이 정상 범위인지 확인 (RTX 5090 기준)
# - 총 사용량: ~22GB / 32GB (68% 사용률)
# - vLLM: ~13GB, STT(Whisper): ~5GB, 임베딩: ~3GB
```

#### 기능 테스트
- 웹 인터페이스 접속: http://localhost:5000
- API 응답: `/api/config` 엔드포인트 정상 응답
- GPU 감지: 설정에서 `"gpu": true` 확인

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

## 🔗 RAG (Retrieval-Augmented Generation) 시스템

### RAG 아키텍처
```
음성 → STT → 텍스트 → 키워드 추출 → 쿼리 확장 → 임베딩
   ↓
벡터 검색 (Qdrant) → Top-K 문서 → 재순위화
   ↓
컨텍스트 + 쿼리 → vLLM → 증강된 응답 → 사용자
```

### RAG 파이프라인 상세

#### 1. 임베딩 서비스 (Port: 8002)
- **모델**: BAAI/bge-m3 (다국어 지원)
- **차원**: 1024
- **배치 크기**: 32
- **최대 길이**: 8192 토큰
- **엔드포인트**:
  - `POST /embed`: 단일 텍스트 임베딩
  - `POST /embed/batch`: 배치 임베딩 처리
  - `GET /health`: 서비스 상태 확인

#### 2. 벡터 데이터베이스 (Qdrant - Ports: 6333/6334)
- **컬렉션**: documents
- **벡터 차원**: 1024
- **거리 메트릭**: Cosine
- **페이로드**: 텍스트, 메타데이터, 소스 정보
- **대시보드**: http://localhost:6333/dashboard

#### 3. RAG 오케스트레이터 (Port: 8003)
- **프레임워크**: FastAPI
- **주요 기능**:
  - 쿼리 확장 (Query Expansion)
  - 하이브리드 검색 (Dense + Sparse)
  - 재순위화 (Reranking)
  - 컨텍스트 증강 생성
- **설정값**:
  - TOP_K_RETRIEVAL: 10
  - TOP_K_RERANK: 5
  - MAX_CONTEXT_LENGTH: 2048

### RAG 데이터 플로우

#### 단계별 처리 과정
1. **쿼리 확장**: 키워드를 관련 용어로 확장
2. **임베딩 생성**: BGE-M3로 쿼리 벡터 생성
3. **벡터 검색**: Qdrant에서 유사 문서 검색
4. **재순위화**: 관련성 점수 기반 재정렬
5. **컨텍스트 구성**: 상위 문서로 컨텍스트 생성
6. **응답 생성**: vLLM으로 최종 응답 생성

### Redis 캐싱 전략
- **쿼리 캐시**: 자주 묻는 질문 (TTL: 1시간)
- **임베딩 캐시**: 계산된 벡터 (TTL: 24시간)
- **세션 관리**: 다중 턴 대화 컨텍스트

### RAG 서비스 시작
```bash
# RAG 관련 서비스만 시작
docker-compose --profile rag up -d

# 개별 서비스 시작
docker start qdrant
docker start embedding-service
docker start rag-orchestrator
docker start redis
```

### RAG API 엔드포인트

#### 임베딩 서비스
```bash
# 텍스트 임베딩 생성
curl -X POST http://localhost:8002/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "인공지능 기술의 발전"}'

# 배치 임베딩
curl -X POST http://localhost:8002/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["텍스트1", "텍스트2"]}'
```

#### RAG 오케스트레이터
```bash
# RAG 쿼리 처리
curl -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "인공지능이란 무엇인가요?",
    "keywords": ["인공지능", "AI", "기계학습"]
  }'

# 벡터 검색
curl -X POST http://localhost:8003/search \
  -H "Content-Type: application/json" \
  -d '{"query": "딥러닝", "top_k": 5}'
```

### 문서 인덱싱
```bash
# 문서를 벡터 DB에 추가
curl -X POST http://localhost:8003/index \
  -H "Content-Type: application/json" \
  -d '{
    "text": "문서 내용...",
    "metadata": {
      "source": "document.pdf",
      "page": 1,
      "date": "2024-01-01"
    }
  }'
```

### RAG 성능 메트릭
- **임베딩 생성**: ~50ms/텍스트
- **벡터 검색**: ~100ms (10K 문서 기준)
- **재순위화**: ~30ms
- **전체 RAG 응답**: <2초

### RAG 트러블슈팅

#### Qdrant Unhealthy 상태
```bash
# Qdrant 상태 확인
docker logs qdrant --tail 50

# Qdrant 재시작
docker restart qdrant

# 데이터 볼륨 확인
docker volume ls | grep qdrant
```

#### 임베딩 서비스 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# 배치 크기 조정 (환경변수)
docker run -e BATCH_SIZE=16 embedding-service
```

#### RAG 응답 지연
1. Redis 캐시 상태 확인
2. Qdrant 인덱스 최적화
3. TOP_K 값 조정

## 🔑 주요 기술

- **Faster-Whisper**: CTranslate2 기반 최적화된 Whisper
- **vLLM**: PagedAttention으로 고속 LLM 서빙
- **WebSocket**: 실시간 양방향 통신
- **VAD**: 실시간 음성 구간 감지
- **BGE-M3**: 다국어 임베딩 모델
- **Qdrant**: 고성능 벡터 데이터베이스
- **Redis**: 인메모리 캐싱
- **Docker**: 컨테이너 기반 배포

## 📄 라이센스

MIT License

## 🤝 기여

Issues와 Pull Requests를 환영합니다!

## 📞 문의

- GitHub: [Cloud-Linuxer/stt-model](https://github.com/Cloud-Linuxer/stt-model)