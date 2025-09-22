# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 Project Overview

**STT + LLM + RAG 실시간 음성 지능형 정보 시스템**

실시간 음성 인식(STT)과 LLM 기반 키워드 추출을 넘어, 추출된 키워드에 대한 관련 정보를 자동으로 검색하고 제공하는 RAG(Retrieval-Augmented Generation) 시스템으로 확장하는 프로젝트입니다.

## 🏗️ System Architecture (Docker Compose Integrated)

### 3-Phase Architecture

#### Phase 1: Core System (Profile: core)
```
음성 입력 → STT(Whisper Medium) → 텍스트 → vLLM(Qwen2.5-7B) → 키워드 추출
```
**Services**: vllm-server, stt-streaming, redis

#### Phase 2: RAG Integration (Profile: rag)
```
키워드 → Embedding(BGE-M3) → Qdrant 검색 → 관련 문서 → RAG Orchestrator → 증강 응답
```
**Services**: +qdrant, embedding-service, rag-orchestrator

#### Phase 3: Full Production (Profile: full)
```
완전 통합 시스템 + API Gateway + Web UI + MongoDB + Nginx
```
**Services**: +mongodb, document-processor, api-gateway, web-ui, nginx

## 🐳 Docker Compose 기반 통합 배포

### 시작 명령어
```bash
# Core 시스템만 시작 (STT + LLM + Redis)
docker-compose --profile core up -d

# RAG 시스템 포함 시작
docker-compose --profile rag up -d

# 전체 프로덕션 시스템 시작
docker-compose --profile full up -d
```

### 서비스 아키텍처 (11개 서비스)

```yaml
services:
  # Core Services (필수)
  - vllm-server       # Qwen2.5-7B, 13GB VRAM, Port 8000
  - stt-streaming     # Whisper Medium, 5GB VRAM, Port 5000
  - redis            # Cache, 2GB RAM, Port 6379

  # RAG Services
  - qdrant           # Vector DB, 4GB RAM, Ports 6333/6334
  - embedding-service # BGE-M3, 3GB VRAM, Port 8002
  - rag-orchestrator  # RAG Logic, CPU, Port 8003

  # Full Stack Services
  - mongodb          # Metadata, 2GB RAM, Port 27017
  - document-processor # Doc Processing, CPU, Port 8004
  - api-gateway      # Central API, CPU, Port 8080
  - web-ui           # Frontend, Port 3000
  - nginx            # Load Balancer, Ports 80/443
```

### 네트워크 구성
```yaml
networks:
  frontend-net:  # Web UI ↔ API Gateway
  backend-net:   # Core services communication
  data-net:      # Database connections
```

## 📊 GPU 메모리 할당 전략 (RTX 5090 32GB)

```yaml
GPU Memory Allocation:
  vLLM (Qwen2.5-7B):     13GB (40% utilization)
  STT (Whisper Medium):   5GB
  Embedding (BGE-M3):     3GB
  ----------------------------
  Total Used:            21GB
  Available Buffer:      11GB (for operations)
```

## 🔄 Data Flow Architecture

### 1. Audio Input Flow
```mermaid
User Speech → WebSocket → STT Service → Text
    ↓
Text → vLLM → Keywords Extraction
    ↓
Keywords → Query Expansion (RAG Orchestrator)
```

### 2. RAG Processing Flow
```mermaid
Expanded Query → Embedding Service → Query Vector
    ↓
Query Vector → Qdrant Search → Top-K Documents
    ↓
Documents → Reranking → Relevant Context
    ↓
Context + Query → vLLM → Generated Response
    ↓
Response → Redis Cache → User
```

### 3. Document Ingestion Flow
```mermaid
Documents → Document Processor → Chunks
    ↓
Chunks → Embedding Service → Vectors
    ↓
Vectors → Qdrant Storage
Metadata → MongoDB
```

## 🚀 Implementation Roadmap (4주 계획)

### Week 1: Core System Stabilization ✅
- [x] Docker Compose 기본 구성
- [x] vLLM + STT 통합
- [x] Redis 캐싱 레이어
- [x] 기본 health monitoring

### Week 2: RAG Foundation 🔄
- [ ] Qdrant 벡터 DB 설정
- [ ] BGE-M3 임베딩 서비스 구현
- [ ] 기본 RAG Orchestrator
- [ ] 문서 ingestion 파이프라인

### Week 3: Intelligence Layer
- [ ] Advanced RAG with reranking
- [ ] Query expansion 및 최적화
- [ ] Multi-turn conversation 지원
- [ ] Context management

### Week 4: Production Ready
- [ ] Nginx 로드 밸런싱
- [ ] MongoDB 메타데이터 관리
- [ ] Monitoring & observability
- [ ] Auto-scaling policies

## 🛠️ Technical Decisions

### Vector Database: Qdrant
```yaml
Why Qdrant:
  - 대규모 확장성 우수
  - Built-in filtering & payload storage
  - Multiple vectors per point (hybrid search)
  - REST & gRPC APIs
  - Production-proven
```

### Embedding Model: BGE-M3
```yaml
Why BGE-M3:
  - 다국어 지원 (한국어 + 영어)
  - 1024 dimension vectors (균형적)
  - Dense & sparse retrieval 지원
  - ~2GB model size (3GB VRAM 할당에 적합)
```

### Caching Strategy: Redis
```yaml
Cache Levels:
  - Query Cache: 자주 묻는 질문 (TTL: 1h)
  - Embedding Cache: 계산된 임베딩 (TTL: 24h)
  - Session Management: 다중 턴 대화
```

### Document Processing
```yaml
Chunking Strategy:
  - Chunk Size: 512 tokens
  - Overlap: 50 tokens
  - Metadata: title, date, source, language
  - Formats: PDF, DOCX, TXT, HTML, Markdown
```

## 📁 Project Structure

```
stt-model/
├── docker-compose.yml     # 11-service orchestration
├── Dockerfile.vllm        # vLLM server
├── Dockerfile.stt         # STT streaming server
├── Dockerfile.embedding   # BGE-M3 embedding service
├── Dockerfile.rag         # RAG orchestrator
├── Dockerfile.processor   # Document processor
├── Dockerfile.gateway     # API gateway
├── Dockerfile.webui       # React frontend
├── config/
│   ├── nginx.conf        # Nginx configuration
│   ├── qdrant.yaml       # Qdrant settings
│   └── rag_config.yaml   # RAG pipeline config
├── rag/
│   ├── orchestrator.py   # Main RAG logic
│   ├── embedding.py      # Embedding service
│   ├── retriever.py      # Search & ranking
│   └── generator.py      # Response generation
├── processors/
│   ├── document.py       # Document chunking
│   ├── metadata.py       # Metadata extraction
│   └── ingestion.py      # Batch processing
├── data/
│   ├── documents/        # Source documents
│   ├── cache/           # Redis cache data
│   └── qdrant/          # Vector DB storage
└── scripts/
    ├── start_core.sh     # Start core services
    ├── start_rag.sh      # Start with RAG
    └── start_full.sh     # Full production start
```

## 🔧 Service Configuration Details

### vLLM Server Configuration
```python
# Dockerfile.vllm
--model Qwen/Qwen2.5-7B-Instruct
--gpu-memory-utilization 0.5  # 50% of GPU = ~16GB
--max-model-len 224
--dtype float16
--enforce-eager
--trust-remote-code
```

### Embedding Service Configuration
```python
# embedding_service.py
class EmbeddingService:
    def __init__(self):
        self.model = FlagModel('BAAI/bge-m3',
                               use_fp16=True,
                               device='cuda')
        self.batch_size = 32
        self.max_length = 512
```

### RAG Orchestrator Configuration
```python
# rag_orchestrator.py
class RAGOrchestrator:
    def __init__(self):
        self.qdrant = QdrantClient("qdrant:6333")
        self.redis = redis.Redis("redis", 6379)
        self.vllm_url = "http://vllm-server:8000"
        self.embedding_url = "http://embedding-service:8002"

    async def process(self, query, keywords):
        # 1. Query expansion
        expanded = await self.expand_query(keywords)
        # 2. Vector search
        results = await self.vector_search(expanded)
        # 3. Reranking
        reranked = await self.rerank(results, query)
        # 4. Generate response
        response = await self.generate(query, reranked)
        return response
```

## 📊 Monitoring & Observability

### Health Checks
```yaml
Services Health Endpoints:
  vLLM:      GET /v1/models → 200 OK
  STT:       GET /api/config → {gpu: true}
  Qdrant:    GET /collections → list
  Redis:     PING → PONG
  Embedding: GET /health → ready
  RAG:       GET /health → operational
```

### Metrics Collection
```yaml
Key Metrics:
  - GPU memory usage & utilization
  - STT processing latency
  - Embedding generation time
  - LLM inference latency
  - Vector search response time
  - Cache hit rates
  - End-to-end pipeline latency
```

### Logging Strategy
```yaml
Logging:
  - Format: Structured JSON
  - Level: ERROR (prod), INFO (dev)
  - Correlation IDs for request tracing
  - Centralized to stdout for Docker
```

## 🚨 Alert Thresholds
```yaml
Alerts:
  - GPU memory > 90%
  - Service response time > 5s
  - Error rate > 1%
  - Cache hit rate < 50%
  - Qdrant query latency > 500ms
```

## 🔐 Security Considerations

```yaml
Security Measures:
  - API key management via environment variables
  - Network isolation (3 separate networks)
  - MongoDB authentication enabled
  - Redis password protection
  - Rate limiting on API Gateway
  - CORS configuration for frontend
  - SSL/TLS via Nginx
```

## 📈 Performance Targets

### System Performance
```yaml
Targets:
  - Audio → Keywords: < 1s
  - Keywords → RAG Response: < 2s
  - Total E2E Latency: < 3s
  - Concurrent Users: 50+
  - Documents in Vector DB: 100K+
  - QPS: 100+ queries/second
```

### Quality Metrics
```yaml
Quality:
  - STT Accuracy: > 95%
  - Keyword Precision: > 85%
  - Retrieval Relevance: > 90%
  - Response Coherence: > 4.0/5.0
  - Source Attribution: 100%
```

## 🎯 Quick Start Commands

```bash
# 1. Build all images
docker-compose build

# 2. Start core services only
docker-compose --profile core up -d

# 3. Check service status
docker-compose ps
docker logs vllm-server --tail 50
docker logs stt-streaming --tail 50

# 4. Add RAG capabilities
docker-compose --profile rag up -d

# 5. Full production deployment
docker-compose --profile full up -d

# 6. Monitor GPU usage
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv

# 7. Access services
# STT Web UI: http://localhost:5000
# vLLM API: http://localhost:8000
# Qdrant UI: http://localhost:6333/dashboard
# API Gateway: http://localhost:8080
# Web UI: http://localhost:3000

# 8. Stop all services
docker-compose down

# 9. Clean up volumes
docker-compose down -v
```

## 🔄 Development Workflow

### Adding Documents to Knowledge Base
```bash
# 1. Place documents in data/documents/
cp *.pdf data/documents/

# 2. Run document processor
docker-compose run document-processor python ingest.py

# 3. Verify in Qdrant
curl http://localhost:6333/collections
```

### Testing RAG Pipeline
```python
# test_rag.py
import requests

# Test STT → Keywords
audio_data = open("test.wav", "rb").read()
response = requests.post("http://localhost:5000/process",
                         files={"audio": audio_data})

# Test RAG response
keywords = response.json()["keywords"]
rag_response = requests.post("http://localhost:8003/query",
                             json={"keywords": keywords})
print(rag_response.json())
```

## 📝 Troubleshooting

### GPU Memory Issues
```bash
# Reduce vLLM memory usage
docker-compose down vllm-server
# Edit Dockerfile.vllm: --gpu-memory-utilization 0.4
docker-compose up -d vllm-server
```

### Slow Vector Search
```bash
# Check Qdrant index
curl http://localhost:6333/collections/documents
# Optimize if needed
curl -X POST http://localhost:6333/collections/documents/index
```

### Cache Issues
```bash
# Clear Redis cache
docker exec redis-cache redis-cli FLUSHALL
```

## 🎯 Final Goal

**"실시간 음성을 통해 즉각적인 지식 검색과 지능형 응답을 제공하는 차세대 대화형 AI 시스템"**

사용자가 말하는 순간, 시스템이:
1. 정확하게 인식하고 (STT)
2. 핵심을 파악하고 (Keyword Extraction)
3. 관련 정보를 찾고 (Vector Search)
4. 맥락을 이해하고 (RAG)
5. 지능적으로 응답하는 (LLM Generation)

완전한 end-to-end 지능형 어시스턴트 구축이 최종 목표입니다.