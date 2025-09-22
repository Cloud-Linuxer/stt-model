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

## 🧪 Comprehensive Test Cases

### Unit Tests

#### 1. STT Service Tests
```python
# tests/test_stt_service.py
import pytest
import requests
import base64
import json

class TestSTTService:
    def test_health_check(self):
        """Test STT service health endpoint"""
        response = requests.get("http://localhost:5000/api/config")
        assert response.status_code == 200
        assert response.json()["gpu"] == True
        assert response.json()["model"] == "medium"

    def test_audio_upload(self):
        """Test audio file upload and transcription"""
        with open("tests/samples/korean_audio.wav", "rb") as f:
            response = requests.post(
                "http://localhost:5000/process",
                files={"audio": f}
            )
        assert response.status_code == 200
        assert "text" in response.json()
        assert len(response.json()["text"]) > 0

    def test_websocket_streaming(self):
        """Test real-time WebSocket streaming"""
        import websocket
        ws = websocket.WebSocket()
        ws.connect("ws://localhost:5000/ws")

        # Send audio chunks
        with open("tests/samples/audio_chunk.raw", "rb") as f:
            audio_data = f.read()
            ws.send(json.dumps({
                "type": "audio",
                "data": base64.b64encode(audio_data).decode()
            }))

        # Receive transcription
        result = json.loads(ws.recv())
        assert result["type"] == "transcription"
        assert "text" in result

    def test_vad_detection(self):
        """Test Voice Activity Detection"""
        # Test with silence
        silence_audio = bytes(16000)  # 1 second of silence
        response = requests.post(
            "http://localhost:5000/process",
            files={"audio": ("silence.wav", silence_audio)}
        )
        assert response.json()["text"] == ""

    def test_hallucination_filter(self):
        """Test hallucination filtering"""
        # Known hallucination patterns should be filtered
        test_texts = ["감사합니다", "MBC 뉴스", "구독과 좋아요"]
        # Test implementation would check filtering logic
```

#### 2. vLLM Keyword Extraction Tests
```python
# tests/test_keyword_extraction.py
import pytest
import requests

class TestKeywordExtraction:
    def test_keyword_extraction_korean(self):
        """Test keyword extraction from Korean text"""
        response = requests.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "prompt": "텍스트: '인공지능 기술이 빠르게 발전하고 있습니다'\n키워드 추출:",
                "max_tokens": 200,
                "temperature": 0.1
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert "인공지능" in result["choices"][0]["text"]
        assert "기술" in result["choices"][0]["text"]

    def test_keyword_importance_scoring(self):
        """Test keyword importance scoring"""
        text = "삼성전자가 새로운 AI 칩을 발표했습니다. 이 칩은 높은 성능을 보입니다."
        response = requests.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "prompt": f"텍스트에서 중요도 점수와 함께 키워드 추출:\n{text}",
                "max_tokens": 200
            }
        )
        keywords = response.json()
        # Should extract "삼성전자", "AI 칩" with high importance

    def test_multilingual_keywords(self):
        """Test Korean-English mixed keyword extraction"""
        text = "Cloud computing과 AI 기술이 결합된 서비스"
        # Test extraction of both Korean and English keywords
```

#### 3. Embedding Service Tests
```python
# tests/test_embedding_service.py
import pytest
import requests
import numpy as np

class TestEmbeddingService:
    def test_text_embedding(self):
        """Test single text embedding generation"""
        response = requests.post(
            "http://localhost:8002/embed",
            json={"text": "인공지능 기술의 발전"}
        )
        assert response.status_code == 200
        embedding = response.json()["embedding"]
        assert len(embedding) == 1024  # BGE-M3 dimension
        assert isinstance(embedding[0], float)

    def test_batch_embedding(self):
        """Test batch embedding processing"""
        texts = [
            "첫 번째 문장입니다",
            "두 번째 문장입니다",
            "세 번째 문장입니다"
        ]
        response = requests.post(
            "http://localhost:8002/embed/batch",
            json={"texts": texts}
        )
        assert response.status_code == 200
        embeddings = response.json()["embeddings"]
        assert len(embeddings) == 3
        assert len(embeddings[0]) == 1024

    def test_similarity_calculation(self):
        """Test semantic similarity between embeddings"""
        # Similar texts should have high cosine similarity
        text1 = "인공지능 기술"
        text2 = "AI 기술"
        text3 = "날씨가 좋네요"

        # Get embeddings
        emb1 = requests.post("http://localhost:8002/embed",
                             json={"text": text1}).json()["embedding"]
        emb2 = requests.post("http://localhost:8002/embed",
                             json={"text": text2}).json()["embedding"]
        emb3 = requests.post("http://localhost:8002/embed",
                             json={"text": text3}).json()["embedding"]

        # Calculate similarities
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

        assert sim_12 > sim_13  # Similar texts should have higher similarity
```

### Integration Tests

#### 4. RAG Pipeline Tests
```python
# tests/test_rag_pipeline.py
import pytest
import requests
import time

class TestRAGPipeline:
    def test_document_ingestion(self):
        """Test document ingestion pipeline"""
        # Upload a document
        with open("tests/samples/test_doc.pdf", "rb") as f:
            response = requests.post(
                "http://localhost:8004/ingest",
                files={"document": f},
                data={"metadata": '{"source": "test", "date": "2024-01-01"}'}
            )
        assert response.status_code == 200
        doc_id = response.json()["document_id"]

        # Verify in Qdrant
        qdrant_response = requests.get(
            f"http://localhost:6333/collections/documents/points/{doc_id}"
        )
        assert qdrant_response.status_code == 200

    def test_vector_search(self):
        """Test vector similarity search"""
        query = "인공지능 기술의 최신 동향"
        response = requests.post(
            "http://localhost:8003/search",
            json={"query": query, "top_k": 5}
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) <= 5
        assert all("score" in r for r in results)
        assert all("text" in r for r in results)

    def test_rag_response_generation(self):
        """Test complete RAG response generation"""
        keywords = ["인공지능", "머신러닝", "딥러닝"]
        response = requests.post(
            "http://localhost:8003/generate",
            json={
                "keywords": keywords,
                "query": "인공지능 기술의 차이점을 설명해주세요"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert "response" in result
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_query_expansion(self):
        """Test query expansion for better retrieval"""
        original_query = "AI"
        response = requests.post(
            "http://localhost:8003/expand_query",
            json={"query": original_query}
        )
        expanded = response.json()["expanded_queries"]
        assert "인공지능" in expanded
        assert "artificial intelligence" in expanded
        assert len(expanded) > 1
```

#### 5. End-to-End Tests
```python
# tests/test_e2e.py
import pytest
import requests
import time

class TestEndToEnd:
    def test_voice_to_knowledge_response(self):
        """Test complete flow: voice → STT → keywords → RAG → response"""
        # Step 1: Upload audio
        with open("tests/samples/question_audio.wav", "rb") as f:
            stt_response = requests.post(
                "http://localhost:5000/process",
                files={"audio": f}
            )
        assert stt_response.status_code == 200
        text = stt_response.json()["text"]
        keywords = stt_response.json()["keywords"]

        # Step 2: Get RAG response
        rag_response = requests.post(
            "http://localhost:8003/query",
            json={
                "text": text,
                "keywords": keywords
            }
        )
        assert rag_response.status_code == 200
        assert "response" in rag_response.json()
        assert len(rag_response.json()["response"]) > 0

        # Measure total latency
        start = time.time()
        # ... perform operations ...
        latency = time.time() - start
        assert latency < 3.0  # Should be under 3 seconds

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation with context"""
        session_id = "test_session_123"

        # First turn
        response1 = requests.post(
            "http://localhost:8003/chat",
            json={
                "session_id": session_id,
                "message": "인공지능이 뭐야?",
                "keywords": ["인공지능"]
            }
        )
        assert "인공지능" in response1.json()["response"]

        # Second turn (should maintain context)
        response2 = requests.post(
            "http://localhost:8003/chat",
            json={
                "session_id": session_id,
                "message": "더 자세히 설명해줘",
                "keywords": ["자세히", "설명"]
            }
        )
        # Should elaborate on AI based on previous context
        assert len(response2.json()["response"]) > len(response1.json()["response"])
```

### Performance Tests

#### 6. Load and Stress Tests
```python
# tests/test_performance.py
import pytest
import requests
import concurrent.futures
import time

class TestPerformance:
    def test_concurrent_users(self):
        """Test system with concurrent users"""
        def make_request(user_id):
            response = requests.post(
                "http://localhost:5000/process",
                files={"audio": open("tests/samples/test.wav", "rb")}
            )
            return response.status_code == 200

        # Test with 50 concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(results)
        success_rate = sum(results) / len(results)
        assert success_rate > 0.95  # 95% success rate

    def test_response_time_sla(self):
        """Test response time SLAs"""
        timings = {
            "stt_to_keywords": [],
            "keywords_to_rag": [],
            "e2e": []
        }

        for _ in range(10):
            # Measure STT → Keywords
            start = time.time()
            stt_response = requests.post(
                "http://localhost:5000/process",
                files={"audio": open("tests/samples/test.wav", "rb")}
            )
            timings["stt_to_keywords"].append(time.time() - start)

            # Measure Keywords → RAG
            start = time.time()
            rag_response = requests.post(
                "http://localhost:8003/query",
                json={"keywords": stt_response.json()["keywords"]}
            )
            timings["keywords_to_rag"].append(time.time() - start)

        # Check SLAs
        assert np.mean(timings["stt_to_keywords"]) < 1.0  # < 1 second
        assert np.mean(timings["keywords_to_rag"]) < 2.0  # < 2 seconds

    def test_memory_usage(self):
        """Test GPU memory usage stays within limits"""
        response = requests.get("http://localhost:5000/api/gpu_status")
        gpu_memory = response.json()["memory_used_gb"]
        assert gpu_memory < 21  # Should stay under 21GB (our allocation)
```

### Failure and Recovery Tests

#### 7. Error Handling Tests
```python
# tests/test_error_handling.py
import pytest
import requests

class TestErrorHandling:
    def test_invalid_audio_format(self):
        """Test handling of invalid audio formats"""
        response = requests.post(
            "http://localhost:5000/process",
            files={"audio": ("test.txt", b"not audio data")}
        )
        assert response.status_code == 400
        assert "error" in response.json()

    def test_service_unavailable_recovery(self):
        """Test graceful degradation when services are down"""
        # Simulate vLLM being down
        # System should still do STT even without keyword extraction
        pass

    def test_qdrant_connection_failure(self):
        """Test fallback when vector DB is unavailable"""
        # Should fallback to basic keyword matching
        pass

    def test_rate_limiting(self):
        """Test rate limiting protection"""
        # Make 100 rapid requests
        responses = []
        for _ in range(100):
            r = requests.post("http://localhost:8080/api/query", json={})
            responses.append(r.status_code)

        # Should get rate limited (429) after threshold
        assert 429 in responses
```

### Data Quality Tests

#### 8. Accuracy and Quality Tests
```python
# tests/test_quality.py
import pytest
import requests

class TestQuality:
    def test_stt_accuracy(self):
        """Test STT accuracy with ground truth"""
        test_cases = [
            ("tests/samples/clear_korean.wav", "안녕하세요 오늘 날씨가 좋네요"),
            ("tests/samples/technical_terms.wav", "인공지능과 머신러닝 기술"),
        ]

        for audio_file, expected_text in test_cases:
            with open(audio_file, "rb") as f:
                response = requests.post(
                    "http://localhost:5000/process",
                    files={"audio": f}
                )
            actual_text = response.json()["text"]

            # Calculate accuracy (e.g., using edit distance)
            accuracy = calculate_accuracy(expected_text, actual_text)
            assert accuracy > 0.95  # 95% accuracy threshold

    def test_retrieval_relevance(self):
        """Test retrieval relevance scoring"""
        # Test with known documents and queries
        test_queries = [
            ("인공지능의 정의", ["ai_definition.pdf", "ml_basics.pdf"]),
            ("딥러닝 알고리즘", ["deep_learning.pdf", "neural_networks.pdf"])
        ]

        for query, expected_docs in test_queries:
            response = requests.post(
                "http://localhost:8003/search",
                json={"query": query, "top_k": 5}
            )
            retrieved = [r["source"] for r in response.json()["results"]]

            # Check if expected documents are in top results
            relevance = len(set(expected_docs) & set(retrieved[:2])) / len(expected_docs)
            assert relevance > 0.8  # 80% relevance
```

## 🧪 Testing Infrastructure

### Test Data Preparation
```bash
# scripts/prepare_test_data.sh
#!/bin/bash

# Create test directories
mkdir -p tests/samples
mkdir -p tests/expected_outputs

# Download sample audio files
wget -O tests/samples/korean_audio.wav https://example.com/sample1.wav
wget -O tests/samples/clear_korean.wav https://example.com/sample2.wav

# Generate test documents
python scripts/generate_test_docs.py

# Populate vector database with test data
docker-compose run document-processor python ingest_test_data.py
```

### Continuous Testing
```yaml
# .github/workflows/test.yml
name: RAG System Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Start services
      run: docker-compose --profile core up -d

    - name: Run unit tests
      run: pytest tests/unit -v

    - name: Run integration tests
      run: pytest tests/integration -v

    - name: Run performance tests
      run: pytest tests/performance -v

    - name: Generate test report
      run: pytest --html=report.html --self-contained-html
```

### Test Coverage Requirements
```yaml
coverage_targets:
  unit_tests: 80%
  integration_tests: 70%
  e2e_tests: 60%
  overall: 75%
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