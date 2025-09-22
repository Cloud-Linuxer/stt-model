#!/usr/bin/env python3
"""
RAG Orchestrator Service
Core RAG logic for retrieval-augmented generation
"""

import os
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import httpx
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Orchestrator", version="1.0.0")

# Environment variables
VLLM_URL = os.getenv("VLLM_URL", "http://vllm-server:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embedding-service:8002")

# Initialize clients
qdrant_client = None
redis_client = None
http_client = None

# Constants
COLLECTION_NAME = "documents"
EMBEDDING_DIM = 1024  # BGE-M3 dimension
TOP_K_RETRIEVAL = 10
RERANK_TOP_K = 5
CACHE_TTL = 3600  # 1 hour

# Pydantic models
class QueryRequest(BaseModel):
    """RAG query request"""
    query: str = Field(..., description="User query")
    keywords: Optional[List[str]] = Field(None, description="Keywords for enhanced retrieval")
    top_k: Optional[int] = Field(TOP_K_RETRIEVAL, description="Number of documents to retrieve")
    session_id: Optional[str] = Field(None, description="Session ID for context tracking")
    language: Optional[str] = Field("ko", description="Language code")

class RAGResponse(BaseModel):
    """RAG response"""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    keywords: List[str] = Field(..., description="Extracted keywords")
    session_id: Optional[str] = Field(None, description="Session ID")
    processing_time: float = Field(..., description="Processing time in seconds")

class DocumentChunk(BaseModel):
    """Document chunk for indexing"""
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    doc_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Chunk ID")

# Helper functions
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_embeddings(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Get embeddings from embedding service"""
    global http_client

    response = await http_client.post(
        f"{EMBEDDING_URL}/embed/batch",
        json={"texts": texts, "is_query": is_query}
    )
    response.raise_for_status()
    data = response.json()
    return data['embeddings']

async def search_vectors(query_vector: List[float], top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
    """Search similar vectors in Qdrant"""
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )

    documents = []
    for hit in results:
        doc = {
            "content": hit.payload.get("content", ""),
            "metadata": hit.payload.get("metadata", {}),
            "score": hit.score,
            "doc_id": hit.payload.get("doc_id", ""),
            "chunk_id": hit.payload.get("chunk_id", "")
        }
        documents.append(doc)

    return documents

async def rerank_documents(query: str, documents: List[Dict]) -> List[Dict]:
    """Rerank documents using cross-encoder approach"""
    # Simple relevance scoring based on keyword overlap
    # In production, use a cross-encoder model
    query_tokens = set(query.lower().split())

    for doc in documents:
        content_tokens = set(doc["content"].lower().split())
        overlap = len(query_tokens.intersection(content_tokens))
        doc["rerank_score"] = overlap / max(len(query_tokens), 1)

    # Sort by rerank score combined with vector score
    documents.sort(key=lambda x: (x["rerank_score"] * 0.5 + x["score"] * 0.5), reverse=True)

    return documents[:RERANK_TOP_K]

async def query_expansion(query: str, keywords: Optional[List[str]] = None) -> List[str]:
    """Expand query with synonyms and related terms"""
    expanded_queries = [query]

    if keywords:
        # Add keyword-based variations
        for keyword in keywords[:3]:  # Limit to top 3 keywords
            expanded_queries.append(f"{query} {keyword}")

    # In production, add synonym expansion using WordNet or similar

    return expanded_queries

async def generate_response(query: str, context: str) -> str:
    """Generate response using vLLM"""
    global http_client

    prompt = f"""다음 문맥을 바탕으로 질문에 답하세요.

문맥:
{context}

질문: {query}

답변:"""

    response = await http_client.post(
        f"{VLLM_URL}/v1/completions",
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.1,
            "top_p": 0.95,
            "stop": ["질문:", "\n\n"]
        }
    )
    response.raise_for_status()
    data = response.json()
    return data['choices'][0]['text'].strip()

async def cache_get(key: str) -> Optional[Dict]:
    """Get cached result"""
    try:
        result = redis_client.get(key)
        if result:
            return json.loads(result)
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
    return None

async def cache_set(key: str, value: Dict, ttl: int = CACHE_TTL):
    """Set cache with TTL"""
    try:
        redis_client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        logger.warning(f"Cache set error: {e}")

# API endpoints
@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    global qdrant_client, redis_client, http_client

    logger.info("🚀 Starting RAG Orchestrator")

    # Initialize HTTP client for all external API calls
    http_client = httpx.AsyncClient(timeout=30.0)

    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=QDRANT_URL)

    # Create collection if not exists
    try:
        collections = qdrant_client.get_collections()
        if COLLECTION_NAME not in [c.name for c in collections.collections]:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            logger.info(f"✅ Created collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"❌ Qdrant initialization error: {e}")

    # Initialize Redis client
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)

    logger.info("✅ RAG Orchestrator ready")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global http_client

    if http_client:
        await http_client.aclose()
        logger.info("✅ HTTP client closed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "rag-orchestrator",
        "timestamp": time.time()
    }

@app.post("/query", response_model=RAGResponse)
async def process_query(request: QueryRequest):
    """Process RAG query"""
    start_time = time.time()

    try:
        # Check cache
        cache_key = f"rag:{request.query}:{request.top_k}"
        cached_result = await cache_get(cache_key)
        if cached_result:
            cached_result['processing_time'] = time.time() - start_time
            return RAGResponse(**cached_result)

        # Query expansion
        expanded_queries = await query_expansion(request.query, request.keywords)

        # Get query embeddings
        query_embeddings = await get_embeddings(expanded_queries, is_query=True)

        # Search for each expanded query and combine results
        all_documents = []
        seen_chunks = set()

        for query_vector in query_embeddings:
            documents = await search_vectors(query_vector, request.top_k)

            for doc in documents:
                chunk_id = doc['chunk_id']
                if chunk_id not in seen_chunks:
                    all_documents.append(doc)
                    seen_chunks.add(chunk_id)

        # Rerank documents
        reranked_documents = await rerank_documents(request.query, all_documents)

        # Prepare context
        context = "\n\n".join([doc['content'] for doc in reranked_documents])

        # Generate response
        answer = await generate_response(request.query, context)

        # Prepare response
        result = {
            "answer": answer,
            "sources": reranked_documents,
            "keywords": request.keywords or [],
            "session_id": request.session_id,
            "processing_time": time.time() - start_time
        }

        # Cache result
        await cache_set(cache_key, result)

        return RAGResponse(**result)

    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_document(chunk: DocumentChunk):
    """Index a document chunk"""
    try:
        # Get embedding for chunk
        embeddings = await get_embeddings([chunk.content], is_query=False)
        embedding = embeddings[0]

        # Create point for Qdrant
        point = PointStruct(
            id=hash(chunk.chunk_id) & 0x7FFFFFFF,  # Convert to positive int
            vector=embedding,
            payload={
                "content": chunk.content,
                "metadata": chunk.metadata,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "indexed_at": time.time()
            }
        )

        # Insert into Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

        return {"status": "success", "chunk_id": chunk.chunk_id}

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete all chunks of a document"""
    try:
        # Delete by filter
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=doc_id
                    )
                ]
            )
        )

        return {"status": "success", "doc_id": doc_id}

    except Exception as e:
        logger.error(f"Deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get collection statistics"""
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)

        return {
            "collection": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
            "config": {
                "size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance
            }
        }

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)