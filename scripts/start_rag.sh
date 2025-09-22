#!/bin/bash
# Start RAG Services (Core + Qdrant + Embedding + RAG Orchestrator)

echo "============================================"
echo "🚀 Starting RAG Services"
echo "============================================"

# Stop existing containers
echo "🔄 Stopping existing containers..."
docker compose --profile rag down

# Build images
echo "🔨 Building Docker images..."
docker compose --profile rag build

# Start RAG services
echo "🚀 Starting RAG services..."
docker compose --profile rag up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 15

# Check service status
echo "📊 Checking service status..."
docker compose --profile rag ps

# Show service endpoints
echo ""
echo "📋 Service endpoints:"
echo "  - STT Service: http://localhost:5000"
echo "  - vLLM API: http://localhost:8000/v1"
echo "  - Qdrant UI: http://localhost:6333/dashboard"
echo "  - Embedding Service: http://localhost:8002"
echo "  - RAG Orchestrator: http://localhost:8003"
echo "  - Redis: redis://localhost:6379"
echo ""

# Monitor health
echo "🔍 Health checks:"
curl -s http://localhost:8002/health | jq . || echo "Embedding service not ready"
curl -s http://localhost:8003/health | jq . || echo "RAG service not ready"
curl -s http://localhost:6333/collections | jq . || echo "Qdrant not ready"

echo ""
echo "✅ RAG services started!"
echo "Use 'docker compose --profile rag logs -f' to view logs"