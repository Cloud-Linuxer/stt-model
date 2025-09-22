#!/bin/bash
# Start Full Stack (All services including UI, Gateway, MongoDB)

echo "============================================"
echo "🚀 Starting Full Stack Services"
echo "============================================"

# Stop existing containers
echo "🔄 Stopping existing containers..."
docker compose --profile full down

# Build images
echo "🔨 Building Docker images..."
docker compose --profile full build

# Start all services
echo "🚀 Starting all services..."
docker compose --profile full up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy (this may take a while)..."
sleep 30

# Check service status
echo "📊 Checking service status..."
docker compose --profile full ps

# Show service endpoints
echo ""
echo "📋 Service endpoints:"
echo "  - Web UI: http://localhost:3000"
echo "  - API Gateway: http://localhost:8080"
echo "  - STT Service: http://localhost:5000"
echo "  - vLLM API: http://localhost:8000/v1"
echo "  - Qdrant UI: http://localhost:6333/dashboard"
echo "  - Embedding Service: http://localhost:8002"
echo "  - RAG Orchestrator: http://localhost:8003"
echo "  - Document Processor: http://localhost:8004"
echo "  - MongoDB: mongodb://localhost:27017"
echo "  - Redis: redis://localhost:6379"
echo ""

# Monitor health
echo "🔍 Health checks:"
curl -s http://localhost:8080/health | jq . || echo "API Gateway not ready"
curl -s http://localhost:8004/health | jq . || echo "Document Processor not ready"
curl -s http://localhost:3000 > /dev/null && echo "Web UI ready" || echo "Web UI not ready"

echo ""
echo "✅ Full stack started!"
echo "🌐 Access the Web UI at http://localhost:3000"
echo ""
echo "Use 'docker compose --profile full logs -f' to view logs"