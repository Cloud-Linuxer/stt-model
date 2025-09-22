#!/bin/bash
# Start Core Services (STT + vLLM + Redis)

echo "============================================"
echo "🚀 Starting Core Services"
echo "============================================"

# Create models directory if not exists
mkdir -p ./models

# Stop existing containers
echo "🔄 Stopping existing containers..."
docker compose --profile core down

# Build images
echo "🔨 Building Docker images..."
docker compose --profile core build

# Start core services
echo "🚀 Starting core services..."
docker compose --profile core up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service status
echo "📊 Checking service status..."
docker compose --profile core ps

# Show service logs
echo ""
echo "📋 Service endpoints:"
echo "  - STT Service: http://localhost:5000"
echo "  - vLLM API: http://localhost:8000/v1"
echo "  - Redis: redis://localhost:6379"
echo ""

# Monitor health
echo "🔍 Health checks:"
curl -s http://localhost:5000/api/config | jq . || echo "STT service not ready yet"
curl -s http://localhost:8000/v1/models | jq . || echo "vLLM service not ready yet"

echo ""
echo "✅ Core services started!"
echo "Use 'docker compose --profile core logs -f' to view logs"