#!/bin/bash

echo "🚀 Starting STT Streaming Service (GPU with CUDNN Fixed)"

# 기존 STT 컨테이너들 정리
echo "🧹 Cleaning up existing STT containers..."
docker stop $(docker ps -q --filter name=stt-final) 2>/dev/null || true
docker rm $(docker ps -aq --filter name=stt-final) 2>/dev/null || true

# Docker 이미지 빌드
echo "🔧 Building STT Docker image..."
docker build -f Dockerfile.stt-fixed -t stt-streaming-fixed:latest . || {
    echo "❌ Failed to build Docker image"
    exit 1
}

# 컨테이너 실행
echo "▶️ Starting STT container..."
docker run -d \
    --name stt-final \
    --gpus all \
    -p 5000:5000 \
    -v /app/models:/app/models \
    -e VLLM_API_URL=http://vllm-server:8000/v1/completions \
    --network stt-network \
    stt-streaming-fixed:latest

# 컨테이너 상태 확인
echo "🔍 Checking container status..."
sleep 5

if docker ps | grep -q stt-final; then
    echo "✅ STT container is running!"
    echo "🌐 Service available at: http://localhost:5000"
    echo "🔌 WebSocket endpoint: ws://localhost:5000/ws"
    echo ""
    echo "📊 Container logs:"
    docker logs stt-final --tail 20
else
    echo "❌ STT container failed to start"
    echo "📋 Error logs:"
    docker logs stt-final 2>&1
    exit 1
fi