#!/bin/bash

echo "🚀 STT + LLM 통합 시스템 시작"
echo "========================================="

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리..."
docker stop vllm-server stt-streaming 2>/dev/null
docker rm vllm-server stt-streaming 2>/dev/null

# vLLM 빌드 및 시작
echo "📦 vLLM 이미지 빌드 중..."
docker build -t vllm-server -f Dockerfile.vllm .

echo "🚀 vLLM 서버 시작 중 (메모리 40% 사용)..."
docker run -d \
    --name vllm-server \
    --gpus all \
    -p 8000:8000 \
    -v /app/models:/app/models \
    --restart unless-stopped \
    vllm-server

# vLLM 준비 대기
echo "⏳ vLLM 모델 로딩 대기중 (최대 3분)..."
for i in {1..180}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✅ vLLM 서버 준비 완료!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "❌ vLLM 서버 시작 실패"
        docker logs vllm-server --tail 50
        exit 1
    fi
    sleep 1
done

# STT 서버 시작
echo "🎤 STT 스트리밍 서버 시작 중..."
docker run -d \
    --name stt-streaming \
    --gpus all \
    -p 5000:5000 \
    -v /home/stt-model:/app \
    -v /app/models:/app/models \
    --network host \
    --restart unless-stopped \
    stt-gpu-force \
    python /app/web_server_streaming.py

# STT 준비 대기
echo "⏳ STT 서버 대기중..."
sleep 10

# 상태 확인
echo ""
echo "📊 시스템 상태:"
echo "============================================"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🎯 GPU 메모리 사용량:"
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv

echo ""
echo "✅ 시스템 준비 완료!"
echo "============================================"
echo "🌐 STT 웹 인터페이스: http://localhost:5000"
echo "🤖 LLM API: http://localhost:8000"
echo ""
echo "📋 로그 확인 명령어:"
echo "  docker logs vllm-server -f"
echo "  docker logs stt-streaming -f"
echo ""
echo "🌐 Cloudflare 터널 시작..."
cloudflared tunnel --url http://localhost:5000