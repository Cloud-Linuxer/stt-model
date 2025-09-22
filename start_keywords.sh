#!/bin/bash

echo "🚀 STT + LLM 키워드 추출 시스템 시작"
echo "=================================="

# 1. vLLM 서버 시작
echo "📦 vLLM 서버 시작 중..."
docker run -d \
    --name vllm-server \
    --gpus all \
    -p 8000:8000 \
    -v /app/models:/app/models \
    --restart unless-stopped \
    vllm-server

# vLLM이 준비될 때까지 대기
echo "⏳ vLLM 모델 로딩 대기중 (최대 2분)..."
for i in {1..120}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✅ vLLM 서버 준비 완료!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "❌ vLLM 서버 시작 실패"
        exit 1
    fi
    sleep 1
done

# 2. STT + 키워드 추출 서버 시작
echo "🎤 STT + 키워드 추출 서버 시작 중..."
docker run -d \
    --name stt-keywords \
    --gpus all \
    -p 5000:5000 \
    -v /home/stt-model:/app \
    -v /app/models:/app/models \
    --network host \
    --restart unless-stopped \
    stt-gpu-force \
    python /app/web_server_keywords.py

# 상태 확인
echo ""
echo "📊 시스템 상태:"
echo "==============="
docker ps | grep -E "vllm-server|stt-keywords"

echo ""
echo "✅ 시스템 준비 완료!"
echo "🌐 웹 인터페이스: http://localhost:5000"
echo "🤖 LLM API: http://localhost:8000"
echo ""
echo "📋 로그 확인:"
echo "  - vLLM: docker logs vllm-server -f"
echo "  - STT: docker logs stt-keywords -f"