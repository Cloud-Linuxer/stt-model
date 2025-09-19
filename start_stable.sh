#!/bin/bash
# 안정적인 STT 서비스 시작 스크립트 (CPU 모드)

echo "🚀 실시간 STT 서비스 시작 (안정화 버전)"
echo "================================"

# 기존 프로세스 정리
echo "🧹 기존 프로세스 정리 중..."
pkill -f cloudflared 2>/dev/null
docker stop $(docker ps -q) 2>/dev/null
rm -f /tmp/*.log
sleep 2

# 1. Docker 컨테이너 먼저 시작 (CPU 모드)
echo "📦 Docker 컨테이너 시작 중 (CPU 모드)..."
docker compose run --rm -d \
    -p 127.0.0.1:5000:5000 \
    -p 127.0.0.1:8766:8766 \
    -e CUDA_VISIBLE_DEVICES="" \
    whisper python web_server.py

# 컨테이너가 시작될 때까지 대기
echo "⏳ 서비스 시작 대기 중..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:5000/api/config > /dev/null 2>&1; then
        echo ""
        echo "✅ Flask 서버 준비 완료!"
        break
    fi
    sleep 1
    echo -n "."
done

# WebSocket 확인
for i in {1..10}; do
    if nc -z 127.0.0.1 8766 2>/dev/null; then
        echo "✅ WebSocket 서버 준비 완료!"
        break
    fi
    sleep 1
done

# 2. Cloudflare 터널 시작 (서비스가 준비된 후)
echo "🌐 Cloudflare 터널 생성 중..."
cloudflared tunnel --url http://127.0.0.1:5000 > /tmp/flask_tunnel.log 2>&1 &
FLASK_PID=$!
cloudflared tunnel --url http://127.0.0.1:8766 > /tmp/ws_tunnel.log 2>&1 &
WS_PID=$!

# 터널 URL 대기
echo "⏳ 터널 URL 생성 대기 중..."
for i in {1..30}; do
    if grep -q "https://.*trycloudflare.com" /tmp/flask_tunnel.log 2>/dev/null && \
       grep -q "https://.*trycloudflare.com" /tmp/ws_tunnel.log 2>/dev/null; then
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# URL 추출
FLASK_URL=$(grep "https://" /tmp/flask_tunnel.log | grep trycloudflare | tail -1 | sed 's/.*https/https/' | sed 's/[[:space:]].*//g' | tr -d '|')
WS_URL=$(grep "https://" /tmp/ws_tunnel.log | grep trycloudflare | tail -1 | sed 's/.*https/https/' | sed 's/[[:space:]].*//g' | tr -d '|')
WS_URL=${WS_URL/https:/wss:}

# 3. WebSocket URL 업데이트
CONTAINER_ID=$(docker ps -q --filter ancestor=stt-model-whisper)
docker exec $CONTAINER_ID sh -c "echo 'export WEBSOCKET_URL=$WS_URL' >> /tmp/ws_url.sh"

echo ""
echo "================================"
echo "✅ 모든 서비스 준비 완료!"
echo "================================"
echo "🌐 웹 인터페이스: $FLASK_URL"
echo "📡 WebSocket: $WS_URL"
echo "🖥️ CPU 모드로 실행 중"
echo "================================"
echo ""
echo "📝 테스트:"
echo "  1. 웹 브라우저에서 $FLASK_URL 접속"
echo "  2. '녹음 시작' 버튼 클릭"
echo "  3. 말하기 시작"
echo ""
echo "종료하려면 Ctrl+C를 누르세요..."

# 종료 처리
cleanup() {
    echo ""
    echo "🛑 서비스 종료 중..."
    kill $FLASK_PID 2>/dev/null
    kill $WS_PID 2>/dev/null
    docker stop $CONTAINER_ID 2>/dev/null
    echo "✅ 종료 완료"
    exit 0
}

trap cleanup INT TERM

# 모니터링
while true; do
    # 컨테이너 상태 확인
    if ! docker ps -q --filter id=$CONTAINER_ID > /dev/null 2>&1; then
        echo "⚠️ Docker 컨테이너가 종료됨!"
        echo "로그 확인: docker logs $CONTAINER_ID"
        cleanup
    fi
    sleep 5
done