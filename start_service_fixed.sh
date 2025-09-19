#!/bin/bash
# 실시간 STT 서비스 시작 스크립트 (수정버전)

echo "🚀 실시간 STT 서비스 시작"
echo "================================"

# 기존 프로세스 정리
echo "🧹 기존 프로세스 정리 중..."
pkill -f cloudflared 2>/dev/null
docker stop $(docker ps -q --filter ancestor=stt-model-whisper) 2>/dev/null
sleep 2

# 1. Cloudflare 터널 먼저 시작 (IPv4 명시)
echo "🌐 Cloudflare 터널 생성 중..."
cloudflared tunnel --url http://127.0.0.1:5000 > /tmp/flask_tunnel.log 2>&1 &
FLASK_PID=$!
cloudflared tunnel --url http://127.0.0.1:8766 > /tmp/ws_tunnel.log 2>&1 &
WS_PID=$!

# 터널이 생성될 때까지 대기
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

# 2. URL 추출
FLASK_URL=$(grep "https://" /tmp/flask_tunnel.log | grep trycloudflare | tail -1 | sed 's/.*https/https/' | sed 's/[[:space:]].*//g' | tr -d '|')
WS_URL=$(grep "https://" /tmp/ws_tunnel.log | grep trycloudflare | tail -1 | sed 's/.*https/https/' | sed 's/[[:space:]].*//g' | tr -d '|')
WS_URL=${WS_URL/https:/wss:}

if [ -z "$FLASK_URL" ] || [ -z "$WS_URL" ]; then
    echo "❌ 터널 URL 생성 실패!"
    pkill -f cloudflared
    exit 1
fi

echo "✅ 터널 생성 완료!"
echo "  - Flask URL: $FLASK_URL"
echo "  - WebSocket URL: $WS_URL"

# 3. Docker 컨테이너 시작 (한 번만!)
echo "📦 Docker 컨테이너 시작 중..."
docker compose run --rm -d \
    -p 127.0.0.1:5000:5000 \
    -p 127.0.0.1:8766:8766 \
    -e WEBSOCKET_URL="$WS_URL" \
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
echo ""

# WebSocket 서버 확인
for i in {1..10}; do
    if nc -z 127.0.0.1 8766 2>/dev/null; then
        echo "✅ WebSocket 서버 준비 완료!"
        break
    fi
    sleep 1
done

# 4. 서비스 상태 확인
echo ""
echo "🔍 서비스 상태 확인..."
CONTAINER_ID=$(docker ps -q --filter ancestor=stt-model-whisper)
if [ -n "$CONTAINER_ID" ]; then
    echo "✅ Docker 컨테이너 실행 중 (ID: ${CONTAINER_ID:0:12})"
else
    echo "❌ Docker 컨테이너가 실행되지 않음!"
    exit 1
fi

# Flask 서버 테스트
if curl -s http://127.0.0.1:5000/api/config | grep -q "ws_url"; then
    echo "✅ Flask API 응답 정상"
else
    echo "⚠️ Flask API 응답 확인 필요"
fi

echo ""
echo "================================"
echo "✅ 모든 서비스 준비 완료!"
echo "================================"
echo "🌐 웹 인터페이스: $FLASK_URL"
echo "📡 WebSocket: $WS_URL"
echo "================================"
echo ""
echo "📝 로그 확인 명령어:"
echo "  - Docker 로그: docker logs $CONTAINER_ID -f"
echo "  - Flask 터널: tail -f /tmp/flask_tunnel.log"
echo "  - WS 터널: tail -f /tmp/ws_tunnel.log"
echo ""
echo "종료하려면 Ctrl+C를 누르세요..."

# 종료 처리
cleanup() {
    echo ""
    echo "🛑 서비스 종료 중..."

    # Cloudflare 터널 종료
    kill $FLASK_PID 2>/dev/null
    kill $WS_PID 2>/dev/null
    pkill -f cloudflared 2>/dev/null

    # Docker 컨테이너 종료
    docker stop $CONTAINER_ID 2>/dev/null

    echo "✅ 종료 완료"
    exit 0
}

trap cleanup INT TERM

# 서비스 모니터링
while true; do
    # 컨테이너 상태 확인
    if ! docker ps -q --filter id=$CONTAINER_ID > /dev/null 2>&1; then
        echo "⚠️ Docker 컨테이너가 종료됨! 재시작 중..."
        docker compose run --rm -d \
            -p 127.0.0.1:5000:5000 \
            -p 127.0.0.1:8766:8766 \
            -e WEBSOCKET_URL="$WS_URL" \
            whisper python web_server.py
        CONTAINER_ID=$(docker ps -q --filter ancestor=stt-model-whisper)
    fi

    # Cloudflare 터널 상태 확인
    if ! kill -0 $FLASK_PID 2>/dev/null; then
        echo "⚠️ Flask 터널이 종료됨! 재시작 중..."
        cloudflared tunnel --url http://127.0.0.1:5000 > /tmp/flask_tunnel.log 2>&1 &
        FLASK_PID=$!
    fi

    if ! kill -0 $WS_PID 2>/dev/null; then
        echo "⚠️ WebSocket 터널이 종료됨! 재시작 중..."
        cloudflared tunnel --url http://127.0.0.1:8766 > /tmp/ws_tunnel.log 2>&1 &
        WS_PID=$!
    fi

    sleep 5
done