#!/bin/bash
# 실시간 STT 서비스 시작 스크립트

echo "🚀 실시간 STT 서비스 시작"
echo "================================"

# 1. Docker 컨테이너 시작
echo "📦 Docker 컨테이너 시작 중..."
docker compose run --rm -d -p 5000:5000 -p 8766:8766 whisper python web_server.py
sleep 10

# 2. Cloudflare 터널 시작
echo "🌐 Cloudflare 터널 생성 중..."
cloudflared tunnel --url http://localhost:5000 > /tmp/flask_tunnel.log 2>&1 &
cloudflared tunnel --url http://localhost:8766 > /tmp/ws_tunnel.log 2>&1 &
sleep 10

# 3. URL 추출
FLASK_URL=$(grep "https://" /tmp/flask_tunnel.log | grep trycloudflare | tail -1 | sed 's/.*https/https/' | sed 's/[[:space:]].*//g' | tr -d '|')
WS_URL=$(grep "https://" /tmp/ws_tunnel.log | grep trycloudflare | tail -1 | sed 's/.*https/https/' | sed 's/[[:space:]].*//g' | tr -d '|')
WS_URL=${WS_URL/https:/wss:}

echo ""
echo "✅ 서비스 준비 완료!"
echo "================================"
echo "📱 웹 인터페이스: $FLASK_URL"
echo "📡 WebSocket: $WS_URL"
echo "================================"

# 4. WebSocket URL을 환경변수로 설정하고 Docker 컨테이너 재시작
echo "⚙️ WebSocket URL 설정 중..."
docker stop $(docker ps -q --filter ancestor=stt-model-whisper) 2>/dev/null
docker compose run --rm -d -p 5000:5000 -p 8766:8766 -e WEBSOCKET_URL=$WS_URL whisper python web_server.py

echo ""
echo "✅ 모든 설정 완료!"
echo "================================"
echo "🌐 접속 URL: $FLASK_URL"
echo "================================"
echo ""
echo "종료하려면 Ctrl+C를 누르세요..."

# 종료 처리
trap "echo '종료 중...'; pkill -f cloudflared; docker stop \$(docker ps -q) 2>/dev/null; exit" INT TERM

# 대기
while true; do
    sleep 1
done