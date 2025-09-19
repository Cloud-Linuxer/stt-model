#!/bin/bash
# 실시간 STT 웹 서비스를 Cloudflare Tunnel과 함께 실행

echo "🚀 실시간 STT 서비스 시작 (Cloudflare Tunnel)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Docker 컨테이너 실행
echo "📦 Docker 컨테이너 시작..."
docker compose run --rm -d \
    -p 5000:5000 \
    -p 8766:8766 \
    --name stt-web \
    whisper python web_server.py &

# 서버 시작 대기
echo "⏳ 서버 시작 대기 중..."
sleep 5

# Cloudflare Tunnel 시작
echo ""
echo "🌐 Cloudflare Tunnel 시작..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Flask 터널
echo "📡 Flask 서버 터널 생성 중..."
cloudflared tunnel --url http://localhost:5000 2>&1 | while IFS= read -r line; do
    echo "$line"
    if [[ $line == *"https://"*.trycloudflare.com* ]]; then
        echo ""
        echo "✅ Flask 웹 인터페이스 URL:"
        echo "$line" | grep -o 'https://[^ ]*'
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
done &
FLASK_TUNNEL_PID=$!

# WebSocket 터널
echo ""
echo "📡 WebSocket 서버 터널 생성 중..."
cloudflared tunnel --url http://localhost:8766 2>&1 | while IFS= read -r line; do
    echo "$line"
    if [[ $line == *"https://"*.trycloudflare.com* ]]; then
        echo ""
        echo "✅ WebSocket 서버 URL (WSS로 변경해서 사용):"
        URL=$(echo "$line" | grep -o 'https://[^ ]*')
        WSS_URL=${URL/https:/wss:}
        echo "   $WSS_URL"
        echo ""
        echo "💡 사용 방법:"
        echo "   1. 위의 Flask URL로 웹 브라우저 접속"
        echo "   2. WebSocket URL 입력란에 위의 WSS URL 입력"
        echo "   3. '연결' 버튼 클릭"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
done &
WS_TUNNEL_PID=$!

echo ""
echo "📌 서비스가 실행 중입니다..."
echo "   - 로컬 Flask: http://localhost:5000"
echo "   - 로컬 WebSocket: ws://localhost:8766"
echo ""
echo "종료하려면 Ctrl+C를 누르세요..."

# 종료 처리
trap cleanup INT TERM

cleanup() {
    echo ""
    echo "🛑 서비스 종료 중..."

    # Cloudflare Tunnel 종료
    kill $FLASK_TUNNEL_PID $WS_TUNNEL_PID 2>/dev/null

    # Docker 컨테이너 종료
    docker stop stt-web 2>/dev/null
    docker rm stt-web 2>/dev/null

    echo "✅ 종료 완료"
    exit 0
}

# 프로세스 대기
wait