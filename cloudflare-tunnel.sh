#!/bin/bash
# Cloudflare Tunnel 실행 스크립트

echo "🌐 Cloudflare Tunnel 시작..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Flask 웹 서버 (포트 5000)
echo "📡 Flask 서버 터널 생성 (포트 5000)..."
cloudflared tunnel --url http://localhost:5000 &
FLASK_PID=$!

# WebSocket 서버 (포트 8766)
echo "📡 WebSocket 서버 터널 생성 (포트 8766)..."
cloudflared tunnel --url http://localhost:8766 --protocol http2 &
WS_PID=$!

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Cloudflare Tunnel 실행 중..."
echo ""
echo "📌 터널 URL은 위 로그에서 확인하세요:"
echo "   - https://[random-subdomain].trycloudflare.com (Flask)"
echo "   - https://[random-subdomain].trycloudflare.com (WebSocket)"
echo ""
echo "⚠️  WebSocket 연결 시 주의사항:"
echo "   - wss:// 프로토콜 사용 (https 터널이므로)"
echo "   - CORS 설정 확인 필요"
echo ""
echo "종료하려면 Ctrl+C를 누르세요..."

# 종료 처리
trap "echo '종료 중...'; kill $FLASK_PID $WS_PID 2>/dev/null; exit" INT TERM

# 프로세스 대기
wait