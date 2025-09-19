#!/bin/bash
# 통합 STT 서비스 시작 스크립트

echo "🚀 통합 실시간 STT 서비스 시작"
echo "================================"

# 기존 프로세스 정리
echo "🧹 기존 프로세스 정리 중..."
pkill -f cloudflared 2>/dev/null
docker stop $(docker ps -q) 2>/dev/null
rm -f /tmp/*.log
sleep 2

# 1. Docker 컨테이너 시작 (통합 버전)
echo "📦 통합 Docker 컨테이너 시작 중..."
docker run --rm -d \
    --name stt-integrated \
    -p 127.0.0.1:5000:5000 \
    -v /home/stt-model/web_server_integrated.py:/app/web_server_integrated.py:ro \
    -v /home/stt-model/templates:/app/templates:ro \
    -e CUDA_VISIBLE_DEVICES="" \
    stt-integrated \
    python /app/web_server_integrated.py

# 서비스 준비 대기
echo "⏳ 서비스 시작 대기 중..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:5000/api/config > /dev/null 2>&1; then
        echo ""
        echo "✅ 통합 서버 준비 완료!"
        break
    fi
    sleep 1
    echo -n "."
done

# 2. Cloudflare 터널 시작 (하나만!)
echo "🌐 Cloudflare 터널 생성 중..."
cloudflared tunnel --url http://127.0.0.1:5000 > /tmp/tunnel.log 2>&1 &
TUNNEL_PID=$!

# 터널 URL 대기
echo "⏳ 터널 URL 생성 대기 중..."
for i in {1..30}; do
    if grep -q "https://.*trycloudflare.com" /tmp/tunnel.log 2>/dev/null; then
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# URL 추출
TUNNEL_URL=$(grep "https://" /tmp/tunnel.log | grep trycloudflare | tail -1 | sed 's/.*https/https/' | sed 's/[[:space:]].*//g' | tr -d '|')

# 설정 확인
echo ""
echo "🔍 설정 확인:"
curl -s http://127.0.0.1:5000/api/config | jq .

echo ""
echo "================================"
echo "✅ 통합 서비스 준비 완료!"
echo "================================"
echo "🌐 웹 인터페이스: $TUNNEL_URL"
echo "📡 WebSocket: 동일 도메인 /ws 경로"
echo "🖥️ CPU 모드로 실행 중"
echo "================================"
echo ""
echo "📝 테스트:"
echo "  1. 웹 브라우저에서 $TUNNEL_URL 접속"
echo "  2. '녹음 시작' 버튼 클릭"
echo "  3. 말하기 시작"
echo "  4. WebSocket은 자동으로 /ws 경로에 연결됩니다"
echo ""
echo "종료하려면 Ctrl+C를 누르세요..."

# 종료 처리
cleanup() {
    echo ""
    echo "🛑 서비스 종료 중..."
    kill $TUNNEL_PID 2>/dev/null
    docker stop stt-integrated 2>/dev/null
    echo "✅ 종료 완료"
    exit 0
}

trap cleanup INT TERM

# 모니터링
while true; do
    # 컨테이너 상태 확인
    if ! docker ps | grep -q stt-integrated; then
        echo "⚠️ Docker 컨테이너가 종료됨!"
        echo "로그 확인: docker logs stt-integrated"
        cleanup
    fi
    sleep 5
done