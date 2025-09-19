# Faster-Whisper Large v3 Docker Setup

Faster-Whisper (CTranslate2 기반) Large v3 모델을 Docker 환경에서 테스트하기 위한 설정입니다.

## 🚀 Faster-Whisper 장점
- **4x 빠른 속도**: OpenAI Whisper보다 4배 빠른 처리
- **50% 메모리 절약**: 효율적인 메모리 사용
- **동일한 정확도**: OpenAI Whisper와 같은 품질
- **추가 기능**: VAD(Voice Activity Detection), 스트리밍, Word timestamps

## 🚀 Quick Start

### 1. Docker 이미지 빌드
```bash
docker-compose build
```

### 2. 컨테이너 실행
```bash
docker-compose up
```

### 3. 인터랙티브 모드로 실행
```bash
docker-compose run --rm whisper bash
```

## 📁 디렉토리 구조
```
.
├── data/         # 테스트할 오디오 파일 (.wav, .mp3)
├── outputs/      # 변환 결과 저장
├── models/       # Whisper 모델 캐시
├── test_whisper.py   # 테스트 스크립트
├── Dockerfile
└── docker-compose.yml
```

## 🎤 오디오 파일 테스트

1. `data/` 디렉토리에 오디오 파일 추가 (.wav 또는 .mp3)
2. 컨테이너 실행:
```bash
docker-compose run --rm whisper python test_whisper.py
```

## 🖥️ GPU 지원

GPU를 사용하려면 `docker-compose.yml`에서 GPU 섹션 주석 해제:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 📋 모델 정보
- **모델**: Whisper Large v3
- **파라미터**: ~1550M
- **언어**: 99개 이상 지원
- **크기**: 약 3GB

## 🔧 커스텀 스크립트 실행
```bash
docker-compose run --rm whisper python your_script.py
```