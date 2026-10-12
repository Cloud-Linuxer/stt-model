#!/usr/bin/env bash
# Start the rtstt stack.  Usage: ./start.sh [cpu|cpu-hq|gpu-min|gpu-max|gpu-multi] [port]
#   cpu      CPU only: Nemotron-3.5 streaming ASR + CAM++ diarization + Hy-MT2-1.8B translation
#   cpu-hq   CPU only, + Qwen3-ASR-1.7B second pass for final tokens (more accurate, finals ~1-2 s later)
#   gpu-min  second-pass ASR + translation on the GPU (~6 GB VRAM); streaming ASR, VAD and
#            diarization stay on the CPU
#   gpu-max  lowest latency / best accuracy: Whisper large-v3-turbo final pass on the GPU with batching
#            and progressive finalization, Hy-MT2 on the GPU (~5 GB VRAM); Nemotron partials on the CPU
#   gpu-multi all 60 supported languages: per-language routing over Nemotron / Whisper turbo + large-v3 /
#            Omnilingual-ASR, translation Hy-MT2 (ko/en/zh/de) + MADLAD-400 (any pair) (~11 GB VRAM)
set -euo pipefail
PROFILE=${1:-cpu}
PORT=${2:-8765}
HERE=$(cd "$(dirname "$0")" && pwd)
MODELS=$HERE/models/mt
IMG=ghcr.io/ggml-org/llama.cpp
LOG=/tmp/rtstt-$PROFILE.log

wait_http() { for _ in $(seq 120); do curl -sf "$1" >/dev/null && return 0; sleep 1; done; echo "timeout: $1" >&2; return 1; }

run_llama() {  # name port image model [llama-server args...]; GPU images get --gpus all
  local name=$1 port=$2 image=$3 model=$4 gpu=(); shift 4
  [[ $image == *cuda* ]] && gpu=(--gpus all)
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --restart unless-stopped "${gpu[@]}" -p "$port:8080" -v "$MODELS:/models:ro" "$image" \
    -m "/models/$model" --host 0.0.0.0 --port 8080 "$@" >/dev/null
  wait_http "http://localhost:$port/health"
}

# translation: Hy-MT2-1.8B (Q4_K_M), CPU or (gpu-min) GPU
if [ "$PROFILE" = gpu-min ] || [ "$PROFILE" = gpu-max ] || [ "$PROFILE" = gpu-multi ]; then
  run_llama rtstt-mt 8095 $IMG:server-cuda Hy-MT2-1.8B-Q4_K_M.gguf -ngl 99 -c 32768 -np 8 --cache-ram 512
else
  run_llama rtstt-mt 8095 $IMG:server Hy-MT2-1.8B-Q4_K_M.gguf -t 8 -c 16384 -np 4 --cache-ram 512
fi
ARGS=(--asr nemotron --llm-url http://localhost:8095/v1 --llm-model hy-mt2 --llm-style hymt)

case $PROFILE in
  cpu) ;;
  cpu-hq)
    run_llama rtstt-asr2 8092 $IMG:server Qwen3-ASR-1.7B-Q8_0.gguf --mmproj /models/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf -t 8 -c 8192 -np 4 --cache-ram 0
    ARGS+=(--final llamacpp --final-url http://localhost:8092/v1) ;;
  gpu-min)
    run_llama rtstt-asr2 8094 $IMG:server-cuda Qwen3-ASR-1.7B-Q8_0.gguf --mmproj /models/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf \
      -ngl 99 -c 4096 -np 4 --cache-ram 0
    ARGS+=(--final llamacpp --final-url http://localhost:8094/v1) ;;
  gpu-max)
    docker rm -f rtstt-asr2 >/dev/null 2>&1 || true
    ARGS+=(--final whisper --device cuda --workers 2 --whisper-batch 16 --vad-interval-s 0.05
           --asr-workers 8 --asr-threads 1 --nemotron-dir "$HERE/models/asr/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-int8-2026-06-11.dl") ;;
  gpu-multi)
    docker rm -f rtstt-asr2 >/dev/null 2>&1 || true
    ARGS+=(--asr multi --device cuda --workers 2 --whisper-batch 16 --vad-interval-s 0.05
           --asr-workers 8 --asr-threads 1 --nemotron-dir "$HERE/models/asr/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-int8-2026-06-11.dl" --madlad-dir "$HERE/models/mt/madlad400-3b-ct2-int8") ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 1 ;;
esac

[ "$PROFILE" = cpu ] && { docker rm -f rtstt-asr2 >/dev/null 2>&1 || true; }
# EXTRA="--workers 4 --whisper-batch 32" ./start.sh gpu-multi  -> appended to the profile's server args (tuning)
[ -n "${EXTRA:-}" ] && ARGS+=($EXTRA)
cd "$HERE"
pkill -f "^\.venvs/cpuasr/bin/python server\.py" || true   # any previous server instance
sleep 1
if [ "$PROFILE" = gpu-max ] || [ "$PROFILE" = gpu-multi ]; then   # faster-whisper on the GPU needs torch's cuDNN/cuBLAS
  NV=$(python3 -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || echo /usr/local/lib/python3.13/site-packages/nvidia)
  LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib nohup .venvs/cpuasr/bin/python server.py "${ARGS[@]}" --port "$PORT" > "$LOG" 2>&1 &
else
  CUDA_VISIBLE_DEVICES="" nohup .venvs/cpuasr/bin/python server.py "${ARGS[@]}" --port "$PORT" > "$LOG" 2>&1 &
fi
for _ in $(seq 60); do grep -q READY "$LOG" && break; grep -q Traceback "$LOG" && { tail -20 "$LOG"; exit 1; }; sleep 1; done
grep READY "$LOG"
echo "profile=$PROFILE log=$LOG"
