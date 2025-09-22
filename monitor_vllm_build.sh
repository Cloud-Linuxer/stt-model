#!/bin/bash

echo "📦 Monitoring vLLM Docker image build..."

while true; do
    if docker images | grep -q vllm-server 2>/dev/null; then
        echo "✅ vLLM Docker image build complete!"
        docker images | grep vllm-server
        exit 0
    fi
    echo "⏳ Still building vLLM Docker image..."
    sleep 10
done