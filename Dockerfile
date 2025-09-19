FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install Faster-Whisper and dependencies
RUN pip3 install --no-cache-dir \
    faster-whisper \
    numpy \
    soundfile \
    librosa \
    tqdm \
    psutil \
    websockets \
    asyncio \
    flask \
    flask-cors \
    flask-sock

# Create directories for model cache and data
RUN mkdir -p /app/models /app/data /app/outputs

# Note: Model will be downloaded at runtime due to RTX 5090 compatibility

# Copy application files
COPY . /app/

# Set environment variables
ENV WHISPER_CACHE_DIR=/app/models
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "test_faster_whisper.py"]