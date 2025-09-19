FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

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

# Install PyTorch with CUDA support (2.5.1+ for RTX 5090 Blackwell support)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

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

# cuDNN is already included in the base image

# Note: Model will be downloaded at runtime due to RTX 5090 compatibility

# Copy application files
COPY . /app/

# Set environment variables
ENV WHISPER_CACHE_DIR=/app/models
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDNN_PATH=/usr/lib/x86_64-linux-gnu

CMD ["python", "test_faster_whisper.py"]