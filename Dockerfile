# ============================================================
# Dockerfile for contrastive-successor-features (METRA)
# Montezuma's Revenge Skill Discovery
# 
# Optimized for Linux x86_64 GPU machines (Hex cluster).
# Uses system Python + venv instead of conda for simplicity.
#
# Build:
#   docker build -t metra-csf .
#
# Run (GPU):
#   docker run --gpus all -v $(pwd)/exp:/app/exp -it metra-csf bash run_montezuma_hex.sh
#
# Run (interactive):
#   docker run --gpus all -it metra-csf bash
# ============================================================

# ── Base image: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04 ──
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ── System dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3-pip \
    # Build tools
    build-essential \
    wget \
    curl \
    git \
    unzip \
    ca-certificates \
    # OpenCV / rendering deps
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    # Video encoding (for moviepy / imageio-ffmpeg)
    ffmpeg \
    # For ale-py / Atari
    cmake \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Create Python virtual environment ──
WORKDIR /app
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Upgrade pip ──
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Install PyTorch with CUDA 12.1 support ──
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Copy requirements file ──
COPY requirements_docker.txt /app/requirements_docker.txt

# ── Install Python dependencies ──
RUN pip install --no-cache-dir -r requirements_docker.txt

# ── Copy the full project ──
COPY . /app

# ── Install local packages (garage + iod) ──
RUN pip install --no-cache-dir --no-deps -e garaged/ && \
    pip install --no-cache-dir --no-deps -e .

# ── Install Atari ROMs ──
RUN pip install --no-cache-dir "gym[atari,accept-rom-license]" 2>/dev/null || true

# ── Set environment variables ──
ENV PYTHONPATH=/app:$PYTHONPATH
ENV MUJOCO_GL=egl
ENV PYTHONUNBUFFERED=1

# ── Default command ──
CMD ["bash"]
