#!/bin/bash
# ============================================================
# Hex Cluster Quick Commands - Copy/paste ready!
# Replace mm3435 with your username if different
# ============================================================

# ── INITIAL SETUP (once) ──

# SSH to cluster
ssh mm3435@ogg.cs.bath.ac.uk

# Clone repo
git clone https://github.com/thamidmiah/contrastive-successor-features.git
cd contrastive-successor-features

# Build image (~15-20 min first time)
docker build -t metra-csf:latest .

# ── INTERACTIVE TESTING ──

# Start interactive shell (good for debugging)
hare run \
  -it \
  --name csf-shell \
  --gpus device=5 \
  --user "$(id -u)":"$(id -g)" \
  -v "$(pwd)":/app \
  --workdir /app \
  metra-csf:latest \
  bash

# Inside container, test:
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from envs.atari.montezuma_room1_wrapper import MontezumaRoom1Wrapper; print('OK')"

# ── RUN TRAINING (BACKGROUND) ──

# Start training in background (recommended)
hare run \
  -d \
  --name csf-montezuma \
  --gpus device=5 \
  --user "$(id -u)":"$(id -g)" \
  -v "$(pwd)":/app \
  --workdir /app \
  metra-csf:latest \
  bash run_montezuma_hex.sh

# ── MONITORING ──

# View logs (follow mode)
hare logs -f csf-montezuma

# View logs (last 100 lines)
hare logs --tail 100 csf-montezuma

# Check if running
hare ps

# Check GPU usage
nvidia-smi

# ── STOP/CLEANUP ──

# Stop container
hare stop csf-montezuma

# Remove container
hare rm csf-montezuma

# Stop and remove in one command
hare rm -f csf-montezuma

# ── RESUME TRAINING ──

# After checkpoint saved, resume with new params
hare run \
  -d \
  --name csf-resume \
  --gpus device=5 \
  --user "$(id -u)":"$(id -g)" \
  -v "$(pwd)":/app \
  --workdir /app \
  metra-csf:latest \
  python run/train.py --resume_from exp/montezuma/checkpoint_500.pkl --resume_epoch 500 ...

# ── MULTIPLE EXPERIMENTS ──

# Run on different GPUs simultaneously
hare run -d --name exp1 --gpus device=3 -v "$(pwd)":/app --workdir /app metra-csf bash run_montezuma_hex.sh
hare run -d --name exp2 --gpus device=4 -v "$(pwd)":/app --workdir /app metra-csf bash run_montezuma_hex.sh
hare run -d --name exp3 --gpus device=5 -v "$(pwd)":/app --workdir /app metra-csf bash run_montezuma_hex.sh

# ── DEBUGGING ──

# Check what's wrong if container exits immediately
hare logs csf-montezuma

# Run command directly in existing container
hare exec csf-montezuma bash
hare exec csf-montezuma nvidia-smi

# Inspect container details
hare inspect csf-montezuma

# ── IMAGE MANAGEMENT ──

# List images
docker images

# Remove old image before rebuilding
docker rmi metra-csf:latest

# Save image to file
docker save metra-csf:latest | gzip > metra-csf.tar.gz

# Load image from file
docker load < metra-csf.tar.gz

# ── DISK SPACE ──

# Check disk usage
df -h
du -sh exp/

# Clean up old containers
docker container prune

# Clean up old images
docker image prune

# Nuclear option (clean everything)
docker system prune -a
