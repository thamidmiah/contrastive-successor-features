#!/bin/bash
# ============================================================
# Build, test, and optionally push the METRA Docker image
#
# Usage:
#   ./docker_build.sh                  # Build only
#   ./docker_build.sh --test           # Build + run smoke test
#   ./docker_build.sh --push REGISTRY  # Build + push to registry
#
# Examples:
#   ./docker_build.sh --test
#   ./docker_build.sh --push ghcr.io/thamidmiah/metra-csf
#   ./docker_build.sh --push your-registry.com/metra-csf
# ============================================================

set -e

IMAGE_NAME="metra-csf"
IMAGE_TAG="latest"

echo "============================================================"
echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "============================================================"

docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo ""
echo "✅ Image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "   Size: $(docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} --format='{{.Size}}' | numfmt --to=iec 2>/dev/null || docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} --format='{{.Size}}')"

# ── Smoke test ──
if [[ "$1" == "--test" ]]; then
    echo ""
    echo "============================================================"
    echo "Running smoke test..."
    echo "============================================================"

    docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} python -c "
import sys
print(f'Python:       {sys.version}')

import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA avail:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'GPU:          {torch.cuda.get_device_name(0)}')

import numpy as np
print(f'NumPy:        {np.__version__}')

import gym
print(f'Gym:          {gym.__version__}')

import ale_py
print(f'ALE-py:       {ale_py.__version__}')

import dowel_wrapper
print('dowel_wrapper: OK')

from envs.atari.montezuma_room1_wrapper import MontezumaRoom1Wrapper
env = MontezumaRoom1Wrapper()
obs = env.reset()
print(f'Env obs shape: {obs.shape}')
env.close()

print()
print('✅ All smoke tests passed!')
"
    echo ""
    echo "✅ Smoke test passed!"
fi

# ── Push to registry ──
if [[ "$1" == "--push" && -n "$2" ]]; then
    REGISTRY="$2"
    REMOTE_TAG="${REGISTRY}:${IMAGE_TAG}"

    echo ""
    echo "============================================================"
    echo "Pushing to: ${REMOTE_TAG}"
    echo "============================================================"

    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REMOTE_TAG}
    docker push ${REMOTE_TAG}

    echo ""
    echo "✅ Pushed successfully: ${REMOTE_TAG}"
    echo ""
    echo "On the remote machine, pull and run with:"
    echo "  docker pull ${REMOTE_TAG}"
    echo "  docker run --gpus all -it ${REMOTE_TAG} bash run_montezuma_hex.sh"
fi

echo ""
echo "============================================================"
echo "Quick reference:"
echo "  Run training:    docker run --gpus all -v \$(pwd)/exp:/app/exp ${IMAGE_NAME} bash run_montezuma_hex.sh"
echo "  Interactive:     docker run --gpus all -it -v \$(pwd)/exp:/app/exp ${IMAGE_NAME} bash"
echo "  With compose:    docker compose run --rm train"
echo "============================================================"
