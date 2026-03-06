#!/bin/bash
# ============================================================
# Hex GPU Cluster Setup Script
# Run this ONCE inside the container to set up the environment
# ============================================================
#
# Usage:
#   hare run --rm -it \
#     --gpus device=0 \
#     -v "$(pwd)":/app \
#     --workdir /app \
#     <base-image> \
#     bash setup_hex.sh
#
# Or interactively:
#   hare run --rm -it --gpus device=0 -v "$(pwd)":/app --workdir /app <base-image>
#   Then inside: bash setup_hex.sh
# ============================================================

set -e

echo "============================================================"
echo "Setting up contrastive-successor-features on Hex cluster"
echo "============================================================"

# ── Fix HOME for non-root containers ──
export HOME=/tmp
echo "HOME set to $HOME"

# ── Check Python version ──
echo ""
echo "Python version:"
python3 --version

# ── Check CUDA availability ──
echo ""
echo "CUDA check:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not yet installed"

# ── Check if we need to create a venv ──
# If running in a container with system Python, create a venv
# If conda is available, use that instead
if command -v conda &> /dev/null; then
    echo ""
    echo "Conda detected. Creating conda environment..."
    conda create -n csf python=3.9 -y
    eval "$(conda shell.bash hook)"
    conda activate csf
    echo "Conda env 'csf' activated"
else
    echo ""
    echo "No conda found. Using system Python with venv..."
    python3 -m venv /tmp/csf_env
    source /tmp/csf_env/bin/activate
    echo "venv activated at /tmp/csf_env"
fi

# ── Install PyTorch with CUDA ──
echo ""
echo "Installing PyTorch with CUDA support..."
# Detect CUDA version and install matching PyTorch
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | sed 's/,//' | cut -d. -f1,2)
echo "Detected CUDA version: ${CUDA_VERSION:-unknown}"

if [[ "$CUDA_VERSION" == "12"* ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11"* ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "WARNING: Could not detect CUDA version. Installing default PyTorch..."
    pip install torch torchvision
fi

# ── Install all other dependencies ──
echo ""
echo "Installing dependencies from requirements_hex.txt..."
pip install -r requirements_hex.txt

# ── Install local packages in editable mode ──
echo ""
echo "Installing local packages (garage, iod)..."
pip install -e garaged/
pip install -e .

# ── Install Atari ROMs ──
echo ""
echo "Installing Atari ROM support..."
pip install "gym[atari,accept-rom-license]" 2>/dev/null || echo "ROM install via gym failed, trying ale-py..."
python3 -c "import ale_py; print(f'ALE-py version: {ale_py.__version__}')"

# ── Verify installation ──
echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"
python3 -c "
import dowel_wrapper
import torch
import numpy as np
import gym

print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
print(f'NumPy:    {np.__version__}')
print(f'Gym:      {gym.__version__}')

# Test Atari env creation
from envs.atari.montezuma_room1_wrapper import MontezumaRoom1Wrapper
env = MontezumaRoom1Wrapper()
obs = env.reset()
print(f'Env obs shape: {obs.shape}')
env.close()
print()
print('All checks passed!')
"

echo ""
echo "============================================================"
echo "Setup complete! Run training with:"
echo "  bash run_montezuma_hex.sh"
echo "============================================================"
