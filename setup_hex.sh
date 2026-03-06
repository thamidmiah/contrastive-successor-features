#!/bin/bash
# ============================================================
# Hex GPU Cluster Setup Script
# Run this ONCE inside the container to set up the environment
# ============================================================

set -e

echo "============================================================"
echo "Setting up contrastive-successor-features on Hex cluster"
echo "============================================================"

# ── Fix permissions for non-root container users ──
export HOME=/tmp
export XDG_CACHE_HOME=/tmp/.cache
export CONDA_PKGS_DIRS=/tmp/conda_pkgs
mkdir -p /tmp/.cache /tmp/conda_pkgs

# Ensure project root is importable
export PYTHONPATH=/app:$PYTHONPATH

echo "HOME set to $HOME"

# ── Check Python version ──
echo ""
echo "Python version:"
python3 --version

# ── Check CUDA availability ──
echo ""
echo "CUDA check:"
python3 - <<EOF || echo "PyTorch not yet installed"
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
EOF

# ── Environment setup ──
if command -v conda &> /dev/null; then
    echo ""
    echo "Conda detected. Creating conda environment..."

    source /opt/conda/etc/profile.d/conda.sh

    conda create -n csf python=3.9 -y
    conda activate csf

    echo "Conda env 'csf' activated"

else
    echo ""
    echo "No conda found. Using Python venv..."

    python3 -m venv /tmp/csf_env
    source /tmp/csf_env/bin/activate

    echo "venv activated at /tmp/csf_env"
fi

# ── Upgrade pip ──
python -m pip install --upgrade pip

# ── Install PyTorch with CUDA support ──
echo ""
echo "Installing PyTorch with CUDA support..."

CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | sed 's/,//' | cut -d. -f1,2)

echo "Detected CUDA version: ${CUDA_VERSION:-unknown}"

if [[ "$CUDA_VERSION" == 12* ]]; then
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == 11* ]]; then
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "WARNING: Could not detect CUDA version. Installing default PyTorch..."
    python -m pip install torch torchvision
fi

# ── Install dependencies ──
echo ""
echo "Installing dependencies from requirements_hex.txt..."

python -m pip install -r requirements_hex.txt

# ── Install local packages ──
echo ""
echo "Installing local packages (garage, iod)..."

python -m pip install -e garaged/
python -m pip install -e .

# ── Install Atari ROM support ──
echo ""
echo "Installing Atari ROM support..."

python -m pip install "gym[atari,accept-rom-license]" 2>/dev/null || echo "ROM install via gym failed"

python3 - <<EOF
import ale_py
print(f"ALE-py version: {ale_py.__version__}")
EOF

# ── Verify installation ──
echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

python3 - <<EOF
import dowel_wrapper
import torch
import numpy as np
import gym

print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)")

if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")

print(f"NumPy:    {np.__version__}")
print(f"Gym:      {gym.__version__}")

from envs.atari.montezuma_room1_wrapper import MontezumaRoom1Wrapper

env = MontezumaRoom1Wrapper()
obs = env.reset()

print(f"Env obs shape: {obs.shape}")

env.close()

print()
print("All checks passed!")
EOF

echo ""
echo "============================================================"
echo "Setup complete!"
echo "Run training with:"
echo "  bash run_montezuma_hex.sh"
echo "============================================================"