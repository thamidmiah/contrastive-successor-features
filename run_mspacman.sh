#!/bin/bash
# Run MsPacman METRA with CNN Encoder (Phase 3)

# Ensure we're in the csf conda environment
source ~/.zshrc
conda activate csf

# Navigate to project directory
cd "/Users/thamidmiah/Documents/ University of Bath/Bath CS/Year 3/Final year project/contrastive-successor-features"

# Run training with CNN encoder enabled
python run/train.py \
  --env atari_mspacman \
  --algo metra \
  --use_discrete_sac 1 \
  --use_cnn_encoder 1 \
  --cnn_type nature \
  --dim_option 8 \
  --discrete 1 \
  --n_epochs 100 \
  --max_path_length 200 \
  --traj_batch_size 8 \
  --trans_optimization_epochs 50 \
  --trans_minibatch_size 128 \
  --use_gpu 0 \
  --sample_cpu 1 \
  --n_parallel 1 \
  --n_epochs_per_save 25 \
  --n_epochs_per_pt_save 25 \
  --n_epochs_per_log 25 \
  --run_group MsPacman_METRA \
  --seed 100
