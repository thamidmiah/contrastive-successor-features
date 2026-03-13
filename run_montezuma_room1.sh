#!/bin/bash
# Montezuma's Revenge Room 1 — Contrastive Successor Features (CSF)
#
# Key design choices for CSF on Atari:
#   - algo: metra_sf          → SF-based policy (Q = psi^T z)
#   - dual_reg: 0             → no METRA dual constraint (SF handles reward)
#   - no_diff_in_rep: 1       → phi(s) not phi(s')-phi(s)  (state identity)
#   - self_normalizing: 1     → L2-normalise phi → unit sphere
#   - turn_off_dones: 1       → ignore episode boundaries in SF bootstrap
#   - discrete: 1             → one-hot skills z ∈ {e_1,...,e_8}
#   - use_cnn_encoder: 1      → shared NatureCNN for pixel observations
#   - alpha_min: 0.05         → entropy floor to prevent skill collapse
#   - trans_optimization_epochs: 30  → fewer updates per rollout (prevents overfit)
#   - sac_min_buffer_size: 2000      → start learning sooner
#   - traj_batch_size: 8             → reasonable for 1 worker

RESUME_DIR="exp/Montezuma-CSF-Dim8-2/sd000_1773322197_montezuma_room1_metra_sf"

python run/train.py \
    --run_group "Montezuma-CSF-Dim8-2" \
    --resume_from "$RESUME_DIR" \
    --resume_epoch 500 \
    --env "montezuma_room1" \
    --algo "metra_sf" \
    --max_path_length 500 \
    --dim_option 8 \
    --discrete 1 \
    --inner 1 \
    --unit_length 0 \
    --num_random_trajectories 48 \
    --model_master_dim 512 \
    --model_master_num_layers 2 \
    --use_cnn_encoder 1 \
    --cnn_type "nature" \
    --frame_stack 4 \
    --n_epochs 10000 \
    --n_epochs_per_eval 100 \
    --n_epochs_per_log 10 \
    --n_epochs_per_save 100 \
    --traj_batch_size 8 \
    --trans_minibatch_size 128 \
    --trans_optimization_epochs 30 \
    --alpha 0.05 \
    --alpha_min 0.05 \
    --sac_scale_reward 1.0 \
    --sac_target_coef 0.5 \
    --sac_discount 0.99 \
    --sac_tau 5e-3 \
    --sac_min_buffer_size 2000 \
    --sac_max_buffer_size 20000 \
    --common_lr 1e-4 \
    --dual_reg 0 \
    --no_diff_in_rep 1 \
    --self_normalizing 1 \
    --turn_off_dones 1 \
    --n_parallel 1 \
    --eval_plot_axis -1 \
    --use_discrete_sac 1 \
    --use_gpu 0 \
    --sample_cpu 1 \
    --seed 0
