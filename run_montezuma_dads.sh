#!/bin/bash
# Montezuma's Revenge Room 1 — DADS Baseline
#
# DADS (Dynamics-Aware Discovery of Skills) baseline for thesis comparison
# against CSF (Contrastive Successor Features).
#
# DADS learns a skill-conditioned dynamics model p(s'|s,z) and uses it
# to compute intrinsic rewards via mutual information:
#   r(s,z) = log p(s'|s,z) - log sum_z' p(s'|s,z')
#
# Key differences from CSF:
#   - algo: dads               → dynamics-based skill discovery
#   - skill_dynamics trained    → MLP predicts encoded state deltas
#   - No traj_encoder loss      → no contrastive/SF objective
#   - inner: 1                  → use inner product reward formulation
#   - dual_reg: 0               → no dual constraint
#   - no_diff_in_rep: 0         → DADS uses state differences (s'-s)
#   - self_normalizing: 0       → no L2 normalisation of representations
#   - num_alt_samples: 100      → number of alternative skills for MI estimate
#
# Uses same CNN encoder, frame stack, and discrete action space as CSF.

python run/train.py \
    --run_group "Montezuma-DADS" \
    --env "montezuma_room1" \
    --algo "dads" \
    --max_path_length 500 \
    --dim_option 4 \
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
    --n_epochs_per_save 50 \
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
    --no_diff_in_rep 0 \
    --self_normalizing 0 \
    --turn_off_dones 1 \
    --num_alt_samples 100 \
    --split_group 65536 \
    --sd_batch_norm 0 \
    --sd_const_std 0 \
    --n_parallel 1 \
    --eval_plot_axis -1 \
    --use_discrete_sac 1 \
    --use_gpu 0 \
    --sample_cpu 1 \
    --seed 0
