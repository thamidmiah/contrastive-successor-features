#!/bin/bash
# Montezuma's Revenge Room 1 - METRA skill discovery
# Improved hyperparameters for visually distinct skills:
#   - dim_option=4 (fewer skills → more distinct)
#   - alpha_min=0.01 (prevent alpha collapse → maintain exploration)
#   - num_random_trajectories=100 (better baseline for phi normalization)
#   - sac_max_buffer_size=50000 (larger buffer for diverse experience)
#   - sac_target_coef=1.0 (default; don't suppress target entropy)
#   - No-op reset randomization (built into MontezumaRoom1Wrapper, noop_max=30)

python run/train.py \
    --run_group "Montezuma-Increased-Trajectories" \
    --env "montezuma_room1" \
    --algo "metra" \
    --max_path_length 500 \
    --dim_option 4 \
    --discrete 1 \
    --inner 1 \
    --unit_length 0 \
    --num_random_trajectories 132 \
    --model_master_dim 512 \
    --use_cnn_encoder 1 \
    --cnn_type "nature" \
    --frame_stack 4 \
    --n_epochs 10000 \
    --n_epochs_per_eval 100 \
    --n_epochs_per_log 10 \
    --n_epochs_per_save 100 \
    --traj_batch_size 16 \
    --trans_minibatch_size 128 \
    --trans_optimization_epochs 25 \
    --alpha 0.05 \
    --alpha_min 0.03 \
    --sac_scale_reward 1.0 \
    --sac_target_coef 0.5 \
    --sac_discount 0.99 \
    --sac_tau 5e-3 \
    --sac_min_buffer_size 3000 \
    --sac_max_buffer_size 30000 \
    --common_lr 1e-4 \
    --dual_reg 1 \
    --dual_lam 24 \
    --dual_slack 1e-2 \
    --turn_off_dones 0 \
    --n_parallel 1 \
    --eval_plot_axis -1 \
    --use_discrete_sac 1 \
    --use_gpu 0 \
    --sample_cpu 1 \
    --seed 0
