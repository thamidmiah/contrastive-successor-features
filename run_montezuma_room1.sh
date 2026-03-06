#!/bin/bash
# Montezuma's Revenge Room 1 - METRA skill discovery
# RESUME from epoch 200 checkpoint with improved v3 hyperparameters:
#   - dual_slack: 0.01 → 1.0  (MOST CRITICAL: force real phi separation)
#   - alpha_min: 0.01 → 0.03  (higher exploration floor)
#   - sac_target_coef: 0.5    (higher entropy target)
#   - dual_lam: 24 initial    (strong Lagrange pressure; restored value from checkpoint ~2.68)
#   - sac_max_buffer_size: 30000 (prevent OOM on 8GB Mac)
#   - n_epochs: 10000 (training continues from epoch 200 → 10000)

#RESUME_DIR="exp/Montezuma-Increased-Trajectories/sd000_1772467785_montezuma_room1_metra"
#     --resume_from "$RESUME_DIR" \
#   --resume_epoch 200 \

python run/train.py \
    --run_group "Montezuma-Hex" \
    --env "montezuma_room1" \
    --algo "metra" \
    --disable_tensorboard 1 \
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
    --dual_slack 1.0 \
    --turn_off_dones 0 \
    --n_parallel 1 \
    --eval_plot_axis -1 \
    --use_discrete_sac 1 \
    --use_gpu 0 \
    --sample_cpu 1 \
    --seed 0
