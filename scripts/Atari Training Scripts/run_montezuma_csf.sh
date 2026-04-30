#!/bin/bash
# Montezuma's Revenge Room 1 — CSF (Contrastive Successor Features)
#
# CSF = SF backbone + InfoNCE contrastive term + dual constraint
# Differences from ViSR (run_montezuma_room1.sh):
#   --self_normalizing 0   → no unit sphere; InfoNCE separates skills
#   --dual_reg 1           → METRA dual constraint to bound ||φ||
#                            (without this OR self_normalizing, φ → ∞)
#   --no_diff_in_rep 1     → φ(s')·z reward (displacement gives ~0 on Atari)
#   --log_sum_exp 1        → InfoNCE contrastive loss
#   --sample_new_z 1       → fresh negative z's each batch
#   --num_negative_z 256   → number of negatives
#   --infonce_lam 1.0      → contrastive weight

python run/train.py \
    --run_group "Montezuma-CSF" \
    --env "montezuma_room1" \
    --algo "metra_sf" \
    --max_path_length 500 \
    --discrete 0 \
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
    --no_diff_in_rep 1 \
    --self_normalizing 0 \
    --dual_reg 1 \
    --dual_lam 30 \
    --dual_slack 1e-3 \
    --turn_off_dones 1 \
    --log_sum_exp 1 \
    --sample_new_z 1 \
    --num_negative_z 256 \
    --infonce_lam 1.0 \
    --n_parallel 1 \
    --use_discrete_sac 1 \
    --use_gpu 0 \
    --sample_cpu 1 \
    --seed 0
