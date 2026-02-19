#!/usr/bin/env python3
"""
Generate training analysis plots from METRA progress.csv.
Only plots that are genuinely useful for understanding skill learning.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def smooth(vals, w=5):
    if len(vals) < w:
        return vals
    return np.convolve(vals, np.ones(w)/w, mode='valid')


def plot_all(csv_path, output_dir=None):
    df = pd.read_csv(csv_path)
    if output_dir is None:
        output_dir = Path(csv_path).parent / "training_plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Shorten column names for convenience
    col = lambda suffix: f'TrainSp/metra/{suffix}'
    ep = df['TotalEpoch'].values

    # =====================================================================
    # FIGURE 1: The Big Picture (2x2) — the 4 metrics that matter most
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('METRA Training Overview', fontsize=16, fontweight='bold')

    # 1a: phi_diff_l2 — ARE SKILLS DIFFERENT?
    ax = axes[0, 0]
    vals = df[col('phi_diff_l2')].values
    ax.plot(ep, vals, color='#2196F3', linewidth=1.5, alpha=0.4, label='raw')
    if len(vals) >= 5:
        ax.plot(ep[2:-2], smooth(vals), color='#2196F3', linewidth=2.5, label='smoothed')
    ax.set_title('phi_diff_l2  (skill differentiation)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||phi(s\') - phi(s)||')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    # Add annotation showing growth
    if len(vals) > 10:
        early = np.mean(vals[:5])
        late = np.mean(vals[-5:])
        ratio = late / early if early > 0 else 0
        ax.annotate(f'{ratio:.0f}x growth', xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=11, fontweight='bold',
                    color='green' if ratio > 5 else 'orange')

    # 1b: phi_l2 — IS PHI NORM STABLE?
    ax = axes[0, 1]
    vals = df[col('phi_l2')].values
    ax.plot(ep, vals, color='#4CAF50', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='target = 1.0')
    ax.set_title('phi_l2  (representation norm stability)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||phi(s)||')
    ax.set_ylim([max(0, vals.min() - 0.2), vals.max() + 0.2])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    # Annotate stability
    std = np.std(vals)
    ax.annotate(f'std = {std:.4f}', xy=(0.95, 0.05), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10,
                color='green' if std < 0.1 else 'red')

    # 1c: PureRewardMean + Std — REWARD SIGNAL QUALITY
    ax = axes[1, 0]
    mean_r = df[col('PureRewardMean')].values
    std_r = df[col('PureRewardStd')].values
    ax.plot(ep, mean_r, color='#FF9800', linewidth=2, label='mean')
    ax.fill_between(ep, mean_r - std_r, mean_r + std_r, color='#FF9800', alpha=0.15, label='±1 std')
    ax.plot(ep, std_r, color='#E91E63', linewidth=1.5, linestyle='--', label='std (skill diversity)')
    ax.set_title('Intrinsic Reward  (METRA objective)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 1d: DualLam — CONSTRAINT SATISFACTION
    ax = axes[1, 1]
    vals = df[col('DualLam')].values
    ax.plot(ep, vals, color='#9C27B0', linewidth=2)
    ax.set_title('DualLam  (Lagrange multiplier)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('lambda')
    ax.grid(True, alpha=0.3)
    trend = 'decreasing (good — constraint being satisfied)' if vals[-1] < vals[0] else 'increasing (constraint violated)'
    ax.annotate(trend, xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top', fontsize=9,
                color='green' if vals[-1] < vals[0] else 'red')

    plt.tight_layout()
    out = output_dir / '1_training_overview.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    # =====================================================================
    # FIGURE 2: Loss Landscape (2x2) — are the optimizers healthy?
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Loss Landscape', fontsize=16, fontweight='bold')

    # 2a: Q-function losses
    ax = axes[0, 0]
    ax.plot(ep, df[col('LossQf1')].values, label='Qf1', linewidth=1.5, alpha=0.8)
    ax.plot(ep, df[col('LossQf2')].values, label='Qf2', linewidth=1.5, alpha=0.8)
    ax.set_title('Q-Function Losses', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2b: Policy + TE losses
    ax = axes[0, 1]
    ax.plot(ep, df[col('LossSacp')].values, label='Policy (SAC)', color='#2196F3', linewidth=1.5)
    ax.plot(ep, df[col('LossTe')].values, label='Traj Encoder', color='#4CAF50', linewidth=1.5)
    ax.set_title('Policy & Traj Encoder Losses', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2c: TeObj (how well constraint is satisfied) + PhiNormReg
    ax = axes[1, 0]
    ax.plot(ep, df[col('TeObjMean')].values, label='TeObj (-> 0 = satisfied)', color='#FF5722', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(ep, df[col('PhiNormReg')].values, label='PhiNormReg', color='#607D8B', linewidth=1.5, linestyle='--')
    ax2.set_ylabel('PhiNormReg', color='#607D8B')
    ax.set_title('Constraint Satisfaction', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('TeObj')
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # 2d: Alpha (SAC entropy)
    ax = axes[1, 1]
    ax.plot(ep, df[col('Alpha')].values, color='#00BCD4', linewidth=2)
    ax.set_title('SAC Alpha (exploration-exploitation)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('alpha')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / '2_loss_landscape.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    # =====================================================================
    # FIGURE 3: Gradient Health (2x1) — are gradients flowing?
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Gradient Health', fontsize=16, fontweight='bold')

    # 3a: Per-component gradient norms
    ax = axes[0]
    components = [
        ('TotalGradNormCnn', 'CNN', '#F44336'),
        ('TotalGradNormTrajEncoder', 'TrajEncoder', '#4CAF50'),
        ('TotalGradNormOptionPolicy', 'Policy', '#2196F3'),
        ('TotalGradNormQf1', 'Qf1', '#FF9800'),
    ]
    for suffix, label, color in components:
        vals = df[col(suffix)].values
        ax.plot(ep, vals, label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.set_title('Gradient Norms by Component', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3b: Before-clip vs total (is clipping active?)
    ax = axes[1]
    before_clip = df[col('GradNormBeforeClip_cnn_traj_encoder')].values
    after_clip = df[col('TotalGradNormCnn')].values + df[col('TotalGradNormTrajEncoder')].values
    ax.plot(ep, before_clip, label='CNN+TE before clip', color='#F44336', linewidth=1.5)
    ax.plot(ep, after_clip, label='CNN+TE after clip', color='#4CAF50', linewidth=1.5)
    ax.axhline(y=10.0, color='gray', linestyle='--', alpha=0.5, label='clip threshold (10)')
    ax.set_title('Gradient Clipping Activity', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / '3_gradient_health.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    # =====================================================================
    # FIGURE 4: Episode Performance — game score over training
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Episode Performance', fontsize=16, fontweight='bold')

    # 4a: Returns
    ax = axes[0]
    avg_ret = df['TrainSp/AverageReturn'].values
    max_ret = df['TrainSp/MaxReturn'].values
    min_ret = df['TrainSp/MinReturn'].values
    ax.fill_between(ep, min_ret, max_ret, alpha=0.15, color='#2196F3', label='min-max range')
    ax.plot(ep, avg_ret, color='#2196F3', linewidth=2, label='average')
    ax.set_title('Episode Returns (game score)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 4b: Path lengths
    ax = axes[1]
    ax.plot(ep, df[col('PathLengthMean')].values, color='#4CAF50', linewidth=2, label='mean')
    ax.fill_between(ep,
                    df[col('PathLengthMin')].values,
                    df[col('PathLengthMax')].values,
                    alpha=0.15, color='#4CAF50', label='min-max')
    ax.set_title('Episode Lengths', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Steps')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / '4_episode_performance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    # =====================================================================
    # Print summary stats
    # =====================================================================
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Epochs trained    : {int(ep[-1])}")
    print(f"  Total env steps   : {int(df['TotalEnvSteps'].iloc[-1]):,}")
    print(f"  Total wall time   : {df['TimeTotal'].iloc[-1] / 3600:.1f} hours")
    print(f"  ---")
    phi_diff_early = np.mean(df[col('phi_diff_l2')].values[:5])
    phi_diff_late = np.mean(df[col('phi_diff_l2')].values[-5:])
    print(f"  phi_diff_l2 early : {phi_diff_early:.6f}")
    print(f"  phi_diff_l2 late  : {phi_diff_late:.6f}")
    print(f"  phi_diff growth   : {phi_diff_late/phi_diff_early:.1f}x")
    print(f"  phi_l2 mean       : {np.mean(df[col('phi_l2')].values):.4f}  (target ~1.0)")
    print(f"  phi_l2 std        : {np.std(df[col('phi_l2')].values):.4f}")
    print(f"  DualLam trend     : {df[col('DualLam')].iloc[0]:.2f} -> {df[col('DualLam')].iloc[-1]:.2f}")
    print(f"  Avg return (last) : {np.mean(avg_ret[-5:]):.1f}")
    print(f"  RewardStd (last)  : {np.mean(df[col('PureRewardStd')].values[-5:]):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot METRA training progress')
    parser.add_argument('--csv', type=str,
                        default="exp/Final-tests/sd100_1771186746_atari_mspacman_metra/progress.csv",
                        help='Path to progress.csv')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same dir as csv)')
    args = parser.parse_args()
    
    print("Generating training analysis plots...")
    plot_all(args.csv, args.output_dir)
    print("\nDone!")
