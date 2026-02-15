#!/usr/bin/env python3
"""
Visualize learned skills from a trained METRA checkpoint.
Shows: skill rollout videos, phi representations, per-dimension analysis, and skill behavior.

Bug fixes applied:
  1. num_skills read from algo.dim_option (not hardcoded 16)
  2. Policy input uses raw flat obs + option (policy handles CNN internally)
  3. Deterministic actions via force_use_mode_actions (not stochastic sampling)
  4. Clean discrete action handling
  5. Shape assertions on phi extraction
"""

# CRITICAL: Import dowel_wrapper before anything else to avoid import errors
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import dowel_wrapper  # Must be first!

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import os
import imageio

from envs.atari.atari_env import AtariEnv


def load_checkpoint(exp_dir, epoch=None):
    """Load trained models from checkpoint."""
    exp_path = Path(exp_dir)
    
    if epoch is not None:
        print(f"Loading checkpoint from epoch {epoch}...")
        itr_path = exp_path / f"itr_{epoch}.pkl"
        if not itr_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {itr_path}")
    else:
        itr_files = sorted(exp_path.glob("itr_*.pkl"))
        if not itr_files:
            raise FileNotFoundError(f"No checkpoints found in {exp_path}")
        itr_path = itr_files[-1]
        epoch = int(itr_path.stem.split('_')[1])
        print(f"Using latest checkpoint: epoch {epoch}")
    
    with open(itr_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Loaded checkpoint from epoch {epoch}")
    return data, epoch


def setup_env():
    """Create Ms. Pac-Man environment matching training config."""
    env = AtariEnv(
        game='MsPacman',
        frame_stack=4,
        screen_size=84,
        grayscale=True,
        normalize_pixels=True,
    )
    print("Created Ms. Pac-Man environment")
    return env


def introspect_algo(algo):
    """Print key properties of the loaded algo for sanity checking."""
    print("\n--- Algo Introspection ---")
    print(f"  dim_option      : {algo.dim_option}")
    print(f"  inner           : {getattr(algo, 'inner', '?')}")
    print(f"  unit_length     : {getattr(algo, 'unit_length', '?')}")
    print(f"  dual_slack      : {getattr(algo, 'dual_slack', '?')}")
    print(f"  device          : {algo.device}")
    policy = algo.option_policy
    print(f"  policy type     : {type(policy).__name__}")
    has_cnn = hasattr(policy, '_obs_preprocessor') and policy._obs_preprocessor is not None
    print(f"  policy has CNN  : {has_cnn}")
    if has_cnn:
        print(f"  policy option_dim: {policy._option_dim}")
    te = algo.traj_encoder
    print(f"  traj_encoder    : {type(te).__name__}")
    mean_mod = getattr(te, '_mean_module', None)
    if mean_mod is not None:
        layers = [m for m in mean_mod.modules() if isinstance(m, torch.nn.Linear)]
        if layers:
            last = layers[-1]
            print(f"  te final Linear : in={last.in_features} -> out={last.out_features}")
            print(f"  te final weight norm: {last.weight.data.norm():.4f}")
    if hasattr(algo, 'cnn_encoder'):
        print(f"  cnn_encoder     : {type(algo.cnn_encoder).__name__}")
    else:
        print(f"  cnn_encoder     : NOT FOUND")
    print("--------------------------\n")


def collect_skill_trajectories(algo, env, num_episodes_per_skill=3, max_steps=500,
                               record_video=True, output_dir=None):
    """Collect trajectories for each skill and optionally record videos.
    
    Key fixes (matching training pipeline exactly):
      - num_skills = algo.dim_option  (Bug 1 fix)
      - Action via policy.get_action(np.concatenate([obs_flat, option]))
        the policy's process_observations() handles CNN internally (Bug 2 fix)
      - Deterministic actions via force_use_mode_actions (Bug 3 fix)
      - Clean discrete action handling (Bug 4 fix)
      - Shape assertions on phi (Bug 5 fix)
    """
    policy = algo.option_policy
    traj_encoder = algo.traj_encoder
    cnn_encoder = algo.cnn_encoder if hasattr(algo, 'cnn_encoder') else None
    num_skills = algo.dim_option
    device = torch.device('cpu')
    
    # Force deterministic actions for reproducible visualisation
    old_force_mode = policy._force_use_mode_actions
    policy._force_use_mode_actions = True
    
    print(f"\nCollecting trajectories for {num_skills} skills x {num_episodes_per_skill} episodes...")
    if record_video:
        print("  Video recording enabled!")
    
    all_trajectories = {}
    all_phis = {}
    all_rewards = []
    
    for skill_idx in range(num_skills):
        # One-hot encode skill
        option = np.zeros(num_skills, dtype=np.float32)
        option[skill_idx] = 1.0
        
        print(f"  Skill {skill_idx}/{num_skills}...")
        
        skill_trajectories = []
        skill_phis = []
        skill_total_rewards = []
        
        for ep in range(num_episodes_per_skill):
            obs = env.reset()
            obs_flat = obs.flatten().astype(np.float32)  # (28224,)
            
            episode_obs = []
            episode_phis = []
            episode_rewards = []
            frames = []
            
            for step in range(max_steps):
                # --- Compute phi for this observation ---
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs_flat).float().unsqueeze(0)  # (1, 28224)
                    if cnn_encoder is not None:
                        obs_4d = obs_tensor.view(1, 4, 84, 84).to(device)
                        encoded = cnn_encoder(obs_4d)  # (1, 512)
                    else:
                        encoded = obs_tensor.to(device)
                    phi = traj_encoder(encoded).mean  # (1, dim_option)
                    assert phi.shape == (1, num_skills), \
                        f"phi shape mismatch: expected (1, {num_skills}), got {phi.shape}"
                    phi_np = phi.cpu().numpy().squeeze(0)  # (dim_option,)
                
                # --- Get action via policy (handles CNN internally) ---
                # Matches metra.py: policy.get_action(np.concatenate([obs, option]))
                policy_input = np.concatenate([obs_flat, option])  # (28228,)
                with torch.no_grad():
                    action, agent_info = policy.get_action(policy_input)
                
                # --- Convert action for discrete Atari ---
                if isinstance(action, np.ndarray):
                    if action.ndim == 0:
                        action_int = int(action)
                    elif action.size == 1:
                        action_int = int(action.item())
                    else:
                        action_int = int(np.argmax(action))
                else:
                    action_int = int(action)
                
                # Record frame for video
                if record_video and ep == 0 and step % 2 == 0:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                
                # Step environment
                next_obs, reward, done, info = env.step(action_int)
                
                episode_obs.append(obs_flat.copy())
                episode_phis.append(phi_np.copy())
                episode_rewards.append(reward)
                
                obs = next_obs
                obs_flat = obs.flatten().astype(np.float32)
                
                if done:
                    break
            
            total_reward = sum(episode_rewards)
            skill_total_rewards.append(total_reward)
            
            skill_trajectories.append({
                'obs': np.array(episode_obs),
                'phis': np.array(episode_phis),
                'rewards': episode_rewards,
                'total_reward': total_reward,
                'length': len(episode_rewards),
            })
            skill_phis.append(np.array(episode_phis))
            
            # Save video for first episode of this skill
            if record_video and ep == 0 and len(frames) > 0 and output_dir is not None:
                video_path = output_dir / f"skill_{skill_idx:02d}_reward_{total_reward:.0f}.mp4"
                try:
                    imageio.mimsave(str(video_path), frames, fps=30)
                    print(f"    Saved video: {video_path.name}")
                except Exception as e:
                    print(f"    [video save failed: {e}]")
            
            print(f"    ep {ep}: reward={total_reward:.1f}  len={len(episode_rewards)}  "
                  f"phi_norm={np.linalg.norm(episode_phis[-1]):.4f}")
        
        avg_reward = np.mean(skill_total_rewards)
        all_rewards.append(avg_reward)
        all_trajectories[skill_idx] = skill_trajectories
        all_phis[skill_idx] = skill_phis
        print(f"    -> Skill {skill_idx} avg reward: {avg_reward:.1f}")
    
    # Restore old mode setting
    policy._force_use_mode_actions = old_force_mode
    
    return all_trajectories, all_phis, all_rewards


def visualize_phi_space(all_phis, num_skills, output_dir):
    """Visualize phi representations in 2D using PCA (or directly if dim <= 2)."""
    print("\nVisualizing phi space...")
    
    # Flatten all phis
    all_phis_flat = []
    skill_labels = []
    
    for skill_idx in range(num_skills):
        for episode_phis in all_phis[skill_idx]:
            all_phis_flat.append(episode_phis)
            skill_labels.extend([skill_idx] * len(episode_phis))
    
    all_phis_flat = np.concatenate(all_phis_flat, axis=0)
    skill_labels = np.array(skill_labels)
    
    # If dim_option <= 2, plot directly; otherwise PCA
    if all_phis_flat.shape[1] <= 2:
        phis_2d = all_phis_flat[:, :2]
        xlabel, ylabel = 'phi_0', 'phi_1'
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        phis_2d = pca.fit_transform(all_phis_flat)
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)'
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = cm.get_cmap('tab10')
    
    for skill_idx in range(num_skills):
        mask = skill_labels == skill_idx
        ax.scatter(
            phis_2d[mask, 0], 
            phis_2d[mask, 1],
            c=[cmap(skill_idx)],
            label=f'Skill {skill_idx}',
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Learned Skill Representations (phi) in 2D', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'phi_space_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def visualize_phi_dimensions(all_phis, num_skills, output_dir):
    """Box-plot of each phi dimension, grouped by skill."""
    print("\nVisualizing phi dimensions...")
    
    dim_option = None
    per_skill_phis = {}
    for skill_idx in range(num_skills):
        stacked = np.concatenate(all_phis[skill_idx], axis=0)
        per_skill_phis[skill_idx] = stacked
        if dim_option is None:
            dim_option = stacked.shape[1]
    
    fig, axes = plt.subplots(1, dim_option, figsize=(5 * dim_option, 6), squeeze=False)
    axes = axes.flatten()
    cmap_fn = cm.get_cmap('tab10')
    
    for d in range(dim_option):
        ax = axes[d]
        data = []
        labels = []
        colours = []
        for s in range(num_skills):
            vals = per_skill_phis[s][:, d]
            data.append(vals)
            labels.append(f'S{s}')
            colours.append(cmap_fn(s))
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(f'phi dim {d}', fontweight='bold')
        ax.set_xlabel('Skill')
        ax.set_ylabel(f'phi[{d}]')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Per-Dimension Phi Distribution by Skill', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'phi_dimensions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def visualize_skill_rewards(all_rewards, output_dir):
    """Bar plot of average rewards per skill."""
    print("\nVisualizing skill rewards...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    skills = np.arange(len(all_rewards))
    cmap_fn = cm.get_cmap('tab10')
    colors = [cmap_fn(i) for i in range(len(all_rewards))]
    
    bars = ax.bar(skills, all_rewards, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, reward in zip(bars, all_rewards):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{reward:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Skill Index', fontsize=12)
    ax.set_ylabel('Average Episode Return', fontsize=12)
    ax.set_title('Performance per Skill', fontsize=14, fontweight='bold')
    ax.set_xticks(skills)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'skill_rewards.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def visualize_phi_evolution(all_phis, num_skills, output_dir):
    """Show how ||phi|| changes over episode timesteps for each skill."""
    print("\nVisualizing phi evolution...")
    
    # Dynamic grid layout based on actual num_skills
    cols = min(num_skills, 4)
    rows = (num_skills + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    cmap_fn = cm.get_cmap('tab10')
    
    for skill_idx in range(num_skills):
        r, c = divmod(skill_idx, cols)
        ax = axes[r][c]
        
        for episode_phis in all_phis[skill_idx]:
            phi_norms = np.linalg.norm(episode_phis, axis=1)
            ax.plot(np.arange(len(phi_norms)), phi_norms,
                    alpha=0.5, color=cmap_fn(skill_idx), linewidth=1)
        
        ax.set_title(f'Skill {skill_idx}', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('||phi||')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_skills, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')
    
    plt.suptitle('Phi Norm Evolution per Skill', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'phi_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    import argparse
    
    
    parser = argparse.ArgumentParser(description='Visualize METRA skills on Atari')
    parser.add_argument('--exp_dir', type=str, 
                        default="exp/Ultimate-run/sd100_1771180736_atari_mspacman_metra",
                        help='Path to experiment directory')
    parser.add_argument('--checkpoint_epoch', type=int, default=None, 
                        help='Epoch to load (default: latest)')
    parser.add_argument('--num_episodes', type=int, default=3,
                        help='Episodes per skill')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Max steps per episode')
    parser.add_argument('--record_videos', action='store_true',
                        help='Record videos of skill gameplay')
    
    args = parser.parse_args()
    
    output_dir = Path(args.exp_dir) / "skill_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    video_dir = None
    if args.record_videos:
        video_dir = Path(args.exp_dir) / 'skill_videos'
        video_dir.mkdir(exist_ok=True)
    
    # Load checkpoint
    data, epoch = load_checkpoint(args.exp_dir, args.checkpoint_epoch)
    
    algo = data.get('algo', None)
    if algo is None:
        print("ERROR: Could not find 'algo' in checkpoint!")
        print("Available keys:", list(data.keys()))
        return
    
    num_skills = algo.dim_option  # Bug 1 fix: read from algo, not CLI
    
    print("="*60)
    print("METRA Skill Visualization (fixed)")
    print("="*60)
    print(f"Experiment  : {args.exp_dir}")
    print(f"Checkpoint  : epoch {epoch}")
    print(f"Skills      : {num_skills}  (from algo.dim_option)")
    print(f"Episodes    : {args.num_episodes} per skill")
    print(f"Max steps   : {args.max_steps}")
    print(f"Videos      : {'YES' if args.record_videos else 'no'}")
    print("="*60)
    
    # Move models to CPU + eval mode
    device = torch.device('cpu')
    algo.option_policy = algo.option_policy.to(device)
    algo.option_policy.eval()
    algo.traj_encoder = algo.traj_encoder.to(device)
    algo.traj_encoder.eval()
    if hasattr(algo, 'cnn_encoder') and algo.cnn_encoder is not None:
        algo.cnn_encoder = algo.cnn_encoder.to(device)
        algo.cnn_encoder.eval()
    
    introspect_algo(algo)
    
    # Setup environment
    env = setup_env()
    
    # Collect trajectories
    all_trajectories, all_phis, all_rewards = collect_skill_trajectories(
        algo, env,
        num_episodes_per_skill=args.num_episodes,
        max_steps=args.max_steps,
        record_video=args.record_videos,
        output_dir=video_dir,
    )
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualisations...")
    print("="*60)
    
    visualize_phi_space(all_phis, num_skills, output_dir)
    visualize_phi_dimensions(all_phis, num_skills, output_dir)
    visualize_skill_rewards(all_rewards, output_dir)
    visualize_phi_evolution(all_phis, num_skills, output_dir)
    
    print("\n" + "="*60)
    print("Visualisation complete!")
    print(f"Outputs saved to: {output_dir}")
    if args.record_videos:
        print(f"Videos saved to: {video_dir}")
    print("="*60)
    print("\nGenerated files:")
    print("  - phi_space_2d.png      : 2D projection of learned representations")
    print("  - phi_dimensions.png    : Per-dimension phi distribution by skill")
    print("  - skill_rewards.png     : Average reward per skill")
    print("  - phi_evolution.png     : ||phi|| norm over episode timesteps")
    if args.record_videos:
        print(f"  - {num_skills} video files  : Gameplay of each skill")


if __name__ == "__main__":
    main()
