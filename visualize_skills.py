#!/usr/bin/env python3
"""
Visualize learned skills from a trained METRA checkpoint.
Shows: skill trajectories, phi representations, and skill behavior.
"""

# CRITICAL: Import dowel_wrapper before anything else to avoid import errors
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import dowel_wrapper  # Must be first!

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import os
import imageio

from envs.atari.atari_env import AtariEnv
from iod.cnn_encoder import NatureCNN


def load_checkpoint(exp_dir, epoch=250):
    """Load trained models from checkpoint."""
    exp_path = Path(exp_dir)
    
    print(f"Loading checkpoint from epoch {epoch}...")
    
    # Load full state
    itr_path = exp_path / f"itr_{epoch}.pkl"
    if not itr_path.exists():
        # Try to find the latest checkpoint
        itr_files = sorted(exp_path.glob("itr_*.pkl"))
        if itr_files:
            itr_path = itr_files[-1]
            epoch = int(itr_path.stem.split('_')[1])
            print(f"Using latest checkpoint: epoch {epoch}")
        else:
            raise FileNotFoundError(f"No checkpoints found in {exp_path}")
    
    with open(itr_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Loaded checkpoint from epoch {epoch}")
    return data, epoch


def setup_env():
    """Create Ms. Pac-Man environment."""
    env = AtariEnv(
        game='MsPacman',  # Not 'MsPacman-v4', just the game name
        frame_stack=4,
        screen_size=84,
        grayscale=True,
        normalize_pixels=True  # Match training config
    )
    print("✓ Created Ms. Pac-Man environment")
    return env


def collect_skill_trajectories(env, policy, traj_encoder, encoder, num_skills=16, num_episodes_per_skill=3, max_steps=500, record_video=True, output_dir=None):
    """Collect trajectories for each skill and optionally record videos."""
    print(f"\nCollecting trajectories for {num_skills} skills...")
    if record_video:
        print("📹 Video recording enabled!")
    
    all_trajectories = []
    all_phis = []
    all_rewards = []
    
    device = next(policy.parameters()).device
    
    for skill_idx in range(num_skills):
        # One-hot encode skill
        skill = np.zeros(num_skills)
        skill[skill_idx] = 1.0
        
        print(f"  Skill {skill_idx}/{num_skills}...", end='')
        
        skill_trajectories = []
        skill_phis = []
        skill_rewards = []
        
        for episode in range(num_episodes_per_skill):
            obs = env.reset()
            trajectory = []
            episode_phis = []
            episode_reward = 0
            frames = []  # For video recording
            
            for step in range(max_steps):
                # Encode observation
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    if encoder is not None:
                        obs_encoded = encoder(obs_tensor)
                    else:
                        obs_encoded = obs_tensor
                    
                    # Get phi representation
                    phi = traj_encoder(obs_encoded).mean.cpu().numpy()[0]
                    episode_phis.append(phi)
                    
                    # Get action
                    obs_with_skill = np.concatenate([obs_encoded.cpu().numpy()[0], skill])
                    policy_output = policy(torch.from_numpy(obs_with_skill).float().to(device))
                    
                    # Handle both single distribution and tuple (dist, info) returns
                    if isinstance(policy_output, tuple):
                        action_dist = policy_output[0]
                    else:
                        action_dist = policy_output
                    
                    action = action_dist.sample().cpu().numpy()
                    if action.ndim > 0:
                        action = action.item() if action.size == 1 else int(np.argmax(action))
                
                # Record frame for video (only first episode of each skill)
                if record_video and episode == 0 and step % 2 == 0:  # Record every 2nd frame to reduce size
                    # Get RGB frame from environment
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                
                # Step environment
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Store state info (for visualization later)
                trajectory.append({
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                })
                
                obs = next_obs
                
                if done:
                    break
            
            # Save video for first episode of this skill
            if record_video and episode == 0 and len(frames) > 0 and output_dir is not None:
                video_path = output_dir / f"skill_{skill_idx:02d}_reward_{episode_reward:.0f}.mp4"
                try:
                    imageio.mimsave(video_path, frames, fps=30)
                except Exception as e:
                    print(f" [video save failed: {e}]", end='')
            
            skill_trajectories.append(trajectory)
            skill_phis.append(np.array(episode_phis))
            skill_rewards.append(episode_reward)
        
        all_trajectories.append(skill_trajectories)
        all_phis.append(skill_phis)
        all_rewards.append(np.mean(skill_rewards))
        
        print(f" avg_reward={np.mean(skill_rewards):.1f}")
    
    return all_trajectories, all_phis, all_rewards


def visualize_phi_space(all_phis, num_skills, output_dir):
    """Visualize phi representations in 2D using PCA."""
    print("\nVisualizing phi space...")
    
    # Flatten all phis
    all_phis_flat = []
    skill_labels = []
    
    for skill_idx, skill_episodes in enumerate(all_phis):
        for episode_phis in skill_episodes:
            all_phis_flat.extend(episode_phis)
            skill_labels.extend([skill_idx] * len(episode_phis))
    
    all_phis_flat = np.array(all_phis_flat)
    skill_labels = np.array(skill_labels)
    
    # PCA to 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    phis_2d = pca.fit_transform(all_phis_flat)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    cmap = cm.get_cmap('tab20' if num_skills > 10 else 'tab10')
    
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
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('Learned Skill Representations (φ) in 2D', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'phi_space_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_skill_rewards(all_rewards, output_dir):
    """Bar plot of average rewards per skill."""
    print("\nVisualizing skill rewards...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    skills = np.arange(len(all_rewards))
    colors = cm.get_cmap('tab20' if len(all_rewards) > 10 else 'tab10')(skills / len(all_rewards))
    
    bars = ax.bar(skills, all_rewards, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, reward) in enumerate(zip(bars, all_rewards)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
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
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_phi_evolution(all_phis, num_skills, output_dir):
    """Show how phi changes over episode for each skill."""
    print("\nVisualizing phi evolution...")
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    cmap = cm.get_cmap('tab20' if num_skills > 10 else 'tab10')
    
    for skill_idx in range(min(num_skills, 16)):
        ax = axes[skill_idx]
        
        # Plot all episodes for this skill
        for episode_idx, episode_phis in enumerate(all_phis[skill_idx]):
            # Compute phi norm over time
            phi_norms = np.linalg.norm(episode_phis, axis=1)
            timesteps = np.arange(len(phi_norms))
            
            ax.plot(timesteps, phi_norms, 
                   alpha=0.5, 
                   color=cmap(skill_idx),
                   linewidth=1)
        
        ax.set_title(f'Skill {skill_idx}', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('||φ||')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.5])  # Since phi should be normalized to ~1
    
    # Hide unused subplots
    for idx in range(num_skills, 16):
        axes[idx].axis('off')
    
    plt.suptitle('Phi Norm Evolution per Skill', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'phi_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    import argparse
    
    # Configuration
    parser = argparse.ArgumentParser(description='Visualize METRA skills on Atari')
    parser.add_argument('--exp_dir', type=str, 
                        default="exp/FULL_DETACH/sd042_1771094518_atari_mspacman_metra",
                        help='Path to experiment directory')
    parser.add_argument('--checkpoint_epoch', type=int, default=250, 
                        help='Epoch to load checkpoint from')
    parser.add_argument('--num_skills', type=int, default=16,
                        help='Number of skills')
    parser.add_argument('--num_episodes', type=int, default=3,
                        help='Episodes per skill')
    parser.add_argument('--record_videos', action='store_true',
                        help='Record videos of skill gameplay')
    
    args = parser.parse_args()
    
    exp_dir = args.exp_dir
    checkpoint_epoch = args.checkpoint_epoch
    num_skills = args.num_skills
    num_episodes_per_skill = args.num_episodes
    max_steps = 500
    
    output_dir = Path(exp_dir) / "skill_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Create video directory if recording
    video_dir = None
    if args.record_videos:
        video_dir = Path(exp_dir) / 'skill_videos'
        video_dir.mkdir(exist_ok=True)
        print(f"\n📹 Videos will be saved to: {video_dir}")
    
    print("="*60)
    print("METRA Skill Visualization")
    print("="*60)
    print(f"Experiment: {exp_dir}")
    print(f"Checkpoint: epoch {checkpoint_epoch if checkpoint_epoch else 'latest'}")
    print(f"Skills: {num_skills}")
    print(f"Episodes per skill: {num_episodes_per_skill}")
    if args.record_videos:
        print(f"Recording videos: YES")
    print("="*60)
    
    # Load checkpoint
    data, epoch = load_checkpoint(exp_dir, checkpoint_epoch)
    
    # Extract models (this depends on how they're saved in your pickle)
    # You may need to adjust these keys based on your actual checkpoint structure
    try:
        # Try to get the algorithm object
        algo = data.get('algo', None)
        if algo is None:
            print("Warning: Could not find 'algo' in checkpoint, trying alternative keys...")
            print("Available keys:", list(data.keys()))
            # Try alternative extraction
            policy = data.get('policy', None)
            encoder = data.get('encoder', None)
            traj_encoder = data.get('traj_encoder', None)
        else:
            policy = algo.option_policy
            traj_encoder = algo.traj_encoder
            encoder = algo.cnn_encoder if hasattr(algo, 'cnn_encoder') else None
    except Exception as e:
        print(f"Error extracting models: {e}")
        print("Checkpoint keys:", list(data.keys()))
        return
    
    if policy is None or traj_encoder is None:
        print("ERROR: Could not extract policy or traj_encoder from checkpoint!")
        print("Available keys:", list(data.keys()))
        if algo is not None:
            print("Algo attributes:", dir(algo))
        return
    
    # Move to eval mode
    policy.eval()
    traj_encoder.eval()
    if encoder is not None:
        encoder.eval()
    
    device = torch.device('cpu')  # Use CPU for visualization
    policy = policy.to(device)
    traj_encoder = traj_encoder.to(device)
    if encoder is not None:
        encoder = encoder.to(device)
    
    # Setup environment
    env = setup_env()
    
    # Collect trajectories
    all_trajectories, all_phis, all_rewards = collect_skill_trajectories(
        env, policy, traj_encoder, encoder, num_skills, num_episodes_per_skill, max_steps,
        record_video=args.record_videos, output_dir=video_dir
    )
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    visualize_phi_space(all_phis, num_skills, output_dir)
    visualize_skill_rewards(all_rewards, output_dir)
    visualize_phi_evolution(all_phis, num_skills, output_dir)
    
    print("\n" + "="*60)
    print("✓ Visualization complete!")
    print(f"✓ Outputs saved to: {output_dir}")
    if args.record_videos:
        print(f"✓ Videos saved to: {video_dir}")
    print("="*60)
    print("\nGenerated files:")
    print("  - phi_space_2d.png      : 2D PCA projection of learned representations")
    print("  - skill_rewards.png     : Average reward per skill")
    print("  - phi_evolution.png     : How phi norm changes during episodes")
    if args.record_videos:
        print(f"  - {num_skills} video files  : Gameplay of each skill")


if __name__ == "__main__":
    main()
