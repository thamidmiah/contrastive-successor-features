#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import dowel_wrapper

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import os
import imageio
import json
from datetime import datetime
from itertools import combinations

from envs.atari.atari_env import AtariEnv
from envs.atari.montezuma_room1_wrapper import MontezumaRoom1Wrapper

def load_checkpoint(exp_dir, epoch=None):
    """Load trained models from checkpoint."""
    exp_path = Path(exp_dir)
    if epoch is not None:
        itr_path = exp_path / f"itr_{epoch}.pkl"
        if not itr_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {itr_path}")
    else:
        itr_files = sorted(exp_path.glob("itr_*.pkl"))
        if not itr_files:
            raise FileNotFoundError(f"No checkpoints found in {exp_path}")
        itr_path = itr_files[-1]
        epoch = int(itr_path.stem.split('_')[1])
    
    with open(itr_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded checkpoint epoch {epoch}")
    return data, epoch


def make_env(noop_max=30):
    base_env = AtariEnv(
        game='MontezumaRevenge',
        frame_stack=4,
        screen_size=84,
        grayscale=True,
        normalize_pixels=True,
    )
    env = MontezumaRoom1Wrapper(base_env, noop_max=noop_max)
    return env


def prepare_algo(data):
    algo = data.get('algo')
    if algo is None:
        raise ValueError("'algo' not found in checkpoint. Keys: " + str(list(data.keys())))
    
    device = torch.device('cpu')
    algo.option_policy = algo.option_policy.to(device)
    algo.option_policy.eval()
    algo.traj_encoder = algo.traj_encoder.to(device)
    algo.traj_encoder.eval()
    if hasattr(algo, 'cnn_encoder') and algo.cnn_encoder is not None:
        algo.cnn_encoder = algo.cnn_encoder.to(device)
        algo.cnn_encoder.eval()
    return algo

@torch.no_grad()
def rollout_skill(algo, env, skill_idx, max_steps=500, record_video=True,
                  option_vector=None):
    """
    Roll out a single skill for one episode.
    
    Args:
        option_vector: If provided, use this z vector directly (for continuous z).
                       If None, use one-hot encoding of skill_idx (discrete z).
    
    Returns dict with: obs, phis, positions, rewards, frames, length, total_reward
    """
    policy = algo.option_policy
    traj_encoder = algo.traj_encoder
    cnn_encoder = algo.cnn_encoder if hasattr(algo, 'cnn_encoder') else None
    num_skills = algo.dim_option
    device = torch.device('cpu')

    # Skill vector: use provided continuous z, or fall back to one-hot
    if option_vector is not None:
        option = option_vector.astype(np.float32)
    else:
        option = np.zeros(num_skills, dtype=np.float32)
        option[skill_idx] = 1.0

    obs = env.reset()
    obs_flat = obs.flatten().astype(np.float32)

    episode_phis = []
    episode_rewards = []
    episode_positions = []
    frames = []

    for step in range(max_steps):
        # Compute phi
        obs_tensor = torch.as_tensor(obs_flat).float().unsqueeze(0)
        if cnn_encoder is not None:
            obs_4d = obs_tensor.view(1, 4, 84, 84).to(device)
            encoded = cnn_encoder(obs_4d)
        else:
            encoded = obs_tensor.to(device)
        phi = traj_encoder(encoded).mean.cpu().numpy().squeeze(0)
        episode_phis.append(phi.copy())

        # Get action (deterministic — force_use_mode_actions)
        policy_input = np.concatenate([obs_flat, option])
        action, _ = policy.get_action(policy_input)

        # Discrete action
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action_int = int(action)
            elif action.size == 1:
                action_int = int(action.item())
            else:
                action_int = int(np.argmax(action))
        else:
            action_int = int(action)

        # Record frame
        if record_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Step
        next_obs, reward, done, info = env.step(action_int)
        episode_rewards.append(reward)

        if 'player_x' in info:
            episode_positions.append([info['player_x'], info['player_y']])

        obs = next_obs
        obs_flat = obs.flatten().astype(np.float32)

        if done:
            break

    return {
        'phis': np.array(episode_phis),
        'rewards': episode_rewards,
        'total_reward': sum(episode_rewards),
        'length': len(episode_rewards),
        'positions': np.array(episode_positions) if episode_positions else None,
        'frames': frames,
    }


def run_evaluation(algo, mode, num_skills, episodes_per_option, max_steps, seed,
                   option_vectors=None):
    """
    Run full evaluation for one mode.
    
    Args:
        mode: 'deterministic' or 'randomised'
        option_vectors: If provided, list of (label, z_vector) tuples for continuous z.
                        If None, uses discrete one-hot skills.
    
    Returns:
        results: dict[skill_idx] -> list of episode dicts
    """
    is_det = (mode == 'deterministic')
    noop_max = 0 if is_det else 30
    
    env = make_env(noop_max=noop_max)
    
    # Force deterministic actions
    old_force = algo.option_policy._force_use_mode_actions
    algo.option_policy._force_use_mode_actions = True

    results = {}
    
    # Determine what skills to evaluate
    if option_vectors is not None:
        skill_list = list(range(len(option_vectors)))
    else:
        skill_list = list(range(num_skills))
    
    for skill_idx in skill_list:
        results[skill_idx] = []
        z_vec = option_vectors[skill_idx][1] if option_vectors is not None else None
        z_label = option_vectors[skill_idx][0] if option_vectors is not None else f"skill {skill_idx}"
        
        for ep in range(episodes_per_option):
            # Seed control: deterministic mode uses fixed seed per (skill, ep)
            if is_det:
                ep_seed = seed + skill_idx * 1000 + ep
                np.random.seed(ep_seed)
                torch.manual_seed(ep_seed)
            else:
                ep_seed = seed + skill_idx * 1000 + ep + 50000
                np.random.seed(ep_seed)
                torch.manual_seed(ep_seed)

            result = rollout_skill(algo, env, skill_idx, max_steps=max_steps,
                                   record_video=True, option_vector=z_vec)
            results[skill_idx].append(result)
            
            pos_str = ""
            if result['positions'] is not None and len(result['positions']) > 0:
                final_pos = result['positions'][-1]
                pos_str = f"  final_pos=({final_pos[0]}, {final_pos[1]})"
            
            print(f"    {z_label} ep {ep}: len={result['length']:3d}  "
                  f"reward={result['total_reward']:.0f}  "
                  f"phi_norm={np.linalg.norm(result['phis'][-1]):.3f}{pos_str}")
    
    algo.option_policy._force_use_mode_actions = old_force
    env.close()
    
    return results


def compute_separability_metrics(results, num_skills):
    centroids = {}
    intra_vars = {}
    coverage = {}
    lengths = {}
    
    for skill_idx in range(num_skills):
        final_phis = []
        all_positions = set()
        ep_lengths = []
        
        for ep_result in results[skill_idx]:
            final_phis.append(ep_result['phis'][-1])
            ep_lengths.append(ep_result['length'])
            
            if ep_result['positions'] is not None:
                for pos in ep_result['positions']:
                    # Quantize to grid cells for coverage
                    all_positions.add((int(pos[0]) // 5, int(pos[1]) // 5))
        
        final_phis = np.array(final_phis)
        centroids[skill_idx] = final_phis.mean(axis=0)
        intra_vars[skill_idx] = final_phis.var(axis=0).sum()
        coverage[skill_idx] = len(all_positions)
        lengths[skill_idx] = np.mean(ep_lengths)
    
    # Pairwise distances between centroids
    pair_dists = {}
    for i, j in combinations(range(num_skills), 2):
        d = np.linalg.norm(centroids[i] - centroids[j])
        pair_dists[(i, j)] = d
    
    mean_pair_dist = np.mean(list(pair_dists.values())) if pair_dists else 0.0
    mean_intra_var = np.mean(list(intra_vars.values()))
    total_coverage = sum(coverage.values())
    
    return {
        'mean_pairwise_phi_dist': mean_pair_dist,
        'per_pair_distances': pair_dists,
        'mean_intra_variance': mean_intra_var,
        'coverage_per_skill': coverage,
        'total_coverage': total_coverage,
        'mean_lengths': lengths,
        'centroids': centroids,
    }

def save_videos(results, num_skills, out_dir):
    """Save ONE video per skill (the longest episode)."""
    video_dir = out_dir / 'videos'
    video_dir.mkdir(exist_ok=True)
    
    for skill_idx in range(num_skills):
        # Pick the episode with the most frames
        best_ep = max(results[skill_idx],
                      key=lambda ep: len(ep.get('frames', [])))
        frames = best_ep.get('frames', [])
        if not frames:
            continue
        path = video_dir / f'skill{skill_idx}.mp4'
        try:
            imageio.mimsave(str(path), frames, fps=30)
        except Exception as e:
            print(f"    [video save failed: {e}]")
    
    print(f"  ✓ Videos saved to {video_dir} (1 per skill)")


def make_montage(results, num_skills, episodes_per_option, out_dir, mode_name, n_cols=6):
    grid = []          # grid[skill_idx] = list of n_cols frames (or None)
    col_timesteps = []  # will be set from the first skill that has frames

    for skill_idx in range(num_skills):
        best_frames = []
        for ep_idx in range(episodes_per_option):
            frames = results[skill_idx][ep_idx].get('frames', [])
            if len(frames) > len(best_frames):
                best_frames = frames

        if not best_frames:
            grid.append([None] * n_cols)
            continue

        n = len(best_frames)
        # Uniform indices spanning [0, n-1]
        if n_cols == 1:
            indices = [n // 2]
        else:
            indices = [int(round(i * (n - 1) / (n_cols - 1))) for i in range(n_cols)]

        row = [best_frames[idx] for idx in indices]
        grid.append(row)

        # Record timestep labels from the first skill that has data
        if not col_timesteps:
            col_timesteps = indices

    # Fill default timestep labels if nothing was set
    if not col_timesteps:
        col_timesteps = list(range(n_cols))

    # Find any valid frame to get dimensions
    any_frame = None
    for row in grid:
        for f in row:
            if f is not None:
                any_frame = f
                break
        if any_frame is not None:
            break

    if any_frame is None:
        print("  [skip] No frames for montage")
        return

    h, w = any_frame.shape[:2]
    pad = 4          # pixels between cells
    header_px = 22  # extra space at top for timestep labels

    rows = num_skills
    cols = n_cols
    montage_h = rows * h + (rows - 1) * pad
    montage_w = cols * w + (cols - 1) * pad
    # Canvas with room for column headers
    canvas_h = montage_h + header_px
    canvas = np.ones((canvas_h, montage_w, 3), dtype=np.uint8) * 40  # dark grey bg

    cmap_fn = cm.get_cmap('tab10')

    for r in range(rows):
        for c in range(cols):
            frame = grid[r][c]
            if frame is None:
                continue
            y0 = header_px + r * (h + pad)
            x0 = c * (w + pad)
            color = np.array(cmap_fn(r)[:3]) * 255
            bordered = frame.copy()
            bordered[:2, :] = color
            bordered[-2:, :] = color
            bordered[:, :2] = color
            bordered[:, -2:] = color
            canvas[y0:y0+h, x0:x0+w] = bordered

    fig, ax = plt.subplots(1, 1, figsize=(2.2 * cols + 1, 2 * rows + 1.2))
    ax.imshow(canvas)
    ax.set_title(f'Skill Montage — {mode_name}\n(rows = skills, cols = time snapshots)',
                 fontweight='bold')
    ax.axis('off')

    # Column headers: timestep labels centred on each column
    for c, t in enumerate(col_timesteps):
        x_center = c * (w + pad) + w // 2
        ax.text(x_center, header_px // 2, f't={t}',
                va='center', ha='center', fontsize=9, color='white',
                fontweight='bold')

    # Row labels (skill index, coloured)
    for r in range(rows):
        y_center = header_px + r * (h + pad) + h // 2
        ax.text(-10, y_center, f'S{r}', va='center', ha='right',
                fontsize=11, fontweight='bold', color=cmap_fn(r))

    plt.tight_layout()
    path = out_dir / f'montage_{mode_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Montage saved: {path.name}")


def plot_phi_space(results, num_skills, out_dir, mode_name):
    """Scatter plot of phi embeddings in 2D."""
    all_phis = []
    labels = []
    
    for skill_idx in range(num_skills):
        for ep_result in results[skill_idx]:
            all_phis.append(ep_result['phis'])
            labels.extend([skill_idx] * len(ep_result['phis']))
    
    all_phis = np.concatenate(all_phis, axis=0)
    labels = np.array(labels)
    dim = all_phis.shape[1]
    
    if dim <= 2:
        phis_2d = all_phis[:, :2]
        xlabel, ylabel = 'φ₀', 'φ₁'
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        phis_2d = pca.fit_transform(all_phis)
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_name = 'tab10' if num_skills <= 10 else 'tab20'
    cmap_fn = cm.get_cmap(cmap_name)
    
    for s in range(num_skills):
        mask = labels == s
        ax.scatter(phis_2d[mask, 0], phis_2d[mask, 1],
                   c=[cmap_fn(s % cmap_fn.N)], label=f'Skill {s}', alpha=0.4, s=15)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'φ-space — {mode_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = out_dir / f'phi_space_{mode_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Phi space: {path.name}")


def plot_phi_trajectory_arrows(results, num_skills, out_dir, mode_name):
    """
    Mean φ-direction arrow per skill.

    For each skill we compute the average start-φ and average end-φ across
    all episodes, project into the top-2 PCA dimensions, and draw one bold
    arrow per skill.  Individual episode lines are shown faintly behind.
    """
    from sklearn.decomposition import PCA

    # ── Collect start / end phi for every trajectory ──
    starts, ends, skill_ids = [], [], []
    for s in range(num_skills):
        for ep in results[s]:
            phis = ep['phis']
            if len(phis) < 2:
                continue
            starts.append(phis[0])
            ends.append(phis[-1])
            skill_ids.append(s)

    if not starts:
        return

    starts = np.array(starts)
    ends   = np.array(ends)
    skill_ids = np.array(skill_ids)
    dim = starts.shape[1]

    # ── PCA projection ──
    all_pts = np.concatenate([starts, ends], axis=0)
    if dim > 2:
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_pts)
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
    else:
        all_2d = all_pts[:, :2]
        xlabel, ylabel = 'φ₀', 'φ₁'

    n = len(starts)
    starts_2d = all_2d[:n]
    ends_2d   = all_2d[n:]

    mean_displacements_2d = np.zeros((num_skills, 2))
    raw_magnitudes = np.zeros(num_skills)          # full-dim magnitude
    for s in range(num_skills):
        mask = skill_ids == s
        if mask.sum() == 0:
            continue
        mean_start_2d = starts_2d[mask].mean(axis=0)
        mean_end_2d   = ends_2d[mask].mean(axis=0)
        mean_displacements_2d[s] = mean_end_2d - mean_start_2d
        # Full-dim magnitude (not PCA-projected) for the annotation
        raw_magnitudes[s] = np.linalg.norm(
            ends[mask].mean(axis=0) - starts[mask].mean(axis=0)
        )

    # ── Plot: unit-length arrows from origin ──
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap_fn = cm.get_cmap('hsv', num_skills + 1)

    for s in range(num_skills):
        if (skill_ids == s).sum() == 0:
            continue
        d = mean_displacements_2d[s]
        length = np.linalg.norm(d)
        if length < 1e-12:
            continue                               
        direction = d / length                    
        ax.annotate('',
                    xy=(direction[0], direction[1]),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=cmap_fn(s),
                                   lw=2.5, mutation_scale=15))
        ax.plot([], [], color=cmap_fn(s), linewidth=2.5,
                label=f'Skill {s}')

    angles = []
    for s in range(num_skills):
        d = mean_displacements_2d[s]
        if np.linalg.norm(d) > 1e-12:
            angles.append(np.arctan2(d[1], d[0]))
    if len(angles) >= 2:
        angles_sorted = np.sort(angles)
        gaps = np.diff(angles_sorted)
        gaps = np.append(gaps, 2 * np.pi - (angles_sorted[-1] - angles_sorted[0]))
        angular_spread = np.degrees(2 * np.pi - np.max(gaps))
    else:
        angular_spread = 0.0

    ax.set_xlabel('Normalised PC1 direction', fontsize=12)
    ax.set_ylabel('Normalised PC2 direction', fontsize=12)
    ax.set_title('Skill φ-Directions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=1, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_aspect('equal')
    plt.tight_layout()

    path = out_dir / f'phi_arrows_{mode_name}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Phi trajectory arrows: {path.name}")


def plot_phi_dimensions(results, num_skills, out_dir, mode_name):
    """Box plot per phi dimension, grouped by skill."""
    per_skill = {}
    dim = None
    for s in range(num_skills):
        stacked = np.concatenate([ep['phis'] for ep in results[s]], axis=0)
        per_skill[s] = stacked
        dim = stacked.shape[1]
    
    fig, axes = plt.subplots(1, dim, figsize=(5 * dim, 6), squeeze=False)
    axes = axes.flatten()
    cmap_fn = cm.get_cmap('tab10')
    
    for d in range(dim):
        ax = axes[d]
        data = [per_skill[s][:, d] for s in range(num_skills)]
        bp = ax.boxplot(data, labels=[f'S{s}' for s in range(num_skills)],
                        patch_artist=True, showfliers=False)
        for patch, s in zip(bp['boxes'], range(num_skills)):
            patch.set_facecolor(cmap_fn(s))
            patch.set_alpha(0.6)
        ax.set_title(f'φ[{d}]', fontweight='bold')
        ax.set_xlabel('Skill')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Per-Dimension φ Distribution — {mode_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / f'phi_dims_{mode_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Phi dimensions: {path.name}")


def plot_heatmaps(results, num_skills, out_dir, mode_name):
    """Position heatmaps per skill."""
    has_pos = any(
        ep['positions'] is not None and len(ep['positions']) > 0
        for s in range(num_skills) for ep in results[s]
    )
    if not has_pos:
        return
    
    cols = min(num_skills, 4)
    rows = (num_skills + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    cmap_fn = cm.get_cmap('tab10')
    
    for s in range(num_skills):
        r, c = divmod(s, cols)
        ax = axes[r][c]
        
        all_pos = []
        for ep in results[s]:
            if ep['positions'] is not None and len(ep['positions']) > 0:
                all_pos.append(ep['positions'])
        
        if not all_pos:
            ax.set_visible(False)
            continue
        
        all_pos = np.concatenate(all_pos, axis=0)
        hm, xedges, yedges = np.histogram2d(
            all_pos[:, 0], all_pos[:, 1],
            bins=32, range=[[0, 160], [0, 255]]
        )
        hm = np.log1p(hm)
        
        im = ax.imshow(hm.T, origin='lower', aspect='auto',
                       extent=[0, 160, 0, 255], cmap='hot', interpolation='nearest')
        avg_len = np.mean([ep['length'] for ep in results[s]])
        ax.set_title(f'Skill {s} (avg {avg_len:.0f} steps)', fontsize=10, color=cmap_fn(s))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    for idx in range(num_skills, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    
    fig.suptitle(f'Position Heatmaps — {mode_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / f'heatmaps_{mode_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Heatmaps: {path.name}")


def plot_trajectory_traces(results, num_skills, out_dir, mode_name):
    """Plot x-y position traces for each skill (overlaid episodes)."""
    has_pos = any(
        ep['positions'] is not None and len(ep['positions']) > 0
        for s in range(num_skills) for ep in results[s]
    )
    if not has_pos:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_fn = cm.get_cmap('tab10')
    
    for s in range(num_skills):
        for ep_idx, ep in enumerate(results[s]):
            if ep['positions'] is None or len(ep['positions']) == 0:
                continue
            pos = ep['positions']
            label = f'Skill {s}' if ep_idx == 0 else None
            ax.plot(pos[:, 0], pos[:, 1], color=cmap_fn(s), alpha=0.5,
                    linewidth=1.5, label=label)
            # Mark start and end
            ax.scatter(pos[0, 0], pos[0, 1], color=cmap_fn(s), marker='o',
                       s=40, edgecolors='black', linewidths=0.5, zorder=5)
            ax.scatter(pos[-1, 0], pos[-1, 1], color=cmap_fn(s), marker='x',
                       s=40, linewidths=1.5, zorder=5)
    
    ax.set_xlabel('Player X', fontsize=12)
    ax.set_ylabel('Player Y', fontsize=12)
    ax.set_title(f'Skill Trajectory Traces — {mode_name}\n(○=start, ×=end)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 255)
    plt.tight_layout()
    
    path = out_dir / f'traces_{mode_name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Trajectory traces: {path.name}")


def write_report(metrics_det, metrics_rand, num_skills, epoch, out_dir,
                 skill_labels=None):
    """Write a text summary report."""
    def skill_name(s):
        if skill_labels and s < len(skill_labels):
            return skill_labels[s]
        return f"Skill {s}"
    
    lines = []
    lines.append("=" * 60)
    lines.append("SKILL EVALUATION REPORT")
    lines.append(f"Checkpoint epoch: {epoch}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Number of skills: {num_skills}")
    lines.append("=" * 60)
    
    for name, metrics in [("DETERMINISTIC", metrics_det), ("RANDOMISED", metrics_rand)]:
        if metrics is None:
            continue
        lines.append(f"\n--- {name} MODE ---")
        lines.append(f"  Mean pairwise φ distance:  {metrics['mean_pairwise_phi_dist']:.4f}")
        lines.append(f"  Mean intra-skill variance: {metrics['mean_intra_variance']:.6f}")
        lines.append(f"  Total state coverage:      {metrics['total_coverage']} cells")
        
        lines.append(f"\n  Per-skill breakdown:")
        for s in range(num_skills):
            cov = metrics['coverage_per_skill'].get(s, 0)
            avg_len = metrics['mean_lengths'].get(s, 0)
            c = metrics['centroids'][s]
            lines.append(f"    {skill_name(s)}: coverage={cov:3d} cells, "
                         f"avg_len={avg_len:.0f}, "
                         f"centroid=[{', '.join(f'{v:.3f}' for v in c)}]")
        
        lines.append(f"\n  Pairwise φ distances:")
        for (i, j), d in sorted(metrics['per_pair_distances'].items()):
            lines.append(f"    {skill_name(i)} <-> {skill_name(j)}: {d:.4f}")
    
    lines.append("\n" + "=" * 60)
    
    report = "\n".join(lines)
    path = out_dir / "evaluation_report.txt"
    with open(path, 'w') as f:
        f.write(report)
    
    print(f"\n{report}")
    print(f"\n  ✓ Report saved: {path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate learned skills — deterministic & randomised modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Both modes, epoch 500
  python evaluate_skills.py --exp_dir exp/MontezumaRoom1-v2/sd000_... --checkpoint_epoch 500

  # Deterministic only, more episodes
  python evaluate_skills.py --exp_dir ... --checkpoint_epoch 500 --mode deterministic --episodes_per_option 10

  # Compare multiple checkpoints
  for e in 100 200 400 600; do
    python evaluate_skills.py --exp_dir ... --checkpoint_epoch $e --mode deterministic
  done
        """,
    )
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--checkpoint_epoch', type=int, default=None,
                        help='Epoch to load (default: latest)')
    parser.add_argument('--episodes_per_option', type=int, default=5,
                        help='Number of episodes per skill (default: 5)')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Max steps per episode (default: 500)')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['deterministic', 'randomised', 'both'],
                        help='Evaluation mode (default: both)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--continuous', action='store_true', default=False,
                        help='Evaluate with continuous z (random unit-sphere directions). '
                             'Use for models trained with --discrete 0.')
    parser.add_argument('--num_continuous_skills', type=int, default=16,
                        help='Number of random z directions to evaluate (only with --continuous)')
    
    args = parser.parse_args()
    
    # ── Load checkpoint ──
    print("\n" + "=" * 60)
    print("SKILL EVALUATION HARNESS")
    print("=" * 60)
    
    data, epoch = load_checkpoint(args.exp_dir, args.checkpoint_epoch)
    algo = prepare_algo(data)
    dim_option = algo.dim_option
    
    # ── Build skill z-vectors ──
    # Continuous: random directions on the unit sphere (matching training distribution)
    # Discrete:   dim_option one-hot vectors (handled inside rollout_skill)
    if args.continuous:
        n = args.num_continuous_skills
        rng = np.random.RandomState(args.seed)  # reproducible
        raw = rng.randn(n, dim_option).astype(np.float32)
        raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)  # unit sphere
        option_vectors = [(f"z{i}", raw[i]) for i in range(n)]
        num_skills = n
        print(f"  Continuous z: {num_skills} random unit-sphere directions (dim={dim_option})")
        for label, vec in option_vectors:
            print(f"    {label}: [{', '.join(f'{v:.2f}' for v in vec)}]")
    else:
        option_vectors = None
        num_skills = dim_option
    
    print(f"  Skills:     {num_skills}")
    print(f"  Episodes:   {args.episodes_per_option} per skill")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Mode:       {args.mode}")
    print(f"  Seed:       {args.seed}")
    if args.continuous:
        print(f"  Z type:     continuous (unit-sphere)")
    else:
        print(f"  Z type:     discrete (one-hot)")
    
    # Output directory
    out_base = Path(args.exp_dir) / "skill_eval" / f"epoch_{epoch}"
    
    metrics_det = None
    metrics_rand = None
    
    # ── DETERMINISTIC MODE ──
    if args.mode in ('deterministic', 'both'):
        print(f"\n{'─' * 60}")
        print(f"MODE A: DETERMINISTIC (noop=0, fixed seed)")
        print(f"{'─' * 60}")
        
        out_det = out_base / "deterministic"
        out_det.mkdir(parents=True, exist_ok=True)
        
        results_det = run_evaluation(
            algo, 'deterministic', num_skills,
            args.episodes_per_option, args.max_steps, args.seed,
            option_vectors=option_vectors
        )
        
        print(f"\n  Generating visualisations...")
        save_videos(results_det, num_skills, out_det)
        make_montage(results_det, num_skills, args.episodes_per_option, out_det, "deterministic")
        plot_phi_space(results_det, num_skills, out_det, "deterministic")
        plot_phi_trajectory_arrows(results_det, num_skills, out_det, "deterministic")
        plot_phi_dimensions(results_det, num_skills, out_det, "deterministic")
        plot_heatmaps(results_det, num_skills, out_det, "deterministic")
        plot_trajectory_traces(results_det, num_skills, out_det, "deterministic")
        
        metrics_det = compute_separability_metrics(results_det, num_skills)
    
    # ── RANDOMISED MODE ──
    if args.mode in ('randomised', 'both'):
        print(f"\n{'─' * 60}")
        print(f"MODE B: RANDOMISED (noop_max=30, varied seeds)")
        print(f"{'─' * 60}")
        
        out_rand = out_base / "randomised"
        out_rand.mkdir(parents=True, exist_ok=True)
        
        results_rand = run_evaluation(
            algo, 'randomised', num_skills,
            args.episodes_per_option, args.max_steps, args.seed,
            option_vectors=option_vectors
        )
        
        print(f"\n  Generating visualisations...")
        save_videos(results_rand, num_skills, out_rand)
        make_montage(results_rand, num_skills, args.episodes_per_option, out_rand, "randomised")
        plot_phi_space(results_rand, num_skills, out_rand, "randomised")
        plot_phi_trajectory_arrows(results_rand, num_skills, out_rand, "randomised")
        plot_phi_dimensions(results_rand, num_skills, out_rand, "randomised")
        plot_heatmaps(results_rand, num_skills, out_rand, "randomised")
        plot_trajectory_traces(results_rand, num_skills, out_rand, "randomised")
        
        metrics_rand = compute_separability_metrics(results_rand, num_skills)
    
    # ── REPORT ──
    skill_labels = [ov[0] for ov in option_vectors] if option_vectors else None
    write_report(metrics_det, metrics_rand, num_skills, epoch, out_base,
                 skill_labels=skill_labels)
    
    print(f"\n  All outputs in: {out_base}")
    print("  Done! ✓")


if __name__ == "__main__":
    main()
