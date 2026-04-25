#!/usr/bin/env python3
"""
Standalone script to produce the 'Skill φ-Directions' arrow plot
from manually supplied φ start/end centroids (or just centroids).

Usage examples
--------------
# Minimal: provide per-skill centroids (arrows from origin in PCA space)
python scripts/plot_phi_arrows.py

# Custom output path
python scripts/plot_phi_arrows.py --output my_arrows.png

Edit the DATA section below to supply your own values.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# ═══════════════════════════════════════════════════════════
# ██  DATA — EDIT HERE  ██
# ═══════════════════════════════════════════════════════════
# Option A: provide per-skill START and END φ-centroids.
#           The arrow is end − start, projected to 2-D via PCA.
#
# Option B: provide only CENTROIDS (the "end" values).
#           start is assumed to be the origin (zeros).
#           This is equivalent to plotting the centroid directions.
#
# Set whichever you have; leave the other as None.

# Each key is the skill index, value is a list / array of floats.
CENTROIDS = {
    0: [0.48,  0.22,  0.03,  0.02,  0.01, -0.01,  0.00,  0.01],
    1: [0.32,  0.38,  0.02,  0.03,  0.00,  0.01,  0.01,  0.00],
    2: [0.18,  0.44,  0.01,  0.02,  0.02,  0.00,  0.00,  0.01],
    3: [0.52,  0.10,  0.02,  0.01,  0.00, -0.01,  0.01,  0.00],
    4: [0.36, -0.08,  0.02,  0.00,  0.01, -0.01,  0.00,  0.00],
    5: [0.14, -0.18,  0.01,  0.01,  0.00,  0.00,  0.01,  0.00],
    6: [0.05,  0.28,  0.02,  0.01,  0.01,  0.00,  0.00,  0.00],
    7: [-0.06, 0.20,  0.01,  0.00,  0.01,  0.01,  0.00,  0.00],
}

# If you have separate start/end φ values, fill these in instead:
START_PHIS = None   # e.g. {0: [...], 1: [...], ...}
END_PHIS   = None   # e.g. {0: [...], 1: [...], ...}

# Title (set to None for default)
TITLE = None   # e.g. "Epoch 200 — Skill φ-Directions"
# ═══════════════════════════════════════════════════════════


def build_data():
    """Return (starts, ends, skill_ids) arrays from the DATA section."""
    if START_PHIS is not None and END_PHIS is not None:
        skills = sorted(START_PHIS.keys())
        starts = np.array([START_PHIS[s] for s in skills], dtype=np.float64)
        ends   = np.array([END_PHIS[s]   for s in skills], dtype=np.float64)
    elif CENTROIDS is not None:
        skills = sorted(CENTROIDS.keys())
        ends   = np.array([CENTROIDS[s] for s in skills], dtype=np.float64)
        starts = np.zeros_like(ends)
    else:
        raise ValueError("Provide either CENTROIDS or START_PHIS + END_PHIS.")
    skill_ids = np.array(skills)
    return starts, ends, skill_ids


def plot_phi_arrows(starts, ends, skill_ids, title=None, output='phi_arrows.png',
                    dpi=150):
    """
    Reproduce the exact 'Skill φ-Directions' plot from evaluate_skills.py.
    """
    from sklearn.decomposition import PCA

    num_skills = len(skill_ids)
    dim = starts.shape[1]

    # ── PCA projection ──
    all_pts = np.concatenate([starts, ends], axis=0)
    if dim > 2:
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_pts)
    else:
        all_2d = all_pts[:, :2]

    n = len(starts)
    starts_2d = all_2d[:n]
    ends_2d   = all_2d[n:]

    # ── Per-skill mean displacement ──
    mean_displacements_2d = ends_2d - starts_2d
    raw_magnitudes = np.linalg.norm(ends - starts, axis=1)

    # ── Plot: unit-length arrows from origin ──
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap_fn = plt.get_cmap('hsv', num_skills + 1)

    for i, s in enumerate(skill_ids):
        d = mean_displacements_2d[i]
        length = np.linalg.norm(d)
        if length < 1e-12:
            continue
        direction = d / length          # unit vector
        ax.annotate('',
                    xy=(direction[0], direction[1]),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=cmap_fn(i),
                                   lw=2.5, mutation_scale=15))
        ax.plot([], [], color=cmap_fn(i), linewidth=2.5,
                label=f'Skill {s}')

    # ── Angular spread (printed to console) ──
    angles = []
    for i in range(num_skills):
        d = mean_displacements_2d[i]
        if np.linalg.norm(d) > 1e-12:
            angles.append(np.arctan2(d[1], d[0]))
    if len(angles) >= 2:
        angles_sorted = np.sort(angles)
        gaps = np.diff(angles_sorted)
        gaps = np.append(gaps, 2 * np.pi - (angles_sorted[-1] - angles_sorted[0]))
        angular_spread = np.degrees(2 * np.pi - np.max(gaps))
    else:
        angular_spread = 0.0
    print(f"Angular spread: {angular_spread:.1f}°")

    # ── Formatting (matches evaluate_skills.py exactly) ──
    ax.set_xlabel('Normalised PC1 direction', fontsize=12)
    ax.set_ylabel('Normalised PC2 direction', fontsize=12)
    ax.set_title(title or 'Skill φ-Directions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=1, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_aspect('equal')
    plt.tight_layout()

    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {output}")

    # ── Print summary table ──
    print(f"\n{'Skill':>6}  {'Raw ‖Δφ‖':>10}  {'2D angle':>10}")
    print("-" * 32)
    for i, s in enumerate(skill_ids):
        d = mean_displacements_2d[i]
        ang = np.degrees(np.arctan2(d[1], d[0]))
        print(f"  {s:>4}  {raw_magnitudes[i]:>10.4f}  {ang:>9.1f}°")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Skill φ-Directions from manually supplied phi values.")
    parser.add_argument('--output', '-o', default='phi_arrows.png',
                        help='Output image path (default: phi_arrows.png)')
    parser.add_argument('--title', '-t', default=None,
                        help='Plot title (default: "Skill φ-Directions")')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI (default: 150)')
    args = parser.parse_args()

    starts, ends, skill_ids = build_data()
    plot_phi_arrows(starts, ends, skill_ids,
                    title=args.title or TITLE,
                    output=args.output,
                    dpi=args.dpi)


if __name__ == '__main__':
    main()
