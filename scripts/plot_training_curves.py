import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str,
    default="exp/Montezuma-CSF-Dim8/sd000_1773275656_montezuma_room1_metra_sf")
parser.add_argument("--method", type=str, default="ViSR")
parser.add_argument("--best_epoch", type=int, default=300)
args = parser.parse_args()

csv_path = Path(args.exp_dir) / "progress.csv"
df = pd.read_csv(csv_path)
epoch = df["TotalEpoch"]

out_dir = Path(args.exp_dir) / "plots"
out_dir.mkdir(exist_ok=True)

METHOD = args.method
BEST_EPOCH = args.best_epoch

# Detect column prefix
if "TrainSp/metra_sf/SfQMean" in df.columns:
    PREFIX = "TrainSp/metra_sf"
else:
    PREFIX = "TrainSp/metra"

def col(name):
    """Return column name with correct prefix, or None if not present."""
    full = f"{PREFIX}/{name}"
    return full if full in df.columns else None

EP_COLOUR = "crimson"


def mark_best(ax):
    ylims = ax.get_ylim()
    ax.axvline(BEST_EPOCH, color=EP_COLOUR, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(BEST_EPOCH + 2, ylims[0] + 0.04 * (ylims[1] - ylims[0]),
            f"ep{BEST_EPOCH}", fontsize=7, color=EP_COLOUR, va="bottom")


# ── 1. Overview — 6 meaningful panels ─────────────────────────────────────────
#   Good-news metrics only: things that should increase or become more negative
fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

panels = [
    (0, 0, "SfQMean",        "SF-Q mean",         "Successor Feature Q-value ↑",   "mediumpurple",  False),
    (0, 1, "LossTe",         "−LossTe",           "Skill Separation (−LossTe) ↑",  "steelblue",     True),
    (0, 2, "PathLengthMean", "Steps / episode",   "Mean Episode Length ↑",         "saddlebrown",   False),
    (1, 0, "LossOp",         "−LossOp",           "Policy Objective (−LossOp) ↑",  "mediumseagreen",True),
    (1, 1, "phi_l2",         "|φ| L2",            "φ L2 Norm (convergence)",       "darkorange",    False),
    (1, 2, "PhiStd",         "φ Std",             "φ Std (convergence)",           "steelblue",     False),
]

for r, c, name, ylabel, title, colour, invert in panels:
    full_col = col(name)
    ax = fig.add_subplot(gs[r, c])
    if full_col:
        series = -df[full_col] if invert else df[full_col]
        ax.plot(epoch, series, color=colour, linewidth=1.4)
    else:
        ax.text(0.5, 0.5, "N/A for this method", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25)
    mark_best(ax)

fig.suptitle(f"{METHOD} Training Curves — Montezuma Room 1", fontsize=13, fontweight="bold")
fig.savefig(out_dir / "training_curves.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: training_curves.png")

# ── 2. Skill separation: −LossTe over time ────────────────────────────────────
#   LossTe is the METRA objective; more negative = skills better separated.
#   We negate it so the curve goes UP = better, which is the natural reading.
fig, ax = plt.subplots(figsize=(8, 4))
neg_loss = -df[col("LossTe")]
ax.plot(epoch, neg_loss, color="steelblue", linewidth=1.8, label="−LossTe (skill separation)")
ax.fill_between(epoch, neg_loss.min(), neg_loss, alpha=0.15, color="steelblue")
ax.axvline(BEST_EPOCH, color=EP_COLOUR, linestyle="--", linewidth=1.2, label=f"ep{BEST_EPOCH} (best eval)")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("−Trajectory Encoder Loss", fontsize=11)
ax.set_title(f"Skill Separation Over Training — {METHOD}\n(higher = skills more distinguishable in φ-space)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
fig.savefig(out_dir / "skill_separation.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: skill_separation.png")

# ── 3. SF-Q mean (policy learning) — only for metra_sf methods ───────────────
if col("SfQMean"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epoch, df[col("SfQMean")], color="mediumpurple", linewidth=1.8, label="SF-Q mean")
    ax.fill_between(epoch, 0, df[col("SfQMean")], alpha=0.15, color="mediumpurple")
    ax.axvline(BEST_EPOCH, color=EP_COLOUR, linestyle="--", linewidth=1.2, label=f"ep{BEST_EPOCH} (best eval)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("SF-Q Mean", fontsize=11)
    ax.set_title(f"Successor Feature Q-value Over Training — {METHOD}\n(higher = policy better at predicting φ returns)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / "sfq_growth.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: sfq_growth.png")

# ── 4. Episode length + SF-Q on twin axes ─────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(epoch, df[col("PathLengthMean")], color="saddlebrown",
         linewidth=1.6, label="Mean episode length")
sfq_c = col("SfQMean")
if sfq_c:
    ax2.plot(epoch, df[sfq_c], color="mediumpurple",
             linewidth=1.6, linestyle="--", label="SF-Q mean")
ax1.axvline(BEST_EPOCH, color="grey", linestyle=":", linewidth=1.0)
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Mean Episode Length (steps)", fontsize=10, color="saddlebrown")
ax2.set_ylabel("SF-Q Mean", fontsize=10, color="mediumpurple")
ax1.set_title(f"Episode Length & SF-Q Value Over Training — {METHOD}", fontsize=12, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.2)
fig.savefig(out_dir / "episode_length_sfq.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: episode_length_sfq.png")

print("\nAll done:", out_dir)
