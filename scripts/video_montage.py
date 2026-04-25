#!/usr/bin/env python3
"""
Generate a temporal montage image from one or more .mp4 video files.

Takes video files and samples N uniformly-spaced frames across the timeline,
arranging them in a single row (one video) or a grid (multiple videos).
Column headers show the timestep, row labels show the video/skill name.

Usage:
  # Single video → one-row montage
  python scripts/video_montage.py path/to/skill0.mp4

  # Multiple videos → grid (one row per video)
  q

  # Whole directory of .mp4 files
  python scripts/video_montage.py path/to/videos/

  # Custom options
  python scripts/video_montage.py path/to/skill0.mp4 --n_cols 8 --output montage.png --dpi 200

  # With custom row labels
  python scripts/video_montage.py skill0.mp4 skill1.mp4 --labels "Go right" "Climb ladder"
"""

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def extract_frames(video_path, n_cols, timesteps=None, every=None):
    """Read a video and return sampled frames.
    
    Priority:
      1. timesteps   — exact frame indices, e.g. [0, 50, 100, 200, 499]
      2. every       — sample every N-th frame, e.g. every=6 → [0,6,12,...]
      3. n_cols      — N uniformly-spaced frames across full video
    """
    try:
        # imageio v3 reads all frames at once as (T, H, W, C)
        import imageio.v3 as iio
        all_frames = iio.imread(str(video_path), plugin='pyav')
        frames = [all_frames[i] for i in range(all_frames.shape[0])]
    except Exception:
        # Fallback to v2 ffmpeg reader
        reader = imageio.get_reader(str(video_path), format='ffmpeg')
        frames = []
        try:
            for frame in reader:
                frames.append(np.array(frame))
        except Exception:
            pass
        finally:
            reader.close()

    if not frames:
        return [], []

    n = len(frames)

    if timesteps is not None:
        # Exact user-specified timesteps — clamp to valid range
        indices = [min(max(0, t), n - 1) for t in timesteps]
    elif every is not None:
        # Every N-th frame
        indices = list(range(0, n, every))
    else:
        # Uniform across full video
        if n_cols == 1:
            indices = [n // 2]
        else:
            indices = [int(round(i * (n - 1) / (n_cols - 1))) for i in range(n_cols)]

    sampled = [frames[idx] for idx in indices]
    return sampled, indices


def make_montage(video_paths, labels, n_cols, output_path, dpi, title,
                 border_width=2, pad=4, header_px=24, show_timesteps=True,
                 timesteps_arg=None, every=None):
    """
    Build and save a montage grid.

    Args:
        video_paths:    list of Path objects to .mp4 files
        labels:         list of row labels (one per video)
        n_cols:         number of temporal snapshots per row
        output_path:    where to save the PNG
        dpi:            output resolution
        title:          figure title
        border_width:   coloured border around each frame (pixels)
        pad:            gap between cells (pixels)
        header_px:      space reserved at top for timestep labels
        show_timesteps: whether to add t=N column headers
        timesteps_arg:  exact frame indices to sample (overrides n_cols)
        every:          sample every N-th frame (overrides n_cols)
    """
    n_rows = len(video_paths)
    cmap_fn = plt.get_cmap('tab10' if n_rows <= 10 else 'tab20')

    # ── Extract frames ──
    grid = []         # grid[row] = list of frames
    timesteps = []    # timestep indices from the first valid video

    for vid_path in video_paths:
        sampled, indices = extract_frames(vid_path, n_cols,
                                          timesteps=timesteps_arg, every=every)
        if not sampled:
            grid.append([None] * n_cols)
            print(f"  [warn] No frames from {vid_path.name}")
            continue
        grid.append(sampled)
        if not timesteps:
            timesteps = indices

    if not timesteps:
        timesteps = list(range(n_cols))

    # When using --timesteps or --every, n_cols is determined by the data
    actual_cols = max(len(row) for row in grid) if grid else n_cols

    # Pad short rows with None so every row has the same length
    for i, row in enumerate(grid):
        if len(row) < actual_cols:
            grid[i] = row + [None] * (actual_cols - len(row))

    # ── Find frame dimensions ──
    any_frame = None
    for row in grid:
        for f in row:
            if f is not None:
                any_frame = f
                break
        if any_frame is not None:
            break

    if any_frame is None:
        print("  [error] No valid frames found in any video.")
        sys.exit(1)

    h, w = any_frame.shape[:2]

    # ── Build canvas ──
    canvas_h = header_px + n_rows * h + (n_rows - 1) * pad
    canvas_w = actual_cols * w + (actual_cols - 1) * pad
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40  # dark grey

    for r in range(n_rows):
        color = (np.array(cmap_fn(r % 10)[:3]) * 255).astype(np.uint8)
        for c in range(actual_cols):
            frame = grid[r][c]
            if frame is None:
                continue

            # Resize if needed
            if frame.shape[0] != h or frame.shape[1] != w:
                from PIL import Image
                frame = np.array(Image.fromarray(frame).resize((w, h)))

            # Ensure RGB
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]

            # Add coloured border
            bordered = frame.copy()
            bw = border_width
            bordered[:bw, :] = color
            bordered[-bw:, :] = color
            bordered[:, :bw] = color
            bordered[:, -bw:] = color

            y0 = header_px + r * (h + pad)
            x0 = c * (w + pad)
            canvas[y0:y0 + h, x0:x0 + w] = bordered

    # ── Plot ──
    fig_w = max(2.2 * actual_cols + 1, 6)
    fig_h = max(2.0 * n_rows + 1.2, 3)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(canvas)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.axis('off')

    # Column headers (timestep labels)
    if show_timesteps:
        for c, t in enumerate(timesteps):
            x_center = c * (w + pad) + w // 2
            ax.text(x_center, header_px // 2, f't={t}',
                    va='center', ha='center', fontsize=9, color='white',
                    fontweight='bold')

    # Row labels
    for r in range(n_rows):
        y_center = header_px + r * (h + pad) + h // 2
        label = labels[r] if r < len(labels) else f'V{r}'
        ax.text(-10, y_center, label, va='center', ha='right',
                fontsize=11, fontweight='bold', color=cmap_fn(r % 10))

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Montage saved: {output_path}  ({n_rows} rows × {actual_cols} cols, dpi={dpi})")


def main():
    parser = argparse.ArgumentParser(
        description='Generate a temporal montage from .mp4 video(s)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  python scripts/video_montage.py exp/.../videos/skill0.mp4

  # Multiple videos (one row each)
  python scripts/video_montage.py exp/.../videos/skill0.mp4 exp/.../videos/skill1.mp4

  # Whole directory
  python scripts/video_montage.py exp/.../videos/

  # Custom columns and output
  python scripts/video_montage.py exp/.../videos/ --n_cols 10 --output my_montage.png

  # Exact timesteps
  python scripts/video_montage.py exp/.../videos/ --timesteps 0 50 100 200 400 499

  # Every 6th frame
  python scripts/video_montage.py exp/.../videos/ --every 6

  # With custom row labels
  python scripts/video_montage.py s0.mp4 s1.mp4 s2.mp4 --labels "Right" "Climb" "Jump"
        """,
    )
    parser.add_argument('inputs', nargs='+',
                        help='One or more .mp4 files, or a directory containing .mp4 files')
    parser.add_argument('--n_cols', type=int, default=6,
                        help='Number of uniformly-spaced snapshots (default: 6). '
                             'Ignored if --timesteps or --every is used.')
    parser.add_argument('--timesteps', '-t', type=int, nargs='+', default=None,
                        help='Exact frame indices to sample, e.g. --timesteps 0 50 100 200 499')
    parser.add_argument('--every', '-e', type=int, default=None,
                        help='Sample every N-th frame, e.g. --every 6 → frames 0,6,12,...')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output PNG path (default: montage.png next to first input)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI (default: 150)')
    parser.add_argument('--title', type=str, default=None,
                        help='Figure title (default: auto-generated)')
    parser.add_argument('--labels', nargs='*', default=None,
                        help='Custom row labels (one per video). '
                             'Default: inferred from filenames.')
    parser.add_argument('--border', type=int, default=2,
                        help='Coloured border width in pixels (default: 2)')
    parser.add_argument('--no-timesteps', action='store_true',
                        help='Hide t=N column headers')
    parser.add_argument('--sort', action='store_true',
                        help='Sort video files alphabetically')

    args = parser.parse_args()

    # ── Resolve inputs ──
    video_paths = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            found = sorted(p.glob('*.mp4'))
            if not found:
                print(f"  [warn] No .mp4 files in {p}")
            video_paths.extend(found)
        elif p.is_file() and p.suffix == '.mp4':
            video_paths.append(p)
        else:
            print(f"  [warn] Skipping {inp} (not a .mp4 file or directory)")

    if not video_paths:
        print("Error: No .mp4 files found.")
        sys.exit(1)

    if args.sort:
        video_paths = sorted(video_paths)

    print(f"  Videos: {len(video_paths)}")
    for vp in video_paths:
        print(f"    {vp.name}")

    # ── Labels ──
    if args.labels:
        labels = args.labels
    else:
        labels = [vp.stem for vp in video_paths]

    # ── Output path ──
    if args.output:
        out = Path(args.output)
    else:
        out = video_paths[0].parent / 'montage.png'

    # ── Sampling info for title ──
    if args.timesteps:
        snap_desc = f'timesteps {args.timesteps}'
    elif args.every:
        snap_desc = f'every {args.every} frames'
    else:
        snap_desc = f'{args.n_cols} snapshots'

    # ── Title ──
    title = args.title or f'Video Montage ({len(video_paths)} videos × {snap_desc})'

    make_montage(
        video_paths=video_paths,
        labels=labels,
        n_cols=args.n_cols,
        output_path=out,
        dpi=args.dpi,
        title=title,
        border_width=args.border,
        show_timesteps=not args.no_timesteps,
        timesteps_arg=args.timesteps,
        every=args.every,
    )


if __name__ == '__main__':
    main()
