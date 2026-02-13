"""Video processing demo: Run the production pipeline on a video file
or fall back to synthetic input, then save output cubes and visualizations.

Usage:
    python experiments/video_demo.py                    # synthetic fallback
    python experiments/video_demo.py path/to/video.mp4  # real video
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from frozen_substrate.redesign import (
    Pipeline, SubstrateConfig, ReadoutConfig, VideoIOConfig,
)
from frozen_substrate.viz import save_cubes_npz, print_cube_summary, render_composite


def synthetic_frames(H, W, n_frames=200):
    """Generate synthetic frames: a Gaussian dot orbiting + a static square."""
    for t in range(n_frames):
        frame = np.zeros((H, W), dtype=np.float32)

        # Static square (low-entropy -- should NOT persist deeply)
        frame[5:15, 5:15] = 0.6

        # Orbiting dot (mid-entropy -- SHOULD persist deeply)
        cy, cx = H / 2.0, W / 2.0
        angle = 2 * np.pi * (t / 100.0)
        x0 = cx + 10.0 * np.cos(angle)
        y0 = cy + 10.0 * np.sin(angle)
        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        frame += np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * 2.0**2)).astype(np.float32)

        # Fast flicker (high-entropy -- should NOT persist deeply)
        if t % 2 == 0:
            frame[35:45, 35:45] = 0.8

        yield frame


def video_frames(path, max_frames=0):
    """Read frames from a video file using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    count = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        count += 1
        if max_frames > 0 and count >= max_frames:
            break
    cap.release()


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "video_demo")
    os.makedirs(out_dir, exist_ok=True)

    scfg = SubstrateConfig.default()
    rcfg = ReadoutConfig.for_substrate(scfg)
    vcfg = VideoIOConfig()
    pipe = Pipeline(scfg, rcfg, vcfg, seed=0)

    # Choose input source
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        video_path = sys.argv[1]
        print(f"Processing video: {video_path}")
        frames = video_frames(video_path, max_frames=500)
    else:
        print("No video file provided -- using synthetic input")
        print("  (pass a video path as argument to process real video)")
        frames = synthetic_frames(scfg.height, scfg.width, n_frames=200)

    cubes = []
    metas = []
    frame_count = 0

    for frame in frames:
        out = pipe.process_frame(frame)
        if out is not None:
            cube, meta = out
            cubes.append(cube)
            metas.append(meta)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  processed {frame_count} frames, {len(cubes)} cubes...", end="\r")

    print(f"\nDone: {frame_count} frames -> {len(cubes)} cubes")

    if not cubes:
        print("No cubes produced (need at least integrate_steps frames).")
        return

    cubes_arr = np.stack(cubes, axis=0)
    print_cube_summary(cubes_arr, metas)

    # Save cubes
    npz_path = os.path.join(out_dir, "cubes.npz")
    save_cubes_npz(npz_path, cubes_arr, metas)
    print(f"Saved cubes: {npz_path}")

    # Render composite visualization
    n_a = len(metas[-1].get("a_layers", ()))
    render_composite(cubes_arr, out_dir, a_channels=n_a)

    print(f"\nAll output written to: {out_dir}")


if __name__ == "__main__":
    main()
