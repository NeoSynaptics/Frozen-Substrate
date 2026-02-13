"""V2 Resonance demo: Compare v1 (passive) vs v2 (resonant) pipeline
on the same synthetic input.

Shows:
1. Feedback resonance amplifies mid-entropy stimuli
2. Channel C distinguishes structured deviation from noise
3. Side-by-side visualization of v1 vs v2 output
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from frozen_substrate.redesign import Pipeline, SubstrateConfig, ReadoutConfig, VideoIOConfig
from frozen_substrate.v2 import PipelineV2, ResonantConfig, ReadoutV2Config


def generate_test_frames(H, W, n_frames=300):
    """Three stimulus types in different spatial regions:
    - Top-left: static square (low entropy)
    - Center: orbiting Gaussian dot (mid entropy)
    - Bottom-right: random flicker (high entropy)
    """
    rng = np.random.default_rng(42)
    frames = []
    for t in range(n_frames):
        frame = np.zeros((H, W), dtype=np.float32)

        # Static square (top-left)
        frame[4:14, 4:14] = 0.6

        # Orbiting Gaussian (center)
        cy, cx = H / 2.0, W / 2.0
        angle = 2 * np.pi * (t / 100.0)
        x0 = cx + 10.0 * np.cos(angle)
        y0 = cy + 10.0 * np.sin(angle)
        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        frame += np.exp(-((x - x0)**2 + (y - y0)**2) / 8.0).astype(np.float32)

        # Random flicker (bottom-right)
        if rng.random() > 0.5:
            frame[36:46, 36:46] = float(rng.random() * 0.8)

        frames.append(frame)
    return frames


def run_v1(frames, H, W):
    scfg = SubstrateConfig(height=H, width=W, n_layers=10, noise_std=0.0)
    rcfg = ReadoutConfig(
        a_layers=(0, 1), b_layers=(3, 4, 5, 6, 7),
        integrate_steps=4, baseline_alpha=0.02,
    )
    vcfg = VideoIOConfig()
    pipe = Pipeline(scfg, rcfg, vcfg, seed=0)

    cubes, metas = [], []
    for frame in frames:
        out = pipe.process_frame(frame)
        if out is not None:
            cube, meta = out
            cubes.append(cube)
            metas.append(meta)
    return np.stack(cubes, axis=0), metas


def run_v2(frames, H, W):
    scfg = ResonantConfig(height=H, width=W, n_layers=10, noise_std=0.0)
    rcfg = ReadoutV2Config(
        a_layers=(0, 1), b_layers=(3, 4, 5, 6, 7),
        integrate_steps=4, baseline_alpha=0.02,
        flood_patch_size=12,
    )
    vcfg = VideoIOConfig()
    pipe = PipelineV2(scfg, rcfg, vcfg, seed=0)

    cubes, metas = [], []
    for frame in frames:
        out = pipe.process_frame(frame)
        if out is not None:
            cube, meta = out
            cubes.append(cube)
            metas.append(meta)
    return np.stack(cubes, axis=0), metas


def plot_comparison(v1_cubes, v2_cubes, v2_metas, out_dir):
    n_a = 2
    n_b = 5

    # Time-averaged Channel B for each pipeline
    v1_b_mean = v1_cubes[:, n_a:n_a+n_b].mean(axis=(0, 1))  # (H, W)
    v2_b_mean = v2_cubes[:, n_a:n_a+n_b].mean(axis=(0, 1))

    # Channel C from v2 (time-averaged)
    n_c = v2_metas[-1]["n_c"]
    if n_c > 0:
        v2_c_mean = v2_cubes[:, n_a+n_b:n_a+n_b+n_c].mean(axis=(0, 1))
    else:
        v2_c_mean = np.zeros_like(v2_b_mean)

    # --- Figure 1: Side-by-side Channel B comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    im0 = axes[0].imshow(v1_b_mean, cmap="inferno", interpolation="nearest")
    axes[0].set_title("V1: Channel B (passive)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(v2_b_mean, cmap="inferno", interpolation="nearest")
    axes[1].set_title("V2: Channel B (resonant)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(v2_c_mean, cmap="plasma", interpolation="nearest")
    axes[2].set_title("V2: Channel C (coherence)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle("V1 (passive) vs V2 (resonant feedback + coherence)", fontsize=12)
    plt.tight_layout()
    path1 = os.path.join(out_dir, "v2_comparison.png")
    plt.savefig(path1, dpi=180)
    plt.close(fig)
    print(f"Saved: {path1}")

    # --- Figure 2: Channel B time traces per region ---
    def region_mean(cubes, ch_start, ch_end, y0, y1, x0, x1):
        return cubes[:, ch_start:ch_end, y0:y1, x0:x1].mean(axis=(1, 2, 3))

    t = np.arange(v1_cubes.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for ax, cubes, label in [(axes[0], v1_cubes, "V1"), (axes[1], v2_cubes, "V2")]:
        static = region_mean(cubes, n_a, n_a+n_b, 4, 14, 4, 14)
        orbit = region_mean(cubes, n_a, n_a+n_b, 15, 35, 15, 35)
        flicker = region_mean(cubes, n_a, n_a+n_b, 36, 46, 36, 46)

        ax.plot(t, static, label="Static square", alpha=0.8)
        ax.plot(t, orbit, label="Orbiting dot", alpha=0.8)
        ax.plot(t, flicker, label="Random flicker", alpha=0.8)
        ax.set_ylabel(f"{label} Channel B")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[1].set_xlabel("Cube index")
    fig.suptitle("Channel B persistence by stimulus type", fontsize=12)
    plt.tight_layout()
    path2 = os.path.join(out_dir, "v2_traces.png")
    plt.savefig(path2, dpi=180)
    plt.close(fig)
    print(f"Saved: {path2}")

    # --- Figure 3: Channel C coherence time traces ---
    if n_c > 0:
        fig, ax = plt.subplots(figsize=(10, 3.5))

        static_c = region_mean(v2_cubes, n_a+n_b, n_a+n_b+n_c, 4, 14, 4, 14)
        orbit_c = region_mean(v2_cubes, n_a+n_b, n_a+n_b+n_c, 15, 35, 15, 35)
        flicker_c = region_mean(v2_cubes, n_a+n_b, n_a+n_b+n_c, 36, 46, 36, 46)

        ax.plot(t, static_c, label="Static square", alpha=0.8)
        ax.plot(t, orbit_c, label="Orbiting dot", alpha=0.8)
        ax.plot(t, flicker_c, label="Random flicker", alpha=0.8)
        ax.set_ylabel("Channel C (coherence)")
        ax.set_xlabel("Cube index")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_title("Temporal coherence: structured vs random deviation")
        plt.tight_layout()
        path3 = os.path.join(out_dir, "v2_coherence.png")
        plt.savefig(path3, dpi=180)
        plt.close(fig)
        print(f"Saved: {path3}")


def main():
    H, W = 50, 50
    n_frames = 300

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {n_frames} test frames ({H}x{W})...")
    frames = generate_test_frames(H, W, n_frames)

    print("Running V1 pipeline (passive)...")
    v1_cubes, v1_metas = run_v1(frames, H, W)
    print(f"  V1: {v1_cubes.shape[0]} cubes, {v1_metas[-1]['cube_channels']} channels")

    print("Running V2 pipeline (resonant)...")
    v2_cubes, v2_metas = run_v2(frames, H, W)
    print(f"  V2: {v2_cubes.shape[0]} cubes, {v2_metas[-1]['cube_channels']} channels")
    print(f"  V2 channels: A={v2_metas[-1]['n_a']}, B={v2_metas[-1]['n_b']}, C={v2_metas[-1]['n_c']}")

    # Compare Channel B strength
    n_a = 2
    n_b = 5
    v1_b = float(v1_cubes[:, n_a:n_a+n_b].mean())
    v2_b = float(v2_cubes[:, n_a:n_a+n_b].mean())
    print(f"\nChannel B comparison:")
    print(f"  V1 mean: {v1_b:.6f}")
    print(f"  V2 mean: {v2_b:.6f}")
    if abs(v1_b) > 1e-8:
        print(f"  Ratio (V2/V1): {v2_b / v1_b:.2f}x")

    # Channel C stats
    n_c = v2_metas[-1]["n_c"]
    if n_c > 0:
        v2_c = v2_cubes[:, n_a+n_b:n_a+n_b+n_c]
        print(f"\nChannel C (coherence):")
        print(f"  Mean: {float(v2_c.mean()):.4f}")
        print(f"  Max:  {float(v2_c.max()):.4f}")

    plot_comparison(v1_cubes, v2_cubes, v2_metas, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
