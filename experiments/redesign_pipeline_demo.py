"""Redesign Pipeline demo: Run the production pipeline on synthetic
moving Gaussian input and emit output cubes."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from frozen_substrate.redesign import (
    Pipeline, SubstrateConfig, ReadoutConfig, VideoIOConfig,
)


def moving_gaussian(H, W, t, period=120, sigma=2.0):
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    angle = 2 * np.pi * (t / period)
    x0 = cx + 12.0 * np.cos(angle)
    y0 = cy + 12.0 * np.sin(angle)

    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g.astype(np.float32)


def main():
    scfg = SubstrateConfig(height=50, width=50, n_layers=10, noise_std=0.0)
    rcfg = ReadoutConfig(
        a_layers=(0, 1),
        b_layers=(3, 4, 5, 6),
        baseline_alpha=0.02,
        spatial_baseline=True,
        integrate_steps=4,
        b_policy="max",
    )
    vcfg = VideoIOConfig(output_fps=12, input_gain=0.35)

    pipe = Pipeline(scfg, rcfg, vcfg, seed=0)

    cubes = []
    metas = []
    for t in range(240):
        frame = moving_gaussian(50, 50, t)
        out = pipe.process_frame(frame)
        if out is not None:
            cube, meta = out
            cubes.append(cube)
            metas.append(meta)

    cubes = np.stack(cubes, axis=0)
    print(f"Produced cube stream: {cubes.shape}")
    print(f"  Channels: {metas[-1]['cube_channels']} "
          f"(A={len(metas[-1]['a_layers'])}, B={len(metas[-1]['b_layers'])})")
    print(f"  Flood events in last window: {metas[-1]['flood_events_in_window']}")
    print(f"  Channel A mean: {cubes[-1, :len(rcfg.a_layers)].mean():.4f}")
    print(f"  Channel B mean: {cubes[-1, len(rcfg.a_layers):].mean():.4f}")


if __name__ == "__main__":
    main()
