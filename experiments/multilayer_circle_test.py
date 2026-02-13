"""Multi-layer circle test: A Gaussian pen traces a circle on L0 while
passive depth layers propagate the activity downward. Produces snapshot
images and a trace plot."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Allow running directly from this file
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from frozen_substrate.multilayer import (
    MultiLayerFrozenSubstrate,
    PassiveLayerParams,
)


def gaussian_spot(H, W, cx, cy, sigma=1.5, amp=1.0):
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return (amp * g).astype(np.float32)


def run(T=2500, H=50, W=50, n_layers=10, seed=0):
    passive = PassiveLayerParams()

    core_kwargs = dict(
        enable_pacemakers=False,
        mod_pruning=1.0,
        survive_drain=0.004,
        survive_gain=0.004,
        metabolic_cost=0.0005,
    )

    substrate = MultiLayerFrozenSubstrate(
        H=H, W=W, n_layers=n_layers, seed=seed,
        passive=passive, core_kwargs=core_kwargs,
    )

    r = 12
    cx0, cy0 = W // 2, H // 2

    snap_ts = [0, 500, 1000, 1500, 2000, T - 1]
    snaps = {t: None for t in snap_ts}

    mean_abs = np.zeros((T, n_layers), dtype=np.float32)

    for t in range(T):
        theta = 2 * np.pi * (t / 1000.0)
        cx = cx0 + int(r * np.cos(theta))
        cy = cy0 + int(r * np.sin(theta))
        stim = gaussian_spot(H, W, cx, cy, sigma=1.6, amp=0.9)
        substrate.inject(stim)
        substrate.step()

        xs = substrate.x_layers
        mean_abs[t, 0] = float(np.mean(np.abs(xs[0]) * substrate.core.alive))
        for k in range(1, n_layers):
            mean_abs[t, k] = float(np.mean(np.abs(xs[k])))

        if t in snaps:
            snaps[t] = [x.copy() for x in xs]

    return substrate, mean_abs, snaps


def plot(mean_abs, snaps, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    T, n_layers = mean_abs.shape

    plt.figure(figsize=(12, 5))
    for k in range(n_layers):
        plt.plot(mean_abs[:, k], label=f"L{k}")
    plt.title("Mean |activity| per layer")
    plt.xlabel("time step")
    plt.ylabel("mean |x|")
    plt.legend(ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trace_multilayer.png"), dpi=150)
    plt.close()

    for t, xs in snaps.items():
        if xs is None:
            continue
        n = len(xs)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = np.array(axes).reshape(-1)
        vmin = min(float(np.min(x)) for x in xs)
        vmax = max(float(np.max(x)) for x in xs)
        for k in range(rows * cols):
            ax = axes[k]
            if k < n:
                ax.imshow(xs[k], cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f"L{k} x (t={t})")
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"snap_multi_{t}.png"), dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    substrate, mean_abs, snaps = run()
    plot(mean_abs, snaps)
    print("Wrote outputs/trace_multilayer.png and outputs/snap_multi_*.png")
