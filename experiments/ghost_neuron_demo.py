"""Ghost Neurons demo: Run the substrate with a Gaussian pen orbit
and visualize ghost neuron novelty as a 3D point cloud."""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from frozen_substrate.multilayer import MultiLayerFrozenSubstrate
from frozen_substrate.gaussian_pen import orbit_gaussian_pen
from frozen_substrate.ghost import GhostNeurons


def run(
    H=50, W=50, n_layers=10,
    ghost_layers=(2, 3, 4, 5),
    warmup_steps=60,
    T=240,
    period=160,
    radius=12.0,
    sigma=2.0,
    inject_amp=0.35,
    alpha_bg=0.01,
    threshold=0.06,
    z_spacing=3.0,
):
    sub = MultiLayerFrozenSubstrate(H=H, W=W, n_layers=n_layers, seed=1)
    ghost = GhostNeurons(ghost_layers=ghost_layers, alpha_bg=alpha_bg)

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    # Warmup baseline (no stimulus)
    for _ in range(warmup_steps):
        sub.step()
        ghost.read(sub.x_layers)

    # Run gaussian pen orbit
    for t in range(T):
        stim = orbit_gaussian_pen(
            H, W, t, cy, cx, radius,
            period=period, sigma=sigma, amplitude=1.0
        )
        sub.inject(stim, amp=inject_amp)
        sub.step()
        G_t, dG_t = ghost.read(sub.x_layers)

    # Build 3D point cloud from novelty (delta) -- follows recent path
    pts = []
    for zi, layer_id in enumerate(ghost_layers):
        m = dG_t[zi] > threshold
        ys, xs = np.where(m)
        for y, x in zip(ys, xs):
            pts.append((int(x), int(y), int(zi), int(layer_id),
                        float(dG_t[zi, y, x]), float(G_t[zi, y, x])))

    return pts, ghost_layers, z_spacing


def plot_3d(pts, ghost_layers, z_spacing, out_path):
    if len(pts) == 0:
        print("No active ghost points found.")
        return

    pts_arr = np.array(pts, dtype=np.float32)
    Xs = pts_arr[:, 0]
    Zs_spaceY = pts_arr[:, 1]
    Ys_depth = pts_arr[:, 2] * z_spacing
    vals = pts_arr[:, 4]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Xs, Ys_depth, Zs_spaceY, c=vals, cmap="viridis",
                    s=26, alpha=0.9, depthshade=True)

    ax.set_xlabel("X (space)")
    ax.set_ylabel("Depth (layers)")
    ax.set_zlabel("Y (space)")
    ax.set_title("Ghost Neurons -- Gaussian Pen Recent Path (Active Only)")

    ax.set_yticks([i * z_spacing for i in range(len(ghost_layers))])
    ax.set_yticklabels([f"L{z}" for z in ghost_layers])

    ax.view_init(elev=18, azim=-45)
    plt.colorbar(sc, label="Novelty (|activation - baseline|)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Active ghost points: {len(pts)}")


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    pts, ghost_layers, z_spacing = run()
    plot_3d(pts, ghost_layers, z_spacing,
            os.path.join(out_dir, "ghost_3d_gaussian_pen.png"))


if __name__ == "__main__":
    main()
