"""Retina + Channel B demo: Four dots with different temporal patterns
test the depth penetration filter.

- Dot A: continuous (static) -- should NOT propagate deep
- Dot B: mid micro-move -- should propagate deepest
- Dot C: fast flicker -- should be suppressed
- Dot D: burst then OFF -- sanity check (decays)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from frozen_substrate.retina import RetinaDepthStack, IntegratedResidual


def gaussian_blob(H, W, y0, x0, sigma=1.6, amp=1.0):
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    g = np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sigma**2))
    return (amp * g).astype(np.float32)


def make_four_dot_frame(t, H=50, W=50, sigma=1.6,
                        dotA=(15, 15), dotB=(15, 34),
                        dotC=(35, 25), dotD=(35, 38)):
    A = gaussian_blob(H, W, dotA[0], dotA[1], sigma=sigma, amp=1.0)
    shift = 0 if ((t // 8) % 2 == 0) else 1
    B = gaussian_blob(H, W, dotB[0], dotB[1] + shift, sigma=sigma, amp=1.0)
    C = gaussian_blob(H, W, dotC[0], dotC[1], sigma=sigma,
                      amp=(1.0 if (t % 2 == 0) else 0.0))
    D = gaussian_blob(H, W, dotD[0], dotD[1], sigma=sigma,
                      amp=(1.0 if t < 25 else 0.0))
    return (A + B + C + D).astype(np.float32)


def patch_max(img, y, x, r=1):
    H, W = img.shape
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    return float(img[y0:y1, x0:x1].max())


def plot_l0_existence(out_path, A0_hist):
    fig = plt.figure(figsize=(10, 3.4))
    plt.plot(A0_hist[:, 0], label="A: continuous (L0)")
    plt.plot(A0_hist[:, 1], label="B: mid move (L0)")
    plt.plot(A0_hist[:, 2], label="C: flicker (L0)")
    plt.plot(A0_hist[:, 3], label="D: burst (L0)")
    plt.xlabel("time step")
    plt.ylabel("L0 patch max")
    plt.title("L0 Retina Buffer -- Existence traces (everything survives in L0)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_depth_traces(out_path, traces, layers_track):
    trace_A, trace_B, trace_C, trace_D = traces
    fig = plt.figure(figsize=(10, 3.5))
    plt.plot(trace_A, label="A: continuous (should NOT propagate)")
    plt.plot(trace_B, label="B: mid micro-move (should propagate deepest)")
    plt.plot(trace_C, label="C: fast flicker (should be suppressed)")
    plt.plot(trace_D, label="D: burst then OFF (sanity)")
    plt.ylim(0, 1 + max(1, max(layers_track) + 1))
    plt.xlabel("time step")
    plt.ylabel("max depth reached (L1..)")
    plt.title("Depth Penetration vs Time -- Channel B on L1+ (|I|), with L0 retina buffer")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_3d_topN(out_path, N_hist, t0, t1, ghost_layers=(0, 1, 2, 3, 4, 5),
                 topN=35, min_plot=0.01, z_spacing=3.0):
    Wn = N_hist[t0:t1]
    Nint = Wn.max(axis=0)
    H, W = Nint.shape[1], Nint.shape[2]

    pts = []
    for zi, li in enumerate(ghost_layers):
        mag = Nint[li]
        flat = mag.ravel()
        k = min(topN, flat.size)
        idx = np.argpartition(flat, -k)[-k:]
        for ind in idx:
            v = float(flat[ind])
            if v < min_plot:
                continue
            y = int(ind // W)
            x = int(ind % W)
            pts.append((x, y, zi, li, v))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if len(pts) == 0:
        ax.text2D(0.2, 0.5, f"No points above min_plot={min_plot}.",
                  transform=ax.transAxes)
        plt.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    pts = np.array(pts, dtype=np.float32)
    sc = ax.scatter(pts[:, 0], pts[:, 2] * z_spacing, pts[:, 1],
                    c=pts[:, 4], cmap="viridis", s=28, alpha=0.95,
                    depthshade=True)
    ax.set_xlabel("X (space)")
    ax.set_ylabel("Depth (layers)")
    ax.set_zlabel("Y (space)")
    ax.set_title(f"Channel B (|I|) -- Active Only (Top-{topN}/layer) -- t=[{t0},{t1})")
    ax.set_yticks([i * z_spacing for i in range(len(ghost_layers))])
    ax.set_yticklabels([f"L{li + 1}" for li in ghost_layers])
    ax.view_init(elev=18, azim=-45)
    plt.colorbar(sc, label="|I| (integrated residual novelty)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    H, W = 50, 50
    L = 8
    dotA = (15, 15)
    dotB = (15, 34)
    dotC = (35, 25)
    dotD = (35, 38)

    stack = RetinaDepthStack(H=H, W=W, L=L, clip_l0=3.0)
    ghost = IntegratedResidual(L=L - 1, H=H, W=W, baseline_alpha=0.04,
                               integral_beta=0.12)

    T = 260
    warmup = 40
    frame_gain = 0.30

    for _ in range(warmup):
        stack.inject_frame_to_l0(np.zeros((H, W), dtype=np.float32), gain=0.0)
        stack.step()
        ghost.step(stack.x[1:])

    N_hist = []
    A0_hist = []

    for t in range(T):
        frame = make_four_dot_frame(t, H=H, W=W, dotA=dotA, dotB=dotB,
                                    dotC=dotC, dotD=dotD)
        stack.inject_frame_to_l0(frame, gain=frame_gain)
        stack.step()
        N = ghost.step(stack.x[1:])
        N_hist.append(N)

        A0_hist.append([
            patch_max(stack.x[0], dotA[0], dotA[1]),
            max(patch_max(stack.x[0], dotB[0], dotB[1]),
                patch_max(stack.x[0], dotB[0], dotB[1] + 1)),
            patch_max(stack.x[0], dotC[0], dotC[1]),
            patch_max(stack.x[0], dotD[0], dotD[1]),
        ])

    N_hist = np.stack(N_hist, axis=0)
    A0_hist = np.array(A0_hist, dtype=np.float32)

    layers_track = tuple(range(0, min(6, L - 1)))
    q_thr = 0.98
    eps_thr = 0.003

    def max_depth_trace(positions, rank_thr=2.7):
        traces = []
        refs = np.zeros((T, len(layers_track)), dtype=np.float32)
        for jj, li in enumerate(layers_track):
            refs[:, jj] = np.quantile(N_hist[:, li], q_thr, axis=(1, 2))
        for t in range(T):
            deepest = -1
            for jj, li in enumerate(layers_track):
                ref = float(max(eps_thr, refs[t, jj]))
                thr = rank_thr * ref
                for (yy, xx) in positions:
                    if float(N_hist[t, li, yy, xx]) > thr:
                        deepest = max(deepest, li)
            traces.append(0 if deepest < 0 else (deepest + 1))
        return np.array(traces, dtype=np.int32)

    trace_A = max_depth_trace([dotA])
    trace_B = max_depth_trace([dotB, (dotB[0], dotB[1] + 1)])
    trace_C = max_depth_trace([dotC])
    trace_D = max_depth_trace([dotD])

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    plot_l0_existence(os.path.join(out_dir, "v3_l0_existence_traces.png"), A0_hist)
    plot_depth_traces(os.path.join(out_dir, "v3_l1plus_depth_traces.png"),
                      (trace_A, trace_B, trace_C, trace_D), layers_track)
    plot_3d_topN(os.path.join(out_dir, "v3_l1plus_3d_early.png"),
                 N_hist, 0, 80, topN=35, min_plot=0.01)
    plot_3d_topN(os.path.join(out_dir, "v3_l1plus_3d_late.png"),
                 N_hist, 170, 255, topN=35, min_plot=0.01)

    print(f"Wrote output plots to {out_dir}/")


if __name__ == "__main__":
    main()
