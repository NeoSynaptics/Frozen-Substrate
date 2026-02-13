"""Simple depth demo: Minimal 20-layer passive substrate showing
depth-scaled propagation with a moving stimulus.

This is a standalone reference implementation that does not use the
main frozen_substrate package -- it's self-contained for clarity.
"""

import numpy as np
import matplotlib.pyplot as plt

# Configuration (v1.0 frozen)
H = 40
W = 40
LAYERS = 20
TIMESTEPS = 120

BASE_LEAK = 0.015
LEAK_K = 0.02

BASE_GAIN = 0.97
GAIN_TAU = 10.0

DIFF_START = 0.20
DIFF_END = 0.12

BASELINE_ALPHA = 0.025
RESIDUAL_THRESHOLD = 0.015


def local_diffusion(x, strength):
    kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(x, 1, mode='edge')
    out = np.zeros_like(x)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(padded[i:i + 3, j:j + 3] * kernel)
    return (1 - strength) * x + strength * out


def depth_params(l):
    leak = BASE_LEAK + LEAK_K * (l / (LAYERS - 1))
    gain = BASE_GAIN * np.exp(-l / GAIN_TAU)
    diff = DIFF_START + (DIFF_END - DIFF_START) * (l / (LAYERS - 1))
    return leak, gain, diff


def depth_centroid(depth_map):
    layers = np.arange(depth_map.shape[1])
    return (depth_map * layers).sum(axis=1) / (depth_map.sum(axis=1) + 1e-8)


def depth_spread(depth_map):
    layers = np.arange(depth_map.shape[1])
    centroid = depth_centroid(depth_map)
    return np.sqrt(((layers - centroid[:, None])**2 * depth_map).sum(axis=1))


def persistence_score(depth_map):
    return depth_map.sum(axis=1)


def inject(t):
    x = np.zeros((H, W))
    cx = int(H / 2 + 4 * np.sin(t / 30))
    cy = int(W / 2 + 4 * np.cos(t / 30))
    x[cx - 1:cx + 2, cy - 1:cy + 2] = 1.0
    return x


def main():
    A = np.zeros((LAYERS, H, W))
    baseline = np.zeros_like(A)

    depth_map = []

    for t in range(TIMESTEPS):
        A[0] = inject(t)
        for l in range(LAYERS):
            leak, gain, diff = depth_params(l)
            A[l] = local_diffusion(A[l], diff)
            A[l] *= (1 - leak)
            if l < LAYERS - 1:
                A[l + 1] += gain * A[l]
            baseline[l] = (1 - BASELINE_ALPHA) * baseline[l] + BASELINE_ALPHA * A[l]
        R = np.abs(A - baseline)
        depth_map.append(R.mean(axis=(1, 2)))
        if t < TIMESTEPS - 1:
            A[1:] *= 0

    depth_map = np.array(depth_map)

    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map.T, aspect='auto')
    plt.title("Depth Map (Mean Residual)")
    plt.xlabel("Time")
    plt.ylabel("Layer")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    print(f"Depth centroid (last step): {depth_centroid(depth_map)[-1]:.2f}")
    print(f"Persistence score (last step): {persistence_score(depth_map)[-1]:.4f}")


if __name__ == '__main__':
    main()
