"""Synthetic video clip generator for benchmark.

Two modes:
- 'easy': Spatially distinct classes (static blob, periodic, drift, noise)
- 'hard': All look like the same blob — only temporal dynamics differ.
  This is where Channel B should shine.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


CLASS_NAMES_EASY = ["static", "periodic", "drift", "noise"]
CLASS_NAMES_HARD = ["noise_only", "static+noise", "drift+noise", "oscil+noise"]


def _gaussian_blob(h: int, w: int, cy: float, cx: float, sigma: float = 6.0) -> np.ndarray:
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)


# ── Easy mode generators (spatially distinct) ──

def _easy_static(rng, h, w, n_frames):
    cy = h * 0.3 + rng.random() * h * 0.4
    cx = w * 0.3 + rng.random() * w * 0.4
    sigma = 4 + rng.random() * 4
    blob = _gaussian_blob(h, w, cy, cx, sigma)
    amp = 0.4 + rng.random() * 0.4
    return np.stack([blob * amp] * n_frames, axis=0)


def _easy_periodic(rng, h, w, n_frames):
    cy = h * 0.3 + rng.random() * h * 0.4
    cx = w * 0.3 + rng.random() * w * 0.4
    sigma = 4 + rng.random() * 4
    blob = _gaussian_blob(h, w, cy, cx, sigma)
    freq = 0.3 + rng.random() * 0.5
    phase = rng.random() * 2 * np.pi
    frames = []
    for t in range(n_frames):
        amp = 0.3 + 0.4 * np.sin(2 * np.pi * freq * t / n_frames + phase)
        frames.append(blob * amp)
    return np.stack(frames, axis=0).astype(np.float32)


def _easy_drift(rng, h, w, n_frames):
    cy_start = h * 0.3 + rng.random() * h * 0.2
    cx_start = w * 0.3 + rng.random() * w * 0.2
    dy = (rng.random() - 0.5) * h * 0.4
    dx = (rng.random() - 0.5) * w * 0.4
    sigma = 4 + rng.random() * 4
    amp = 0.4 + rng.random() * 0.3
    frames = []
    for t in range(n_frames):
        frac = t / max(n_frames - 1, 1)
        frames.append(_gaussian_blob(h, w, cy_start + dy * frac, cx_start + dx * frac, sigma) * amp)
    return np.stack(frames, axis=0).astype(np.float32)


def _easy_noise(rng, h, w, n_frames):
    return (rng.random((n_frames, h, w)) * 0.6).astype(np.float32)


EASY_GENERATORS = [_easy_static, _easy_periodic, _easy_drift, _easy_noise]


# ── Hard mode generators (signal-in-noise paradigm) ──
#
# All classes share the SAME noise level, making them visually similar
# in raw frames. The classes differ in the underlying TEMPORAL SIGNAL
# buried in that noise.
#
# The substrate's depth cascade should separate persistent signals
# (blob dynamics) from transient noise — this is its core value prop:
# "Information = persistence of deviation from expectation under degradation"

NOISE_LEVEL = 0.25   # spatially-correlated noise magnitude
BLOB_AMP = 0.5       # blob peak amplitude (similar to noise → low SNR)


def _hard_base_blob(rng, h, w):
    """Shared blob params — all classes look alike spatially."""
    cy = h * 0.4 + rng.random() * h * 0.2
    cx = w * 0.4 + rng.random() * w * 0.2
    sigma = 5.0 + rng.random() * 2.0
    return cy, cx, sigma


def _spatial_noise(rng, h, w):
    """Spatially-correlated noise (3x3 smooth) — looks natural, not pixel noise."""
    raw = rng.standard_normal((h, w)).astype(np.float32) * NOISE_LEVEL
    # Simple 3x3 box blur for spatial correlation
    padded = np.pad(raw, 1, mode="reflect")
    out = np.zeros_like(raw)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            out += padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
    return (out / 9.0).astype(np.float32)


def _hard_noise_only(rng, h, w, n_frames):
    """Pure noise — no persistent signal. Substrate should show low Channel B
    because noise doesn't survive blur+leak cascades."""
    frames = [_spatial_noise(rng, h, w) for _ in range(n_frames)]
    return np.stack(frames, axis=0).astype(np.float32)


def _hard_static_in_noise(rng, h, w, n_frames):
    """Fixed blob hidden in noise. The blob is constant so substrate EMA
    converges → no Channel B from blob. Only noise residuals (transient)."""
    cy, cx, sigma = _hard_base_blob(rng, h, w)
    blob = _gaussian_blob(h, w, cy, cx, sigma) * BLOB_AMP
    frames = [blob + _spatial_noise(rng, h, w) for _ in range(n_frames)]
    return np.stack(frames, axis=0).astype(np.float32)


def _hard_drift_in_noise(rng, h, w, n_frames):
    """Slowly drifting blob hidden in noise. Drift creates coherent spatial
    deviation (edge signal) that PERSISTS through depth cascade. Noise doesn't.
    Channel B should show persistent signal at blob edges."""
    cy, cx, sigma = _hard_base_blob(rng, h, w)
    dy = (rng.random() - 0.5) * 8.0   # 3-4 pixel total drift
    dx = (rng.random() - 0.5) * 8.0
    frames = []
    for t in range(n_frames):
        frac = t / max(n_frames - 1, 1)
        blob = _gaussian_blob(h, w, cy + dy * frac, cx + dx * frac, sigma) * BLOB_AMP
        frames.append(blob + _spatial_noise(rng, h, w))
    return np.stack(frames, axis=0).astype(np.float32)


def _hard_oscillation_in_noise(rng, h, w, n_frames):
    """Blob oscillating in amplitude, hidden in noise. Amplitude changes
    create spatially-coherent residuals that persist through depth cascade.
    Channel B should show sustained activity at blob location."""
    cy, cx, sigma = _hard_base_blob(rng, h, w)
    blob = _gaussian_blob(h, w, cy, cx, sigma)
    freq = 0.5 + rng.random() * 1.5
    phase = rng.random() * 2 * np.pi
    frames = []
    for t in range(n_frames):
        amp = BLOB_AMP + 0.2 * np.sin(2 * np.pi * freq * t / n_frames + phase)
        frames.append(blob * amp + _spatial_noise(rng, h, w))
    return np.stack(frames, axis=0).astype(np.float32)


HARD_GENERATORS = [_hard_noise_only, _hard_static_in_noise,
                   _hard_drift_in_noise, _hard_oscillation_in_noise]


# ── Dataset generation ──

def generate_dataset(
    n_clips: int, h: int, w: int, n_frames: int, seed: int = 0, mode: str = "hard"
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Generate balanced synthetic dataset.

    Parameters
    ----------
    mode : str
        'easy' — spatially distinct classes (baseline can solve easily)
        'hard' — spatially identical, temporally distinct (needs temporal features)
    """
    rng = np.random.default_rng(seed)
    generators = HARD_GENERATORS if mode == "hard" else EASY_GENERATORS
    n_classes = len(generators)
    clips_per_class = n_clips // n_classes

    clips = []
    labels = []
    for cls_idx, gen_fn in enumerate(generators):
        for _ in range(clips_per_class):
            clips.append(gen_fn(rng, h, w, n_frames))
            labels.append(cls_idx)

    order = rng.permutation(len(clips))
    clips = [clips[i] for i in order]
    labels = np.array([labels[i] for i in order], dtype=np.int64)

    return clips, labels


def class_names(mode: str = "hard") -> list:
    return CLASS_NAMES_HARD if mode == "hard" else CLASS_NAMES_EASY
