"""PyTorch dataset wrappers for the benchmark."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_channel_stats(
    clip_b: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std across all clips for normalization.

    Parameters
    ----------
    clip_b : list of np.ndarray, each shape (T, n_b, H, W)

    Returns
    -------
    mean : np.ndarray, shape (n_b,)
    std : np.ndarray, shape (n_b,)
    """
    # Stack all channel B data: (total_frames, n_b, H, W)
    all_b = np.concatenate(clip_b, axis=0)
    n_b = all_b.shape[1]
    mean = np.zeros(n_b, dtype=np.float32)
    std = np.ones(n_b, dtype=np.float32)
    for c in range(n_b):
        ch = all_b[:, c]
        mean[c] = ch.mean()
        s = ch.std()
        std[c] = s if s > 1e-8 else 1.0
    return mean, std


class SubstrateDataset(Dataset):
    """Dataset that serves either raw-only or raw+channelB tensors.

    Each sample is one video clip's worth of temporally-aligned features.
    """

    def __init__(
        self,
        clip_raw: List[np.ndarray],
        clip_b: List[np.ndarray],
        labels: np.ndarray,
        mode: str = "a+b",
        b_mean: Optional[np.ndarray] = None,
        b_std: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        clip_raw : list of np.ndarray
            Per-clip raw frames, each shape (T, 1, H, W).
        clip_b : list of np.ndarray
            Per-clip Channel B features, each shape (T, n_b, H, W).
        labels : np.ndarray
            Shape (N_clips,). Integer class labels.
        mode : str
            'a' for raw-only, 'a+b' for raw + Channel B.
        b_mean, b_std : optional np.ndarray, shape (n_b,)
            Per-channel normalization stats for Channel B.
            If provided, each B channel is z-scored: (x - mean) / std.
        """
        assert mode in ("a", "a+b", "late_fusion"), f"mode must be 'a', 'a+b', or 'late_fusion', got {mode!r}"
        self.clip_raw = clip_raw
        self.clip_b = clip_b
        self.labels = labels
        self.mode = mode
        self.b_mean = b_mean
        self.b_std = b_std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        raw = torch.from_numpy(self.clip_raw[idx])   # (T, 1, H, W)
        label = int(self.labels[idx])

        if self.mode == "a":
            # (T, 1, H, W) → permute to (1, T, H, W) for Conv3d
            x = raw.permute(1, 0, 2, 3)  # (1, T, H, W)
            return x, label

        b = torch.from_numpy(self.clip_b[idx].copy())  # (T, n_b, H, W)
        # Per-channel normalization of Channel B
        if self.b_mean is not None and self.b_std is not None:
            for c in range(b.shape[1]):
                b[:, c] = (b[:, c] - self.b_mean[c]) / self.b_std[c]

        if self.mode == "late_fusion":
            # Return raw and channel_b as separate tensors
            raw_out = raw.permute(1, 0, 2, 3)   # (1, T, H, W)
            b_out = b.permute(1, 0, 2, 3)       # (n_b, T, H, W)
            return (raw_out, b_out), label

        # Early fusion: concatenate along channel dim
        combined = torch.cat([raw, b], dim=1)   # (T, 1+n_b, H, W)
        x = combined.permute(1, 0, 2, 3)        # (1+n_b, T, H, W)
        return x, label
