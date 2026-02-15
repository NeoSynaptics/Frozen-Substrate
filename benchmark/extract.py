"""Feature extraction: video clips → aligned (raw_frames, channel_b) pairs."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from frozen_substrate.redesign.config import SubstrateConfig, ReadoutConfig, VideoIOConfig
from frozen_substrate.redesign.pipeline import Pipeline
from frozen_substrate.redesign.video import frame_to_stim


def extract_clip_features(
    clip_frames: np.ndarray,
    substrate_cfg: SubstrateConfig,
    readout_cfg: ReadoutConfig,
    video_cfg: VideoIOConfig,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single video clip through the substrate pipeline.

    Parameters
    ----------
    clip_frames : np.ndarray
        Shape (T, H, W) or (T, H, W, 3). Raw video frames.
    substrate_cfg, readout_cfg, video_cfg : configs
        Substrate pipeline configuration.
    seed : int
        RNG seed for substrate.

    Returns
    -------
    raw_aligned : np.ndarray
        Shape (N, 1, grid_h, grid_w). One raw frame per integration window.
    channel_b : np.ndarray
        Shape (N, n_b, grid_h, grid_w). Channel B features per window.
    """
    pipe = Pipeline(substrate_cfg, readout_cfg, video_cfg, seed=seed)
    grid_hw = (substrate_cfg.height, substrate_cfg.width)
    n_a = len(readout_cfg.a_layers)

    raw_aligned = []
    channel_b = []

    last_raw = None
    for frame in clip_frames:
        # Get the preprocessed raw frame (same processing as substrate sees)
        last_raw = frame_to_stim(frame, grid_hw, video_cfg)

        result = pipe.process_frame(frame)
        if result is not None:
            cube, _meta = result
            # raw_aligned: last frame of integration window
            raw_aligned.append(last_raw[np.newaxis])  # (1, H, W)
            # channel_b: extract B channels from cube
            channel_b.append(cube[n_a:])  # (n_b, H, W)

    if not raw_aligned:
        # Clip too short to produce any cubes
        n_b = len(readout_cfg.b_layers)
        h, w = substrate_cfg.height, substrate_cfg.width
        return np.zeros((0, 1, h, w), dtype=np.float32), np.zeros((0, n_b, h, w), dtype=np.float32)

    return np.stack(raw_aligned, axis=0), np.stack(channel_b, axis=0)


def compute_temporal_diff(clip_raw_list):
    """Compute frame-to-frame temporal differences for each clip.

    This is the simplest possible temporal feature — one line of numpy.
    If the substrate doesn't beat this, the multi-layer cascade adds
    complexity without benefit.

    Parameters
    ----------
    clip_raw_list : list of np.ndarray, each shape (T, 1, H, W)

    Returns
    -------
    diff_list : list of np.ndarray, each shape (T, 1, H, W)
        diff[0] = zeros (no previous frame), diff[t] = raw[t] - raw[t-1].
    """
    diff_list = []
    for raw in clip_raw_list:
        diff = np.zeros_like(raw)
        if raw.shape[0] > 1:
            diff[1:] = raw[1:] - raw[:-1]
        diff_list.append(diff)
    return diff_list


def extract_dataset(
    clips: List[np.ndarray],
    labels: np.ndarray,
    substrate_cfg: SubstrateConfig,
    readout_cfg: ReadoutConfig,
    video_cfg: VideoIOConfig,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for an entire dataset of clips.

    Returns
    -------
    all_raw : np.ndarray, shape (N_total, 1, H, W)
    all_b : np.ndarray, shape (N_total, n_b, H, W)
    all_labels : np.ndarray, shape (N_clips,)
        One label per clip (clips produce variable cubes, but we use clip-level labels).
    clip_raw : list of np.ndarray — per-clip raw features
    clip_b : list of np.ndarray — per-clip channel B features
    """
    clip_raw_list = []
    clip_b_list = []

    for i, (clip, label) in enumerate(zip(clips, labels)):
        raw, b = extract_clip_features(clip, substrate_cfg, readout_cfg, video_cfg, seed=seed)
        clip_raw_list.append(raw)
        clip_b_list.append(b)

        if verbose and (i + 1) % 50 == 0:
            print(f"  Extracted {i + 1}/{len(clips)} clips")

    return clip_raw_list, clip_b_list, labels
