"""Real video benchmark using UCSD Pedestrian 2 (anomaly detection).

Downloads the UCSD Ped2 dataset (~35MB) and runs A vs A+C vs A+B ablation
on a binary classification task: normal pedestrians vs anomalies (cyclists,
skateboarders, etc).

This tests whether the substrate can detect temporal anomalies in real video
where the visual appearance is similar but the motion patterns differ.
"""

from __future__ import annotations

import os
import sys
import json
import time
import tarfile
import urllib.request
import glob as glob_mod

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frozen_substrate.redesign.config import SubstrateConfig, ReadoutConfig, VideoIOConfig

from .config import BenchmarkConfig
from .extract import extract_dataset, compute_temporal_diff
from .train import run_ablation
from .sklearn_probe import run_sklearn_probe


UCSD_URLS = [
    "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz",
    "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset_v1p2.tar.gz",
]


def _find_ped2_dir(data_dir: str) -> str | None:
    """Search for UCSDped2 directory under data_dir."""
    for root, dirs, files in os.walk(data_dir):
        if "UCSDped2" in dirs:
            return os.path.join(root, "UCSDped2")
        # Also check if we're already in the right place
        if os.path.basename(root) == "UCSDped2" and "Train" in dirs and "Test" in dirs:
            return root
    return None


def _download_ucsd_ped2(data_dir: str) -> str:
    """Download and extract UCSD Ped2 dataset. Returns path to UCSDped2 dir."""
    existing = _find_ped2_dir(data_dir)
    if existing:
        print(f"  Dataset already exists: {existing}")
        return existing

    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "UCSD_Anomaly_Dataset.tar.gz")

    if not os.path.isfile(tar_path):
        print(f"  Downloading UCSD Ped2 (~35MB)...")
        downloaded = False
        for url in UCSD_URLS:
            print(f"    Trying: {url}")
            try:
                urllib.request.urlretrieve(url, tar_path, _download_progress)
                print()
                downloaded = True
                break
            except Exception as e:
                print(f"\n    Failed: {e}")

        if not downloaded:
            print()
            print("  All download URLs failed.")
            print("  To manually download, get the dataset from:")
            print("    https://www.kaggle.com/datasets/aryashah2k/ucsd-pedestrian-database")
            print(f"  Extract UCSDped2/ into: {data_dir}/")
            raise SystemExit(1)

    print(f"  Extracting to {data_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)

    dataset_path = _find_ped2_dir(data_dir)
    if not dataset_path:
        print(f"  ERROR: UCSDped2 directory not found after extraction.")
        print(f"  Contents of {data_dir}:")
        for item in os.listdir(data_dir):
            print(f"    {item}")
        raise SystemExit(1)

    return dataset_path


def _download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)


def _load_frames_from_dir(frame_dir: str, max_frames: int = 0) -> np.ndarray:
    """Load grayscale frames from a directory of .tif/.png images.

    Returns array of shape (N, H, W) with values in [0, 1].
    """
    # Find frame files (UCSD uses .tif)
    patterns = ["*.tif", "*.tiff", "*.png", "*.jpg"]
    frame_files = []
    for pat in patterns:
        frame_files.extend(sorted(glob_mod.glob(os.path.join(frame_dir, pat))))
    if not frame_files:
        return np.array([])

    if max_frames > 0:
        frame_files = frame_files[:max_frames]

    try:
        import cv2
        frames = []
        for f in frame_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames.append(img.astype(np.float32) / 255.0)
        return np.stack(frames, axis=0) if frames else np.array([])
    except ImportError:
        # Fallback: use PIL
        from PIL import Image
        frames = []
        for f in frame_files:
            img = Image.open(f).convert("L")
            frames.append(np.array(img, dtype=np.float32) / 255.0)
        return np.stack(frames, axis=0) if frames else np.array([])


def _load_gt_mask(gt_dir: str, n_frames: int) -> np.ndarray:
    """Load ground truth frame-level labels from GT directory.

    Returns boolean array of shape (n_frames,) — True = anomalous frame.
    """
    if not os.path.isdir(gt_dir):
        return np.zeros(n_frames, dtype=bool)

    # GT dirs contain binary mask images for anomalous frames only
    gt_files = sorted(glob_mod.glob(os.path.join(gt_dir, "*.bmp")) +
                      glob_mod.glob(os.path.join(gt_dir, "*.tif")) +
                      glob_mod.glob(os.path.join(gt_dir, "*.png")))

    if not gt_files:
        return np.zeros(n_frames, dtype=bool)

    # Frame numbers with GT masks = anomalous frames
    anomalous = np.zeros(n_frames, dtype=bool)
    for gf in gt_files:
        # Extract frame number from filename (e.g., "001.bmp" -> 0)
        basename = os.path.splitext(os.path.basename(gf))[0]
        try:
            frame_idx = int(basename) - 1  # 1-indexed to 0-indexed
            if 0 <= frame_idx < n_frames:
                anomalous[frame_idx] = True
        except ValueError:
            pass

    return anomalous


def _extract_clips(frames: np.ndarray, clip_len: int = 16, stride: int = 8):
    """Extract overlapping clips from a video.

    Returns list of clips, each shape (clip_len, H, W).
    """
    n_frames = frames.shape[0]
    clips = []
    for start in range(0, n_frames - clip_len + 1, stride):
        clips.append(frames[start:start + clip_len])
    return clips


def _load_ucsd_ped2(dataset_path: str, clip_len: int = 16, stride: int = 8):
    """Load UCSD Ped2 as binary classification clips.

    Returns (clips, labels, split_indices) where:
    - clips: list of np.ndarray, each (clip_len, H, W)
    - labels: np.ndarray of 0 (normal) or 1 (anomalous)
    - train_mask: boolean array — True for training clips
    """
    train_dir = os.path.join(dataset_path, "Train")
    test_dir = os.path.join(dataset_path, "Test")

    all_clips = []
    all_labels = []
    is_train = []

    # --- Training videos (all normal) ---
    train_subdirs = sorted(glob_mod.glob(os.path.join(train_dir, "Train*")))
    # Filter out _gt directories
    train_subdirs = [d for d in train_subdirs if os.path.isdir(d) and "_gt" not in d]

    print(f"  Loading {len(train_subdirs)} training videos (all normal)...")
    for subdir in train_subdirs:
        frames = _load_frames_from_dir(subdir)
        if len(frames) == 0:
            continue
        clips = _extract_clips(frames, clip_len, stride)
        for clip in clips:
            all_clips.append(clip)
            all_labels.append(0)  # normal
            is_train.append(True)

    n_train_clips = len(all_clips)
    print(f"    {n_train_clips} normal clips from training videos")

    # --- Test videos (normal + anomalous) ---
    test_subdirs = sorted(glob_mod.glob(os.path.join(test_dir, "Test*")))
    test_subdirs = [d for d in test_subdirs if os.path.isdir(d) and "_gt" not in d]

    print(f"  Loading {len(test_subdirs)} test videos (normal + anomalous)...")
    n_anomalous = 0
    n_normal_test = 0
    for subdir in test_subdirs:
        frames = _load_frames_from_dir(subdir)
        if len(frames) == 0:
            continue

        # Load ground truth
        vid_name = os.path.basename(subdir)
        gt_dir = os.path.join(test_dir, vid_name + "_gt")
        gt_mask = _load_gt_mask(gt_dir, len(frames))

        clips = _extract_clips(frames, clip_len, stride)
        for i, clip in enumerate(clips):
            start = i * stride
            end = start + clip_len
            # Clip is anomalous if >50% of frames are anomalous
            clip_gt = gt_mask[start:end]
            is_anomalous = clip_gt.mean() > 0.5

            all_clips.append(clip)
            all_labels.append(1 if is_anomalous else 0)
            is_train.append(False)

            if is_anomalous:
                n_anomalous += 1
            else:
                n_normal_test += 1

    print(f"    {n_normal_test} normal + {n_anomalous} anomalous clips from test videos")

    return all_clips, np.array(all_labels), np.array(is_train)


def run_real_video_benchmark(cfg: BenchmarkConfig, data_dir: str = "data"):
    """Run A vs A+C vs A+B benchmark on UCSD Ped2."""
    print()
    print("Frozen Substrate Benchmark -- Real Video (UCSD Ped2)")
    print("=" * 60)

    # --- Download dataset ---
    dataset_path = _download_ucsd_ped2(data_dir)

    # --- Load clips ---
    all_clips, all_labels, is_train = _load_ucsd_ped2(
        dataset_path, clip_len=cfg.clip_frames, stride=cfg.clip_frames // 2
    )

    if len(all_clips) == 0:
        print("  ERROR: No clips loaded. Check dataset directory.")
        return

    n_normal = (all_labels == 0).sum()
    n_anomalous = (all_labels == 1).sum()
    print(f"  Total: {len(all_clips)} clips ({n_normal} normal, {n_anomalous} anomalous)")

    if n_anomalous < 5:
        print("  WARNING: Very few anomalous clips detected.")
        print("  Ground truth may not have loaded correctly.")

    # --- Split into train/test ---
    # Use training videos for train (all normal), test videos for eval
    # Also add some test-set normal clips to training for balance
    rng = np.random.default_rng(cfg.seed)

    train_idx = np.where(is_train)[0]
    test_idx = np.where(~is_train)[0]

    # From test set, find anomalous clips — we need some in training too
    test_anomalous = test_idx[all_labels[test_idx] == 1]
    test_normal = test_idx[all_labels[test_idx] == 0]

    # Split test anomalous: 60% train, 40% test
    rng.shuffle(test_anomalous)
    split = int(len(test_anomalous) * 0.6)
    train_anom = test_anomalous[:split]
    test_anom = test_anomalous[split:]

    # Split test normal: 40% added to train, 60% for test
    rng.shuffle(test_normal)
    split_n = int(len(test_normal) * 0.4)
    train_norm_extra = test_normal[:split_n]
    test_norm = test_normal[split_n:]

    # Combine training indices
    final_train_idx = np.concatenate([train_idx, train_anom, train_norm_extra])
    final_test_idx = np.concatenate([test_anom, test_norm])
    rng.shuffle(final_train_idx)
    rng.shuffle(final_test_idx)

    train_clips_sel = [all_clips[i] for i in final_train_idx]
    train_labels_sel = all_labels[final_train_idx]
    test_clips_sel = [all_clips[i] for i in final_test_idx]
    test_labels_sel = all_labels[final_test_idx]

    n_train_0 = (train_labels_sel == 0).sum()
    n_train_1 = (train_labels_sel == 1).sum()
    n_test_0 = (test_labels_sel == 0).sum()
    n_test_1 = (test_labels_sel == 1).sum()
    print(f"  Train: {len(train_labels_sel)} clips ({n_train_0} normal, {n_train_1} anomalous)")
    print(f"  Test:  {len(test_labels_sel)} clips ({n_test_0} normal, {n_test_1} anomalous)")

    # Override config for binary task
    cfg_real = BenchmarkConfig(
        grid_size=cfg.grid_size,
        n_layers=cfg.n_layers,
        integrate_steps=cfg.integrate_steps,
        n_train=len(train_labels_sel),
        n_test=len(test_labels_sel),
        clip_frames=cfg.clip_frames,
        num_classes=2,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        seed=cfg.seed,
        output_dir=cfg.output_dir,
    )

    # --- Extract substrate features ---
    print("  Extracting substrate features...", end="", flush=True)
    t0 = time.time()

    scfg = SubstrateConfig(height=cfg.grid_size, width=cfg.grid_size, n_layers=cfg.n_layers)
    rcfg = ReadoutConfig.for_substrate(scfg)
    vcfg = VideoIOConfig(input_gain=1.0, normalize=True)  # Real video: use normalization

    train_raw, train_b, train_labels_out = extract_dataset(
        train_clips_sel, train_labels_sel, scfg, rcfg, vcfg, seed=cfg.seed, verbose=False
    )
    test_raw, test_b, test_labels_out = extract_dataset(
        test_clips_sel, test_labels_sel, scfg, rcfg, vcfg, seed=cfg.seed, verbose=False
    )

    print(f" done ({time.time() - t0:.1f}s)")

    # --- Compute temporal differences ---
    print("  Computing temporal differences...")
    train_diff = compute_temporal_diff(train_raw)
    test_diff = compute_temporal_diff(test_raw)

    if train_raw and len(train_raw) > 0:
        print(f"  Raw shape per clip:  {train_raw[0].shape}")
        print(f"  ChB shape per clip:  {train_b[0].shape}")
        print(f"  Diff shape per clip: {train_diff[0].shape}")

    # --- Run ablation ---
    print()
    print("Neural Network Ablation")
    print("-" * 60)
    results = run_ablation(
        train_raw, train_b, train_labels_out,
        test_raw, test_b, test_labels_out,
        cfg_real,
        train_diff=train_diff, test_diff=test_diff,
    )

    # --- Print results ---
    class_names = ["normal", "anomalous"]
    print()
    print("Results Summary")
    print("-" * 60)

    acc_a = results['a']['accuracy']
    print(f"  Model A         (raw only):           {acc_a:.2f}   (chance = 0.50)")
    if 'ac_early' in results:
        acc_ac_e = results['ac_early']['accuracy']
        acc_ac_l = results['ac_late']['accuracy']
        print(f"  Model A+C early (raw + temp diff):    {acc_ac_e:.2f}")
        print(f"  Model A+C late  (raw + temp diff):    {acc_ac_l:.2f}")
    acc_ab_e = results['ab_early']['accuracy']
    acc_ab_l = results['ab_late']['accuracy']
    print(f"  Model A+B early (raw + substrate):    {acc_ab_e:.2f}")
    print(f"  Model A+B late  (raw + substrate):    {acc_ab_l:.2f}")

    # --- Sklearn probe ---
    print()
    print("Random Forest Probe (feature-level)")
    print("-" * 60)
    sklearn_results = run_sklearn_probe(
        train_raw, train_b, train_labels_out,
        test_raw, test_b, test_labels_out,
        class_names,
        train_diff=train_diff, test_diff=test_diff,
    )

    # --- Save results ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    result_file = os.path.join(cfg.output_dir, "real_video_results.json")
    output = {
        "experiment": "real_video_ucsd_ped2",
        "dataset": "UCSD Ped2",
        "task": "binary classification (normal vs anomalous)",
        "config": {
            "grid_size": cfg.grid_size,
            "n_layers": cfg.n_layers,
            "n_train": len(train_labels_sel),
            "n_test": len(test_labels_sel),
            "clip_frames": cfg.clip_frames,
            "epochs": cfg.epochs,
            "seed": cfg.seed,
        },
        "nn_results": {k: v for k, v in results.items()},
        "sklearn_results": sklearn_results,
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print("=" * 60)
    if 'ac_early' in results:
        best_ac = max(results['ac_early']['accuracy'], results['ac_late']['accuracy'])
        best_ab = max(results['ab_early']['accuracy'], results['ab_late']['accuracy'])
        if best_ab > best_ac + 0.02:
            print(f"  VERDICT: Substrate beats temporal diff by {(best_ab - best_ac)*100:.0f} pts on real video")
        elif best_ac > best_ab + 0.02:
            print(f"  VERDICT: Temporal diff beats substrate by {(best_ac - best_ab)*100:.0f} pts on real video")
        else:
            print(f"  VERDICT: Comparable on real video (within 2 pts)")
    print(f"  Saved: {result_file}")
    print()

    return output
