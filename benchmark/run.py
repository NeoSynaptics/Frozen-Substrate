"""Main entry point for the Frozen Substrate benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Add parent dir to path so `frozen_substrate` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frozen_substrate.redesign.config import SubstrateConfig, ReadoutConfig, VideoIOConfig

from .config import BenchmarkConfig
from .synthetic import generate_dataset, class_names
from .extract import extract_dataset, compute_temporal_diff
from .train import run_ablation
from .sklearn_probe import run_sklearn_probe


def _fmt_delta(val):
    return f"{'+' if val >= 0 else ''}{val*100:.0f}"


def run_synthetic(cfg: BenchmarkConfig, mode: str = "hard") -> dict:
    """Run the full synthetic benchmark with temporal diff baseline."""
    names = class_names(mode)

    print()
    print(f"Frozen Substrate Benchmark -- Synthetic '{mode}' (4 classes)")
    print("=" * 60)

    # --- Generate data ---
    print(f"  Generating synthetic clips...   {cfg.n_train} train / {cfg.n_test} test")
    train_clips, train_labels = generate_dataset(
        cfg.n_train, cfg.grid_size, cfg.grid_size, cfg.clip_frames, seed=cfg.seed, mode=mode
    )
    test_clips, test_labels = generate_dataset(
        cfg.n_test, cfg.grid_size, cfg.grid_size, cfg.clip_frames, seed=cfg.seed + 1000, mode=mode
    )

    # --- Extract substrate features ---
    print("  Extracting substrate features...", end="", flush=True)
    t0 = time.time()

    scfg = SubstrateConfig(height=cfg.grid_size, width=cfg.grid_size, n_layers=cfg.n_layers)
    rcfg = ReadoutConfig.for_substrate(scfg)
    vcfg = VideoIOConfig(input_gain=1.0, normalize=False)

    train_raw, train_b, train_labels = extract_dataset(
        train_clips, train_labels, scfg, rcfg, vcfg, seed=cfg.seed, verbose=False
    )
    test_raw, test_b, test_labels = extract_dataset(
        test_clips, test_labels, scfg, rcfg, vcfg, seed=cfg.seed, verbose=False
    )

    print(f" done ({time.time() - t0:.1f}s)")

    # --- Compute temporal differences (the simple baseline) ---
    print("  Computing temporal differences (frame[t] - frame[t-1])...")
    train_diff = compute_temporal_diff(train_raw)
    test_diff = compute_temporal_diff(test_raw)

    # Show shapes
    if train_raw and len(train_raw) > 0:
        print(f"  Raw shape per clip:  {train_raw[0].shape}")
        print(f"  ChB shape per clip:  {train_b[0].shape}")
        print(f"  Diff shape per clip: {train_diff[0].shape}")

    # --- Run ablation (A vs A+C vs A+B) ---
    print()
    print("Neural Network Ablation")
    print("-" * 60)
    results = run_ablation(
        train_raw, train_b, train_labels,
        test_raw, test_b, test_labels,
        cfg,
        train_diff=train_diff, test_diff=test_diff,
    )

    # --- Print results ---
    print()
    print("Results Summary")
    print("-" * 60)

    acc_a = results['a']['accuracy']
    chance = 1.0 / cfg.num_classes

    print(f"  Model A         (raw only):           {acc_a:.2f}   (chance = {chance:.2f})")
    if 'ac_early' in results:
        acc_ac_e = results['ac_early']['accuracy']
        acc_ac_l = results['ac_late']['accuracy']
        print(f"  Model A+C early (raw + temp diff):    {acc_ac_e:.2f}   ({_fmt_delta(acc_ac_e - acc_a)} pts)")
        print(f"  Model A+C late  (raw + temp diff):    {acc_ac_l:.2f}   ({_fmt_delta(acc_ac_l - acc_a)} pts)")
    acc_ab_e = results['ab_early']['accuracy']
    acc_ab_l = results['ab_late']['accuracy']
    print(f"  Model A+B early (raw + substrate):    {acc_ab_e:.2f}   ({_fmt_delta(acc_ab_e - acc_a)} pts)")
    print(f"  Model A+B late  (raw + substrate):    {acc_ab_l:.2f}   ({_fmt_delta(acc_ab_l - acc_a)} pts)")

    # Per-class
    print()
    if 'ac_early' in results:
        print(f"  Per-class:         {'A':>5s}  {'A+C e':>5s}  {'A+C l':>5s}  {'A+B e':>5s}  {'A+B l':>5s}")
    else:
        print(f"  Per-class:         {'A':>5s}  {'A+B e':>5s}  {'A+B l':>5s}")
    for c in range(cfg.num_classes):
        name = names[c] if c < len(names) else f"class_{c}"
        ca = results['a']['per_class'].get(c, 0)
        row = f"    {name:16s}   {ca:.2f}"
        if 'ac_early' in results:
            ce = results['ac_early']['per_class'].get(c, 0)
            cl = results['ac_late']['per_class'].get(c, 0)
            row += f"   {ce:.2f}   {cl:.2f}"
        be = results['ab_early']['per_class'].get(c, 0)
        bl = results['ab_late']['per_class'].get(c, 0)
        row += f"   {be:.2f}   {bl:.2f}"
        print(row)

    # --- Sklearn probe ---
    print()
    print("Random Forest Probe (feature-level)")
    print("-" * 60)
    sklearn_results = run_sklearn_probe(
        train_raw, train_b, train_labels,
        test_raw, test_b, test_labels,
        names,
        train_diff=train_diff, test_diff=test_diff,
    )

    # --- Save results ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    result_file = os.path.join(cfg.output_dir, "synthetic_results.json")
    output = {
        "experiment": "synthetic_ablation_with_baseline",
        "mode": mode,
        "config": {
            "grid_size": cfg.grid_size,
            "n_layers": cfg.n_layers,
            "n_train": cfg.n_train,
            "n_test": cfg.n_test,
            "clip_frames": cfg.clip_frames,
            "epochs": cfg.epochs,
            "seed": cfg.seed,
        },
        "nn_results": {k: v for k, v in results.items()},
        "sklearn_results": sklearn_results,
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # --- Final verdict ---
    print()
    print("=" * 60)
    if 'ac_early' in results:
        best_ac = max(results['ac_early']['accuracy'], results['ac_late']['accuracy'])
        best_ab = max(results['ab_early']['accuracy'], results['ab_late']['accuracy'])
        if best_ab > best_ac + 0.02:
            print(f"  VERDICT: Substrate beats temporal diff by {(best_ab - best_ac)*100:.0f} pts")
        elif best_ac > best_ab + 0.02:
            print(f"  VERDICT: Temporal diff beats substrate by {(best_ac - best_ab)*100:.0f} pts")
            print(f"           The multi-layer cascade may not add value beyond frame[t]-frame[t-1]")
        else:
            print(f"  VERDICT: Substrate and temporal diff are comparable (within 2 pts)")
    print(f"  Saved: {result_file}")
    print()

    return output


def main():
    parser = argparse.ArgumentParser(description="Frozen Substrate Benchmark")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic benchmark")
    parser.add_argument("--real", action="store_true", help="Run real video benchmark (UCSD Ped2)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--n-train", type=int, default=400, help="Training clips (default: 400)")
    parser.add_argument("--n-test", type=int, default=100, help="Test clips (default: 100)")
    parser.add_argument("--grid-size", type=int, default=50, help="Substrate grid size (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--mode", type=str, default="hard", choices=["easy", "hard"],
                        help="Synthetic mode: 'easy' (spatial) or 'hard' (temporal)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory for real video")
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        grid_size=args.grid_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    if args.synthetic:
        run_synthetic(cfg, mode=args.mode)
    elif args.real:
        from .real_video import run_real_video_benchmark
        run_real_video_benchmark(cfg, data_dir=args.data_dir)
    else:
        print("Usage:")
        print("  python -m benchmark.run --synthetic            Synthetic benchmark")
        print("  python -m benchmark.run --synthetic --mode easy  Easy mode (spatial)")
        print("  python -m benchmark.run --real                 Real video (UCSD Ped2)")
        print()
        print("Options:")
        print("  --epochs N        Training epochs (default: 30)")
        print("  --n-train N       Training clips (default: 400)")
        print("  --n-test N        Test clips (default: 100)")
        print("  --grid-size N     Substrate grid size (default: 50)")
        print("  --seed N          Random seed (default: 42)")
        print("  --data-dir DIR    Data directory for real video (default: data)")


if __name__ == "__main__":
    main()
