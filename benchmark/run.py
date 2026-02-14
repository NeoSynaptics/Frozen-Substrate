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
from .extract import extract_dataset
from .train import run_ablation
from .sklearn_probe import run_sklearn_probe


def run_synthetic(cfg: BenchmarkConfig, mode: str = "hard") -> dict:
    """Run the full synthetic benchmark."""
    names = class_names(mode)

    print()
    print(f"Frozen Substrate Benchmark — Synthetic '{mode}' (4 classes)")
    print("=" * 52)

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
    # Disable per-frame normalization: synthetic data is already in [0, 1] range.
    # Per-frame normalization destroys amplitude-based temporal differences.
    vcfg = VideoIOConfig(input_gain=1.0, normalize=False)

    train_raw, train_b, train_labels = extract_dataset(
        train_clips, train_labels, scfg, rcfg, vcfg, seed=cfg.seed, verbose=False
    )
    test_raw, test_b, test_labels = extract_dataset(
        test_clips, test_labels, scfg, rcfg, vcfg, seed=cfg.seed, verbose=False
    )

    print(f" done ({time.time() - t0:.1f}s)")

    # Show shapes
    if train_raw and len(train_raw) > 0:
        print(f"  Raw shape per clip: {train_raw[0].shape}")
        print(f"  ChB shape per clip: {train_b[0].shape}")

    # --- Run ablation ---
    results_a, results_ab_early, results_ab_late = run_ablation(
        train_raw, train_b, train_labels,
        test_raw, test_b, test_labels,
        cfg,
    )

    # --- Print results ---
    print()
    print("Results")
    print("-" * 60)

    acc_a = results_a["accuracy"]
    acc_early = results_ab_early["accuracy"]
    acc_late = results_ab_late["accuracy"]
    delta_early = acc_early - acc_a
    delta_late = acc_late - acc_a
    best_delta = max(delta_early, delta_late)
    best_label = "early" if delta_early >= delta_late else "late"

    chance = 1.0 / cfg.num_classes
    print(f"  Model A       (raw only):       accuracy = {acc_a:.2f}   (chance = {chance:.2f})")
    print(f"  Model A+B     (early fusion):   accuracy = {acc_early:.2f}   ({'+' if delta_early >= 0 else ''}{delta_early*100:.0f} points)")
    print(f"  Model A+B     (late fusion):    accuracy = {acc_late:.2f}   ({'+' if delta_late >= 0 else ''}{delta_late*100:.0f} points)")
    print()
    print(f"  Per-class breakdown:   {'':12s}   {'A':>5s}   {'Early':>5s}   {'Late':>5s}")
    for c in range(cfg.num_classes):
        ca = results_a["per_class"].get(c, 0)
        ce = results_ab_early["per_class"].get(c, 0)
        cl = results_ab_late["per_class"].get(c, 0)
        best = max(ce - ca, cl - ca)
        name = names[c] if c < len(names) else f"class_{c}"
        marker = " <-- substrate helps" if best > 0.10 else ""
        print(f"    {name:16s}   {ca:.2f}   {ce:.2f}   {cl:.2f}{marker}")

    # --- Sklearn probe (feature-level, no SGD instability) ---
    print()
    print("Random Forest Probe (feature-level)")
    print("-" * 60)
    sklearn_results = run_sklearn_probe(
        train_raw, train_b, train_labels,
        test_raw, test_b, test_labels,
        names,
    )

    # --- Save results ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    result_file = os.path.join(cfg.output_dir, "synthetic_results.json")
    output = {
        "experiment": "synthetic_ablation",
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
        "model_a": {
            "accuracy": acc_a,
            "per_class": {names[c]: results_a["per_class"][c] for c in range(cfg.num_classes)},
        },
        "model_ab_early": {
            "accuracy": acc_early,
            "per_class": {names[c]: results_ab_early["per_class"][c] for c in range(cfg.num_classes)},
        },
        "model_ab_late": {
            "accuracy": acc_late,
            "per_class": {names[c]: results_ab_late["per_class"][c] for c in range(cfg.num_classes)},
        },
        "delta_early": delta_early,
        "delta_late": delta_late,
        "sklearn_probe": sklearn_results,
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)

    print()
    if best_delta > 0:
        print(f"  Conclusion: Channel B adds +{best_delta*100:.0f} accuracy points ({best_label} fusion).")
    elif best_delta == 0:
        print(f"  Conclusion: Channel B had no effect on accuracy.")
    else:
        print(f"  Conclusion: Channel B decreased accuracy by {abs(best_delta)*100:.0f} points.")

    print(f"  Saved: {result_file}")
    print()

    return output


def main():
    parser = argparse.ArgumentParser(description="Frozen Substrate Benchmark")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic benchmark")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--n-train", type=int, default=400, help="Training clips (default: 400)")
    parser.add_argument("--n-test", type=int, default=100, help="Test clips (default: 100)")
    parser.add_argument("--grid-size", type=int, default=50, help="Substrate grid size (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--mode", type=str, default="hard", choices=["easy", "hard"],
                        help="Synthetic mode: 'easy' (spatial) or 'hard' (temporal)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
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
    else:
        print("Usage: python -m benchmark.run --synthetic")
        print()
        print("Options:")
        print("  --synthetic       Run synthetic 4-class benchmark")
        print("  --mode easy|hard  'easy' (spatial) or 'hard' (temporal, default)")
        print("  --epochs N        Training epochs (default: 30)")
        print("  --n-train N       Number of training clips (default: 400)")
        print("  --n-test N        Number of test clips (default: 100)")
        print("  --grid-size N     Substrate grid size (default: 50)")
        print("  --seed N          Random seed (default: 42)")


if __name__ == "__main__":
    main()
