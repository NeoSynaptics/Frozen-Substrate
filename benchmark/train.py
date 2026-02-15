"""Training loop and evaluation for the benchmark."""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import VideoClassifier, LateFusionClassifier
from .dataset import SubstrateDataset, compute_channel_stats
from .config import BenchmarkConfig


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: BenchmarkConfig,
    device: torch.device,
    late_fusion: bool = False,
) -> float:
    """Train model for cfg.epochs, return final training loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    final_loss = 0.0
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            optimizer.zero_grad()
            if late_fusion:
                raw, b = x
                logits = model(raw.to(device), b.to(device))
            else:
                logits = model(x.to(device))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)

    return final_loss


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    late_fusion: bool = False,
) -> Dict[str, object]:
    """Evaluate model, return accuracy and per-class metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            if late_fusion:
                raw, b = x
                logits = model(raw.to(device), b.to(device))
            else:
                logits = model(x.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float((all_preds == all_labels).mean())

    per_class = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[c] = float((all_preds[mask] == c).mean())
        else:
            per_class[c] = 0.0

    return {"accuracy": accuracy, "per_class": per_class}


def run_ablation(
    train_raw, train_b, train_labels,
    test_raw, test_b, test_labels,
    cfg: BenchmarkConfig,
    train_diff=None, test_diff=None,
) -> Dict[str, Dict]:
    """Run the full ablation: A vs A+C (temporal diff) vs A+B (substrate).

    Returns dict with keys: 'a', 'ab_early', 'ab_late', and optionally
    'ac_early', 'ac_late' if temporal diff data is provided.
    """
    device = torch.device("cpu")

    n_b = train_b[0].shape[1] if len(train_b) > 0 and train_b[0].shape[0] > 0 else 4

    b_mean, b_std = compute_channel_stats(train_b)
    print(f"  Channel B stats: mean={b_mean}, std={b_std}")

    results = {}

    # --- Model A (raw only) ---
    print("  Training Model A (raw only)...", end="", flush=True)
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    model_a = VideoClassifier(in_channels=1, num_classes=cfg.num_classes).to(device)
    ds_train_a = SubstrateDataset(train_raw, train_b, train_labels, mode="a")
    ds_test_a = SubstrateDataset(test_raw, test_b, test_labels, mode="a")
    train_model(model_a, DataLoader(ds_train_a, batch_size=cfg.batch_size, shuffle=True), cfg, device)
    results['a'] = evaluate_model(model_a, DataLoader(ds_test_a, batch_size=cfg.batch_size), cfg.num_classes, device)
    print(f" done ({time.time() - t0:.1f}s)")

    # --- Model A+C (raw + temporal diff) ---
    if train_diff is not None:
        diff_mean, diff_std = compute_channel_stats(train_diff)
        print(f"  Temporal diff stats: mean={diff_mean}, std={diff_std}")

        print("  Training Model A+C early (raw + temp diff)...", end="", flush=True)
        t0 = time.time()
        torch.manual_seed(cfg.seed)
        model_ac = VideoClassifier(in_channels=2, num_classes=cfg.num_classes).to(device)
        ds_train_ac = SubstrateDataset(train_raw, train_diff, train_labels, mode="a+b",
                                        b_mean=diff_mean, b_std=diff_std)
        ds_test_ac = SubstrateDataset(test_raw, test_diff, test_labels, mode="a+b",
                                       b_mean=diff_mean, b_std=diff_std)
        train_model(model_ac, DataLoader(ds_train_ac, batch_size=cfg.batch_size, shuffle=True), cfg, device)
        results['ac_early'] = evaluate_model(model_ac, DataLoader(ds_test_ac, batch_size=cfg.batch_size), cfg.num_classes, device)
        print(f" done ({time.time() - t0:.1f}s)")

        print("  Training Model A+C late (raw + temp diff)...", end="", flush=True)
        t0 = time.time()
        torch.manual_seed(cfg.seed)
        model_ac_late = LateFusionClassifier(raw_channels=1, b_channels=1, num_classes=cfg.num_classes).to(device)
        ds_train_ac_late = SubstrateDataset(train_raw, train_diff, train_labels, mode="late_fusion",
                                             b_mean=diff_mean, b_std=diff_std)
        ds_test_ac_late = SubstrateDataset(test_raw, test_diff, test_labels, mode="late_fusion",
                                            b_mean=diff_mean, b_std=diff_std)
        train_model(model_ac_late, DataLoader(ds_train_ac_late, batch_size=cfg.batch_size, shuffle=True), cfg, device, late_fusion=True)
        results['ac_late'] = evaluate_model(model_ac_late, DataLoader(ds_test_ac_late, batch_size=cfg.batch_size), cfg.num_classes, device, late_fusion=True)
        print(f" done ({time.time() - t0:.1f}s)")

    # --- Model A+B early fusion (raw + substrate) ---
    print("  Training Model A+B early (raw + substrate)...", end="", flush=True)
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    model_ab_early = VideoClassifier(in_channels=1 + n_b, num_classes=cfg.num_classes).to(device)
    ds_train_ab = SubstrateDataset(train_raw, train_b, train_labels, mode="a+b",
                                    b_mean=b_mean, b_std=b_std)
    ds_test_ab = SubstrateDataset(test_raw, test_b, test_labels, mode="a+b",
                                   b_mean=b_mean, b_std=b_std)
    train_model(model_ab_early, DataLoader(ds_train_ab, batch_size=cfg.batch_size, shuffle=True), cfg, device)
    results['ab_early'] = evaluate_model(model_ab_early, DataLoader(ds_test_ab, batch_size=cfg.batch_size), cfg.num_classes, device)
    print(f" done ({time.time() - t0:.1f}s)")

    # --- Model A+B late fusion (raw + substrate) ---
    print("  Training Model A+B late (raw + substrate)...", end="", flush=True)
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    model_ab_late = LateFusionClassifier(raw_channels=1, b_channels=n_b, num_classes=cfg.num_classes).to(device)
    ds_train_late = SubstrateDataset(train_raw, train_b, train_labels, mode="late_fusion",
                                      b_mean=b_mean, b_std=b_std)
    ds_test_late = SubstrateDataset(test_raw, test_b, test_labels, mode="late_fusion",
                                     b_mean=b_mean, b_std=b_std)
    train_model(model_ab_late, DataLoader(ds_train_late, batch_size=cfg.batch_size, shuffle=True), cfg, device, late_fusion=True)
    results['ab_late'] = evaluate_model(model_ab_late, DataLoader(ds_test_late, batch_size=cfg.batch_size), cfg.num_classes, device, late_fusion=True)
    print(f" done ({time.time() - t0:.1f}s)")

    return results
