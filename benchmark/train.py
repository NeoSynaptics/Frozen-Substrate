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
) -> Tuple[Dict, Dict, Dict]:
    """Run the full A vs A+B ablation (early + late fusion).

    Returns (results_a, results_ab_early, results_ab_late).
    """
    device = torch.device("cpu")

    # Determine input channels
    n_b = train_b[0].shape[1] if len(train_b) > 0 and train_b[0].shape[0] > 0 else 4

    # Compute per-channel normalization stats from training Channel B
    b_mean, b_std = compute_channel_stats(train_b)
    print(f"  Channel B stats: mean={b_mean}, std={b_std}")

    # --- Model A (raw only) ---
    print("  Training Model A (raw only)...", end="", flush=True)
    t0 = time.time()

    torch.manual_seed(cfg.seed)
    model_a = VideoClassifier(in_channels=1, num_classes=cfg.num_classes).to(device)
    ds_train_a = SubstrateDataset(train_raw, train_b, train_labels, mode="a")
    ds_test_a = SubstrateDataset(test_raw, test_b, test_labels, mode="a")
    loader_train_a = DataLoader(ds_train_a, batch_size=cfg.batch_size, shuffle=True)
    loader_test_a = DataLoader(ds_test_a, batch_size=cfg.batch_size, shuffle=False)

    train_model(model_a, loader_train_a, cfg, device)
    results_a = evaluate_model(model_a, loader_test_a, cfg.num_classes, device)

    print(f" done ({time.time() - t0:.1f}s)")

    # --- Model A+B early fusion ---
    print("  Training Model A+B early fusion...", end="", flush=True)
    t0 = time.time()

    torch.manual_seed(cfg.seed)
    model_ab_early = VideoClassifier(in_channels=1 + n_b, num_classes=cfg.num_classes).to(device)
    ds_train_ab = SubstrateDataset(train_raw, train_b, train_labels, mode="a+b",
                                    b_mean=b_mean, b_std=b_std)
    ds_test_ab = SubstrateDataset(test_raw, test_b, test_labels, mode="a+b",
                                   b_mean=b_mean, b_std=b_std)
    loader_train_ab = DataLoader(ds_train_ab, batch_size=cfg.batch_size, shuffle=True)
    loader_test_ab = DataLoader(ds_test_ab, batch_size=cfg.batch_size, shuffle=False)

    train_model(model_ab_early, loader_train_ab, cfg, device)
    results_ab_early = evaluate_model(model_ab_early, loader_test_ab, cfg.num_classes, device)

    print(f" done ({time.time() - t0:.1f}s)")

    # --- Model A+B late fusion ---
    print("  Training Model A+B late fusion...", end="", flush=True)
    t0 = time.time()

    torch.manual_seed(cfg.seed)
    model_ab_late = LateFusionClassifier(
        raw_channels=1, b_channels=n_b, num_classes=cfg.num_classes
    ).to(device)
    ds_train_late = SubstrateDataset(train_raw, train_b, train_labels, mode="late_fusion",
                                      b_mean=b_mean, b_std=b_std)
    ds_test_late = SubstrateDataset(test_raw, test_b, test_labels, mode="late_fusion",
                                     b_mean=b_mean, b_std=b_std)
    loader_train_late = DataLoader(ds_train_late, batch_size=cfg.batch_size, shuffle=True)
    loader_test_late = DataLoader(ds_test_late, batch_size=cfg.batch_size, shuffle=False)

    train_model(model_ab_late, loader_train_late, cfg, device, late_fusion=True)
    results_ab_late = evaluate_model(model_ab_late, loader_test_late, cfg.num_classes, device,
                                      late_fusion=True)

    print(f" done ({time.time() - t0:.1f}s)")

    return results_a, results_ab_early, results_ab_late
