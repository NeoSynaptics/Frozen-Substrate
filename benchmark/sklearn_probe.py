"""Feature-level probe: do Channel B statistics carry discriminative information?

Uses scikit-learn Random Forest — no SGD instability, fast, interpretable.
This complements the Conv3D ablation with a cleaner signal-vs-noise test.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def extract_summary_features(clip_raw, clip_b):
    """Extract summary statistics from raw frames and Channel B.

    Parameters
    ----------
    clip_raw : list of np.ndarray, each shape (T, 1, H, W)
    clip_b : list of np.ndarray, each shape (T, n_b, H, W)

    Returns
    -------
    raw_features : np.ndarray, shape (N, n_raw_feats)
    b_features : np.ndarray, shape (N, n_b_feats)
    """
    raw_feats = []
    b_feats = []

    for raw, b in zip(clip_raw, clip_b):
        # raw: (T, 1, H, W)
        T = raw.shape[0]

        # --- Raw features ---
        rf = []
        for t in range(T):
            frame = raw[t, 0]  # (H, W)
            rf.extend([frame.mean(), frame.std(), frame.max(), frame.min()])

        # Temporal features: pixel-wise stats across frames
        raw_stack = raw[:, 0]  # (T, H, W)
        # Mean temporal variance (how much each pixel changes over time)
        rf.append(raw_stack.std(axis=0).mean())
        # Mean temporal range
        rf.append((raw_stack.max(axis=0) - raw_stack.min(axis=0)).mean())
        # Temporal autocorrelation (frame-to-frame similarity)
        if T > 1:
            diffs = np.array([np.abs(raw_stack[t+1] - raw_stack[t]).mean() for t in range(T-1)])
            rf.append(diffs.mean())
            rf.append(diffs.std())
        else:
            rf.extend([0.0, 0.0])

        raw_feats.append(rf)

        # --- Channel B features ---
        n_b = b.shape[1]
        bf = []
        for l in range(n_b):
            layer = b[:, l]  # (T, H, W)
            # Per-layer statistics
            bf.append(layer.mean())
            bf.append(layer.std())
            bf.append(layer.max())
            bf.append(np.abs(layer).mean())  # mean absolute value
            # Temporal stats of this layer
            bf.append(layer.mean(axis=(1, 2)).std())  # variability of spatial mean over time
            # Spatial structure: ratio of max to mean (peakedness)
            lmean = np.abs(layer).mean()
            bf.append(layer.max() / lmean if lmean > 1e-10 else 0.0)

        b_feats.append(bf)

    return np.array(raw_feats, dtype=np.float32), np.array(b_feats, dtype=np.float32)


def run_sklearn_probe(train_raw, train_b, train_labels,
                      test_raw, test_b, test_labels,
                      class_names, n_estimators=200):
    """Run Random Forest ablation: raw features vs raw+ChB features."""

    print("  Extracting summary features...")
    train_raw_f, train_b_f = extract_summary_features(train_raw, train_b)
    test_raw_f, test_b_f = extract_summary_features(test_raw, test_b)

    print(f"    Raw features: {train_raw_f.shape[1]} dims")
    print(f"    ChB features: {train_b_f.shape[1]} dims")

    # Replace any NaN/Inf
    train_raw_f = np.nan_to_num(train_raw_f)
    test_raw_f = np.nan_to_num(test_raw_f)
    train_b_f = np.nan_to_num(train_b_f)
    test_b_f = np.nan_to_num(test_b_f)

    # --- Model A: raw features only ---
    print("  Training RF (raw only)...", end="", flush=True)
    rf_a = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf_a.fit(train_raw_f, train_labels)
    pred_a = rf_a.predict(test_raw_f)
    acc_a = accuracy_score(test_labels, pred_a)
    print(f" accuracy = {acc_a:.2f}")

    # --- Model B: Channel B features only ---
    print("  Training RF (ChB only)...", end="", flush=True)
    rf_b = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf_b.fit(train_b_f, train_labels)
    pred_b = rf_b.predict(test_b_f)
    acc_b = accuracy_score(test_labels, pred_b)
    print(f" accuracy = {acc_b:.2f}")

    # --- Model A+B: raw + Channel B features ---
    print("  Training RF (raw + ChB)...", end="", flush=True)
    train_ab = np.concatenate([train_raw_f, train_b_f], axis=1)
    test_ab = np.concatenate([test_raw_f, test_b_f], axis=1)
    rf_ab = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf_ab.fit(train_ab, train_labels)
    pred_ab = rf_ab.predict(test_ab)
    acc_ab = accuracy_score(test_labels, pred_ab)
    print(f" accuracy = {acc_ab:.2f}")

    # --- Per-class breakdown ---
    n_classes = len(class_names)
    print()
    print(f"  {'':16s}   {'A':>5s}   {'B':>5s}   {'A+B':>5s}")
    for c in range(n_classes):
        mask = test_labels == c
        ca = accuracy_score(test_labels[mask], pred_a[mask])
        cb = accuracy_score(test_labels[mask], pred_b[mask])
        cab = accuracy_score(test_labels[mask], pred_ab[mask])
        best_delta = max(cab - ca, cb - ca)
        marker = " <-- substrate helps" if best_delta > 0.05 else ""
        print(f"    {class_names[c]:16s}   {ca:.2f}   {cb:.2f}   {cab:.2f}{marker}")

    # --- Feature importances ---
    print()
    print("  Top 5 important features (A+B model):")
    feat_names = [f"raw_{i}" for i in range(train_raw_f.shape[1])] + \
                 [f"chb_{i}" for i in range(train_b_f.shape[1])]
    importances = rf_ab.feature_importances_
    top5 = np.argsort(importances)[-5:][::-1]
    for idx in top5:
        print(f"    {feat_names[idx]:12s}  importance = {importances[idx]:.3f}")

    return {
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "accuracy_ab": acc_ab,
        "delta": acc_ab - acc_a,
    }
