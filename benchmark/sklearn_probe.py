"""Feature-level probe: does Channel B beat simple temporal differencing?

Uses scikit-learn Random Forest — no SGD instability, fast, interpretable.
This is the CRITICAL experiment: if RF(diff) >= RF(ChB), then the multi-layer
substrate cascade adds complexity without benefit over frame[t] - frame[t-1].
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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
        T = raw.shape[0]

        # --- Raw features ---
        rf = []
        for t in range(T):
            frame = raw[t, 0]
            rf.extend([frame.mean(), frame.std(), frame.max(), frame.min()])

        raw_stack = raw[:, 0]  # (T, H, W)
        rf.append(raw_stack.std(axis=0).mean())
        rf.append((raw_stack.max(axis=0) - raw_stack.min(axis=0)).mean())
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
            layer = b[:, l]
            bf.append(layer.mean())
            bf.append(layer.std())
            bf.append(layer.max())
            bf.append(np.abs(layer).mean())
            bf.append(layer.mean(axis=(1, 2)).std())
            lmean = np.abs(layer).mean()
            bf.append(layer.max() / lmean if lmean > 1e-10 else 0.0)

        b_feats.append(bf)

    return np.array(raw_feats, dtype=np.float32), np.array(b_feats, dtype=np.float32)


def extract_diff_features(clip_diff):
    """Extract summary statistics from temporal difference frames.

    Parameters
    ----------
    clip_diff : list of np.ndarray, each shape (T, 1, H, W)

    Returns
    -------
    diff_features : np.ndarray, shape (N, n_diff_feats)
    """
    feats = []
    for diff in clip_diff:
        T = diff.shape[0]
        df = []
        for t in range(T):
            frame = diff[t, 0]
            df.extend([
                frame.mean(),
                frame.std(),
                np.abs(frame).mean(),   # mean absolute change
                frame.max(),
                frame.min(),
            ])

        # Aggregate temporal stats
        diff_stack = diff[:, 0]  # (T, H, W)
        df.append(np.abs(diff_stack).mean())           # overall motion magnitude
        df.append(diff_stack.std())                     # motion variability
        df.append(np.abs(diff_stack).max())             # peak motion
        # Spatial concentration of motion
        abs_sum = np.abs(diff_stack).sum(axis=0)        # (H, W) accumulated motion map
        total = abs_sum.sum()
        if total > 1e-10:
            # Entropy-like measure: concentrated motion vs diffuse
            p = abs_sum / total
            df.append(-(p * np.log(p + 1e-10)).sum())
        else:
            df.append(0.0)

        feats.append(df)

    return np.array(feats, dtype=np.float32)


def _fit_predict(X_train, y_train, X_test, n_estimators=200):
    """Fit RF and return predictions."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf.predict(X_test), rf


def run_sklearn_probe(train_raw, train_b, train_labels,
                      test_raw, test_b, test_labels,
                      class_names,
                      train_diff=None, test_diff=None,
                      n_estimators=200):
    """Run Random Forest ablation: raw vs diff vs ChB vs combinations.

    The critical comparison: does the substrate beat temporal differencing?
    """

    print("  Extracting summary features...")
    train_raw_f, train_b_f = extract_summary_features(train_raw, train_b)
    test_raw_f, test_b_f = extract_summary_features(test_raw, test_b)

    # Clean NaN/Inf
    train_raw_f = np.nan_to_num(train_raw_f)
    test_raw_f = np.nan_to_num(test_raw_f)
    train_b_f = np.nan_to_num(train_b_f)
    test_b_f = np.nan_to_num(test_b_f)

    has_diff = train_diff is not None
    if has_diff:
        train_diff_f = np.nan_to_num(extract_diff_features(train_diff))
        test_diff_f = np.nan_to_num(extract_diff_features(test_diff))

    print(f"    Raw features:  {train_raw_f.shape[1]} dims")
    if has_diff:
        print(f"    Diff features: {train_diff_f.shape[1]} dims")
    print(f"    ChB features:  {train_b_f.shape[1]} dims")

    # --- Train all RF models ---
    models = {}

    print("  Training RF (raw only)...", end="", flush=True)
    pred, rf = _fit_predict(train_raw_f, train_labels, test_raw_f, n_estimators)
    acc = accuracy_score(test_labels, pred)
    models['raw'] = {'pred': pred, 'acc': acc, 'rf': rf}
    print(f" {acc:.2f}")

    if has_diff:
        print("  Training RF (diff only)...", end="", flush=True)
        pred, rf = _fit_predict(train_diff_f, train_labels, test_diff_f, n_estimators)
        acc = accuracy_score(test_labels, pred)
        models['diff'] = {'pred': pred, 'acc': acc, 'rf': rf}
        print(f" {acc:.2f}")

    print("  Training RF (ChB only)...", end="", flush=True)
    pred, rf = _fit_predict(train_b_f, train_labels, test_b_f, n_estimators)
    acc = accuracy_score(test_labels, pred)
    models['chb'] = {'pred': pred, 'acc': acc, 'rf': rf}
    print(f" {acc:.2f}")

    if has_diff:
        print("  Training RF (raw + diff)...", end="", flush=True)
        pred, rf = _fit_predict(
            np.concatenate([train_raw_f, train_diff_f], axis=1), train_labels,
            np.concatenate([test_raw_f, test_diff_f], axis=1), n_estimators)
        acc = accuracy_score(test_labels, pred)
        models['raw+diff'] = {'pred': pred, 'acc': acc, 'rf': rf}
        print(f" {acc:.2f}")

    print("  Training RF (raw + ChB)...", end="", flush=True)
    pred, rf = _fit_predict(
        np.concatenate([train_raw_f, train_b_f], axis=1), train_labels,
        np.concatenate([test_raw_f, test_b_f], axis=1), n_estimators)
    acc = accuracy_score(test_labels, pred)
    models['raw+chb'] = {'pred': pred, 'acc': acc, 'rf': rf}
    print(f" {acc:.2f}")

    if has_diff:
        print("  Training RF (raw + diff + ChB)...", end="", flush=True)
        pred, rf = _fit_predict(
            np.concatenate([train_raw_f, train_diff_f, train_b_f], axis=1), train_labels,
            np.concatenate([test_raw_f, test_diff_f, test_b_f], axis=1), n_estimators)
        acc = accuracy_score(test_labels, pred)
        models['raw+diff+chb'] = {'pred': pred, 'acc': acc, 'rf': rf}
        print(f" {acc:.2f}")

    # --- Per-class breakdown ---
    n_classes = len(class_names)
    print()
    if has_diff:
        header = f"  {'':16s}   {'raw':>5s}   {'diff':>5s}   {'ChB':>5s}   {'r+d':>5s}   {'r+B':>5s}   {'all':>5s}"
    else:
        header = f"  {'':16s}   {'raw':>5s}   {'ChB':>5s}   {'r+B':>5s}"
    print(header)

    for c in range(n_classes):
        mask = test_labels == c
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        row = f"    {name:16s}"
        for key in (['raw', 'diff', 'chb', 'raw+diff', 'raw+chb', 'raw+diff+chb'] if has_diff
                     else ['raw', 'chb', 'raw+chb']):
            if key in models:
                ca = accuracy_score(test_labels[mask], models[key]['pred'][mask])
                row += f"   {ca:.2f}"
        print(row)

    # --- Critical comparison ---
    if has_diff:
        print()
        print("  CRITICAL: Substrate vs Temporal Differencing")
        print("  " + "-" * 50)
        d_acc = models['diff']['acc']
        b_acc = models['chb']['acc']
        rd_acc = models['raw+diff']['acc']
        rb_acc = models['raw+chb']['acc']
        rdb_acc = models['raw+diff+chb']['acc']

        winner_alone = "SUBSTRATE" if b_acc > d_acc else "TEMP DIFF" if d_acc > b_acc else "TIE"
        winner_aug = "SUBSTRATE" if rb_acc > rd_acc else "TEMP DIFF" if rd_acc > rb_acc else "TIE"
        adds_value = rdb_acc > rd_acc

        print(f"    Alone:      diff={d_acc:.2f}  vs  ChB={b_acc:.2f}  -->  {winner_alone}")
        print(f"    With raw:   raw+diff={rd_acc:.2f}  vs  raw+ChB={rb_acc:.2f}  -->  {winner_aug}")
        print(f"    Combined:   raw+diff+ChB={rdb_acc:.2f}  (vs raw+diff={rd_acc:.2f})")
        if adds_value:
            print(f"    --> Substrate adds +{(rdb_acc - rd_acc)*100:.0f} pts beyond temporal diff")
        else:
            print(f"    --> Substrate adds NOTHING beyond temporal differencing")

    # Build results dict
    result = {f"accuracy_{k}": v['acc'] for k, v in models.items()}
    return result
