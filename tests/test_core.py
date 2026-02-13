"""Smoke tests for FrozenCoreV3."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from frozen_substrate.core import FrozenCoreV3


def test_core_runs_without_nan():
    core = FrozenCoreV3(H=20, W=20, seed=42, enable_pacemakers=False)
    core.run(steps=500)

    assert np.isfinite(core.x).all(), "x contains non-finite values"
    assert np.isfinite(core.C).all(), "C contains non-finite values"
    assert np.isfinite(core.s).all(), "s contains non-finite values"
    assert core.t == 500
    print("PASS: test_core_runs_without_nan")


def test_core_external_drive():
    core = FrozenCoreV3(H=16, W=16, seed=0, enable_external_drive=True)

    stim = np.zeros((16, 16), dtype=np.float32)
    stim[7:9, 7:9] = 1.0

    for _ in range(100):
        core.add_drive(stim, gain=0.3)
        core.step()

    # The driven region should have higher activity
    center = float(np.abs(core.x[6:10, 6:10]).mean())
    edge = float(np.abs(core.x[0:4, 0:4]).mean())
    assert center > edge, f"Center ({center:.4f}) should be more active than edge ({edge:.4f})"
    print("PASS: test_core_external_drive")


def test_core_traces():
    core = FrozenCoreV3(H=10, W=10, seed=1)
    core.run(steps=200)

    assert len(core.trace_mean_abs_e) == 200
    assert len(core.trace_alive_frac) == 200
    assert len(core.trace_mean_C) == 200
    assert all(0.0 <= v <= 1.0 for v in core.trace_alive_frac)
    print("PASS: test_core_traces")


def test_core_snapshot():
    core = FrozenCoreV3(H=10, W=10, seed=3)
    core.run(steps=50)
    snap = core.snapshot()

    expected_keys = {"x", "alive", "error", "C", "A_fast", "A_mid", "A_slow", "survival", "fatigue"}
    assert set(snap.keys()) == expected_keys
    for k, v in snap.items():
        assert v.shape == (10, 10), f"{k} has wrong shape: {v.shape}"
    print("PASS: test_core_snapshot")


def test_core_reset():
    core = FrozenCoreV3(H=10, W=10, seed=5)
    core.run(steps=100)
    core.reset(seed=5)

    assert core.t == 0
    assert np.allclose(core.x, 0.0)
    assert np.allclose(core.C, 0.0)
    assert len(core.trace_mean_abs_e) == 0
    print("PASS: test_core_reset")


if __name__ == "__main__":
    test_core_runs_without_nan()
    test_core_external_drive()
    test_core_traces()
    test_core_snapshot()
    test_core_reset()
    print("\nAll core tests passed.")
