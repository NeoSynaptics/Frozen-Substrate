"""Smoke tests for SubstrateStack and Readout."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from frozen_substrate.redesign.config import SubstrateConfig, ReadoutConfig
from frozen_substrate.redesign.stack import SubstrateStack
from frozen_substrate.redesign.readout import Readout
from frozen_substrate.redesign.ops import blur3x3, global_rms_normalize


def test_stack_shapes():
    cfg = SubstrateConfig(height=16, width=16, n_layers=5, noise_std=0.0)
    stack = SubstrateStack(cfg, seed=0)

    assert stack.x.shape == (5, 16, 16)

    stim = np.ones((16, 16), dtype=np.float32) * 0.5
    stack.inject_l0(stim, gain=0.3)
    stack.step()

    assert stack.x.shape == (5, 16, 16)
    assert np.isfinite(stack.x).all()
    print("PASS: test_stack_shapes")


def test_stack_depth_decay():
    cfg = SubstrateConfig(height=16, width=16, n_layers=8, noise_std=0.0)
    stack = SubstrateStack(cfg, seed=0)

    for _ in range(50):
        stim = np.ones((16, 16), dtype=np.float32) * 0.5
        stack.inject_l0(stim, gain=0.3)
        stack.step()

    # Activity should generally decrease with depth
    layer_means = [float(np.abs(stack.x[l]).mean()) for l in range(8)]
    # L0 should be most active (receives direct injection)
    assert layer_means[0] > layer_means[-1], \
        f"L0 ({layer_means[0]:.4f}) should be > L7 ({layer_means[-1]:.4f})"
    print("PASS: test_stack_depth_decay")


def test_readout_cube_shape():
    cfg = SubstrateConfig(height=12, width=12, n_layers=6, noise_std=0.0)
    rcfg = ReadoutConfig(
        a_layers=(0, 1), b_layers=(2, 3, 4),
        integrate_steps=3,
    )
    stack = SubstrateStack(cfg, seed=0)
    readout = Readout(rcfg, n_layers=6, height=12, width=12)

    frames = []
    for _ in range(3):
        stack.inject_l0(np.random.default_rng(42).standard_normal((12, 12)).astype(np.float32), gain=0.2)
        stack.step()
        frames.append(stack.x.copy())

    window = np.stack(frames, axis=0)
    cube, meta = readout.emit_cube(window)

    assert cube.shape == (5, 12, 12), f"Bad cube shape: {cube.shape}"
    assert meta["cube_channels"] == 5
    assert meta["a_layers"] == (0, 1)
    assert meta["b_layers"] == (2, 3, 4)
    print("PASS: test_readout_cube_shape")


def test_blur3x3():
    x = np.zeros((8, 8), dtype=np.float32)
    x[4, 4] = 1.0
    blurred = blur3x3(x)
    assert blurred.shape == (8, 8)
    assert blurred[4, 4] > blurred[0, 0]
    assert float(blurred.sum()) > 0
    print("PASS: test_blur3x3")


def test_rms_normalize():
    x = np.random.default_rng(0).standard_normal((10, 10)).astype(np.float32) * 5.0
    normed = global_rms_normalize(x, target=0.25, eps=1e-6)
    rms = float(np.sqrt(np.mean(normed**2)))
    assert abs(rms - 0.25) < 0.01, f"RMS should be ~0.25, got {rms:.4f}"
    print("PASS: test_rms_normalize")


def test_stack_reset():
    cfg = SubstrateConfig(height=8, width=8, n_layers=4)
    stack = SubstrateStack(cfg, seed=0)

    for _ in range(20):
        stack.inject_l0(np.ones((8, 8), dtype=np.float32), gain=0.5)
        stack.step()

    stack.reset()
    assert np.allclose(stack.x, 0.0)
    print("PASS: test_stack_reset")


if __name__ == "__main__":
    test_stack_shapes()
    test_stack_depth_decay()
    test_readout_cube_shape()
    test_blur3x3()
    test_rms_normalize()
    test_stack_reset()
    print("\nAll stack tests passed.")
