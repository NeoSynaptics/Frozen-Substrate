"""Smoke tests for the production pipeline."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from frozen_substrate.redesign import Pipeline, SubstrateConfig, ReadoutConfig, VideoIOConfig


def test_pipeline_produces_cubes():
    scfg = SubstrateConfig(height=16, width=16, n_layers=6, noise_std=0.0)
    rcfg = ReadoutConfig(
        a_layers=(0, 1), b_layers=(2, 3, 4),
        integrate_steps=4, baseline_alpha=0.05,
    )
    vcfg = VideoIOConfig()
    pipe = Pipeline(scfg, rcfg, vcfg, seed=0)

    cubes = []
    for t in range(20):
        frame = np.random.default_rng(t).standard_normal((16, 16)).astype(np.float32)
        out = pipe.process_frame(frame)
        if out is not None:
            cube, meta = out
            cubes.append(cube)
            assert cube.shape == (5, 16, 16), f"Bad shape: {cube.shape}"
            assert meta["cube_channels"] == 5

    assert len(cubes) == 5, f"Expected 5 cubes from 20 frames / 4 steps, got {len(cubes)}"
    print("PASS: test_pipeline_produces_cubes")


def test_buffer_does_not_leak():
    scfg = SubstrateConfig(height=8, width=8, n_layers=4, noise_std=0.0)
    rcfg = ReadoutConfig(
        a_layers=(0,), b_layers=(1, 2),
        integrate_steps=3,
    )
    vcfg = VideoIOConfig()
    pipe = Pipeline(scfg, rcfg, vcfg, seed=0)

    for t in range(100):
        frame = np.zeros((8, 8), dtype=np.float32)
        pipe.process_frame(frame)
        # Buffer should never exceed integrate_steps
        assert len(pipe._buffer) < rcfg.integrate_steps, \
            f"Buffer leaked: {len(pipe._buffer)} entries at t={t}"

    print("PASS: test_buffer_does_not_leak")


def test_pipeline_reset():
    scfg = SubstrateConfig(height=8, width=8, n_layers=4)
    rcfg = ReadoutConfig(a_layers=(0,), b_layers=(1, 2), integrate_steps=2)
    vcfg = VideoIOConfig()
    pipe = Pipeline(scfg, rcfg, vcfg, seed=0)

    # Process some frames
    for t in range(10):
        pipe.process_frame(np.ones((8, 8), dtype=np.float32) * 0.5)

    pipe.reset()
    assert pipe.frame_count == 0
    assert len(pipe._buffer) == 0
    assert np.allclose(pipe.substrate.x, 0.0)
    print("PASS: test_pipeline_reset")


def test_config_presets():
    fast = SubstrateConfig.fast()
    assert fast.height == 32 and fast.width == 32 and fast.n_layers == 6

    hi = SubstrateConfig.high_res()
    assert hi.height == 128 and hi.width == 128 and hi.n_layers == 15

    default = SubstrateConfig.default()
    assert default.height == 50 and default.width == 50 and default.n_layers == 10

    rcfg = ReadoutConfig.for_substrate(fast)
    assert len(rcfg.a_layers) > 0
    assert len(rcfg.b_layers) > 0
    assert max(rcfg.b_layers) < fast.n_layers
    print("PASS: test_config_presets")


if __name__ == "__main__":
    test_pipeline_produces_cubes()
    test_buffer_does_not_leak()
    test_pipeline_reset()
    test_config_presets()
    print("\nAll pipeline tests passed.")
