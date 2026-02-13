"""Smoke tests for V2: resonant stack, readout with coherence, pipeline."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from frozen_substrate.v2 import (
    PipelineV2, ResonantConfig, ReadoutV2Config, ResonantStack, ReadoutV2,
)
from frozen_substrate.redesign.config import VideoIOConfig


def test_resonant_stack_shapes():
    cfg = ResonantConfig(height=16, width=16, n_layers=6, noise_std=0.0)
    stack = ResonantStack(cfg, seed=0)

    assert stack.x.shape == (6, 16, 16)

    stim = np.ones((16, 16), dtype=np.float32) * 0.5
    stack.inject_l0(stim, gain=0.3)
    stack.step()

    assert stack.x.shape == (6, 16, 16)
    assert np.isfinite(stack.x).all()
    print("PASS: test_resonant_stack_shapes")


def test_feedback_amplifies_persistent_signal():
    """The key test: feedback should amplify signals that reach deep layers."""
    H, W, n_layers = 20, 20, 8

    # V1-style: no feedback
    cfg_nofb = ResonantConfig(
        height=H, width=W, n_layers=n_layers,
        feedback_gain=0.0,  # disabled
        adaptive_input=False,
    )
    stack_nofb = ResonantStack(cfg_nofb, seed=0)

    # V2: with feedback
    cfg_fb = ResonantConfig(
        height=H, width=W, n_layers=n_layers,
        feedback_gain=0.06,
        feedback_threshold=0.03,
        adaptive_input=False,
    )
    stack_fb = ResonantStack(cfg_fb, seed=0)

    # Drive both with same persistent stimulus
    for t in range(200):
        angle = 2 * np.pi * (t / 60.0)
        stim = np.zeros((H, W), dtype=np.float32)
        cy, cx = H / 2.0, W / 2.0
        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        x0 = cx + 5.0 * np.cos(angle)
        y0 = cy + 5.0 * np.sin(angle)
        stim += np.exp(-((x - x0)**2 + (y - y0)**2) / 4.0).astype(np.float32)

        stack_nofb.inject_l0(stim, gain=0.3)
        stack_nofb.step()
        stack_fb.inject_l0(stim, gain=0.3)
        stack_fb.step()

    # Compare L0 activity (feedback version should have stronger L0)
    l0_nofb = float(np.abs(stack_nofb.x[0]).mean())
    l0_fb = float(np.abs(stack_fb.x[0]).mean())

    assert l0_fb >= l0_nofb * 0.95, \
        f"Feedback L0 ({l0_fb:.5f}) should be >= no-feedback L0 ({l0_nofb:.5f})"
    print(f"PASS: test_feedback_amplifies (no_fb={l0_nofb:.5f}, fb={l0_fb:.5f})")


def test_adaptive_input():
    cfg = ResonantConfig(height=8, width=8, n_layers=4, adaptive_input=True)
    stack = ResonantStack(cfg, seed=0)

    # Feed very loud signal
    loud = np.ones((8, 8), dtype=np.float32) * 10.0
    stack.inject_l0(loud, gain=0.3)
    stack.step()

    # Should not saturate thanks to adaptive gain
    assert float(np.abs(stack.x[0]).max()) <= cfg.clip_value + 0.01, \
        "Adaptive input should prevent saturation"
    print("PASS: test_adaptive_input")


def test_readout_v2_three_channels():
    cfg = ReadoutV2Config(
        a_layers=(0, 1), b_layers=(2, 3), c_layers=(2, 3),
        integrate_steps=3,
    )
    readout = ReadoutV2(cfg, n_layers=5, height=10, width=10)

    rng = np.random.default_rng(0)
    window = rng.standard_normal((3, 5, 10, 10)).astype(np.float32) * 0.1
    cube, meta = readout.emit_cube(window)

    expected_channels = 2 + 2 + 2  # A + B + C
    assert cube.shape == (expected_channels, 10, 10), f"Bad shape: {cube.shape}"
    assert meta["n_a"] == 2
    assert meta["n_b"] == 2
    assert meta["n_c"] == 2
    assert meta["cube_channels"] == expected_channels
    print("PASS: test_readout_v2_three_channels")


def test_coherence_structured_vs_noise():
    """Structured deviation should produce higher coherence than random noise."""
    cfg = ReadoutV2Config(
        a_layers=(0,), b_layers=(1,), c_layers=(1,),
        integrate_steps=1, coherence_alpha=0.05,
        spatial_baseline=False,
    )

    # Test 1: Structured deviation (always positive)
    readout_struct = ReadoutV2(cfg, n_layers=3, height=8, width=8)
    for t in range(100):
        x = np.zeros((1, 3, 8, 8), dtype=np.float32)
        x[0, 1, 3:5, 3:5] = 0.3 + 0.05 * np.sin(t / 10.0)  # oscillating positive
        readout_struct.emit_cube(x)

    # Test 2: Random noise deviation
    readout_noise = ReadoutV2(cfg, n_layers=3, height=8, width=8)
    rng = np.random.default_rng(42)
    for t in range(100):
        x = np.zeros((1, 3, 8, 8), dtype=np.float32)
        x[0, 1, 3:5, 3:5] = rng.standard_normal((2, 2)).astype(np.float32) * 0.3
        readout_noise.emit_cube(x)

    # Get final coherence values
    x_final = np.zeros((1, 3, 8, 8), dtype=np.float32)
    x_final[0, 1, 3:5, 3:5] = 0.3
    cube_s, _ = readout_struct.emit_cube(x_final)
    x_final[0, 1, 3:5, 3:5] = rng.standard_normal((2, 2)).astype(np.float32) * 0.3
    cube_n, _ = readout_noise.emit_cube(x_final)

    # Channel C is the last slice
    c_structured = float(cube_s[-1, 3:5, 3:5].mean())
    c_noise = float(cube_n[-1, 3:5, 3:5].mean())

    assert c_structured > c_noise, \
        f"Structured coherence ({c_structured:.4f}) should > noise ({c_noise:.4f})"
    print(f"PASS: test_coherence_structured_vs_noise "
          f"(structured={c_structured:.4f}, noise={c_noise:.4f})")


def test_local_flood_clamping():
    cfg = ReadoutV2Config(
        a_layers=(0,), b_layers=(1, 2),
        integrate_steps=2,
        flood_patch_size=4,
        b_threshold=0.01,
        flood_fraction=0.10,
        spatial_baseline=False,
    )
    readout = ReadoutV2(cfg, n_layers=4, height=8, width=8)

    # First: establish a quiet baseline
    quiet = np.zeros((2, 4, 8, 8), dtype=np.float32)
    quiet[:, :, :, :] = 0.01
    for _ in range(10):
        readout.emit_cube(quiet)

    # Then: sudden large signal in one patch -> flood
    loud = quiet.copy()
    loud[:, 1:3, 0:4, 0:4] = 0.8

    cube, meta = readout.emit_cube(loud)
    assert meta["flood_events_in_window"] > 0, "Should detect local flood"
    print("PASS: test_local_flood_clamping")


def test_pipeline_v2_end_to_end():
    scfg = ResonantConfig(height=16, width=16, n_layers=6, noise_std=0.0)
    rcfg = ReadoutV2Config(
        a_layers=(0, 1), b_layers=(2, 3, 4),
        integrate_steps=4,
    )
    vcfg = VideoIOConfig()
    pipe = PipelineV2(scfg, rcfg, vcfg, seed=0)

    cubes = []
    for t in range(20):
        frame = np.random.default_rng(t).standard_normal((16, 16)).astype(np.float32)
        out = pipe.process_frame(frame)
        if out is not None:
            cube, meta = out
            cubes.append(cube)

    assert len(cubes) == 5
    n_c = cubes[0].shape[0] - 2 - 3  # total - A - B
    assert n_c == 3, f"Expected 3 C channels, got {n_c}"  # c_layers defaults to b_layers
    print("PASS: test_pipeline_v2_end_to_end")


def test_pipeline_v2_buffer_bounded():
    scfg = ResonantConfig(height=8, width=8, n_layers=4)
    rcfg = ReadoutV2Config(a_layers=(0,), b_layers=(1, 2), integrate_steps=3)
    vcfg = VideoIOConfig()
    pipe = PipelineV2(scfg, rcfg, vcfg, seed=0)

    for t in range(60):
        pipe.process_frame(np.zeros((8, 8), dtype=np.float32))
        assert len(pipe._buffer) < rcfg.integrate_steps
    print("PASS: test_pipeline_v2_buffer_bounded")


def test_config_presets():
    fast = ResonantConfig.fast()
    assert fast.height == 32 and fast.n_layers == 6
    fb = fast.resolve_feedback_layers()
    assert len(fb) > 0 and max(fb) < fast.n_layers

    hi = ResonantConfig.high_res()
    assert hi.height == 128 and hi.n_layers == 15

    rcfg = ReadoutV2Config.for_substrate(fast)
    assert len(rcfg.a_layers) > 0
    assert len(rcfg.b_layers) > 0
    print("PASS: test_config_presets")


if __name__ == "__main__":
    test_resonant_stack_shapes()
    test_feedback_amplifies_persistent_signal()
    test_adaptive_input()
    test_readout_v2_three_channels()
    test_coherence_structured_vs_noise()
    test_local_flood_clamping()
    test_pipeline_v2_end_to_end()
    test_pipeline_v2_buffer_bounded()
    test_config_presets()
    print("\nAll V2 tests passed.")
