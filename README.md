# Frozen Substrate

A bio-inspired computational substrate where a 2D grid of neuron-like cells self-organize through local prediction, plasticity, and survival dynamics. Activity propagates through passive depth layers, creating a persistence filter that selectively passes mid-entropy stimuli.

**Core idea:** *Information = persistence of deviation from expectation under degradation.*

## Usage

```bash
pip install -r requirements.txt

# Run synthetic demo (no dependencies beyond numpy)
python -m frozen_substrate demo

# Run demo and render Channel A/B as PNG sequences
python -m frozen_substrate demo --render -o output

# Process a video file (requires opencv-python)
python -m frozen_substrate process video.mp4 -o output

# Process webcam input
python -m frozen_substrate process --webcam -o output

# Show available presets and options
python -m frozen_substrate info
```

### Output

The pipeline produces `(N, C, H, W)` tensor cubes saved as `.npz`:

| Channel | Layers | Meaning |
|---------|--------|---------|
| **Channel A** | Shallow (L0-L2) | "What exists" -- raw substrate state |
| **Channel B** | Mid-depth (L3-L6) | "What matters" -- persistence of deviation from expectation |

Load output in Python:
```python
import numpy as np
data = np.load("output/cubes.npz")
cubes = data["cubes"]     # shape (N, C, H, W)
a_layers = data["a_layers"]
b_layers = data["b_layers"]
```

### Presets

| Preset | Grid | Layers | Use case |
|--------|------|--------|----------|
| `default` | 50x50 | 10 | General purpose |
| `fast` | 32x32 | 6 | Quick experiments, real-time |
| `high_res` | 128x128 | 15 | High spatial detail |

```bash
python -m frozen_substrate demo --preset fast
python -m frozen_substrate process video.mp4 --preset high_res
```

## Architecture

### FrozenCoreV3 (Layer 0 -- Active Substrate)

A 2D grid where each cell:

- **Predicts its own next state** using a local self-prediction coefficient
- Has **8-neighbor weighted synapses** that learn via Hebbian-like rules gated by prediction error
- Maintains **survival energy** -- cells that consistently fail to predict are pruned
- Emits **surprise** (prediction error above baseline) into a **Coordination Field (C)** that diffuses spatially
- The field has **three temporal projections** (fast/mid/slow EMA) that modulate coupling, learning, and death at different timescales

### Multi-Layer Depth Stack (L1+ -- Passive Layers)

Passive layers receive blurred, depth-attenuated feedforward from L0:
- No learning, no death -- stable over long runs (10k+ steps)
- Depth-scaled gain decay prevents saturation
- Activity penetration depth serves as a proxy for stimulus persistence

### Production Pipeline (Redesign)

A deterministic, mechanically-defined rewrite for production use:
- No learning, no semantics -- pure local operators
- `SubstrateStack`: multi-layer lossy transformation with RMS normalization
- `Readout`: emits Channel A (existence) + Channel B (residual persistence) cubes
- `Pipeline`: end-to-end video frames -> substrate -> output tensor stream
- Flood clamping prevents Channel B saturation

### Retina + Channel B (Advanced Mode)

- **L0 retina buffer**: EMA accumulator that registers existence
- **Z-score deviation detection** with band-pass gating on L0 output
- **Channel B** (`IntegratedResidual`): tracks temporal deviation from baseline across depth

Key result: **mid-entropy stimuli penetrate deepest** (micro-motion > static > fast flicker), matching biological persistence behavior.

### Ghost Neurons

An alternative readout that tracks deviation from an EMA background model. Conceptually equivalent to Channel B in the production pipeline -- both compute `|activation - EMA(activation)|`. The standalone class is useful for research experiments with the full FrozenCoreV3 substrate.

## Project Structure

```
frozen_substrate/         # Core Python package
    __main__.py           # CLI entry point (python -m frozen_substrate)
    core.py               # FrozenCoreV3 -- the active substrate
    multilayer.py         # Passive multi-layer depth stack
    retina.py             # Retina buffer + Channel B (integrated residual)
    viz.py                # Output visualization and save/load utilities
    coupling.py           # Layer coupling utilities + two-layer demo
    gaussian_pen.py       # Stimulus generators (Gaussian bump, orbit, ring)
    analysis.py           # Plotting utilities for two-layer experiments
    ghost/                # Ghost Neurons readout module
    redesign/             # Production pipeline (deterministic, no learning)
        config.py         # Frozen dataclass configs with presets
        stack.py          # SubstrateStack (clean multi-layer)
        readout.py        # Channel A/B readout with flood clamping
        pipeline.py       # End-to-end: video -> substrate -> cubes
        video.py          # Frame preprocessing

experiments/              # Runnable demos
    multilayer_circle_test.py     # Gaussian pen circle + depth propagation
    retina_channelB_demo.py       # 4-dot persistence test
    simple_depth_demo.py          # Minimal standalone 20-layer demo
    ghost_neuron_demo.py          # Ghost neuron novelty 3D visualization
    redesign_pipeline_demo.py     # Production pipeline on synthetic input
    video_demo.py                 # Video processing demo (file or synthetic)

tests/                    # Smoke tests
    test_core.py          # FrozenCoreV3 stability and correctness
    test_stack.py         # SubstrateStack, Readout, ops
    test_pipeline.py      # End-to-end pipeline, buffer management, presets

reference/                # V3 minimal reference implementation (self-contained)
    config.py / substrate.py / run_demo.py / metrics.py

outputs/                  # Sample output images
```

## Running Experiments

```bash
# Multi-layer circle test (produces depth snapshots)
python experiments/multilayer_circle_test.py

# Retina + Channel B demo (4-dot persistence test)
python experiments/retina_channelB_demo.py

# Simple standalone depth demo
python experiments/simple_depth_demo.py

# Ghost neuron novelty visualization
python experiments/ghost_neuron_demo.py

# Production pipeline demo
python experiments/redesign_pipeline_demo.py

# Video processing demo (synthetic or provide a video path)
python experiments/video_demo.py [optional_video.mp4]
```

## Running Tests

```bash
python tests/test_core.py
python tests/test_stack.py
python tests/test_pipeline.py
```

## Sample Outputs

### Multi-Layer Activity Propagation
![trace](outputs/trace_multilayer.png)

### Depth Snapshots at t=1000
![snapshot](outputs/snap_multi_1000.png)

### Channel B -- Depth Penetration by Stimulus Type
![depth traces](outputs/v3_l1plus_depth_traces.png)

### Channel B -- 3D Novelty Structure (Late)
![3d late](outputs/v3_l1plus_3d_late.png)

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Frozen Core** | The active L0 substrate with prediction, plasticity, and survival |
| **Coordination Field** | Spatial field driven by surprise that gates plasticity and pruning |
| **Channel A** | Existence signal -- shallow layer states |
| **Channel B** | Persistence signal -- integrated residual novelty across depth |
| **Depth Penetration** | How deep a stimulus propagates -- proxy for "perceptual importance" |
| **Ghost Neurons** | EMA-baseline deviation readout (= Channel B concept for research mode) |
| **Flood Clamping** | Automatic saturation detection and scaling to keep Channel B stable |

## Status

Functional research tool. The core dynamics are stable, the production pipeline processes video end-to-end, and the depth filter produces the expected selective behavior: mid-entropy stimuli (micro-motion) persist deepest, while static and high-flicker inputs are suppressed.
