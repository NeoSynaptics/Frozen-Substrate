# Frozen Substrate

A bio-inspired computational substrate where a 2D grid of neuron-like cells self-organize through local prediction, plasticity, and survival dynamics. Activity propagates through passive depth layers, creating a persistence filter that selectively passes mid-entropy stimuli.

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

### Retina + Channel B (Advanced Mode)

- **L0 retina buffer**: EMA accumulator that registers existence
- **Z-score deviation detection** with band-pass gating on L0 output
- **Channel B** (`IntegratedResidual`): tracks temporal deviation from baseline across depth -- a persistence metric

Key result: **mid-entropy stimuli penetrate deepest** (micro-motion > static > fast flicker), matching biological persistence behavior.

## Project Structure

```
frozen_substrate/         # Core Python package
    core.py               # FrozenCoreV3 -- the active substrate
    multilayer.py         # Passive multi-layer depth stack
    retina.py             # Retina buffer + Channel B (integrated residual)
    coupling.py           # Layer coupling utilities + two-layer demo
    gaussian_pen.py       # Stimulus generators (Gaussian bump, orbit, ring)
    analysis.py           # Plotting utilities for two-layer experiments

experiments/              # Runnable demos
    multilayer_circle_test.py   # Gaussian pen traces a circle, depth propagation
    retina_channelB_demo.py     # 4-dot test: static vs moving vs flicker vs burst
    simple_depth_demo.py        # Minimal 20-layer standalone depth demo

reference/                # V3 minimal reference implementation (self-contained)
    config.py / substrate.py / run_demo.py / metrics.py

outputs/                  # Sample output images
```

## Quick Start

```bash
pip install -r requirements.txt

# Multi-layer circle test (produces depth snapshots)
python experiments/multilayer_circle_test.py

# Retina + Channel B demo (4-dot persistence test)
python experiments/retina_channelB_demo.py

# Simple standalone depth demo
python experiments/simple_depth_demo.py
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
| **Channel A** | Existence signal (L0 retina buffer) |
| **Channel B** | Persistence signal (integrated residual novelty across depth) |
| **Depth Penetration** | How deep a stimulus propagates -- proxy for "perceptual importance" |

## Status

Research code. The core dynamics are stable and the depth filter produces the expected selective behavior. Parameter sweeps and further experiments are ongoing.
