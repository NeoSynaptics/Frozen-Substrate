
# Frozen Substrate v1.0 (Reference)

This is a locked reference implementation of the Frozen Substrate
with explicit persistence (Channel B) and existence (L0).

Grid: 40x40
Depth: 20 layers
Goal: Stable propagation regime (interface, not demo)

## Key Properties
- Depth-scaled leakage and gain
- Local diffusion only
- Passive Channel B (baseline + residual)
- Mid-entropy persistence survives deepest

## Contents
- substrate.py        : core dynamics
- config.py           : all tunable parameters (v1.0 frozen)
- run_demo.py         : synthetic motion demo
- metrics.py          : interpretable metrics & plots

This version is intended to be trusted, reused, and scaled.
