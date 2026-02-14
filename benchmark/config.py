"""Benchmark experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Hyperparameters for the A vs A+B ablation experiment."""

    # Substrate
    grid_size: int = 50
    n_layers: int = 10
    integrate_steps: int = 4

    # Synthetic data
    n_train: int = 400
    n_test: int = 100
    clip_frames: int = 16       # frames per synthetic clip
    num_classes: int = 4        # static, periodic, drift, noise

    # Training
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    seed: int = 42

    # Output
    output_dir: str = "benchmark_results"

    @property
    def cubes_per_clip(self) -> int:
        return self.clip_frames // self.integrate_steps
