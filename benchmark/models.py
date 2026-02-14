"""Models for video classification benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_encoder(in_channels: int, feat_dim: int = 64) -> nn.Sequential:
    """Small Conv3D encoder → feature vector."""
    return nn.Sequential(
        nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
        nn.BatchNorm3d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),

        nn.Conv3d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm3d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),

        nn.Conv3d(64, feat_dim, kernel_size=3, padding=1),
        nn.BatchNorm3d(feat_dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool3d(1),
    )


class VideoClassifier(nn.Module):
    """Single-stream Conv3D classifier (for Model A: raw only).

    Input shape: (B, C, T, H, W).
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = _make_encoder(in_channels, feat_dim=128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class LateFusionClassifier(nn.Module):
    """Two-stream late fusion: separate encoders for raw and Channel B.

    Each modality gets its own Conv3D encoder → feature vector.
    Features are concatenated → classifier. This prevents Channel B's
    extra channels from overwhelming the raw signal in early layers.

    Input: tuple (raw, channel_b).
    """

    def __init__(self, raw_channels: int, b_channels: int, num_classes: int,
                 feat_dim: int = 64):
        super().__init__()
        self.raw_encoder = _make_encoder(raw_channels, feat_dim)
        self.b_encoder = _make_encoder(b_channels, feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, raw: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        f_raw = self.raw_encoder(raw).view(raw.size(0), -1)
        f_b = self.b_encoder(b).view(b.size(0), -1)
        combined = torch.cat([f_raw, f_b], dim=1)
        return self.classifier(combined)
