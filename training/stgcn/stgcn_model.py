from __future__ import annotations

import torch
from torch import nn


COCO_17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def adjacency(num_joints: int = 17) -> torch.Tensor:
    matrix = torch.eye(num_joints, dtype=torch.float32)
    for left, right in COCO_17_EDGES:
        matrix[left, right] = matrix[right, left] = 1.0
    degree = matrix.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * matrix * inv_sqrt[None, :]


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels), nn.Dropout(dropout),
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = torch.einsum("nctv,vw->nctw", x, graph)
        return self.relu(self.temporal(self.spatial(x)) + residual)


class STGCN(nn.Module):
    """Minimal single-person COCO-17 ST-GCN; input is N,C,T,V,M."""

    def __init__(self, num_classes: int, hidden_channels=(64, 64, 128), dropout=0.3):
        super().__init__()
        self.register_buffer("graph", adjacency())
        channels = (3, *hidden_channels)
        self.blocks = nn.ModuleList(
            STGCNBlock(channels[index], channels[index + 1], dropout)
            for index in range(len(channels) - 1)
        )
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[3] != 17:
            raise ValueError(f"Expected N,C,T,V,M with C=3,V=17; received {tuple(x.shape)}")
        x = x.mean(dim=4)
        for block in self.blocks:
            x = block(x, self.graph)
        return self.classifier(x.mean(dim=(2, 3)))
