from __future__ import annotations

import torch
from torch import nn


class StreamingGRU(nn.Module):
    """Unidirectional GRU for COCO-17 pose activity recognition.

    Training input is ``(N, T, 51)``. During live inference, ``step`` accepts
    one ``(N, 51)`` pose vector and returns the updated recurrent state.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int = 51,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if input_size != 51:
            raise ValueError("This deployment model expects COCO-17 x/y/confidence (51 features)")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected (N,T,{self.input_size}); received {tuple(x.shape)}"
            )
        output, hidden = self.gru(x, hidden)
        logits = self.classifier(self.output_dropout(output[:, -1]))
        return logits, hidden

    def step(
        self,
        frame: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frame.ndim == 2:
            frame = frame.unsqueeze(1)
        if frame.ndim != 3 or frame.shape[1] != 1:
            raise ValueError("A streaming step expects (N,51) or (N,1,51)")
        return self.forward(frame, hidden)

