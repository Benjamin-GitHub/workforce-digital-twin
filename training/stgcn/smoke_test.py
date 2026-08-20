from __future__ import annotations

import torch

from stgcn_model import STGCN


torch.manual_seed(42)
model = STGCN(num_classes=6, hidden_channels=(8, 8), dropout=0.0)
inputs = [torch.randn(2, 3, frames, 17, 1) for frames in (16, 32)]
logits = [model(x) for x in inputs]
loss = sum(torch.nn.functional.cross_entropy(output, torch.tensor([0, 5])) for output in logits)
loss.backward()
assert all(output.shape == (2, 6) for output in logits)
print(
    "smoke test passed: "
    f"inputs={[tuple(x.shape) for x in inputs]} "
    f"outputs={[tuple(output.shape) for output in logits]} loss={loss.item():.5f}"
)
