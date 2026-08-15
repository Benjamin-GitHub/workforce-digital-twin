from __future__ import annotations

import torch

from stgcn_model import STGCN


torch.manual_seed(42)
model = STGCN(num_classes=6, hidden_channels=(8, 8), dropout=0.0)
x = torch.randn(2, 3, 32, 17, 1)
logits = model(x)
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 5]))
loss.backward()
assert logits.shape == (2, 6)
print(f"smoke test passed: input={tuple(x.shape)} output={tuple(logits.shape)} loss={loss.item():.5f}")
