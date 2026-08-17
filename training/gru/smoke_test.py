from __future__ import annotations

import torch
from torch import nn

from gru_model import StreamingGRU


def main() -> None:
    torch.manual_seed(42)
    model = StreamingGRU(num_classes=6)
    model.train()
    x = torch.randn(4, 16, 51)
    y = torch.tensor([0, 1, 2, 5])
    logits, hidden = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    assert logits.shape == (4, 6)
    assert hidden.shape == (2, 4, 128)
    assert torch.isfinite(loss)

    # A unidirectional GRU must give the same result when the sequence is fed
    # one step at a time. This is the core streaming-deployment guarantee.
    model.eval()
    with torch.inference_mode():
        full_logits, _ = model(x[:1])
        state = None
        step_logits = None
        for index in range(x.shape[1]):
            step_logits, state = model.step(x[:1, index], state)
        torch.testing.assert_close(full_logits, step_logits, rtol=1e-5, atol=1e-6)
    print("GRU smoke test passed: forward, backward, and streaming equivalence")


if __name__ == "__main__":
    main()

