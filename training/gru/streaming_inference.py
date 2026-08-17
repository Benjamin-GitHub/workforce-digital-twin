from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from gru_data import normalize_live_coco17
from gru_model import StreamingGRU


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    # Batch-1, one-step GRU inference is often faster on CPU than GPU dispatch.
    return torch.device("cpu")


@dataclass
class WorkerState:
    hidden: torch.Tensor | None = None
    observations: int = 0
    last_seen: float = field(default_factory=monotonic)


class StreamingGRUPredictor:
    def __init__(
        self,
        checkpoint_path: Path | str,
        device: str = "auto",
        minimum_observations: int = 5,
        reset_gap_seconds: float = 2.0,
    ) -> None:
        self.device = select_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.classes = list(checkpoint["classes"])
        self.model = StreamingGRU(**checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.minimum_observations = minimum_observations
        self.reset_gap_seconds = reset_gap_seconds
        self.states: dict[str, WorkerState] = {}

    def reset(self, worker_id: str | None = None) -> None:
        if worker_id is None:
            self.states.clear()
        else:
            self.states.pop(worker_id, None)

    @torch.inference_mode()
    def update(
        self,
        worker_id: str,
        coco17_pose: np.ndarray,
        now: float | None = None,
    ) -> dict:
        current_time = monotonic() if now is None else now
        state = self.states.get(worker_id)
        if state is None or current_time - state.last_seen > self.reset_gap_seconds:
            state = WorkerState(last_seen=current_time)
            self.states[worker_id] = state

        pose = np.asarray(coco17_pose, dtype=np.float32)
        if pose.shape != (17, 3):
            raise ValueError(f"Expected a (17,3) x/y/confidence pose; received {pose.shape}")
        body_indices = np.asarray([0, *range(5, 17)])
        if not np.any(pose[body_indices, 2] > 0):
            return {
                "label": "unknown",
                "confidence": 0.0,
                "ready": False,
                "observations": state.observations,
                "probabilities": {name: 0.0 for name in self.classes},
            }

        features = normalize_live_coco17(pose)
        tensor = torch.from_numpy(features).to(self.device).view(1, 51)
        logits, hidden = self.model.step(tensor, state.hidden)
        state.hidden = hidden.detach()
        state.observations += 1
        state.last_seen = current_time
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        index = int(probabilities.argmax())
        ready = state.observations >= self.minimum_observations
        return {
            "label": self.classes[index] if ready else "unknown",
            "confidence": float(probabilities[index]) if ready else 0.0,
            "ready": ready,
            "observations": state.observations,
            "probabilities": dict(zip(self.classes, probabilities.astype(float).tolist())),
        }
