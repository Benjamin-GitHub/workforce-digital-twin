from typing import Dict, Optional

from .models import WorkerState


class WorkerStateManager:
    def __init__(self):
        self._workers: Dict[str, WorkerState] = {}

    def set_worker(self, worker: WorkerState) -> WorkerState:
        self._workers[worker.worker_id] = worker
        return worker

    def get_worker(self, worker_id: str) -> Optional[WorkerState]:
        return self._workers.get(worker_id)

    def get_all_workers(self) -> list[WorkerState]:
        return list(self._workers.values())


worker_state_manager = WorkerStateManager()
