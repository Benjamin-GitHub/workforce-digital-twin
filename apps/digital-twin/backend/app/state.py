from datetime import datetime, timezone
import os
from typing import Dict, Optional

from .models import WorkerState


class WorkerStateManager:
    def __init__(self):
        self._workers: Dict[str, WorkerState] = {}

    def set_worker(self, worker: WorkerState) -> WorkerState:
        self._workers[worker.worker_id] = worker
        return worker

    def set_mobile(self, worker_id: str, mobile) -> WorkerState:
        worker = self._workers.get(worker_id) or WorkerState(worker_id=worker_id)
        worker.mobile = mobile
        self._workers[worker_id] = worker
        return self._with_mobile_freshness(worker)

    def get_worker(self, worker_id: str) -> Optional[WorkerState]:
        worker = self._workers.get(worker_id)
        return self._with_mobile_freshness(worker) if worker else None

    def get_all_workers(self) -> list[WorkerState]:
        return [self._with_mobile_freshness(worker) for worker in self._workers.values()]

    @staticmethod
    def _with_mobile_freshness(worker: WorkerState) -> WorkerState:
        if worker.mobile is None:
            return worker
        now = datetime.now(timezone.utc)
        last_seen = worker.mobile.last_seen
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)
        age_s = max(0.0, (now - last_seen).total_seconds())
        stale_after = float(os.getenv("MOBILE_STALE_AFTER_S", "5"))
        disconnected_after = float(os.getenv("MOBILE_DISCONNECTED_AFTER_S", "30"))
        state = "connected" if age_s <= stale_after else "stale"
        if age_s > disconnected_after:
            state = "disconnected"
        worker.mobile.age_s = round(age_s, 3)
        worker.mobile.connection_state = state
        return worker


worker_state_manager = WorkerStateManager()
