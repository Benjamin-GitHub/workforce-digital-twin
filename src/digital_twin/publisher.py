"""Non-blocking HTTP publisher for Digital Twin worker state updates."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


Payload = dict[str, object]
Transport = Callable[[str, Payload, float], None]


def build_worker_state(
    *,
    worker_id: str,
    track_id: int,
    camera_id: str,
    activity: str,
    confidence: float,
    fps: float | None = None,
) -> Payload:
    """Map live pose output onto the backend's existing WorkerState schema."""

    payload: Payload = {
        "worker_id": worker_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "tracking": {
            "track_id": int(track_id),
            "camera_id": camera_id,
            "online": True,
        },
        "activity": {
            "baseline": activity,
            "baseline_confidence": max(0.0, min(1.0, float(confidence))),
            "stgcn": "unknown",
            "stgcn_confidence": 0.0,
            "display_activity": activity,
        },
    }
    if fps is not None:
        payload["edge"] = {"fps": max(0.0, float(fps))}
    return payload


def post_worker_state(api_url: str, payload: Payload, timeout: float) -> None:
    endpoint = api_url.rstrip("/") + "/workers"
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"HTTP {response.status}")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(str(error)) from error


class WorkerStatePublisher:
    """Rate-limited latest-value publisher running on one daemon thread."""

    def __init__(
        self,
        api_url: str,
        interval: float = 1.0,
        timeout: float = 1.0,
        transport: Transport = post_worker_state,
    ) -> None:
        if interval <= 0 or timeout <= 0:
            raise ValueError("interval and timeout must be positive")
        self.api_url = api_url
        self.interval = interval
        self.timeout = timeout
        self.transport = transport
        self._condition = threading.Condition()
        self._pending: Payload | None = None
        self._stopping = False
        self._last_queued = float("-inf")
        self._last_error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="digital-twin-publisher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, payload: Payload) -> bool:
        """Queue the newest state if the configured cadence has elapsed."""

        now = time.monotonic()
        with self._condition:
            if self._stopping or now - self._last_queued < self.interval:
                return False
            self._last_queued = now
            self._pending = payload
            self._condition.notify()
        return True

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self._thread.join(timeout=self.timeout + 0.5)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._pending is None and self._stopping:
                    return
                payload = self._pending
                self._pending = None
            try:
                self.transport(self.api_url, payload, self.timeout)
                if self._last_error is not None:
                    print("Digital Twin publisher reconnected.")
                self._last_error = None
            except Exception as error:  # network failures must not stop inference
                message = str(error)
                if message != self._last_error:
                    print(f"Digital Twin publish warning: {message}")
                self._last_error = message

