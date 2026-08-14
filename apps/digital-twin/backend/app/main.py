from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from .models import (
    ActivityState,
    EdgeState,
    PPEState,
    TrackingState,
    WorkerState,
)
from .state import worker_state_manager
from .database import Base, engine
from .db_models import WorkerEvent
from .history import get_worker_history, save_worker_event
from .websocket_manager import websocket_manager

app = FastAPI(
    title="Workforce Digital Twin API",
    version="0.1.0",
)

Base.metadata.create_all(bind=engine)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.get("/")
def root():
    return {
        "service": "Workforce Digital Twin API",
        "status": "online",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/workers")
def get_workers():
    return worker_state_manager.get_all_workers()


@app.get("/workers/{worker_id}")
def get_worker(worker_id: str):
    worker = worker_state_manager.get_worker(worker_id)

    if worker is None:
        raise HTTPException(
            status_code=404,
            detail=f"Worker '{worker_id}' not found",
        )

    return worker


@app.get("/workers/{worker_id}/history")
def worker_history(
    worker_id: str,
    limit: int = 100,
):
    events = get_worker_history(
        worker_id=worker_id,
        limit=limit,
    )

    return [
        {
            "id": event.id,
            "worker_id": event.worker_id,
            "timestamp": event.timestamp,
            "activity": event.activity,
            "activity_confidence": event.activity_confidence,
            "track_id": event.track_id,
            "camera_id": event.camera_id,
        }
        for event in events
    ]


@app.post("/workers", response_model=WorkerState)
async def update_worker(worker: WorkerState):
    saved_worker = worker_state_manager.set_worker(worker)

    save_worker_event(saved_worker)

    await websocket_manager.broadcast(
        {
            "type": "worker_update",
            "worker": saved_worker.model_dump(mode="json"),
        }
    )

    return saved_worker


@app.on_event("startup")
def create_demo_worker():
    demo_worker = WorkerState(
        worker_id="worker01",
        tracking=TrackingState(
            track_id=1,
            camera_id="esp32_cam_01",
            online=True,
        ),
        ppe=PPEState(
            helmet=True,
            vest=True,
            gloves=None,
            boots=None,
        ),
        activity=ActivityState(
            baseline="walking",
            baseline_confidence=0.71,
            stgcn="unknown",
            stgcn_confidence=0.0,
            display_activity="walking",
        ),
        edge=EdgeState(
            fps=11.0,
            cpu_temperature=54.2,
            throttled=False,
        ),
    )

    worker_state_manager.set_worker(demo_worker)

