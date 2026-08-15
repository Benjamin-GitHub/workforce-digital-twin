from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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
from .stgcn import temporal_pose_buffer

app = FastAPI(
    title="Workforce Digital Twin API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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


@app.get("/workers/{worker_id}/pose")
def get_worker_pose(worker_id: str):
    worker = worker_state_manager.get_worker(worker_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    if worker.pose is None:
        raise HTTPException(
            status_code=404,
            detail=f"Worker '{worker_id}' has no live pose",
        )
    return worker.pose


@app.get("/workers/{worker_id}/stgcn-sequence")
def get_stgcn_sequence_diagnostic(worker_id: str):
    if worker_state_manager.get_worker(worker_id) is None:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    return temporal_pose_buffer.diagnostic(worker_id)


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
    previous_worker = worker_state_manager.get_worker(
        worker.worker_id
    )

    saved_worker = worker_state_manager.set_worker(worker)

    if saved_worker.pose is not None:
        temporal_pose_buffer.add(saved_worker.worker_id, saved_worker.pose)

    activity_changed = (
        previous_worker is None
        or previous_worker.activity.display_activity
        != saved_worker.activity.display_activity
    )

    if activity_changed:
        save_worker_event(saved_worker)

    await websocket_manager.broadcast(
        {
            "type": "worker_update",
            "worker": saved_worker.model_dump(mode="json"),
            "activity_changed": activity_changed,
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
