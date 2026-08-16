import asyncio
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import (
    ActivityState,
    EdgeState,
    PPEState,
    TrackingState,
    WorkerState,
    SessionStartRequest,
)
from .state import worker_state_manager
from .database import Base, engine
from .db_models import WorkerEvent
from .history import get_worker_history, save_worker_event
from .websocket_manager import websocket_manager
from .stgcn import stgcn_service, temporal_pose_buffer
from .mqtt_mobile import MobileMqttSubscriber
from .sessions import session_recorder

app = FastAPI(
    title="Workforce Digital Twin API",
    version="0.1.0",
)
main_loop = None


def accept_mobile_telemetry(mobile):
    worker = worker_state_manager.set_mobile(mobile.worker_id, mobile)
    session_recorder.record_mobile(mobile)
    if main_loop is not None:
        asyncio.run_coroutine_threadsafe(
            websocket_manager.broadcast({
                "type": "worker_update",
                "worker": worker.model_dump(mode="json"),
                "activity_changed": False,
            }),
            main_loop,
        )


mobile_mqtt_subscriber = MobileMqttSubscriber(accept_mobile_telemetry)

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


@app.get("/stgcn/status")
def get_stgcn_status():
    return stgcn_service.status()


@app.post("/sessions/start")
def start_session(request: SessionStartRequest):
    try:
        return session_recorder.start(
            worker_id=request.worker_id, source_mode=request.source_mode,
            notes=request.notes, expected_activity=request.expected_activity,
            cadence_hz=request.cadence_hz, max_samples=request.max_samples,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/sessions/stop")
def stop_session():
    try:
        return session_recorder.stop()
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/sessions/status")
def session_status():
    return session_recorder.status()


@app.get("/sessions/{session_id}/summary")
def session_summary(session_id: str):
    try:
        return session_recorder.summary(session_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found") from error


@app.get("/sessions/{session_id}/export")
def export_session(session_id: str, format: str = Query(default="csv", pattern="^(csv|json)$")):
    try:
        path = session_recorder.export(session_id, format)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found") from error
    media_type = "text/csv" if format == "csv" else "application/json"
    return FileResponse(path, media_type=media_type, filename=path.name)


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


@app.get("/workers/{worker_id}/ppe")
def get_worker_ppe(worker_id: str):
    worker = worker_state_manager.get_worker(worker_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    return worker.ppe


@app.get("/workers/{worker_id}/mobile")
def get_worker_mobile(worker_id: str):
    worker = worker_state_manager.get_worker(worker_id)
    if worker is None or worker.mobile is None:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' has no mobile telemetry")
    return worker.mobile


@app.get("/workers/{worker_id}/stgcn-sequence")
def get_stgcn_sequence_diagnostic(worker_id: str):
    if worker_state_manager.get_worker(worker_id) is None:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    return temporal_pose_buffer.diagnostic(worker_id)


@app.get("/workers/{worker_id}/stgcn-prediction")
def get_latest_stgcn_prediction(worker_id: str):
    if worker_state_manager.get_worker(worker_id) is None:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    prediction = stgcn_service.latest(worker_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' has no ST-GCN prediction")
    return prediction


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

    # ST-GCN output is backend-owned. Keep the last good prediction while a new
    # sequence is warming up or when a contained inference error occurs.
    if previous_worker is not None:
        worker.activity.stgcn = previous_worker.activity.stgcn
        worker.activity.stgcn_confidence = previous_worker.activity.stgcn_confidence
        worker.mobile = previous_worker.mobile

    saved_worker = worker_state_manager.set_worker(worker)

    if saved_worker.pose is not None:
        added = temporal_pose_buffer.add(saved_worker.worker_id, saved_worker.pose)
        sequence = temporal_pose_buffer.tensor(saved_worker.worker_id) if added else None
        prediction = stgcn_service.predict(saved_worker.worker_id, sequence) if sequence else None
        if prediction is not None:
            saved_worker.activity.stgcn = prediction.activity
            saved_worker.activity.stgcn_confidence = prediction.confidence
            worker_state_manager.set_worker(saved_worker)

    session_recorder.record_vision(saved_worker)

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
def load_stgcn_model():
    stgcn_service.load()


@app.on_event("startup")
async def start_mobile_mqtt():
    global main_loop
    main_loop = asyncio.get_running_loop()
    mobile_mqtt_subscriber.start()


@app.on_event("shutdown")
def stop_mobile_mqtt():
    mobile_mqtt_subscriber.stop()


@app.on_event("startup")
def create_demo_worker():
    demo_worker = WorkerState(
        worker_id="worker01",
        source="replay",
        tracking=TrackingState(
            track_id=1,
            camera_id="esp32_cam_01",
            online=True,
        ),
        ppe=PPEState(),
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
