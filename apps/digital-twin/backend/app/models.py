from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PPEObservation(BaseModel):
    detected: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PPEState(BaseModel):
    helmet: PPEObservation = Field(default_factory=PPEObservation)
    vest: PPEObservation = Field(default_factory=PPEObservation)
    gloves: PPEObservation = Field(default_factory=PPEObservation)
    boots: PPEObservation = Field(default_factory=PPEObservation)
    observed_at: Optional[datetime] = None
    association_method: Optional[str] = None


class ActivityState(BaseModel):
    baseline: str = "unknown"
    baseline_confidence: float = 0.0

    stgcn: str = "unknown"
    stgcn_confidence: float = 0.0

    display_activity: str = "unknown"


class TrackingState(BaseModel):
    track_id: Optional[int] = None
    camera_id: Optional[str] = None
    online: bool = False


class EdgeState(BaseModel):
    fps: Optional[float] = None
    cpu_temperature: Optional[float] = None
    throttled: bool = False


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class MobileLocation(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy_m: Optional[float] = Field(default=None, ge=0.0)
    gps_enabled: bool = False
    permission_state: str = "unknown"
    zone: Optional[str] = None


class MobileTelemetry(BaseModel):
    worker_id: str
    device_id: str
    mqtt_client_id: Optional[str] = None
    source: Literal["android"] = "android"
    timestamp: datetime
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    connection_state: Literal["connected", "stale", "disconnected"] = "connected"
    age_s: float = Field(default=0.0, ge=0.0)
    accelerometer: Vector3
    gyroscope: Vector3
    location: MobileLocation = Field(default_factory=MobileLocation)
    battery_level: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    association_method: Literal["manual_pairing", "configured"] = "configured"
    association_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PoseKeypoint(BaseModel):
    """One COCO keypoint in original image-space pixel coordinates."""

    x: float
    y: float
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PoseState(BaseModel):
    """Transient live pose; this is never written to activity history."""

    frame_number: int = Field(ge=0)
    captured_at: datetime
    coordinate_space: Literal["image_pixels"] = "image_pixels"
    layout: Literal["coco_17"] = "coco_17"
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    keypoints: list[PoseKeypoint] = Field(min_length=17, max_length=17)


class WorkerState(BaseModel):
    worker_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    source: Literal["live", "replay"] = "live"

    tracking: TrackingState = Field(default_factory=TrackingState)
    ppe: PPEState = Field(default_factory=PPEState)
    activity: ActivityState = Field(default_factory=ActivityState)
    edge: EdgeState = Field(default_factory=EdgeState)
    pose: Optional[PoseState] = None
    mobile: Optional[MobileTelemetry] = None
