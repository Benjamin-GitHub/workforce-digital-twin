from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PPEState(BaseModel):
    helmet: Optional[bool] = None
    vest: Optional[bool] = None
    gloves: Optional[bool] = None
    boots: Optional[bool] = None


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

    tracking: TrackingState = Field(default_factory=TrackingState)
    ppe: PPEState = Field(default_factory=PPEState)
    activity: ActivityState = Field(default_factory=ActivityState)
    edge: EdgeState = Field(default_factory=EdgeState)
    pose: Optional[PoseState] = None
