from datetime import datetime
from typing import Optional

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


class WorkerState(BaseModel):
    worker_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    tracking: TrackingState = Field(default_factory=TrackingState)
    ppe: PPEState = Field(default_factory=PPEState)
    activity: ActivityState = Field(default_factory=ActivityState)
    edge: EdgeState = Field(default_factory=EdgeState)

