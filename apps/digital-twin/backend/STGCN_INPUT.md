# ST-GCN input preparation milestone

The Raspberry Pi publishes one Ultralytics pose per worker update. The backend
validates the payload as exactly 17 COCO joints in original-image pixel space:

`nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle`.

Each joint contains `x`, `y`, and optional confidence. Missing confidence and
joints below the live `0.30` threshold are zeroed, eye/ear joints 1-4 are masked,
and remaining confidence is binarized before normalization. Coordinates are centred on the hip
midpoint (shoulder midpoint or visible-joint centroid as fallback) and divided
by torso length (visible-joint bounding-box diagonal, then `1.0`, as fallback).

The Mac backend buffers the checkpoint-defined number of newer, unique frames
independently per worker. A
ready sequence is prepared in the conventional ST-GCN `N,C,T,V,M` layout:
`(1, 3, T, 17, 1)`. Immediately before inference, eye/ear nodes 1-4 are zeroed
and positive confidence values are binarized, matching local-video training.
The checkpoint is loaded once at startup on MPS when available, otherwise CPU.
Predictions update `activity.stgcn` and `activity.stgcn_confidence`; baseline and
display activity remain unchanged.

Diagnostic endpoint:

```text
GET /workers/{worker_id}/stgcn-sequence
GET /stgcn/status
GET /workers/{worker_id}/stgcn-prediction
```

Before `T` pose updates, `ready` is false and `tensor_shape` is null. At `T`,
`ready` is true and `tensor_shape` is `[1,3,T,17,1]`.

The backend defaults to the tracked ST-GCN and GRU checkpoints, so inference
works in a clean clone. The Mac launcher prefers the locally trained 5 Hz
checkpoints when they exist and falls back to the tracked checkpoints when they
do not. Set `STGCN_CHECKPOINT` and `GRU_CHECKPOINT` to select explicit alternate
checkpoints without replacing either default.

Model input cadence is disabled by default. Set `MODEL_INPUT_HZ=5` to accept at
most one pose in each source-timestamp 5 Hz bucket. Duplicate or non-increasing
source timestamps are rejected regardless of cadence configuration. Set
`MODEL_RESET_GAP_SECONDS` to change the default 2-second source-time gap that
resets both models for one worker. Source, camera, track, and frame-rewind changes
also reset that worker before the next accepted pose. Diagnostics are available at
`GET /workers/{worker_id}/model-input`.
