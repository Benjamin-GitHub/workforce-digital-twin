# ST-GCN input preparation milestone

The Raspberry Pi publishes one Ultralytics pose per worker update. The backend
validates the payload as exactly 17 COCO joints in original-image pixel space:

`nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle`.

Each joint contains `x`, `y`, and optional confidence. Missing confidence is
treated as zero during preprocessing. Coordinates are centred on the hip
midpoint (shoulder midpoint or visible-joint centroid as fallback) and divided
by torso length (visible-joint bounding-box diagonal as fallback). Confidence
is retained as the third channel.

The Mac backend buffers 32 newer, unique frames independently per worker. A
ready sequence is prepared in the conventional ST-GCN `N,C,T,V,M` layout:
`(1, 3, 32, 17, 1)`. This milestone does not load or train a model and does not
modify baseline or display activity.

Diagnostic endpoint:

```text
GET /workers/{worker_id}/stgcn-sequence
```

Before 32 pose updates, `ready` is false and `tensor_shape` is null. At 32,
`ready` is true and `tensor_shape` is `[1,3,32,17,1]`.
