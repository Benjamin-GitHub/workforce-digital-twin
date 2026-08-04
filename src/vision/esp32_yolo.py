import time
import cv2
from ultralytics import YOLO

STREAM_URL = "http://192.168.1.171:81/stream"
MODEL_PATH = "/home/rp5/workforce-digital-twin/models/pytorch/yolo26n.pt"
PERSON_CLASS_ID = 0

FRAME_INTERVAL = 3

model = YOLO(MODEL_PATH)
capture = cv2.VideoCapture(STREAM_URL)

if not capture.isOpened():
    raise RuntimeError(f"Unable to open stream: {STREAM_URL}")

frame_number = 0

camera_frame_count = 0
camera_fps = 0.0
camera_timer = time.perf_counter()

processed_frame_count = 0
processed_fps = 0.0
processed_timer = time.perf_counter()

while True:
    success, frame = capture.read()

    if not success:
        print("Unable to receive frame")
        break

    # Measure all frames received from the ESP32
    camera_frame_count += 1
    camera_elapsed = time.perf_counter() - camera_timer

    if camera_elapsed >= 1.0:
        camera_fps = camera_frame_count / camera_elapsed
        camera_frame_count = 0
        camera_timer = time.perf_counter()

    frame_number += 1

    # Process frame 1, 4, 7, 10... when FRAME_INTERVAL = 3
    if (frame_number - 1) % FRAME_INTERVAL != 0:
        continue

    inference_start = time.perf_counter()

    results = model.predict(
        source=frame,
        imgsz=320,
        conf=0.40,
        #classes=[PERSON_CLASS_ID],
        device="cpu",
        verbose=False,
    )

    inference_ms = (
        time.perf_counter() - inference_start
    ) * 1000

    # Measure actual frames analysed by YOLO
    processed_frame_count += 1
    processed_elapsed = time.perf_counter() - processed_timer

    if processed_elapsed >= 1.0:
        processed_fps = processed_frame_count / processed_elapsed
        processed_frame_count = 0
        processed_timer = time.perf_counter()

    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        f"Camera FPS: {camera_fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        annotated_frame,
        f"YOLO FPS: {processed_fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Inference: {inference_ms:.1f} ms",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow("ESP32 YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
