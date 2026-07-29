import time
import cv2
from ultralytics import YOLO

STREAM_URL = "http://192.168.1.171:81/stream"
MODEL_PATH = "yolo26n.pt"

# COCO class 0 is "person".
PERSON_CLASS_ID = 0

model = YOLO(MODEL_PATH)

capture = cv2.VideoCapture(STREAM_URL)

if not capture.isOpened():
    raise RuntimeError(f"Unable to open ESP32 stream: {STREAM_URL}")

previous_time = time.perf_counter()

while True:
    success, frame = capture.read()

    if not success:
        print("Unable to receive a frame. Reconnecting...")
        capture.release()
        time.sleep(2)
        capture = cv2.VideoCapture(STREAM_URL)
        continue

    # Reduce inference resolution to lower CPU load.
    results = model.predict(
        source=frame,
        imgsz=320,
        conf=0.40,
        classes=[PERSON_CLASS_ID],
        device="cpu",
        verbose=False,
    )

    annotated_frame = results[0].plot()

    current_time = time.perf_counter()
    elapsed = current_time - previous_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    previous_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow("ESP32 YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
