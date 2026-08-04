import os
import time

import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO


CAMERA_URL = os.getenv(
    "CAMERA_URL",
    "http://192.168.1.171:81/stream",
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/rp5/workforce-digital-twin/models/pytorch/yolo26n.pt",
)

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "320"))
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.40"))
PORT = int(os.getenv("PORT", "5000"))

app = Flask(__name__)

print(f"Loading YOLO model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Raspberry Pi YOLO Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            margin: 0;
            background: #111;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }

        h1 {
            font-size: 1.4rem;
            margin: 16px;
        }

        img {
            width: min(96vw, 960px);
            height: auto;
            border: 1px solid #555;
        }
    </style>
</head>
<body>
    <h1>ESP32 Camera — YOLO Detection</h1>
    <img src="/video_feed" alt="YOLO annotated camera stream">
</body>
</html>
"""


def open_camera() -> cv2.VideoCapture:
    """Open the ESP32 MJPEG stream."""
    print(f"Connecting to camera: {CAMERA_URL}")

    capture = cv2.VideoCapture(CAMERA_URL)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return capture


def generate_frames():
    """Read camera frames, run YOLO and return an MJPEG stream."""
    capture = None

    while True:
        if capture is None or not capture.isOpened():
            capture = open_camera()

            if not capture.isOpened():
                print("Could not open camera stream. Retrying...")
                capture.release()
                capture = None
                time.sleep(2)
                continue

        success, frame = capture.read()

        if not success or frame is None:
            print("Camera frame unavailable. Reconnecting...")

            capture.release()
            capture = None
            time.sleep(1)
            continue

        try:
            result = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device="cpu",
                verbose=False,
            )[0]

            annotated_frame = result.plot()

            encoded, jpeg = cv2.imencode(
                ".jpg",
                annotated_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )

            if not encoded:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

        except Exception as error:
            print(f"YOLO processing error: {error}")
            time.sleep(0.2)


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print(f"Web stream listening on port {PORT}")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
