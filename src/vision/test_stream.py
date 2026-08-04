import cv2

#STREAM_URL = "http://192.168.1.171:81/stream"

STREAM_URL = "http://192.168.8.101/stream"

capture = cv2.VideoCapture(STREAM_URL)

if not capture.isOpened():
    raise RuntimeError(f"Could not open stream: {STREAM_URL}")

while True:
    success, frame = capture.read()

    if not success:
        print("Frame could not be read")
        break

    print(f"Received frame: {frame.shape}")

    cv2.imshow("ESP32 Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
