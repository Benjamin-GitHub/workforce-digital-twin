# Android mobile telemetry

The Android phone is an explicitly configured sensor source for a stable `worker_id`. Its
app-generated `device_id` is independent of the camera's temporary `track_id`; mobile data is
not used by the activity classifiers.

## Mac setup

Start (or reuse) an MQTT broker reachable by the phone. For a local Mosquitto installation:

```bash
brew services start mosquitto
```

Create and run the backend environment:

```bash
cd ~/workforce-digital-twin/apps/digital-twin/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MQTT_HOST=127.0.0.1 MQTT_PORT=1883 \
  MOBILE_STALE_AFTER_S=5 MOBILE_DISCONNECTED_AFTER_S=30 \
  uvicorn app.main:app --reload
```

If the broker requires credentials, add `MQTT_USERNAME=... MQTT_PASSWORD=...`. The subscriber
uses `digitaltwin/workers/+/mobile` and also accepts legacy `factory/workers/+` messages.

Run the dashboard in another terminal:

```bash
cd ~/workforce-digital-twin/apps/digital-twin/frontend
pnpm install
pnpm dev
```

## Android build and run

```bash
cd ~/AndroidStudioProjects/WorkerSensorApp
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew testDebugUnitTest assembleDebug lintDebug
~/Library/Android/sdk/platform-tools/adb install -r app/build/outputs/apk/debug/app-debug.apk
~/Library/Android/sdk/platform-tools/adb shell am start -n com.example.workersensorapp/.MainActivity
```

In the app, enter the Mac/broker LAN address and existing MQTT credentials, assign `worker01`,
leave Zone blank unless a real zone is configured, and press Connect. The generated device UUID
is stored in DataStore and reused after app/process/device restarts. Uninstalling or clearing the
app's data intentionally creates a new identity.

## Wire format and diagnostics

Topic:

```text
digitaltwin/workers/worker01/mobile
```

Representative payload (`timestamp` is Unix epoch milliseconds):

```json
{
  "worker_id": "worker01",
  "device_id": "12345678-abcd-4abc-8abc-123456789abc",
  "mqtt_client_id": "android-worker01-phone-12345678",
  "source": "android",
  "timestamp": 1770000000000,
  "connection_state": "connected",
  "association_method": "configured",
  "association_confidence": 1.0,
  "accelerometer": {"x": 0.12, "y": -0.04, "z": 9.79},
  "gyroscope": {"x": 0.01, "y": 0.02, "z": -0.01},
  "gps": {"gps_enabled": false, "permission_state": "denied"},
  "battery_level": 83
}
```

Publish a diagnostic sample without the phone:

```bash
mosquitto_pub -h 127.0.0.1 -q 1 \
  -t digitaltwin/workers/worker01/mobile \
  -m '{"worker_id":"worker01","device_id":"test-device-01","mqtt_client_id":"android-worker01-phone-test0001","source":"android","timestamp":1770000000000,"accelerometer":{"x":0.12,"y":-0.04,"z":9.79},"gyroscope":{"x":0.01,"y":0.02,"z":-0.01},"gps":{"gps_enabled":false,"permission_state":"denied"},"association_method":"configured","association_confidence":1.0}'
```

Verify both the mobile branch and the full worker association:

```bash
curl -s http://127.0.0.1:8000/workers/worker01/mobile | python3 -m json.tool
curl -s http://127.0.0.1:8000/workers/worker01 | python3 -m json.tool
```

The backend calculates `age_s` and `connection_state`. With the defaults, state becomes `stale`
after 5 seconds and `disconnected` after 30 seconds, then returns to `connected` on the next valid
message. Only the latest mobile reading is held in memory; raw 10 Hz IMU samples are not written
to SQLite and activity-history semantics remain unchanged.
