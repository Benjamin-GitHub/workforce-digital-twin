#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

STREAM_URL="${1:-}"
DURATION="${2:-120}"
OUTPUT_NAME="${3:-esp32_test_2min}"

if [[ -z "$STREAM_URL" ]]; then
    echo "Usage:"
    echo "  $0 <stream_url> [duration_seconds] [output_name]"
    echo
    echo "Example:"
    echo "  $0 http://192.168.1.100:81/stream 120 esp32_test_2min"
    exit 2
fi

OUTPUT_DIR="benchmarks/live-video/recordings/raw"
META_DIR="benchmarks/live-video/recordings/metadata"

mkdir -p "$OUTPUT_DIR" "$META_DIR"

VIDEO_FILE="$OUTPUT_DIR/${OUTPUT_NAME}.mp4"
LOG_FILE="$META_DIR/${OUTPUT_NAME}_ffmpeg.log"
INFO_FILE="$META_DIR/${OUTPUT_NAME}.json"

START_TIME="$(date -Is)"
START_TEMP="$(vcgencmd measure_temp 2>/dev/null || true)"
START_THROTTLE="$(vcgencmd get_throttled 2>/dev/null || true)"

echo "Recording:"
echo "  URL: $STREAM_URL"
echo "  Duration: $DURATION seconds"
echo "  Output: $VIDEO_FILE"

ffmpeg \
    -hide_banner \
    -y \
    -i "$STREAM_URL" \
    -t "$DURATION" \
    -an \
    -c:v libx264 \
    -preset veryfast \
    -crf 20 \
    -pix_fmt yuv420p \
    "$VIDEO_FILE" \
    2>&1 | tee "$LOG_FILE"

END_TIME="$(date -Is)"
END_TEMP="$(vcgencmd measure_temp 2>/dev/null || true)"
END_THROTTLE="$(vcgencmd get_throttled 2>/dev/null || true)"

VIDEO_FILE="$VIDEO_FILE" \
STREAM_URL="$STREAM_URL" \
DURATION="$DURATION" \
START_TIME="$START_TIME" \
END_TIME="$END_TIME" \
START_TEMP="$START_TEMP" \
END_TEMP="$END_TEMP" \
START_THROTTLE="$START_THROTTLE" \
END_THROTTLE="$END_THROTTLE" \
python - <<'PY' > "$INFO_FILE"
import json
import os

metadata = {
    "stream_url": os.environ["STREAM_URL"],
    "requested_duration_seconds": int(os.environ["DURATION"]),
    "video_file": os.environ["VIDEO_FILE"],
    "started": os.environ["START_TIME"],
    "finished": os.environ["END_TIME"],
    "start_temperature": os.environ["START_TEMP"],
    "end_temperature": os.environ["END_TEMP"],
    "start_throttling": os.environ["START_THROTTLE"],
    "end_throttling": os.environ["END_THROTTLE"],
}

print(json.dumps(metadata, indent=2))
PY

echo
echo "Recording complete:"
echo "$VIDEO_FILE"
