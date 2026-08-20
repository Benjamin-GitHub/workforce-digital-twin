#!/usr/bin/env bash
set -Eeuo pipefail

STREAM_URL="${STREAM_URL:-http://192.168.1.171:81/stream}"
NEW_CLIPS_PER_CLASS="${1:-5}"
CLIP_SECONDS="${CLIP_SECONDS:-20}"
COUNTDOWN_SECONDS="${COUNTDOWN_SECONDS:-5}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/activity-dataset/recordings/raw}"

CLASSES=(
  walking
  bending
  carrying
  material_handling
)

declare -A SPOKEN_NAMES=(
  [walking]="walking"
  [bending]="bending"
  [carrying]="carrying"
  [material_handling]="material handling"
)

if ! [[ "$NEW_CLIPS_PER_CLASS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: number of new clips must be a positive integer."
  exit 1
fi

for command_name in ffmpeg ffprobe awk; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Error: required command not found: $command_name"
    exit 1
  fi
done

if pgrep -af 'python.*src\.vision\.pose_activity' >/dev/null 2>&1; then
  echo "Error: pose_activity is currently using the camera stream."
  echo
  pgrep -af 'python.*src\.vision\.pose_activity'
  echo
  echo "Stop it with Ctrl+C in its original terminal, then run this script again."
  exit 1
fi

speak() {
  local message="$1"

  if command -v espeak-ng >/dev/null 2>&1; then
    espeak-ng "$message" >/dev/null 2>&1 || true
  elif command -v espeak >/dev/null 2>&1; then
    espeak "$message" >/dev/null 2>&1 || true
  fi
}

countdown() {
  local activity_name="$1"

  echo
  echo "Prepare for: $activity_name"
  speak "Prepare for $activity_name"

  for ((remaining=COUNTDOWN_SECONDS; remaining>=1; remaining--)); do
    echo "Starting in $remaining..."
    speak "$remaining"
    sleep 1
  done

  echo "START: $activity_name"
  speak "Start $activity_name"
}

next_available_index() {
  local class_name="$1"
  local class_directory="$OUTPUT_ROOT/$class_name"
  local index=1
  local candidate

  while true; do
    candidate="$(printf \
      '%s/%s_%02d_%ss_raw.mkv' \
      "$class_directory" \
      "$class_name" \
      "$index" \
      "$CLIP_SECONDS")"

    if [[ ! -e "$candidate" ]]; then
      printf '%d\n' "$index"
      return
    fi

    ((index++))
  done
}

record_clip() {
  local class_name="$1"
  local sequence_number="$2"
  local spoken_name="${SPOKEN_NAMES[$class_name]}"
  local class_directory="$OUTPUT_ROOT/$class_name"
  local index
  local final_path
  local temporary_path
  local measured_duration

  mkdir -p "$class_directory"

  index="$(next_available_index "$class_name")"

  final_path="$(printf \
    '%s/%s_%02d_%ss_raw.mkv' \
    "$class_directory" \
    "$class_name" \
    "$index" \
    "$CLIP_SECONDS")"

  temporary_path="$(mktemp \
    "$class_directory/.${class_name}_${index}_recording_XXXXXX.mkv")"

  echo
  echo "Class: $class_name"
  echo "Pilot clip: $sequence_number/$NEW_CLIPS_PER_CLASS"
  echo "Output: $final_path"
  echo
  read -r -p "Press Enter when you and the camera are ready..."

  countdown "$spoken_name"

  if ! ffmpeg \
      -hide_banner \
      -loglevel warning \
      -rw_timeout 15000000 \
      -fflags +genpts \
      -use_wallclock_as_timestamps 1 \
      -i "$STREAM_URL" \
      -map 0:v:0 \
      -t "$CLIP_SECONDS" \
      -c:v copy \
      -an \
      -y \
      "$temporary_path"; then
    echo
    echo "Recording failed. Incomplete file retained at:"
    echo "$temporary_path"
    return 1
  fi

  measured_duration="$(
    ffprobe \
      -v error \
      -show_entries format=duration \
      -of default=noprint_wrappers=1:nokey=1 \
      "$temporary_path" 2>/dev/null || true
  )"

  if [[ -z "$measured_duration" ]]; then
    echo "Could not verify recording duration."
    echo "File retained at: $temporary_path"
    return 1
  fi

  if ! awk \
      -v measured="$measured_duration" \
      -v expected="$CLIP_SECONDS" \
      'BEGIN { exit !(measured >= expected - 1.0) }'; then
    echo "Recording is shorter than expected: ${measured}s"
    echo "File retained at: $temporary_path"
    return 1
  fi

  mv "$temporary_path" "$final_path"

  echo "Saved: $final_path"
  echo "Verified duration: ${measured_duration}s"
  speak "$spoken_name recording complete"
}

mkdir -p "$OUTPUT_ROOT"

echo "ESP32 stream: $STREAM_URL"
echo "Output root: $OUTPUT_ROOT"
echo "New clips per class: $NEW_CLIPS_PER_CLASS"
echo "Clip duration: $CLIP_SECONDS seconds"
echo
echo "Activities will be presented in a varied order for each round."

for ((round=1; round<=NEW_CLIPS_PER_CLASS; round++)); do
  echo
  echo "============================================================"
  echo "ROUND $round OF $NEW_CLIPS_PER_CLASS"
  echo "============================================================"

  if command -v shuf >/dev/null 2>&1; then
    mapfile -t ROUND_CLASSES < <(
      printf '%s\n' "${CLASSES[@]}" | shuf
    )
  else
    ROUND_CLASSES=("${CLASSES[@]}")
  fi

  for class_name in "${ROUND_CLASSES[@]}"; do
    record_clip "$class_name" "$round"
  done
done

echo
echo "============================================================"
echo "RECORDING COMPLETE"
echo "============================================================"

for class_name in "${CLASSES[@]}"; do
  count="$(
    find "$OUTPUT_ROOT/$class_name" \
      -maxdepth 1 \
      -type f \
      -name "${class_name}_*_${CLIP_SECONDS}s_raw.mkv" \
      | wc -l
  )"

  printf '%-22s %s verified files\n' "$class_name" "$count"
done

speak "All activity recordings are complete"
