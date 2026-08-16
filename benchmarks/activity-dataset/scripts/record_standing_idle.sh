#!/usr/bin/env bash

set -u

# ============================================================
# ESP32 Activity Dataset Recorder
#
# Records:
#   15 x Standing clips
#   15 x Idle clips
#
# Each clip:
#   20 seconds recording
#   5 seconds preparation/countdown beforehand
#
# Video is copied directly from the ESP32 MJPEG stream
# without re-encoding.
# ============================================================

CAMERA_URL="${CAMERA_URL:-http://192.168.1.171:81/stream}"

PROJECT_ROOT="$HOME/workforce-digital-twin"

BASE_DIR="$PROJECT_ROOT/benchmarks/activity-dataset/recordings/raw"

STANDING_DIR="$BASE_DIR/standing"
IDLE_DIR="$BASE_DIR/idle"

CLIP_SECONDS=20
BREAK_SECONDS=5
CLIP_COUNT=15

MANIFEST="$BASE_DIR/recording_manifest.csv"

mkdir -p "$STANDING_DIR"
mkdir -p "$IDLE_DIR"

# ------------------------------------------------------------
# Create CSV manifest if it does not already exist
# ------------------------------------------------------------

if [ ! -f "$MANIFEST" ]; then
    echo "activity,clip_number,start_time,end_time,duration_seconds,file" \
        > "$MANIFEST"
fi


countdown() {

    echo
    echo "Prepare for the next clip..."

    for ((sec=BREAK_SECONDS; sec>=1; sec--)); do
        printf "\rRecording starts in %d seconds... " "$sec"
        sleep 1
    done

    printf "\rRECORDING NOW!                    \n"
    printf "\a"
}


record_set() {

    ACTIVITY="$1"
    OUTPUT_DIR="$2"

    echo
    echo "============================================================"
    echo " Recording activity: $ACTIVITY"
    echo " Clips:              $CLIP_COUNT"
    echo " Clip length:        ${CLIP_SECONDS}s"
    echo " Preparation:        ${BREAK_SECONDS}s"
    echo "============================================================"
    echo

    read -rp "Press ENTER when you are ready to begin $ACTIVITY..."

    for n in $(seq -w 1 "$CLIP_COUNT"); do

        countdown

        OUTPUT_FILE="$OUTPUT_DIR/${ACTIVITY}_${n}_20s_raw.mkv"

        START_TIME=$(date --iso-8601=seconds)

        echo
        echo "[$ACTIVITY $n/$CLIP_COUNT]"
        echo "Recording for ${CLIP_SECONDS} seconds..."
        echo

        ffmpeg \
            -hide_banner \
            -loglevel warning \
            -y \
            -rw_timeout 10000000 \
            -fflags +genpts \
            -i "$CAMERA_URL" \
            -t "$CLIP_SECONDS" \
            -map 0:v:0 \
            -an \
            -c:v copy \
            "$OUTPUT_FILE"

        FFMPEG_STATUS=$?

        END_TIME=$(date --iso-8601=seconds)

        if [ "$FFMPEG_STATUS" -eq 0 ]; then

            echo
            echo "Saved:"
            echo "$OUTPUT_FILE"

            echo \
"$ACTIVITY,$n,$START_TIME,$END_TIME,$CLIP_SECONDS,$OUTPUT_FILE" \
                >> "$MANIFEST"

        else

            echo
            echo "WARNING: Recording $ACTIVITY $n failed."
            echo "Check the ESP32 camera connection."

        fi

    done
}


# ============================================================
# STANDING
# ============================================================

record_set "standing" "$STANDING_DIR"

echo
echo "============================================================"
echo " STANDING SET COMPLETE"
echo "============================================================"
echo
echo "You can now change/preparate for the IDLE recordings."
echo

read -rp "Press ENTER when you are ready for the IDLE set..."


# ============================================================
# IDLE
# ============================================================

record_set "idle" "$IDLE_DIR"


# ============================================================
# Finished
# ============================================================

echo
echo "============================================================"
echo " ALL RECORDINGS COMPLETE"
echo "============================================================"
echo
echo "Standing clips:"
find "$STANDING_DIR" -maxdepth 1 -type f -name "*.mkv" | wc -l

echo
echo "Idle clips:"
find "$IDLE_DIR" -maxdepth 1 -type f -name "*.mkv" | wc -l

echo
echo "Manifest:"
echo "$MANIFEST"
echo
