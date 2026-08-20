# Step 15: synchronized multimodal session logging

This records vision/ST-GCN and Android telemetry for later evaluation. It does not fuse inputs,
retrain either model, or change `display_activity`.

## Storage and cadence

The dedicated SQLite tables `multimodal_sessions`, `session_vision_samples`, and
`session_mobile_samples` live in `data/digital_twin.db`; activity history is unchanged. Vision
and mobile inputs are independently downsampled to 5 Hz by default (configurable from 0-10 Hz
when starting a session), with a default hard cap of 18,000 accepted samples per source. Cadence
drops and repeated source timestamps are counted. A 5 Hz, one-hour session therefore stores at
most 18,000 vision and 18,000 mobile rows.

Alignment occurs only during summary/export. Each vision sample gets the nearest Android source
timestamp within `SESSION_ALIGNMENT_TOLERANCE_MS` (default 1000 ms). No match outside that window
is fabricated: mobile fields are null and `mobile_missing=true`. Clock ages use each source time
against its own backend receive time, so unsynchronized clocks remain visible.

## API

Start a controlled `worker01` session:

```bash
curl -sS -X POST http://127.0.0.1:8000/sessions/start \
  -H 'Content-Type: application/json' \
  -d '{"worker_id":"worker01","source_mode":"LIVE","notes":"controlled six-activity run","cadence_hz":5,"max_samples":18000}' \
  | python3 -m json.tool
```

Status and stop:

```bash
curl -sS http://127.0.0.1:8000/sessions/status | python3 -m json.tool
curl -sS -X POST http://127.0.0.1:8000/sessions/stop | python3 -m json.tool
```

Use the returned `session_id`:

```bash
SESSION_ID='session-...'
curl -sS "http://127.0.0.1:8000/sessions/${SESSION_ID}/summary" | python3 -m json.tool
curl -sS "http://127.0.0.1:8000/sessions/${SESSION_ID}/export?format=csv" -o "${SESSION_ID}.csv"
curl -sS "http://127.0.0.1:8000/sessions/${SESSION_ID}/export?format=json" -o "${SESSION_ID}.json"
```

Every export request also writes the canonical file on the Mac under
`apps/digital-twin/backend/data/session_exports/<session_id>.csv` (or `.json`). CSV has one row
per accepted vision sample with model outputs, IMU/GPS, identities, source/receive timestamps,
ages, delta, missing, and stale flags.
