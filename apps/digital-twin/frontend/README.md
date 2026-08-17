# Workforce Digital Twin Dashboard

Live monitoring dashboard for worker activity, PPE state, tracking information,
edge telemetry, and persistent activity-transition history. It reads the
current worker state from the FastAPI backend and receives live updates over a
WebSocket connection.

## Prerequisites

- Node.js `>=22.13.0`
- pnpm `11.x`
- Digital Twin backend running at `http://127.0.0.1:8000`

## Quick Start

From the repository root, start the Mac backend and frontend together with:

```bash
./scripts/run_digital_twin_mac.sh
```

The launcher reuses an MQTT broker already listening on port 1883 or starts the
installed Mosquitto broker, binds the backend to port 8000 for LAN publishing,
and serves the dashboard on port 3000. Press `Ctrl+C` once to stop the backend,
frontend, and any broker started by the launcher. A pre-existing broker is left
running.

For an authenticated MQTT broker, copy `secret.h.example` to `secret.h`, enter
the Mosquitto username and password, and restrict the file to your user:

```bash
cp secret.h.example secret.h
chmod 600 secret.h
```

`secret.h` is ignored by Git and is loaded automatically by the launcher. Only
the placeholder `secret.h.example` is intended to be committed.

To run only the frontend:

```bash
pnpm install
pnpm run dev
```

Open `http://localhost:3000`. To use another backend URL, copy `.env.example`
to `.env.local` and update the API/WebSocket URLs. `NEXT_PUBLIC_STALE_SECONDS`
controls when an online worker is visibly marked stale (10 seconds by default).

The backend-created seed worker is explicitly labelled `REPLAY`; edge payloads
that omit `source` retain the worker-state default of `LIVE`.

## Useful Commands

- `pnpm run dev`: start the local dashboard
- `pnpm run build`: verify the production build
- `pnpm test`: build and verify the rendered dashboard shell
