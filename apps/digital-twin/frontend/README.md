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

```bash
pnpm install
pnpm run dev
```

Open `http://localhost:3000`. To use another backend URL, copy `.env.example`
to `.env.local` and update `NEXT_PUBLIC_API_URL`.

## Useful Commands

- `pnpm run dev`: start the local dashboard
- `pnpm run build`: verify the production build
- `pnpm test`: build and verify the rendered dashboard shell
