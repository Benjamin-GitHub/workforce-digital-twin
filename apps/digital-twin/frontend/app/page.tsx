"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type Worker = {
  worker_id: string; timestamp: string; source?: "live" | "replay";
  tracking: { track_id: number | null; camera_id: string | null; online: boolean };
  ppe: Record<"helmet" | "vest" | "gloves" | "boots", boolean | null>;
  activity: { baseline: string; baseline_confidence: number; stgcn: string; stgcn_confidence: number; display_activity: string };
  edge: { fps: number | null; cpu_temperature: number | null; throttled: boolean };
};
type HistoryEvent = { id: number; worker_id: string; timestamp: string; activity: string; activity_confidence: number; track_id: number | null; camera_id: string | null };
type ModelStatus = { loaded: boolean; device: string | null; window_size: number; error: string | null };
type Connection = "connecting" | "connected" | "disconnected";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? API_URL.replace(/^http/, "ws") + "/ws";
const STALE_SECONDS = Number(process.env.NEXT_PUBLIC_STALE_SECONDS ?? 10);
const activityColors: Record<string, string> = { walking: "#35d49a", carrying: "#f5a55b", material_handling: "#a985ed", bending: "#ef7272", standing: "#65a8f5", idle: "#82908c", unknown: "#82908c" };
const label = (value?: string) => (value ?? "unknown").replaceAll("_", " ");
const percent = (value?: number) => `${Math.round((value ?? 0) * 100)}%`;
const clock = (value: string) => new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(value));
const duration = (seconds: number) => seconds < 60 ? `${Math.max(0, Math.round(seconds))}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;

function freshness(worker: Worker, now: number) {
  const seconds = Math.max(0, (now - new Date(worker.timestamp).getTime()) / 1000);
  if (!worker.tracking.online) return { state: "offline", text: "Offline" };
  if (seconds > STALE_SECONDS) return { state: "stale", text: `Stale · ${duration(seconds)} ago` };
  return { state: "online", text: seconds < 1 ? "Updated now" : `${duration(seconds)} ago` };
}

function WorkerAvatar({ worker, stale }: { worker: Worker; stale: boolean }) {
  const helmet = worker.ppe.helmet; const vest = worker.ppe.vest;
  return <div className={`body-visual ${stale ? "is-stale" : ""}`} aria-label={`Worker schematic. Helmet ${helmet === true ? "detected" : helmet === false ? "missing" : "unknown"}; vest ${vest === true ? "detected" : vest === false ? "missing" : "unknown"}.`}>
    <div className={`hard-hat ${helmet === true ? "detected" : helmet === false ? "missing" : "unknown"}`}><span>HELMET</span></div><div className="figure-head" />
    <div className="figure-body"><div className={`safety-vest ${vest === true ? "detected" : vest === false ? "missing" : "unknown"}`}><i /><span>VEST</span></div></div>
    <div className="figure-arm left" /><div className="figure-arm right" /><div className="figure-leg left" /><div className="figure-leg right" /><div className="floor-shadow" />
  </div>;
}

function PPEItem({ name, value, reliable }: { name: string; value: boolean | null; reliable: boolean }) {
  const state = value === true ? "detected" : value === false ? "missing" : "na";
  const text = reliable ? (value === true ? "Detected" : value === false ? "Missing" : "Not assessed") : (value == null ? "N/A · experimental" : `${value ? "Detected" : "Not detected"} · low confidence`);
  return <div className={`ppe-row ${state}`}><span className="ppe-icon">{value === true ? "✓" : value === false ? "!" : "—"}</span><div><strong>{name}</strong><small>{text}</small></div>{reliable && <b>PRIMARY</b>}</div>;
}

export default function Home() {
  const [workers, setWorkers] = useState<Worker[]>([]); const [selectedId, setSelectedId] = useState("worker01");
  const [history, setHistory] = useState<HistoryEvent[]>([]); const [model, setModel] = useState<ModelStatus | null>(null);
  const [connection, setConnection] = useState<Connection>("connecting"); const [error, setError] = useState<string | null>(null); const [now, setNow] = useState(0);
  const worker = workers.find((item) => item.worker_id === selectedId) ?? workers[0];
  const workerId = worker?.worker_id;
  const loadHistory = useCallback(async (id: string) => { const response = await fetch(`${API_URL}/workers/${encodeURIComponent(id)}/history?limit=12`); if (!response.ok) throw new Error("History unavailable"); setHistory(await response.json()); }, []);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    Promise.all([fetch(`${API_URL}/workers`), fetch(`${API_URL}/stgcn/status`)]).then(async ([workersResponse, statusResponse]) => {
      if (!workersResponse.ok) throw new Error("Backend unavailable"); const data: Worker[] = await workersResponse.json();
      setWorkers(data); setModel(statusResponse.ok ? await statusResponse.json() : null); setSelectedId((current) => data.length && !data.some((item) => item.worker_id === current) ? data[0].worker_id : current); setError(null);
    }).catch(() => setError(`Backend unavailable at ${API_URL}`)); return () => clearInterval(timer);
  }, []);
  useEffect(() => { if (!workerId) return; void Promise.resolve().then(() => loadHistory(workerId)).catch(() => setError("Activity history is temporarily unavailable")); }, [workerId, loadHistory]);
  useEffect(() => {
    let socket: WebSocket | null = null; let retry: ReturnType<typeof setTimeout> | undefined; let stopped = false;
    const connect = () => { setConnection("connecting"); socket = new WebSocket(WS_URL); socket.onopen = () => setConnection("connected");
      socket.onmessage = (event) => { try { const message = JSON.parse(event.data); if (message.type !== "worker_update") return; const incoming = message.worker as Worker; setWorkers((current) => [...current.filter((item) => item.worker_id !== incoming.worker_id), incoming]); setError(null); if (message.activity_changed) loadHistory(incoming.worker_id).catch(() => undefined); } catch { setError("A malformed live update was ignored"); } };
      socket.onclose = () => { setConnection("disconnected"); if (!stopped) retry = setTimeout(connect, 2500); }; socket.onerror = () => socket?.close(); };
    connect(); return () => { stopped = true; clearTimeout(retry); socket?.close(); };
  }, [loadHistory]);

  const fresh = worker ? freshness(worker, now) : { state: "offline", text: "No worker data" }; const source = (worker?.source ?? "live").toUpperCase();
  const timeline = useMemo(() => history.map((event, index) => { const end = index === 0 ? worker?.timestamp : history[index - 1]?.timestamp; const seconds = end ? (new Date(end).getTime() - new Date(event.timestamp).getTime()) / 1000 : 0; return { ...event, segmentDuration: duration(seconds) }; }), [history, worker?.timestamp]);

  return <main className="dashboard"><aside className="worker-sidebar">
    <div className="brand"><div className="brand-icon">DT</div><div><small>WORKFORCE</small><strong>DIGITAL TWIN</strong></div></div><div className="sidebar-heading"><span>WORKERS</span><b>{workers.length}</b></div>
    <div className="worker-list">{workers.map((item) => { const itemFresh = freshness(item, now); return <button key={item.worker_id} className={`worker-list-card ${item.worker_id === worker?.worker_id ? "selected" : ""}`} onClick={() => setSelectedId(item.worker_id)}><div className="worker-initial">{item.worker_id.slice(-2).toUpperCase()}</div><div><strong>{item.worker_id}</strong><small>{item.tracking.camera_id ?? "No camera"}</small></div><span className={`source-tag ${item.source ?? "live"}`}>{(item.source ?? "live").toUpperCase()}</span><em className={itemFresh.state}>{itemFresh.text}</em></button>; })}{!workers.length && <p className="empty-copy">Waiting for worker state…</p>}</div>
    <div className="connection-card"><span className={`connection-dot ${connection}`} /><div><strong>Backend {connection}</strong><small>WebSocket · {WS_URL.replace(/^wss?:\/\//, "")}</small></div></div>
  </aside><section className="main-area">
    <header className="topbar"><div><p>OPERATIONS / WORKER MONITORING</p><h1>Live worker overview</h1></div><div className="header-badges"><span className={`badge source ${source.toLowerCase()}`}>{source}</span><span className={`badge ${fresh.state}`}>{fresh.text}</span><span className={`badge model ${model?.loaded ? "ready" : "not-ready"}`}>ST-GCN {model?.loaded ? "READY" : "NOT READY"}</span></div></header>{error && <div className="notice" role="status">{error}</div>}
    <div className="dashboard-grid"><section className="visual-panel panel"><div className="panel-title"><div><span>WORKER TWIN</span><h2>{worker?.worker_id ?? "Awaiting data"}</h2></div><div className={`pulse-label ${fresh.state}`}><i />{fresh.state}</div></div>
      {worker ? <WorkerAvatar worker={worker} stale={fresh.state !== "online"} /> : <div className="avatar-placeholder">No current worker state</div>}<div className="display-activity"><span>DISPLAY ACTIVITY</span><strong style={{ color: activityColors[worker?.activity.display_activity ?? "unknown"] }}>{label(worker?.activity.display_activity)}</strong><small>Policy unchanged · backend-owned</small></div>
      {worker && <div className="ppe-summary"><PPEItem name="Helmet" value={worker.ppe.helmet} reliable /><PPEItem name="Vest" value={worker.ppe.vest} reliable /><PPEItem name="Gloves" value={worker.ppe.gloves} reliable={false} /><PPEItem name="Boots" value={worker.ppe.boots} reliable={false} /></div>}
    </section><aside className="status-panel panel"><div className="panel-title"><div><span>LIVE STATUS</span><h2>Worker detail</h2></div><span className={`connection-chip ${connection}`}>{connection}</span></div>
      <div className="identity-grid"><div><span>WORKER ID</span><strong>{worker?.worker_id ?? "—"}</strong></div><div><span>TRACK ID</span><strong>{worker?.tracking.track_id ?? "—"}</strong></div><div><span>CAMERA ID</span><strong>{worker?.tracking.camera_id ?? "—"}</strong></div><div><span>LAST UPDATE</span><strong>{worker ? clock(worker.timestamp) : "—"}</strong></div></div>
      <div className="recognizer-heading"><span>ACTIVITY RECOGNISERS</span>{worker && worker.activity.baseline !== worker.activity.stgcn && <b>DISAGREEMENT</b>}</div><div className="recognizer-grid"><div><span>FROZEN BASELINE</span><strong>{label(worker?.activity.baseline)}</strong><div className="meter"><i style={{ width: percent(worker?.activity.baseline_confidence) }} /></div><small>{percent(worker?.activity.baseline_confidence)} confidence</small></div><div><span>ST-GCN</span><strong>{label(worker?.activity.stgcn)}</strong><div className="meter purple"><i style={{ width: percent(worker?.activity.stgcn_confidence) }} /></div><small>{percent(worker?.activity.stgcn_confidence)} confidence</small></div></div>
      <div className="model-row"><div><span>MODEL STATUS</span><strong>{model?.loaded ? "Loaded / ready" : "Unavailable"}</strong></div><div><span>DEVICE</span><strong>{model?.device?.toUpperCase() ?? "—"}</strong></div><div><span>WINDOW</span><strong>{model?.window_size ? `${model.window_size} frames` : "—"}</strong></div></div>
      <div className="telemetry"><div className="telemetry-main"><span>EDGE FRAME RATE</span><strong>{worker?.edge.fps != null ? worker.edge.fps.toFixed(1) : "—"}<small> FPS</small></strong></div><div><span>CPU TEMP</span><strong>{worker?.edge.cpu_temperature != null ? `${worker.edge.cpu_temperature.toFixed(1)}°C` : "—"}</strong></div><div><span>THROTTLED</span><strong className={worker?.edge.throttled ? "bad" : "good"}>{worker ? (worker.edge.throttled ? "YES" : "NO") : "—"}</strong></div></div>
    </aside></div>
    <section className="timeline-panel panel"><div className="panel-title"><div><span>RECENT HISTORY</span><h2>Activity timeline</h2></div><small>Recorded backend transitions only</small></div><ol className="timeline">{timeline.map((event, index) => <li key={event.id}><time>{clock(event.timestamp)}</time><i style={{ background: activityColors[event.activity] ?? activityColors.unknown }} /><div><strong>{label(event.activity)}</strong><small>{percent(event.activity_confidence)} confidence · {event.camera_id ?? "camera unknown"}</small></div><span>{index === 0 ? "CURRENT SEGMENT" : event.segmentDuration}</span></li>)}{!timeline.length && <li className="empty-copy">No activity transitions have been recorded for this worker.</li>}</ol></section>
  </section></main>;
}
