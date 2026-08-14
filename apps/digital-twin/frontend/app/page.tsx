"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type Worker = {
  worker_id: string;
  timestamp: string;
  tracking: { track_id: number | null; camera_id: string | null; online: boolean };
  ppe: Record<"helmet" | "vest" | "gloves" | "boots", boolean | null>;
  activity: { baseline: string; baseline_confidence: number; stgcn: string; stgcn_confidence: number; display_activity: string };
  edge: { fps: number | null; cpu_temperature: number | null; throttled: boolean };
};

type HistoryEvent = {
  id: number; worker_id: string; timestamp: string; activity: string;
  activity_confidence: number; track_id: number | null; camera_id: string | null;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const WS_URL = API_URL.replace(/^http/, "ws") + "/ws";
const activityColors: Record<string, string> = {
  walking: "#0f6b50", carrying: "#b76020", material_handling: "#7b54a3",
  bending: "#b14444", standing: "#3e6ea7", idle: "#78817d", unknown: "#9aa39f",
};

function confidence(worker: Worker) {
  return Math.max(worker.activity.baseline_confidence, worker.activity.stgcn_confidence);
}

function timeLabel(value: string) {
  return new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(value));
}

function durationLabel(seconds: number) {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

export default function Home() {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [selectedId, setSelectedId] = useState("worker01");
  const [history, setHistory] = useState<HistoryEvent[]>([]);
  const [connection, setConnection] = useState<"connecting" | "live" | "offline">("connecting");
  const [error, setError] = useState<string | null>(null);

  const worker = workers.find((item) => item.worker_id === selectedId) ?? workers[0];

  const loadHistory = useCallback(async (workerId: string) => {
    const response = await fetch(`${API_URL}/workers/${encodeURIComponent(workerId)}/history?limit=12`);
    if (!response.ok) throw new Error("History could not be loaded");
    setHistory(await response.json());
  }, []);

  useEffect(() => {
    let active = true;
    fetch(`${API_URL}/workers`)
      .then((response) => { if (!response.ok) throw new Error("Backend unavailable"); return response.json(); })
      .then((data: Worker[]) => {
        if (!active) return;
        setWorkers(data);
        if (data.length && !data.some((item) => item.worker_id === selectedId)) setSelectedId(data[0].worker_id);
        setError(null);
      })
      .catch(() => active && setError(`Start the Digital Twin API at ${API_URL}`));
    return () => { active = false; };
  }, [selectedId]);

  useEffect(() => {
    if (!worker) return;
    loadHistory(worker.worker_id).catch(() => setError("Activity history is temporarily unavailable"));
  }, [worker?.worker_id, loadHistory]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    const connect = () => {
      setConnection("connecting");
      socket = new WebSocket(WS_URL);
      socket.onopen = () => setConnection("live");
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type !== "worker_update") return;
        const incoming = message.worker as Worker;
        setWorkers((current) => [...current.filter((item) => item.worker_id !== incoming.worker_id), incoming]);
        setError(null);
        if (message.activity_changed && incoming.worker_id === selectedId) loadHistory(incoming.worker_id).catch(() => undefined);
      };
      socket.onclose = () => {
        setConnection("offline");
        if (!stopped) retry = setTimeout(connect, 2500);
      };
      socket.onerror = () => socket?.close();
    };
    connect();
    return () => { stopped = true; if (retry) clearTimeout(retry); socket?.close(); };
  }, [loadHistory, selectedId]);

  const timeline = useMemo(() => history.map((event, index) => {
    const newer = history[index - 1];
    const seconds = newer ? (new Date(newer.timestamp).getTime() - new Date(event.timestamp).getTime()) / 1000 : worker ? (new Date(worker.timestamp).getTime() - new Date(event.timestamp).getTime()) / 1000 : 0;
    return { ...event, duration: durationLabel(seconds) };
  }), [history, worker]);

  const conf = worker ? confidence(worker) : 0;
  return (
    <main className="dashboard-shell">
      <aside className="sidebar">
        <div className="brand-mark" aria-hidden="true">DT</div>
        <div><p className="eyebrow">Workforce Intelligence</p><h1>Digital Twin</h1></div>
        <nav aria-label="Dashboard navigation">
          <a className="nav-item active" href="#overview">Overview</a>
          <a className="nav-item" href="#timeline">Activity timeline</a>
          <a className="nav-item" href="#edge">Edge telemetry</a>
        </nav>
        <div className="sidebar-foot">
          <span className={`status-dot ${connection}`} />
          <div><strong>{connection === "live" ? "Live system" : connection === "connecting" ? "Connecting" : "Offline"}</strong><small>{connection === "live" ? "Streaming worker state" : "Waiting for backend"}</small></div>
        </div>
      </aside>

      <section className="workspace" id="overview">
        <header className="topbar">
          <div><p className="eyebrow">Operations overview</p><h2>Worker monitoring</h2><p className="subtitle">Real-time activity, compliance and edge performance.</p></div>
          <div className={`live-pill ${connection}`}><span className={`status-dot ${connection}`} /> {connection}</div>
        </header>

        {workers.length > 1 && <div className="worker-tabs" aria-label="Select worker">{workers.map((item) => <button className={item.worker_id === worker?.worker_id ? "selected" : ""} key={item.worker_id} onClick={() => setSelectedId(item.worker_id)}>{item.worker_id}</button>)}</div>}
        {error && <div className="notice" role="status">{error}</div>}

        <div className="hero-grid">
          <article className="worker-card">
            <div className="worker-heading">
              <div className="avatar">{worker?.worker_id.slice(-2).toUpperCase() ?? "--"}</div>
              <div><p className="eyebrow">Tracked worker</p><h3>{worker?.worker_id ?? "Awaiting data"}</h3></div>
              <span className={`online-badge ${worker?.tracking.online ? "" : "is-offline"}`}>{worker?.tracking.online ? "Online" : "Offline"}</span>
            </div>
            <div className="activity-panel" style={{ "--activity-color": activityColors[worker?.activity.display_activity ?? "unknown"] } as React.CSSProperties}>
              <p>Current activity</p>
              <strong>{worker?.activity.display_activity.replaceAll("_", " ") ?? "Unknown"}</strong>
              <span>{Math.round(conf * 100)}% confidence</span>
              <div className="confidence-track" aria-label={`${Math.round(conf * 100)} percent confidence`}><i style={{ width: `${conf * 100}%` }} /></div>
            </div>
            <div className="worker-meta">
              <div><span>Track ID</span><strong>{worker?.tracking.track_id ?? "—"}</strong></div>
              <div><span>Camera</span><strong>{worker?.tracking.camera_id ?? "—"}</strong></div>
              <div><span>Last update</span><strong>{worker ? timeLabel(worker.timestamp) : "—"}</strong></div>
            </div>
          </article>

          <article className="signal-card" id="edge">
            <p className="eyebrow">Live edge signal</p>
            <div className="signal-orbit"><span>{worker?.edge.fps?.toFixed(1) ?? "—"}<small>FPS</small></span></div>
            <p>{worker?.tracking.camera_id ?? "No camera connected"}</p>
            <div className="edge-row"><span>CPU temperature</span><strong>{worker?.edge.cpu_temperature != null ? `${worker.edge.cpu_temperature.toFixed(1)}°C` : "—"}</strong></div>
            <div className="edge-row"><span>Throttling</span><strong className={worker?.edge.throttled ? "danger" : "good"}>{worker?.edge.throttled ? "Detected" : "Normal"}</strong></div>
          </article>
        </div>

        <div className="detail-grid">
          <section className="panel" aria-labelledby="ppe-heading">
            <div className="panel-heading"><div><p className="eyebrow">Compliance</p><h3 id="ppe-heading">PPE status</h3></div><span>{worker ? Object.values(worker.ppe).filter(Boolean).length : 0}/4 detected</span></div>
            <div className="ppe-grid">{(["helmet", "vest", "gloves", "boots"] as const).map((item) => <div className={`ppe-item ${worker?.ppe[item] === true ? "pass" : worker?.ppe[item] === false ? "fail" : "unknown"}`} key={item}><i>{worker?.ppe[item] === true ? "✓" : worker?.ppe[item] === false ? "!" : "?"}</i><span>{item}</span><small>{worker?.ppe[item] === true ? "Detected" : worker?.ppe[item] === false ? "Missing" : "Not assessed"}</small></div>)}</div>
          </section>

          <section className="panel" id="timeline" aria-labelledby="timeline-heading">
            <div className="panel-heading"><div><p className="eyebrow">Persistent history</p><h3 id="timeline-heading">Activity timeline</h3></div><span>Latest {timeline.length}</span></div>
            <ol className="timeline-list">{timeline.length ? timeline.map((event, index) => <li key={event.id}><i style={{ background: activityColors[event.activity] ?? activityColors.unknown }} /><div><strong>{event.activity.replaceAll("_", " ")}</strong><span>{timeLabel(event.timestamp)} · {Math.round(event.activity_confidence * 100)}% confidence</span></div><small>{index === 0 ? "current" : event.duration}</small></li>) : <li className="empty-row">No activity transitions recorded yet.</li>}</ol>
          </section>
        </div>
      </section>
    </main>
  );
}
