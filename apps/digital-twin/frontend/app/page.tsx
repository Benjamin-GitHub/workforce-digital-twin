"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type PPEObservation = { detected: boolean | null; confidence: number | null };
type PPEValue = PPEObservation | boolean | null | undefined;
type PPE = {
  helmet?: PPEValue; vest?: PPEValue; gloves?: PPEValue; boots?: PPEValue;
  observed_at?: string | null; association_method?: string | null;
};

type Worker = {
  worker_id: string; timestamp: string; source?: "live" | "replay";
  tracking: { track_id: number | null; camera_id: string | null; online: boolean };
  ppe?: PPE | null;
  activity: {
    baseline: string; baseline_confidence: number;
    stgcn: string; stgcn_confidence: number;
    gru: string; gru_confidence: number;
    display_activity: string;
  };
  edge: { fps: number | null; cpu_temperature: number | null; throttled: boolean };
  mobile?: {
    device_id: string; mqtt_client_id: string | null; timestamp: string; last_seen: string;
    connection_state: "connected" | "stale" | "disconnected"; age_s: number;
    accelerometer: { x: number; y: number; z: number };
    gyroscope: { x: number; y: number; z: number };
    location: { latitude: number | null; longitude: number | null; accuracy_m: number | null; gps_enabled: boolean; permission_state: string; zone: string | null };
    battery_level: number | null; association_method: string; association_confidence: number | null;
  } | null;
};
type HistoryEvent = { id: number; worker_id: string; timestamp: string; activity: string; activity_confidence: number; track_id: number | null; camera_id: string | null };
type STGCNStatus = { loaded: boolean; device: string | null; window_size: number; error: string | null };
type GRUStatus = { loaded: boolean; device: string | null; minimum_observations: number | null; sequence_length: number | null; error: string | null };
type STGCNSequence = { worker_id: string; ready: boolean; frames_collected: number; window_size: number };
type GRUPrediction = { worker_id: string; activity: string; confidence: number; probabilities: Record<string, number>; ready: boolean; observations: number };
type ModelDetails = {
  workerId: string;
  stgcn: STGCNSequence | null;
  gru: GRUPrediction | null;
  stgcnError: boolean;
  gruError: boolean;
};
type CardState = "ready" | "warming" | "unknown" | "unavailable" | "error";
type Connection = "connecting" | "connected" | "disconnected";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? API_URL.replace(/^http/, "ws") + "/ws";
const STALE_SECONDS = Number(process.env.NEXT_PUBLIC_STALE_SECONDS ?? 10);
const MOBILE_STALE_SECONDS = Number(process.env.NEXT_PUBLIC_MOBILE_STALE_SECONDS ?? 5);
const MOBILE_DISCONNECTED_SECONDS = Number(process.env.NEXT_PUBLIC_MOBILE_DISCONNECTED_SECONDS ?? 30);
const activityColors: Record<string, string> = { walking: "#35d49a", carrying: "#f5a55b", material_handling: "#a985ed", bending: "#ef7272", standing: "#65a8f5", idle: "#82908c", unknown: "#82908c" };
const label = (value?: string) => (value ?? "unknown").replaceAll("_", " ");
const percent = (value?: number) => `${Math.round((value ?? 0) * 100)}%`;
const clock = (value: string) => new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(value));
const duration = (seconds: number) => seconds < 60 ? `${Math.max(0, Math.round(seconds))}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
const vector = (value?: { x: number; y: number; z: number }) => value ? `${value.x.toFixed(2)}, ${value.y.toFixed(2)}, ${value.z.toFixed(2)}` : "—";
const ppeObservation = (value: PPEValue): PPEObservation => {
  if (value != null && typeof value === "object") {
    return { detected: value.detected ?? null, confidence: value.confidence ?? null };
  }
  return { detected: typeof value === "boolean" ? value : null, confidence: null };
};

function freshness(worker: Worker, now: number) {
  const seconds = Math.max(0, (now - new Date(worker.timestamp).getTime()) / 1000);
  if (!worker.tracking.online) return { state: "offline", text: "Offline" };
  if (seconds > STALE_SECONDS) return { state: "stale", text: `Stale · ${duration(seconds)} ago` };
  return { state: "online", text: seconds < 1 ? "Updated now" : `${duration(seconds)} ago` };
}

function mobileConnection(worker: Worker | undefined, now: number) {
  if (!worker?.mobile) return "not paired";
  const age = Math.max(0, (now - new Date(worker.mobile.last_seen).getTime()) / 1000);
  if (age > MOBILE_DISCONNECTED_SECONDS) return "disconnected";
  if (age > MOBILE_STALE_SECONDS) return "stale";
  return "connected";
}

function WorkerAvatar({ worker, stale }: { worker: Worker; stale: boolean }) {
  const helmet = ppeObservation(worker.ppe?.helmet).detected; const vest = ppeObservation(worker.ppe?.vest).detected;
  return <div className={`body-visual ${stale ? "is-stale" : ""}`} aria-label={`Worker schematic. Helmet ${helmet === true ? "detected" : helmet === false ? "missing" : "unknown"}; vest ${vest === true ? "detected" : vest === false ? "missing" : "unknown"}.`}>
    <div className={`hard-hat ${helmet === true ? "detected" : helmet === false ? "missing" : "unknown"}`}><span>HELMET</span></div><div className="figure-head" />
    <div className="figure-body"><div className={`safety-vest ${vest === true ? "detected" : vest === false ? "missing" : "unknown"}`}><i /><span>VEST</span></div></div>
    <div className="figure-arm left" /><div className="figure-arm right" /><div className="figure-leg left" /><div className="figure-leg right" /><div className="floor-shadow" />
  </div>;
}

function PPEItem({ name, value, confidence, reliable, observedAt, now }: { name: string; value: boolean | null; confidence: number | null; reliable: boolean; observedAt: string | null; now: number }) {
  const state = value === true ? "detected" : value === false ? "missing" : "na";
  const result = value === true ? "DETECTED" : value === false ? "NOT DETECTED" : "UNKNOWN";
  const age = observedAt ? duration(Math.max(0, (now - new Date(observedAt).getTime()) / 1000)) : null;
  const text = `${result}${confidence == null ? "" : ` · ${percent(confidence)}`}${age ? ` · ${age} ago` : ""}${reliable ? "" : " · experimental"}`;
  return <div className={`ppe-row ${state}`}><span className="ppe-icon">{value === true ? "✓" : value === false ? "!" : "—"}</span><div><strong>{name}</strong><small>{text}</small></div>{reliable && <b>PRIMARY</b>}</div>;
}

function ActivityModelCard({ name, activity, confidence, state, detail, tone }: { name: string; activity: string; confidence: number; state: CardState; detail: string; tone: string }) {
  const stateLabel = state === "ready" ? "Ready" : state === "warming" ? "Warming up" : state[0].toUpperCase() + state.slice(1);
  const displayedActivity = state === "ready" ? label(activity) : stateLabel;
  const displayedConfidence = state === "ready" ? percent(confidence) : "0%";
  return <article className={`model-card ${tone} ${state}`}>
    <div className="model-card-heading"><span>{name}</span><b>{stateLabel}</b></div>
    <strong className="model-activity">{displayedActivity}</strong>
    <div className="meter"><i style={{ width: displayedConfidence }} /></div>
    <div className="model-card-meta"><span>{displayedConfidence} confidence</span><small>{detail}</small></div>
  </article>;
}

export default function Home() {
  const [workers, setWorkers] = useState<Worker[]>([]); const [selectedId, setSelectedId] = useState("worker01");
  const [history, setHistory] = useState<HistoryEvent[]>([]);
  const [stgcnStatus, setStgcnStatus] = useState<STGCNStatus | null>(null); const [gruStatus, setGruStatus] = useState<GRUStatus | null>(null);
  const [modelDetails, setModelDetails] = useState<ModelDetails | null>(null);
  const [connection, setConnection] = useState<Connection>("connecting"); const [error, setError] = useState<string | null>(null); const [now, setNow] = useState(0);
  const worker = workers.find((item) => item.worker_id === selectedId) ?? workers[0];
  const helmet = ppeObservation(worker?.ppe?.helmet); const vest = ppeObservation(worker?.ppe?.vest);
  const gloves = ppeObservation(worker?.ppe?.gloves); const boots = ppeObservation(worker?.ppe?.boots);
  const workerId = worker?.worker_id;
  const selectedWorkerRef = useRef<string | undefined>(workerId); const modelDetailsRequest = useRef(0);
  const loadHistory = useCallback(async (id: string) => { const response = await fetch(`${API_URL}/workers/${encodeURIComponent(id)}/history?limit=12`); if (!response.ok) throw new Error("History unavailable"); setHistory(await response.json()); }, []);
  const loadModelDetails = useCallback(async (id: string) => {
    const requestId = ++modelDetailsRequest.current;
    const encodedId = encodeURIComponent(id);
    try {
      const [stgcnResponse, gruResponse] = await Promise.all([
        fetch(`${API_URL}/workers/${encodedId}/stgcn-sequence`),
        fetch(`${API_URL}/workers/${encodedId}/gru-prediction`),
      ]);
      const details: ModelDetails = {
        workerId: id,
        stgcn: stgcnResponse.ok ? await stgcnResponse.json() : null,
        gru: gruResponse.ok ? await gruResponse.json() : null,
        stgcnError: !stgcnResponse.ok && stgcnResponse.status !== 404,
        gruError: !gruResponse.ok && gruResponse.status !== 404,
      };
      if (requestId === modelDetailsRequest.current && selectedWorkerRef.current === id) setModelDetails(details);
    } catch {
      if (requestId === modelDetailsRequest.current && selectedWorkerRef.current === id) {
        setModelDetails({ workerId: id, stgcn: null, gru: null, stgcnError: true, gruError: true });
      }
    }
  }, []);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    Promise.all([fetch(`${API_URL}/workers`), fetch(`${API_URL}/stgcn/status`), fetch(`${API_URL}/gru/status`)]).then(async ([workersResponse, stgcnResponse, gruResponse]) => {
      if (!workersResponse.ok) throw new Error("Backend unavailable"); const data: Worker[] = await workersResponse.json();
      setWorkers(data); setStgcnStatus(stgcnResponse.ok ? await stgcnResponse.json() : null); setGruStatus(gruResponse.ok ? await gruResponse.json() : null); setSelectedId((current) => data.length && !data.some((item) => item.worker_id === current) ? data[0].worker_id : current); setError(null);
    }).catch(() => setError(`Backend unavailable at ${API_URL}`)); return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    selectedWorkerRef.current = workerId;
    if (!workerId) return;
    void Promise.resolve()
      .then(() => Promise.all([loadHistory(workerId), loadModelDetails(workerId)]))
      .catch(() => setError("Worker details are temporarily unavailable"));
  }, [workerId, loadHistory, loadModelDetails]);
  useEffect(() => {
    let socket: WebSocket | null = null; let retry: ReturnType<typeof setTimeout> | undefined; let stopped = false;
    const connect = () => { setConnection("connecting"); socket = new WebSocket(WS_URL); socket.onopen = () => setConnection("connected");
      socket.onmessage = (event) => { try { const message = JSON.parse(event.data); if (message.type !== "worker_update") return; const incoming = message.worker as Worker; setWorkers((current) => [...current.filter((item) => item.worker_id !== incoming.worker_id), incoming]); setError(null); if (incoming.worker_id === selectedWorkerRef.current) void loadModelDetails(incoming.worker_id); if (message.activity_changed) loadHistory(incoming.worker_id).catch(() => undefined); } catch { setError("A malformed live update was ignored"); } };
      socket.onclose = () => { setConnection("disconnected"); if (!stopped) retry = setTimeout(connect, 2500); }; socket.onerror = () => socket?.close(); };
    connect(); return () => { stopped = true; clearTimeout(retry); socket?.close(); };
  }, [loadHistory, loadModelDetails]);

  const fresh = worker ? freshness(worker, now) : { state: "offline", text: "No worker data" }; const source = (worker?.source ?? "live").toUpperCase();
  const timeline = useMemo(() => history.map((event, index) => { const end = index === 0 ? worker?.timestamp : history[index - 1]?.timestamp; const seconds = end ? (new Date(end).getTime() - new Date(event.timestamp).getTime()) / 1000 : 0; return { ...event, segmentDuration: duration(seconds) }; }), [history, worker?.timestamp]);
  const details = modelDetails?.workerId === workerId ? modelDetails : null;
  const baselineState: CardState = !worker || worker.activity.baseline === "unknown" ? "unknown" : "ready";
  const stgcnState: CardState = !worker ? "unknown" : stgcnStatus?.error || details?.stgcnError ? "error" : !stgcnStatus?.loaded ? "unavailable" : worker.activity.stgcn === "unknown" || details?.stgcn?.ready === false ? "warming" : "ready";
  const gruState: CardState = !worker ? "unknown" : gruStatus?.error || details?.gruError ? "error" : !gruStatus?.loaded ? "unavailable" : worker.activity.gru === "unknown" || details?.gru?.ready === false ? "warming" : "ready";
  const stgcnWindow = details?.stgcn?.window_size ?? stgcnStatus?.window_size ?? 16;
  const gruWindow = gruStatus?.minimum_observations ?? gruStatus?.sequence_length ?? 16;
  const stgcnFrames = details?.stgcn?.ready === false ? details.stgcn.frames_collected : 0;
  const gruObservations = details?.gru?.ready === false ? details.gru.observations : 0;
  const readyActivities = [
    baselineState === "ready" ? worker?.activity.baseline : null,
    stgcnState === "ready" ? worker?.activity.stgcn : null,
    gruState === "ready" ? worker?.activity.gru : null,
  ].filter((activity): activity is string => Boolean(activity));
  const modelsDisagree = new Set(readyActivities).size > 1;

  return <main className="dashboard"><aside className="worker-sidebar">
    <div className="brand"><div className="brand-icon">DT</div><div><small>WORKFORCE</small><strong>DIGITAL TWIN</strong></div></div><div className="sidebar-heading"><span>WORKERS</span><b>{workers.length}</b></div>
    <div className="worker-list">{workers.map((item) => { const itemFresh = freshness(item, now); return <button key={item.worker_id} className={`worker-list-card ${item.worker_id === worker?.worker_id ? "selected" : ""}`} onClick={() => setSelectedId(item.worker_id)}><div className="worker-initial">{item.worker_id.slice(-2).toUpperCase()}</div><div><strong>{item.worker_id}</strong><small>{item.tracking.camera_id ?? "No camera"}</small></div><span className={`source-tag ${item.source ?? "live"}`}>{(item.source ?? "live").toUpperCase()}</span><em className={itemFresh.state}>{itemFresh.text}</em></button>; })}{!workers.length && <p className="empty-copy">Waiting for worker state…</p>}</div>
    <div className="connection-card"><span className={`connection-dot ${connection}`} /><div><strong>Backend {connection}</strong><small>WebSocket · {WS_URL.replace(/^wss?:\/\//, "")}</small></div></div>
  </aside><section className="main-area">
    <header className="topbar"><div><p>OPERATIONS / WORKER MONITORING</p><h1>Live worker overview</h1></div><div className="header-badges"><span className={`badge source ${source.toLowerCase()}`}>{source}</span><span className={`badge ${fresh.state}`}>{fresh.text}</span><span className={`badge model ${stgcnStatus?.loaded ? "ready" : "not-ready"}`}>ST-GCN {stgcnStatus?.loaded ? "READY" : "NOT READY"}</span><span className={`badge model ${gruStatus?.loaded ? "ready" : "not-ready"}`}>GRU {gruStatus?.loaded ? "READY" : "NOT READY"}</span></div></header>{error && <div className="notice" role="status">{error}</div>}
    <div className="dashboard-grid"><section className="visual-panel panel"><div className="panel-title"><div><span>WORKER TWIN</span><h2>{worker?.worker_id ?? "Awaiting data"}</h2></div><div className={`pulse-label ${fresh.state}`}><i />{fresh.state}</div></div>
      {worker ? <WorkerAvatar worker={worker} stale={fresh.state !== "online"} /> : <div className="avatar-placeholder">No current worker state</div>}<div className="display-activity"><span>DISPLAY ACTIVITY</span><strong style={{ color: activityColors[worker?.activity.display_activity ?? "unknown"] }}>{label(worker?.activity.display_activity)}</strong><small>Policy unchanged · backend-owned</small></div>
      {worker && <div className="ppe-summary"><PPEItem name="Helmet" value={helmet.detected} confidence={helmet.confidence} reliable observedAt={worker.ppe?.observed_at ?? null} now={now} /><PPEItem name="Vest" value={vest.detected} confidence={vest.confidence} reliable observedAt={worker.ppe?.observed_at ?? null} now={now} /><PPEItem name="Gloves" value={gloves.detected} confidence={gloves.confidence} reliable={false} observedAt={worker.ppe?.observed_at ?? null} now={now} /><PPEItem name="Boots" value={boots.detected} confidence={boots.confidence} reliable={false} observedAt={worker.ppe?.observed_at ?? null} now={now} /></div>}
    </section><aside className="status-panel panel"><div className="panel-title"><div><span>LIVE STATUS</span><h2>Worker detail</h2></div><span className={`connection-chip ${connection}`}>{connection}</span></div>
      <div className="identity-grid"><div><span>WORKER ID</span><strong>{worker?.worker_id ?? "—"}</strong></div><div><span>TRACK ID</span><strong>{worker?.tracking.track_id ?? "—"}</strong></div><div><span>CAMERA ID</span><strong>{worker?.tracking.camera_id ?? "—"}</strong></div><div><span>LAST UPDATE</span><strong>{worker ? clock(worker.timestamp) : "—"}</strong></div></div>
      <div className="telemetry"><div className="telemetry-main"><span>EDGE FRAME RATE</span><strong>{worker?.edge.fps != null ? worker.edge.fps.toFixed(1) : "—"}<small> FPS</small></strong></div><div><span>CPU TEMP</span><strong>{worker?.edge.cpu_temperature != null ? `${worker.edge.cpu_temperature.toFixed(1)}°C` : "—"}</strong></div><div><span>THROTTLED</span><strong className={worker?.edge.throttled ? "bad" : "good"}>{worker ? (worker.edge.throttled ? "YES" : "NO") : "—"}</strong></div></div>
      <div className="recognizer-heading"><span>ANDROID SENSOR SOURCE</span><b>{mobileConnection(worker, now).toUpperCase()}</b></div>
      <div className="mobile-grid">
        <div><span>DEVICE</span><strong>{worker?.mobile ? `${worker.mobile.device_id.slice(0, 8)}…` : "—"}</strong><small>{worker?.mobile?.mqtt_client_id ?? "No MQTT client"}</small></div>
        <div><span>ASSOCIATION</span><strong>{label(worker?.mobile?.association_method)}</strong><small>{worker?.mobile?.association_confidence != null ? `${percent(worker.mobile.association_confidence)} confidence` : "Explicit configuration"}</small></div>
        <div><span>ACCEL X/Y/Z</span><strong>{vector(worker?.mobile?.accelerometer)}</strong><small>m/s²</small></div>
        <div><span>GYRO X/Y/Z</span><strong>{vector(worker?.mobile?.gyroscope)}</strong><small>rad/s</small></div>
        <div><span>GPS / ZONE</span><strong>{worker?.mobile?.location.latitude != null && worker.mobile.location.longitude != null ? `${worker.mobile.location.latitude.toFixed(5)}, ${worker.mobile.location.longitude.toFixed(5)}` : "No fix"}</strong><small>{worker?.mobile?.location.accuracy_m != null ? `±${worker.mobile.location.accuracy_m.toFixed(1)} m` : worker?.mobile?.location.permission_state ?? "—"}{worker?.mobile?.location.zone ? ` · ${worker.mobile.location.zone}` : ""}</small></div>
        <div><span>LAST MOBILE UPDATE</span><strong>{worker?.mobile ? duration(Math.max(0, (now - new Date(worker.mobile.last_seen).getTime()) / 1000)) + " ago" : "—"}</strong><small>{worker?.mobile?.battery_level != null ? `Battery ${Math.round(worker.mobile.battery_level)}%` : "Battery unavailable"}</small></div>
      </div>
    </aside></div>
    <section className="model-comparison panel" aria-labelledby="model-comparison-title"><div className="panel-title"><div><span>ACTIVITY RECOGNISERS</span><h2 id="model-comparison-title">Model comparison</h2></div>{modelsDisagree && <span className="disagreement-indicator">Models disagree</span>}</div>
      <div className="model-comparison-grid">
        <ActivityModelCard name="YOLO Pose Baseline" activity={worker?.activity.baseline ?? "unknown"} confidence={worker?.activity.baseline_confidence ?? 0} state={baselineState} detail="Rule-based pose heuristic" tone="baseline" />
        <ActivityModelCard name="ST-GCN" activity={worker?.activity.stgcn ?? "unknown"} confidence={worker?.activity.stgcn_confidence ?? 0} state={stgcnState} detail={stgcnState === "warming" ? `${stgcnFrames} / ${stgcnWindow} frames` : stgcnState === "ready" ? `${stgcnWindow}-frame window · ${stgcnStatus?.device?.toUpperCase() ?? "device unknown"}` : stgcnState === "error" ? "Model or diagnostics error" : stgcnState === "unavailable" ? "Model not loaded" : "Waiting for worker data"} tone="stgcn" />
        <ActivityModelCard name="GRU" activity={worker?.activity.gru ?? "unknown"} confidence={worker?.activity.gru_confidence ?? 0} state={gruState} detail={gruState === "warming" ? `${gruObservations} / ${gruWindow} observations` : gruState === "ready" ? `${details?.gru?.observations ?? gruWindow} observations · ${gruStatus?.device?.toUpperCase() ?? "device unknown"}` : gruState === "error" ? "Model or diagnostics error" : gruState === "unavailable" ? "Model not loaded" : "Waiting for worker data"} tone="gru" />
      </div>
    </section>
    <section className="timeline-panel panel"><div className="panel-title"><div><span>RECENT HISTORY</span><h2>Activity timeline</h2></div><small>Recorded backend transitions only</small></div><ol className="timeline">{timeline.map((event, index) => <li key={event.id}><time>{clock(event.timestamp)}</time><i style={{ background: activityColors[event.activity] ?? activityColors.unknown }} /><div><strong>{label(event.activity)}</strong><small>{percent(event.activity_confidence)} confidence · {event.camera_id ?? "camera unknown"}</small></div><span>{index === 0 ? "CURRENT SEGMENT" : event.segmentDuration}</span></li>)}{!timeline.length && <li className="empty-copy">No activity transitions have been recorded for this worker.</li>}</ol></section>
  </section></main>;
}
