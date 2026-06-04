"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { io, Socket } from "socket.io-client";

// ── Types ───────────────────────────────────────────────────────────────────
interface DetectionEvent {
  src_ip: string;
  dst_ip: string;
  attack_type: string;
  label_id: number;
  confidence: number;
  all_probs: Record<string, number>;
  is_blocked: boolean;
  sdn_action: string | null;
  timestamp: string;
}

interface Device {
  ip: string;
  first_seen: string;
  last_seen: string;
  status: "Normal" | "Attack" | "Blocked";
  attack_type: string;
  is_blocked: number;
  packet_count: number;
}

interface Stats {
  totalFlows: number;
  attackFlows: number;
  blockedFlows: number;
  totalDevices: number;
  blockedDevices: number;
  attackBreakdown: Array<{ _id: string; count: number }>;
}

interface Packet {
  id: string;
  src_ip: string;
  attack_type: string;
  label_id: number;
  is_blocked: boolean;
  progress: number; // 0..1
  startTime: number;
  duration: number;
}

const BACKEND_URL =
  typeof window !== "undefined"
    ? `http://${window.location.hostname}:3001`
    : "http://localhost:3001";

const ATTACK_COLOR: Record<string, string> = {
  Normal: "#10b981",
  DDoS:   "#ef4444",
};

const ATTACK_LABEL: Record<string, string> = {
  Normal: "Normal",
  DDoS:   "DDoS",
};

// ── Animated Network Topology ────────────────────────────────────────────────
function NetworkTopology({
  devices,
  packets,
  connected,
}: {
  devices: Device[];
  packets: Packet[];
  connected: boolean;
}) {
  // Layout constants
  const W = 900;
  const H = 380;

  const SERVER_X = W * 0.5;
  const SERVER_Y = H * 0.5;
  const GNN_X = W * 0.82;
  const GNN_Y = H * 0.5;

  // Up to 6 device slots arranged in a column on the left
  const MAX_DEVICES = 6;
  const shown = devices.slice(0, MAX_DEVICES);
  const devicePositions = shown.map((_, i) => {
    const total = Math.max(shown.length, 1);
    const spacing = Math.min(60, (H - 60) / total);
    const startY = H / 2 - (spacing * (total - 1)) / 2;
    return { x: W * 0.12, y: startY + i * spacing };
  });

  // Path midpoints (hop through "gateway" at 35%)
  const GW_X = W * 0.35;
  const GW_Y = H * 0.5;

  return (
    <div className="relative w-full rounded-2xl overflow-hidden border border-zinc-800 bg-zinc-950">
      <div className="flex items-center gap-3 px-5 py-4 border-b border-zinc-800">
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
        <h2 className="font-bold text-white text-sm tracking-widest uppercase">
          Live Network Topology
        </h2>
        <span
          className={`ml-auto text-xs font-mono px-3 py-1 rounded-full border ${
            connected
              ? "border-emerald-500/40 text-emerald-400 bg-emerald-500/10"
              : "border-red-500/40 text-red-400 bg-red-500/10"
          }`}
        >
          {connected ? "● SOCKET LIVE" : "○ DISCONNECTED"}
        </span>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ height: "320px" }}
      >
        {/* Grid */}
        {Array.from({ length: 18 }).map((_, i) => (
          <line
            key={`vg${i}`}
            x1={i * 50}
            y1={0}
            x2={i * 50}
            y2={H}
            stroke="#27272a"
            strokeWidth={0.5}
          />
        ))}
        {Array.from({ length: 8 }).map((_, i) => (
          <line
            key={`hg${i}`}
            x1={0}
            y1={i * 50}
            x2={W}
            y2={i * 50}
            stroke="#27272a"
            strokeWidth={0.5}
          />
        ))}

        {/* Backbone: Gateway → Server → GNN (always visible) */}
        <line
          x1={GW_X}
          y1={GW_Y}
          x2={SERVER_X}
          y2={SERVER_Y}
          stroke="#3f3f46"
          strokeWidth={2}
          strokeDasharray="6 4"
        />
        <line
          x1={SERVER_X}
          y1={SERVER_Y}
          x2={GNN_X}
          y2={GNN_Y}
          stroke="#3f3f46"
          strokeWidth={2}
          strokeDasharray="6 4"
        />

        {/* Device → Gateway lines */}
        {devicePositions.map((pos, i) => {
          const dev = shown[i];
          const color =
            dev.is_blocked === 1
              ? "#f59e0b"
              : dev.status === "Attack"
              ? ATTACK_COLOR[dev.attack_type] || "#ef4444"
              : "#27272a";
          return (
            <line
              key={`dl${i}`}
              x1={pos.x}
              y1={pos.y}
              x2={GW_X}
              y2={GW_Y}
              stroke={color}
              strokeWidth={dev.is_blocked === 1 || dev.status === "Attack" ? 1.5 : 1}
              strokeOpacity={0.5}
            />
          );
        })}

        {/* ── Animated Packets ── */}
        {packets.map((pkt) => {
          const devIdx = shown.findIndex((d) => d.ip === pkt.src_ip);
          const devPos =
            devIdx >= 0
              ? devicePositions[devIdx]
              : { x: W * 0.12, y: H * 0.5 };

          const color = pkt.is_blocked
            ? "#f59e0b"
            : ATTACK_COLOR[pkt.attack_type] || "#10b981";

          // Path: device → gateway → server → gnn
          // Segment boundaries: 0..0.35 | 0.35..0.55 | 0.55..1
          let cx: number, cy: number;
          const p = pkt.progress;

          if (p < 0.35) {
            const t = p / 0.35;
            cx = devPos.x + (GW_X - devPos.x) * t;
            cy = devPos.y + (GW_Y - devPos.y) * t;
          } else if (p < 0.62) {
            const t = (p - 0.35) / 0.27;
            cx = GW_X + (SERVER_X - GW_X) * t;
            cy = GW_Y + (SERVER_Y - GW_Y) * t;
          } else {
            const t = (p - 0.62) / 0.38;
            cx = SERVER_X + (GNN_X - SERVER_X) * t;
            cy = SERVER_Y + (GNN_Y - SERVER_Y) * t;
          }

          const r = pkt.label_id !== 0 ? 6 : 4;
          const blocked = pkt.is_blocked && p > 0.58;

          if (blocked) return null; // Drop the packet visually at server

          return (
            <g key={pkt.id}>
              {/* Glow */}
              <circle
                cx={cx}
                cy={cy}
                r={r * 2.5}
                fill={color}
                opacity={0.2}
              />
              {/* Dot */}
              <circle cx={cx} cy={cy} r={r} fill={color} />
            </g>
          );
        })}

        {/* ── Device nodes ── */}
        {devicePositions.map((pos, i) => {
          const dev = shown[i];
          const color =
            dev.is_blocked === 1
              ? "#f59e0b"
              : dev.status === "Attack"
              ? ATTACK_COLOR[dev.attack_type] || "#ef4444"
              : "#10b981";
          return (
            <g key={dev.ip}>
              <circle cx={pos.x} cy={pos.y} r={14} fill={color} opacity={0.12} />
              <circle
                cx={pos.x}
                cy={pos.y}
                r={9}
                fill="#09090b"
                stroke={color}
                strokeWidth={2}
              />
              <text
                x={pos.x}
                y={pos.y + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={7}
                fill={color}
                fontFamily="monospace"
              >
                ESP
              </text>
              <text
                x={pos.x - 20}
                y={pos.y}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={9}
                fill="#a1a1aa"
                fontFamily="monospace"
              >
                {dev.ip.split(".").slice(-2).join(".")}
              </text>
              {dev.is_blocked === 1 && (
                <text
                  x={pos.x}
                  y={pos.y - 18}
                  textAnchor="middle"
                  fontSize={9}
                  fill="#f59e0b"
                  fontFamily="monospace"
                >
                  ✕ BLOCKED
                </text>
              )}
            </g>
          );
        })}

        {/* ── SDN Gateway node ── */}
        <g>
          <circle cx={GW_X} cy={GW_Y} r={22} fill="#3b82f6" opacity={0.1} />
          <circle
            cx={GW_X}
            cy={GW_Y}
            r={14}
            fill="#09090b"
            stroke="#3b82f6"
            strokeWidth={2}
          />
          <text
            x={GW_X}
            y={GW_Y}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={7}
            fill="#3b82f6"
            fontFamily="monospace"
            fontWeight="bold"
          >
            SDN
          </text>
          <text
            x={GW_X}
            y={GW_Y + 26}
            textAnchor="middle"
            fontSize={9}
            fill="#71717a"
            fontFamily="monospace"
          >
            Gateway
          </text>
        </g>

        {/* ── Backend Server node ── */}
        <g>
          <circle cx={SERVER_X} cy={SERVER_Y} r={26} fill="#ec4899" opacity={0.1} />
          <circle
            cx={SERVER_X}
            cy={SERVER_Y}
            r={18}
            fill="#09090b"
            stroke="#ec4899"
            strokeWidth={2.5}
          />
          <text
            x={SERVER_X}
            y={SERVER_Y}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={7}
            fill="#ec4899"
            fontFamily="monospace"
            fontWeight="bold"
          >
            NODE
          </text>
          <text
            x={SERVER_X}
            y={SERVER_Y + 30}
            textAnchor="middle"
            fontSize={9}
            fill="#71717a"
            fontFamily="monospace"
          >
            Backend :3001
          </text>
        </g>

        {/* ── GNN node ── */}
        <g>
          <circle cx={GNN_X} cy={GNN_Y} r={22} fill="#a855f7" opacity={0.1} />
          <circle
            cx={GNN_X}
            cy={GNN_Y}
            r={14}
            fill="#09090b"
            stroke="#a855f7"
            strokeWidth={2}
          />
          <text
            x={GNN_X}
            y={GNN_Y}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={7}
            fill="#a855f7"
            fontFamily="monospace"
            fontWeight="bold"
          >
            GNN
          </text>
          <text
            x={GNN_X}
            y={GNN_Y + 26}
            textAnchor="middle"
            fontSize={9}
            fill="#71717a"
            fontFamily="monospace"
          >
            Classifier :5001
          </text>
        </g>

        {/* Legend: Normal / DDoS / Blocked */}
        <g transform={`translate(${W - 160}, 14)`}>
          {[
            { color: "#10b981", label: "Normal" },
            { color: "#ef4444", label: "DDoS" },
            { color: "#f59e0b", label: "Blocked" },
          ].map(({ color, label }, i) => (
            <g key={label} transform={`translate(0, ${i * 18})`}>
              <circle cx={7} cy={7} r={5} fill={color} />
              <text x={16} y={11} fontSize={10} fill="#a1a1aa" fontFamily="monospace">
                {label}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

// ── Stats card ──────────────────────────────────────────────────────────────
function StatCard({
  label,
  value,
  accent,
  sub,
}: {
  label: string;
  value: number | string;
  accent: string;
  sub?: string;
}) {
  return (
    <div
      className="rounded-2xl border p-5 flex flex-col gap-2"
      style={{
        borderColor: accent + "30",
        background: accent + "08",
      }}
    >
      <p className="text-xs font-bold uppercase tracking-widest" style={{ color: accent }}>
        {label}
      </p>
      <p className="text-4xl font-black text-white leading-none">{value}</p>
      {sub && <p className="text-xs text-zinc-500">{sub}</p>}
    </div>
  );
}

// ── Flow row ─────────────────────────────────────────────────────────────────
function FlowRow({ event, fresh }: { event: DetectionEvent; fresh: boolean }) {
  const isAttack = event.label_id !== 0;
  const color = ATTACK_COLOR[event.attack_type] || "#10b981";
  return (
    <tr
      className="border-b border-zinc-800 transition-all duration-500"
      style={{
        background: fresh
          ? isAttack
            ? "rgba(239,68,68,0.07)"
            : "rgba(16,185,129,0.04)"
          : "transparent",
      }}
    >
      <td className="px-4 py-2.5 font-mono text-xs text-zinc-500 whitespace-nowrap">
        {new Date(event.timestamp).toLocaleTimeString("en-GB")}
      </td>
      <td className="px-4 py-2.5 font-mono text-sm text-zinc-200">{event.src_ip}</td>
      <td className="px-4 py-2.5 font-mono text-xs text-zinc-500">{event.dst_ip}</td>
      <td className="px-4 py-2.5">
        <span
          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full font-bold text-xs"
          style={{ background: color + "20", color }}
        >
          {isAttack ? "⚠" : "✓"} {ATTACK_LABEL[event.attack_type] || event.attack_type}
        </span>
      </td>
      <td className="px-4 py-2.5">
        <div className="flex items-center gap-2">
          <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{ width: `${event.confidence * 100}%`, background: color }}
            />
          </div>
          <span className="text-xs font-mono text-zinc-400">
            {(event.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </td>
      <td className="px-4 py-2.5 text-xs font-bold">
        {event.is_blocked ? (
          <span className="text-amber-400">✕ BLOCKED</span>
        ) : isAttack ? (
          <span style={{ color }}>DETECTED</span>
        ) : (
          <span className="text-zinc-600">PASS</span>
        )}
      </td>
    </tr>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function Home() {
  const socketRef = useRef<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const [logs, setLogs] = useState<DetectionEvent[]>([]);
  const [devices, setDevices] = useState<Device[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [packets, setPackets] = useState<Packet[]>([]);
  const [freshIds, setFreshIds] = useState<Set<string>>(new Set());
  const packetTimer = useRef<NodeJS.Timeout | null>(null);

  // ── Fetch initial data ──────────────────────────────────────────────────
  const fetchData = useCallback(async () => {
    try {
      const [devRes, statsRes, flowRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/devices`),
        fetch(`${BACKEND_URL}/api/stats`),
        fetch(`${BACKEND_URL}/api/flows?limit=50`),
      ]);
      const devData = await devRes.json();
      const statsData = await statsRes.json();
      const flowData = await flowRes.json();
      setDevices(devData.devices || []);
      setStats(statsData.stats || null);

      const flows: DetectionEvent[] = (flowData.flows || []).map((f: Record<string, unknown>) => ({
        src_ip: f.src_ip,
        dst_ip: f.dst_ip,
        attack_type: f.attack_type,
        label_id: f.label_id,
        confidence: f.confidence,
        all_probs: {},
        is_blocked: f.is_blocked === 1,
        sdn_action: null,
        timestamp: f.timestamp,
      }));
      setLogs(flows);
    } catch {
      // backend may be offline
    }
  }, []);

  // ── Animate packets ─────────────────────────────────────────────────────
  const spawnPacket = useCallback((event: DetectionEvent) => {
    const id = `${Date.now()}-${Math.random()}`;
    const duration = event.label_id !== 0 ? 1400 : 1800; // attacks faster
    const newPkt: Packet = {
      id,
      src_ip: event.src_ip,
      attack_type: event.attack_type,
      label_id: event.label_id,
      is_blocked: event.is_blocked,
      progress: 0,
      startTime: Date.now(),
      duration,
    };
    setPackets((prev) => [...prev.slice(-15), newPkt]); // cap at 15 live packets
  }, []);

  // Ticker for packet animation
  useEffect(() => {
    const tick = () => {
      const now = Date.now();
      setPackets((prev) =>
        prev
          .map((p) => ({ ...p, progress: Math.min(1, (now - p.startTime) / p.duration) }))
          .filter((p) => p.progress < 1)
      );
    };
    const id = setInterval(tick, 40); // ~25fps
    return () => clearInterval(id);
  }, []);

  // ── Socket.IO ───────────────────────────────────────────────────────────
  useEffect(() => {
    fetchData();

    const socket = io(BACKEND_URL, { transports: ["websocket", "polling"] });
    socketRef.current = socket;

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));

    socket.on("detection", (event: DetectionEvent) => {
      const id = `${event.timestamp}-${event.src_ip}`;

      // Update log list
      setLogs((prev) => [event, ...prev].slice(0, 150));

      // Flash the row
      setFreshIds((s) => new Set([...s, id]));
      setTimeout(() => setFreshIds((s) => { const n = new Set(s); n.delete(id); return n; }), 1200);

      // Update device state
      setDevices((prev) => {
        const exists = prev.find((d) => d.ip === event.src_ip);
        if (exists) {
          return prev.map((d) =>
            d.ip === event.src_ip
              ? {
                  ...d,
                  status: event.is_blocked ? "Blocked" : event.label_id !== 0 ? "Attack" : "Normal",
                  attack_type: event.attack_type,
                  is_blocked: event.is_blocked ? 1 : 0,
                  last_seen: event.timestamp,
                  packet_count: d.packet_count + 1,
                }
              : d
          );
        }
        return [
          ...prev,
          {
            ip: event.src_ip,
            first_seen: event.timestamp,
            last_seen: event.timestamp,
            status: event.is_blocked ? "Blocked" : event.label_id !== 0 ? "Attack" : "Normal",
            attack_type: event.attack_type,
            is_blocked: event.is_blocked ? 1 : 0,
            packet_count: 1,
          },
        ];
      });

      // Update stats counters
      setStats((prev) => {
        if (!prev) return prev;
        const bd = [...(prev.attackBreakdown || [])];
        const idx = bd.findIndex((b) => b._id === event.attack_type);
        if (idx >= 0) bd[idx] = { ...bd[idx], count: bd[idx].count + 1 };
        else bd.push({ _id: event.attack_type, count: 1 });
        return {
          ...prev,
          totalFlows: prev.totalFlows + 1,
          attackFlows: event.label_id !== 0 ? prev.attackFlows + 1 : prev.attackFlows,
          blockedFlows: event.is_blocked ? prev.blockedFlows + 1 : prev.blockedFlows,
          attackBreakdown: bd,
        };
      });

      // Spawn packet animation
      spawnPacket(event);
    });

    socket.on("unblocked", ({ ip }: { ip: string }) => {
      setDevices((prev) =>
        prev.map((d) =>
          d.ip === ip ? { ...d, status: "Normal", attack_type: "Normal", is_blocked: 0 } : d
        )
      );
    });

    return () => { socket.disconnect(); };
  }, [fetchData, spawnPacket]);

  const handleUnblock = async (ip: string) => {
    await fetch(`${BACKEND_URL}/api/devices/unblock/${ip}`, { method: "POST" });
  };

  // Derived stats (binary: Normal vs DDoS only)
  const ddosCount   = stats?.attackBreakdown?.find((b) => b._id === "DDoS")?.count ?? 0;
  const normalCount = stats?.attackBreakdown?.find((b) => b._id === "Normal")?.count ?? 0;
  const recentDdos   = logs.filter((l) => l.attack_type === "DDoS").length;
  const recentNormal = logs.filter((l) => l.attack_type === "Normal").length;
  const ddosPct = stats && stats.totalFlows > 0
    ? Math.round((ddosCount / stats.totalFlows) * 100)
    : 0;

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-zinc-800/70 bg-zinc-950/90 backdrop-blur-xl">
        <div className="max-w-[1400px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-pink-600 to-purple-600 flex items-center justify-center text-lg">
              🛡
            </div>
            <div>
              <h1 className="text-xl font-extrabold text-white tracking-tight">
                GNN-SDN Sentinel
              </h1>
              <p className="text-[11px] text-zinc-500 font-mono">
                IoT Intrusion Detection · Real-Time
              </p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="hidden md:flex gap-8 text-xs text-zinc-400 font-mono">
              <span>Backend <span className="text-white">:3001</span></span>
              <span>GNN <span className="text-white">:5001</span></span>
              <span>Dashboard <span className="text-white">:3000</span></span>
            </div>
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-bold ${
              connected
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                : "border-red-500/40 bg-red-500/10 text-red-400"
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-emerald-400 animate-pulse" : "bg-red-400"}`} />
              {connected ? "LIVE" : "OFFLINE"}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-6 py-8 space-y-8">

        {/* Stats Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            label="Total Flows"
            value={(stats?.totalFlows ?? 0).toLocaleString()}
            accent="#3b82f6"
            sub="all processed traffic"
          />
          <StatCard
            label="DDoS Detected"
            value={ddosCount}
            accent="#ef4444"
            sub={`${recentDdos} in last 50 flows · ${ddosPct}% of total`}
          />
          <StatCard
            label="Normal Traffic"
            value={normalCount}
            accent="#10b981"
            sub={`${recentNormal} in last 50 flows`}
          />
          <StatCard
            label="Devices Blocked"
            value={stats?.blockedDevices ?? 0}
            accent="#f59e0b"
            sub={`${stats?.totalDevices ?? 0} total devices`}
          />
        </div>

        {/* Network Topology */}
        <NetworkTopology devices={devices} packets={packets} connected={connected} />

        {/* Bottom Grid: Devices + Flow Log */}
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">

          {/* Device List */}
          <div className="xl:col-span-1 rounded-2xl border border-zinc-800 bg-zinc-900/30 overflow-hidden">
            <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
              <h2 className="font-bold text-sm uppercase tracking-widest text-white">
                IoT Devices
              </h2>
              <span className="text-xs text-zinc-500 font-mono">{devices.length} nodes</span>
            </div>
            <div className="p-3 space-y-2 max-h-[480px] overflow-y-auto">
              {devices.length === 0 && (
                <div className="text-center py-12 text-zinc-500 text-sm">
                  <div className="text-3xl mb-2">📡</div>
                  Waiting for devices…
                </div>
              )}
              {devices.map((dev) => {
                const color =
                  dev.is_blocked === 1
                    ? "#f59e0b"
                    : dev.status === "Attack"
                    ? ATTACK_COLOR[dev.attack_type] || "#ef4444"
                    : "#10b981";
                return (
                  <div
                    key={dev.ip}
                    className="rounded-xl p-3 border transition-all"
                    style={{
                      borderColor: color + "40",
                      background: color + "08",
                    }}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-sm font-bold text-white">
                        {dev.ip}
                      </span>
                      <span
                        className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-full"
                        style={{ background: color + "20", color }}
                      >
                        {dev.status === "Attack"
                          ? ATTACK_LABEL[dev.attack_type] || dev.attack_type
                          : dev.status}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-zinc-500">
                      <span>{dev.packet_count.toLocaleString()} flows</span>
                      <span>
                        {Math.floor(
                          (Date.now() - new Date(dev.last_seen).getTime()) / 1000
                        )}
                        s ago
                      </span>
                    </div>
                    {dev.is_blocked === 1 && (
                      <button
                        onClick={() => handleUnblock(dev.ip)}
                        className="mt-2 w-full py-1 rounded-lg text-[11px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500/20 transition-colors"
                      >
                        🔓 Unblock Device
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Flow Log */}
          <div className="xl:col-span-3 rounded-2xl border border-zinc-800 bg-zinc-900/30 overflow-hidden">
            <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
              <h2 className="font-bold text-sm uppercase tracking-widest text-white">
                Live Flow Intelligence
              </h2>
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-emerald-400">{logs.length} flows</span>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-zinc-800 text-[10px] uppercase tracking-widest text-zinc-500 bg-zinc-900/60">
                  <tr>
                    {["Time", "Source IP", "Destination", "Classification", "Confidence", "Action"].map(
                      (h) => (
                        <th key={h} className="px-4 py-3 text-left font-semibold">
                          {h}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60">
                  {logs.length === 0 && (
                    <tr>
                      <td colSpan={6} className="py-16 text-center text-zinc-500">
                        <div className="text-3xl mb-2">📡</div>
                        Waiting for traffic…
                      </td>
                    </tr>
                  )}
                  {logs.slice(0, 60).map((log, i) => {
                    const id = `${log.timestamp}-${log.src_ip}`;
                    return (
                      <FlowRow key={`${id}-${i}`} event={log} fresh={freshIds.has(id)} />
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
