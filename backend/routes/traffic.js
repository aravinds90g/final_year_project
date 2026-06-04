/**
 * traffic.js — POST /api/traffic route  (Binary: Normal vs DDoS)
 * ================================================================
 * Receives flow records from IoT devices,
 * calls GNN for Normal/DDoS prediction, stores result, emits to dashboard.
 *
 * SDN Actions:
 *   Normal → Allow (do nothing)
 *   DDoS   → Block source IP
 */

const express = require("express");
const axios   = require("axios");
const router  = express.Router();

const {
  insertFlow,
  upsertDevice,
  blockDevice,
  getDevice,
} = require("../db/database");

// In-memory cache of blocked IPs to avoid repeated DB lookups
const blockedIPs = new Set();

const GNN_API_URL = process.env.GNN_API_URL || "http://localhost:5001";
const RYU_API_URL = process.env.RYU_API_URL || "http://localhost:8080";

// SDN actions — binary: Normal or DDoS only
const SDN_ACTIONS = {
  Normal: null,        // do nothing
  DDoS:   "block_src", // block source IP
};

/**
 * normalizeFlow — accepts both simple and full payload formats.
 *
 * Simple (e.g. bare-minimum Arduino sketch):
 *   { ip, packets, bytes, device_id }
 *
 * Full (NetFlow-style, preferred):
 *   { src_ip, dst_ip, in_bytes, out_bytes, in_pkts, out_pkts,
 *     flow_duration_ms, protocol, src_port, dst_port, tcp_flags,
 *     src_to_dst_bps }
 */
function normalizeFlow(body) {
  const src_ip = body.src_ip || body.ip || null;
  const dst_ip = body.dst_ip || "192.168.1.100";

  const totalPkts  = body.packets  || 0;
  const totalBytes = body.bytes    || 0;

  const in_bytes         = body.in_bytes         || totalBytes;
  const out_bytes        = body.out_bytes        || Math.floor(totalBytes * 0.3);
  const in_pkts          = body.in_pkts          || Math.ceil(totalPkts / 2);
  const out_pkts         = body.out_pkts         || Math.floor(totalPkts / 2);
  const flow_duration_ms = body.flow_duration_ms || body.duration_ms || 5000;
  const protocol         = body.protocol         || 6;   // TCP
  const src_port         = body.src_port         || Math.floor(Math.random() * (65535 - 1024) + 1024);
  const dst_port         = body.dst_port         || 80;
  const tcp_flags        = body.tcp_flags        || 0;
  const src_to_dst_bps   = body.src_to_dst_bps   ||
    (flow_duration_ms > 0 ? Math.round((in_bytes * 8) / (flow_duration_ms / 1000)) : 0);

  return { src_ip, dst_ip, in_bytes, out_bytes, in_pkts, out_pkts,
           flow_duration_ms, protocol, src_port, dst_port, tcp_flags, src_to_dst_bps };
}

/**
 * POST /api/traffic
 */
router.post("/", async (req, res) => {
  const io = req.app.get("io");   // Socket.IO instance

  try {
    // ── Check if traffic came from Inline Gateway ────────────────────
    // In the Inline IPS architecture, the Gateway queries the GNN and 
    // provides the prediction directly in the payload.
    const isFromGateway = !!req.body.prediction;
    const rawFlow = isFromGateway ? req.body.flow : req.body;
    
    const flow = normalizeFlow(rawFlow);

    // ── Resolve src_ip ────────────────────────────────────────────────
    const socketIP = req.socket.remoteAddress?.replace("::ffff:", "");
    flow.src_ip = flow.src_ip || socketIP;

    if (!flow.src_ip) {
      return res.status(400).json({ error: "Could not determine source IP" });
    }

    // ── Call GNN inference API (Fallback if not inline) ───────────────
    let prediction;
    if (isFromGateway) {
      prediction = req.body.prediction;
    } else {
      try {
        const gnnRes = await axios.post(`${GNN_API_URL}/predict`, flow, {
          timeout: 5000,
        });
        prediction = gnnRes.data;
      } catch (gnnErr) {
        console.warn("[traffic] GNN API unreachable, defaulting to Normal");
        prediction = {
          attack_type: "Normal",
          label_id:    0,
          confidence:  1.0,
          all_probs:   { Normal: 1.0, DDoS: 0.0 },
          blocked:     false,
        };
      }
    }

    const isAttack   = prediction.label_id === 1;  // 1 = DDoS
    const attackType = prediction.attack_type;      // "Normal" or "DDoS"
    const confidence = prediction.confidence;
    const sdnAction  = SDN_ACTIONS[attackType] || null;
    
    // In inline IPS, Gateway blocks it instantly. Otherwise, we block if it's an attack.
    let isBlocked = isFromGateway ? req.body.blocked_by_gateway : isAttack;

    // ── Update Blocked DB Status ──────────────────────────────────────
    if (isBlocked && !blockedIPs.has(flow.src_ip)) {
      await blockDevice(flow.src_ip);
      blockedIPs.add(flow.src_ip);
      console.log(`[traffic] DB recorded block for ${flow.src_ip}`);
    }

    // ── Save to MongoDB ───────────────────────────────────────────────
    await insertFlow({
      src_ip:           flow.src_ip,
      dst_ip:           flow.dst_ip,
      in_bytes:         flow.in_bytes         || 0,
      out_bytes:        flow.out_bytes        || 0,
      in_pkts:          flow.in_pkts          || 0,
      out_pkts:         flow.out_pkts         || 0,
      flow_duration_ms: flow.flow_duration_ms || 0,
      protocol:         flow.protocol         || 6,
      src_port:         flow.src_port         || 0,
      dst_port:         flow.dst_port         || 80,
      tcp_flags:        flow.tcp_flags        || 0,
      src_to_dst_bps:   flow.src_to_dst_bps   || 0,
      attack_type:      attackType,
      label_id:         prediction.label_id,
      confidence:       prediction.confidence,
      is_blocked:       isBlocked ? 1 : 0,
    });

    // Update device registry
    await upsertDevice(flow.src_ip, {
      status:       isBlocked ? "Blocked" : (isAttack ? "Attack" : "Normal"),
      attack_type:  attackType,
      is_blocked:   isBlocked ? 1 : 0,
    });

    // ── Emit real-time event to dashboard ─────────────────────────────
    const event = {
      src_ip:      flow.src_ip,
      dst_ip:      flow.dst_ip,
      attack_type: attackType,
      label_id:    prediction.label_id,
      confidence:  prediction.confidence,
      all_probs:   prediction.all_probs,
      is_blocked:  isBlocked,
      sdn_action:  sdnAction,
      timestamp:   new Date().toISOString(),
    };
    io.emit("detection", event);

    // ── Response to device ────────────────────────────────────────────
    if (isBlocked) {
      return res.status(403).json({
        status:      "blocked",
        attack_type: attackType,
        message:     "DDoS source blocked by SDN controller",
      });
    }

    return res.status(200).json({
      status:      "ok",
      attack_type: attackType,
      confidence:  prediction.confidence,
      sdn_action:  sdnAction,
    });

  } catch (err) {
    console.error("[traffic] Error:", err.message);
    return res.status(500).json({ error: "Internal server error" });
  }
});

module.exports = router;
module.exports.clearBlockedIP = (ip) => blockedIPs.delete(ip);
