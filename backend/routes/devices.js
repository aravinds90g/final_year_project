/**
 * devices.js — Device management routes
 * ========================================
 * GET  /api/devices          — all known IoT devices + status
 * GET  /api/flows            — recent flow records
 * GET  /api/stats            — aggregate counts
 * POST /api/devices/unblock/:ip — manually unblock a device
 */

const express = require("express");
const axios   = require("axios");
const router  = express.Router();

const {
  getAllDevices,
  getRecentFlows,
  getStats,
  unblockDevice,
} = require("../db/database");
const { clearBlockedIP } = require("./traffic");


const RYU_API_URL = process.env.RYU_API_URL || "http://localhost:8080";

// GET /api/devices
router.get("/", async (req, res) => {
  try {
    const devices = await getAllDevices();
    res.json({ devices });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/flows?limit=100
router.get("/flows", async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 100, 500);
    const flows = await getRecentFlows(limit);
    res.json({ flows });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/stats
router.get("/stats", async (req, res) => {
  try {
    const stats = await getStats();
    res.json({ stats });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/devices/unblock/:ip
router.post("/unblock/:ip", async (req, res) => {
  const { ip } = req.params;
  try {
    // Unblock in Ryu
    try {
      await axios.delete(`${RYU_API_URL}/unblock/${ip}`, { timeout: 3000 });
    } catch (_) {
      console.warn(`[devices] Ryu unavailable when unblocking ${ip}`);
    }

    // Update local DB
    await unblockDevice(ip);
    clearBlockedIP(ip);   // remove from in-memory cache so traffic flows again

    const io = req.app.get("io");
    io.emit("unblocked", { ip, timestamp: new Date().toISOString() });

    res.json({ status: "unblocked", ip });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
