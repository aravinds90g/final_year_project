/**
 * server.js — Main Node.js Backend
 * ==================================
 * Port: 3001
 *
 * Architecture:
 *   ESP8266 ──POST /api/traffic──► Express ──► GNN API (FastAPI :5001)
 *                                      │
 *                                      ├──► MongoDB
 *                                      │
 *                                      ├──► Ryu SDN Controller (:8080)
 *                                      │
 *                                      └──► Socket.IO ──► Dashboard (:3000)
 */

require("dotenv").config();
const express   = require("express");
const http      = require("http");
const cors      = require("cors");
const morgan    = require("morgan");
const { Server } = require("socket.io");

const { initDB } = require("./db/database");
const trafficRoutes = require("./routes/traffic");
const deviceRoutes  = require("./routes/devices");

const app    = express();
const server = http.createServer(app);
const PORT   = process.env.PORT || 3001;

// ── Socket.IO ────────────────────────────────────────────────────────────
const io = new Server(server, {
  cors: {
    origin: "*",   // allow Next.js dashboard on any port
    methods: ["GET", "POST"],
  },
});

// Make io available to routes via app
app.set("io", io);

// ── Middleware ────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

// ── Routes ────────────────────────────────────────────────────────────────
app.use("/api/traffic", trafficRoutes);   // ESP8266 sends here
app.use("/api/devices", deviceRoutes);    // dashboard reads here
app.use("/api/flows",   deviceRoutes);    // alias for flows route
app.use("/api/stats",   deviceRoutes);    // alias for stats route

// Health check
app.get("/health", (req, res) => {
  res.json({
    status:    "ok",
    timestamp: new Date().toISOString(),
    port:      PORT,
  });
});

// ── Socket.IO events ──────────────────────────────────────────────────────
io.on("connection", (socket) => {
  console.log(`[socket.io] Dashboard connected: ${socket.id}`);

  socket.on("disconnect", () => {
    console.log(`[socket.io] Dashboard disconnected: ${socket.id}`);
  });

  // Dashboard can request a manual simulation trigger for demo
  socket.on("trigger_attack", (data) => {
    console.log(`[socket.io] Manual attack trigger:`, data);
    // This will be processed by the frontend demo buttons
    io.emit("trigger_attack", data);
  });
});

// ── Start ─────────────────────────────────────────────────────────────────
async function start() {
  try {
    // Initialize MongoDB
    await initDB();

    server.listen(PORT, () => {
      console.log(`\n╔══════════════════════════════════════════════╗`);
      console.log(`║  GNN-SDN Backend running on port ${PORT}        ║`);
      console.log(`╠══════════════════════════════════════════════╣`);
      console.log(`║  POST /api/traffic    ← ESP8266 sends here  ║`);
      console.log(`║  GET  /api/devices    ← device list          ║`);
      console.log(`║  GET  /api/flows      ← flow history         ║`);
      console.log(`║  GET  /api/stats      ← attack counts        ║`);
      console.log(`║  GNN  → http://localhost:5001/predict        ║`);
      console.log(`║  SDN  → http://localhost:8080/block/:ip      ║`);
      console.log(`╚══════════════════════════════════════════════╝\n`);
    });
  } catch (err) {
    console.error("Failed to start server:", err.message);
    process.exit(1);
  }
}

start();

module.exports = { app, server, io };
