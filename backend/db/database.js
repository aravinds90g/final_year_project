/**
 * database.js — MongoDB setup using mongodb driver
 * =================================================
 * Collections:
 *   flows   — every traffic record received from ESP8266 + GNN result
 *   devices — known IoT devices and their current status
 */

const { MongoClient } = require("mongodb");

const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017";
const DB_NAME = process.env.DB_NAME || "iot_intrusion_detection";

let db = null;
let flowsCollection = null;
let devicesCollection = null;

/**
 * Initialize MongoDB connection
 */
async function initDB() {
  try {
    const client = new MongoClient(MONGO_URI);
    await client.connect();
    db = client.db(DB_NAME);

    // Create collections if they don't exist
    const collections = await db.listCollections().toArray();
    const collectionNames = collections.map((c) => c.name);

    if (!collectionNames.includes("flows")) {
      await db.createCollection("flows");
    }
    if (!collectionNames.includes("devices")) {
      await db.createCollection("devices");
    }

    flowsCollection = db.collection("flows");
    devicesCollection = db.collection("devices");

    // Create indexes for better query performance
    await flowsCollection.createIndex({ timestamp: -1 });
    await flowsCollection.createIndex({ src_ip: 1 });
    await flowsCollection.createIndex({ dst_ip: 1 });
    await flowsCollection.createIndex({ attack_type: 1 });

    await devicesCollection.createIndex({ ip: 1 });
    await devicesCollection.createIndex({ status: 1 });
    await devicesCollection.createIndex({ last_seen: -1 });

    console.log(`✓ MongoDB connected to ${MONGO_URI}/${DB_NAME}`);
    return db;
  } catch (err) {
    console.error("✗ MongoDB connection failed:", err.message);
    throw err;
  }
}

/**
 * Insert a flow record
 */
async function insertFlow(payload) {
  if (!flowsCollection) throw new Error("Database not initialized");

  const doc = {
    timestamp: new Date(),
    src_ip: payload.src_ip,
    dst_ip: payload.dst_ip,
    in_bytes: payload.in_bytes || 0,
    out_bytes: payload.out_bytes || 0,
    in_pkts: payload.in_pkts || 0,
    out_pkts: payload.out_pkts || 0,
    flow_duration_ms: payload.flow_duration_ms || 0,
    protocol: payload.protocol || 6,
    src_port: payload.src_port || 0,
    dst_port: payload.dst_port || 80,
    tcp_flags: payload.tcp_flags || 0,
    src_to_dst_bps: payload.src_to_dst_bps || 0,
    attack_type: payload.attack_type,
    label_id: payload.label_id,
    confidence: payload.confidence,
    is_blocked: payload.is_blocked ? 1 : 0,
  };

  const result = await flowsCollection.insertOne(doc);
  return result;
}

/**
 * Upsert a device record (insert or update)
 */
async function upsertDevice(ip, payload) {
  if (!devicesCollection) throw new Error("Database not initialized");

  const now = new Date();
  const update = {
    $set: {
      ip: ip,
      last_seen: now,
      status: payload.status || "Normal",
      attack_type: payload.attack_type || "Normal",
      is_blocked: payload.is_blocked ? 1 : 0,
    },
    $setOnInsert: {
      first_seen: now,
    },
    $inc: {
      packet_count: 1,
    },
  };

  const result = await devicesCollection.updateOne(
    { ip: ip },
    update,
    { upsert: true }
  );

  return result;
}

/**
 * Block a device (set is_blocked = 1)
 */
async function blockDevice(ip) {
  if (!devicesCollection) throw new Error("Database not initialized");

  const result = await devicesCollection.updateOne(
    { ip: ip },
    { $set: { is_blocked: 1, status: "Blocked" } }
  );

  return result;
}

/**
 * Unblock a device (set is_blocked = 0)
 */
async function unblockDevice(ip) {
  if (!devicesCollection) throw new Error("Database not initialized");

  const result = await devicesCollection.updateOne(
    { ip: ip },
    { $set: { is_blocked: 0, status: "Normal" } }
  );

  return result;
}

/**
 * Get a single device by IP
 */
async function getDevice(ip) {
  if (!devicesCollection) throw new Error("Database not initialized");
  return devicesCollection.findOne({ ip });
}

/**
 * Get all devices
 */
async function getAllDevices() {
  if (!devicesCollection) throw new Error("Database not initialized");

  const devices = await devicesCollection
    .find({})
    .sort({ last_seen: -1 })
    .toArray();

  return devices;
}

/**
 * Get recent flow records
 */
async function getRecentFlows(limit = 100) {
  if (!flowsCollection) throw new Error("Database not initialized");

  const flows = await flowsCollection
    .find({})
    .sort({ timestamp: -1 })
    .limit(Math.min(limit, 500))
    .toArray();

  return flows;
}

/**
 * Get aggregate statistics
 */
async function getStats() {
  if (!flowsCollection || !devicesCollection)
    throw new Error("Database not initialized");

  const totalFlows = await flowsCollection.countDocuments();
  const attackFlows = await flowsCollection.countDocuments({
    attack_type: { $ne: "Normal" },
  });
  const blockedFlows = await flowsCollection.countDocuments({
    is_blocked: 1,
  });
  const totalDevices = await devicesCollection.countDocuments();
  const blockedDevices = await devicesCollection.countDocuments({
    is_blocked: 1,
  });

  // Attack type breakdown
  const attackBreakdown = await flowsCollection
    .aggregate([
      { $group: { _id: "$attack_type", count: { $sum: 1 } } },
      { $sort: { count: -1 } },
    ])
    .toArray();

  return {
    totalFlows,
    attackFlows,
    blockedFlows,
    totalDevices,
    blockedDevices,
    attackBreakdown,
  };
}

module.exports = {
  initDB,
  insertFlow,
  upsertDevice,
  blockDevice,
  unblockDevice,
  getDevice,
  getAllDevices,
  getRecentFlows,
  getStats,
};
