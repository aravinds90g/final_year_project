# 🛡️ Real-Time IoT Intrusion Detection & Mitigation
## Using Graph Neural Networks (GAT) + Software-Defined Networking (Ryu)

---

## 🏗️ Architecture

## 🏗️ Architecture

```
Hardware Setup:
  [ ESP32 ] ──(Wi-Fi)──►  Laptop 1 (SDN Gateway)  ────(LAN/Wi-Fi)────►  Laptop 2 (Main Server)

Laptop 1 (Bridge / SDN Gateway):
   │  Proxy script (sdn/gateway.py on port 3002)
   │  Enforces real iptables DROP rules
   ▼
Laptop 2 (Main Server):
   │  Node.js Backend (port 3001)     ←── MongoDB Database
   │    │  POST /predict
   │    ▼
   │  Python/FastAPI GNN Engine (port 5001)
   │    │  Graph Attention Network (GAT)
   │    ▼
   │  Ryu SDN Controller (port 8080)
   │    │  POST /block/<ip>  → OpenFlow DROP rule (or sends command back to Laptop 1)
   │
   └─► Socket.IO events → Next.js Dashboard (port 3000)
```

---

## 📁 Project Structure

```
FinalYearProject/
├── Dataset/
│   └── NF-ToN-IoT-V2.parquet   ← 200MB, 43 features, pre-labeled
│
├── gnn/                          ← Python GNN Engine
│   ├── preprocess.py             ← Parquet → PyG graph
│   ├── model.py                  ← GAT architecture (2 layers, 8 heads)
│   ├── train.py                  ← Training with early stopping
│   ├── inference_api.py          ← FastAPI inference server
│   ├── requirements.txt
│   ├── gnn_model.pth             ← [generated after training]
│   └── scaler.pkl                ← [generated after training]
│
├── backend/                      ← Node.js Backend
│   ├── server.js
│   ├── routes/
│   │   ├── traffic.js
│   │   └── devices.js
│   ├── db/
│   │   └── database.js
│   └── package.json
│
├── esp32/
│   ├── normal_device/
│   │   └── normal_device.ino    ← Arduino firmware (Normal 5s delay)
│   └── ddos_device/
│       └── ddos_device.ino      ← Arduino firmware (DDoS 100ms delay)
│
├── sdn/
│   ├── controller.py            ← Ryu OpenFlow controller
│   └── gateway.py               ← Transparent proxy running on Laptop 1
│
└── dashboard/                   ← Next.js 14 Dashboard
    ├── app/
    │   ├── page.tsx             ← Main dashboard
    │   ├── globals.css
    │   └── components/
    │       ├── DeviceGraph.tsx  ← Canvas force-directed graph
    │       ├── AttackLog.tsx    ← Real-time log table
    │       ├── StatsPanel.tsx   ← Aggregate stats
    │       └── DeviceCard.tsx   ← Per-device status
    └── package.json
```

---

## 🧠 GNN Model — Graph Attention Network (GAT)

| Property | Value |
|---|---|
| **Architecture** | GAT — 2 layers, 8 attention heads → 4 attention heads |
| **Input features** | 8 per node: bytes, packets, protocol, TCP flags, duration, dst ports |
| **Output** | 5-class softmax |
| **Dataset** | NF-ToN-IoT-V2 (200MB Parquet) |
| **Loss** | CrossEntropyLoss + class weights |

**5 Classes:**
| ID | Class | Dataset Labels | SDN Action |
|---|---|---|---|
| 0 | Normal | Normal | Allow |
| 1 | DDoS | DDoS, DoS | Block source IP |
| 2 | Data_Exfiltration | Password, Ransomware | Block dst IP |
| 3 | Botnet | Backdoor, Injection, XSS, MITM | Isolate device |
| 4 | Port_Scan | Scanning, Recon | Alert only |

---

## � Memory Optimizations (16GB RAM Configuration)

The GNN training has been optimized for systems with limited memory (16GB) through several techniques:

### Key Optimizations

| Optimization | Impact | How It Works |
|---|---|---|
| **Temporal Batching** | ↓ 70% memory | Splits data into hourly subgraphs instead of one huge graph |
| **Smaller Model** | ↓ 30% memory | Hidden dims 32→64, attention heads 4/2→8/4 |
| **Gradient Clipping** | Stability | Prevents gradient explosions, enables early stopping |
| **Batch Processing** | ↓ Process X graphs/epoch | Configure `--batch-size` parameter |
| **Data Sampling** | ↓ 80% faster | Default 20% of dataset; adjust `--sample` parameter |

### Memory Usage by Configuration

```
16GB RAM (Recommended):
  - Sample: 20% (656K records)
  - Temporal batches: 24 hourly graphs
  - Memory: ~2-3 GB
  - Training time: 3-5 min/epoch

32GB RAM:
  - Sample: 50% (1.6M records)
  - Temporal batches: 48 hourly graphs
  - Memory: ~6-8 GB
  - Training time: 8-12 min/epoch

48GB+ RAM (Full dataset):
  - Sample: 100% (13.1M records)
  - Temporal batches: 120+ hourly graphs
  - Memory: ~15-20 GB
  - Training time: ~30 min/epoch
```

### How to Adjust for Your System

```bash
# Check available memory first
free -h

# 16GB → 20% sample (default)
python3 train.py --sample 0.2 --batch-size 2

# 32GB → 50% sample
python3 train.py --sample 0.5 --batch-size 4

# 64GB → full dataset
python3 train.py --sample 1.0 --batch-size 8
```

### Technical Details

- **Temporal Batching**: Instead of one 13M-node graph, creates 24 smaller graphs (one per hour) with ~400-500 nodes each
- **Gradient Clipping**: `max_norm=1.0` prevents model divergence without mixed precision
- **Batch Size**: Process multiple temporal graphs per optimizer step (default 4)
- **Feature Scaling**: Single global scaler trained on full dataset before batching

---

### Prerequisites
```bash
# Python dependencies
pip install torch torch-geometric fastapi uvicorn pandas pyarrow scikit-learn networkx joblib

# Node.js (v18+)
node --version  # >= 18
npm --version

# MongoDB (choose one)
# Option A: Local MongoDB server
# https://docs.mongodb.com/manual/installation/

# Option B: MongoDB Atlas (cloud - recommended for production)
# https://www.mongodb.com/cloud/atlas

# SDN Controller (choose one)
# Option A: Mock controller (recommended for Python 3.12)
pip install flask

# Option B: Ryu (requires Python 3.10 or 3.11)
# pip install ryu  # Note: Not compatible with Python 3.12

# Arduino IDE with ESP32 board support + ArduinoJson library
```

> **MongoDB Setup Note**: The backend now uses MongoDB instead of SQLite. 
> Set `MONGO_URI` in `backend/.env` to your MongoDB connection string.
> Default: `mongodb://localhost:27017`

---

### Step 1 — Train the GNN Model

> **⚠️ MEMORY OPTIMIZED FOR 16GB RAM**
> 
> This version uses **temporal batching** (hourly subgraphs) instead of loading the entire graph into memory. 
> Default sample size is reduced to **20%** to fit in memory. Increase `--sample` incrementally if you have more RAM.

```bash
cd gnn

# ✅ RECOMMENDED FOR 16GB RAM (20% data, ~3-5 min)
python3 train.py --sample 0.2 --epochs 50

# For 32GB+ RAM: increase dataset
python3 train.py --sample 0.5 --epochs 100

# Full dataset (requires 48GB+ RAM)
python3 train.py --sample 1.0 --epochs 150

# Output: gnn_model.pth + scaler.pkl + training_log.json
```

---

### Step 2 — Start the GNN Inference API

```bash
cd gnn
python3 inference_api.py

# Verify:
curl http://localhost:5001/health
```

---

### Step 3 — Start the Node.js Backend (with MongoDB)

**Prerequisites:**
```bash
# Make sure MongoDB is running
mongod  # in a separate terminal, or use MongoDB Atlas (cloud)

# Update backend .env if needed
# Default: MONGO_URI=mongodb://localhost:27017
```

```bash
cd backend
npm install
npm start

# Or with development mode (auto-reload):
npm run dev

# Verify:
curl http://localhost:3001/health
```

---

### Step 4 — Start the SDN Gateway on Laptop 1

On your **second laptop (Laptop 1)**, which connects to the ESP32:

```bash
# Provide the IP address of Laptop 2 (Backend)
# Note: We use .venv/bin/python3 because sudo ignores the active virtual environment
sudo .venv/bin/python3 sdn/gateway.py --backend http://<LAPTOP_2_IP>:3001 --port 3002
```

> **Note:** Run with `sudo` so it can execute real `iptables` commands to physically block traffic from attacking ESP32s.

---

### Step 5 — Start the SDN Controller on Laptop 2 (Optional for demo)

**Option A: Mock Controller (Python 3.12)**
```bash
cd sdn
python3 mock_controller.py

# Verify:
curl http://localhost:8080/health
```

**Option B: Ryu (Python 3.10/3.11 only)**
```bash
cd sdn
ryu-manager controller.py --observe-links

# Verify:
curl http://localhost:8080/health
```

> The mock controller provides the same REST API endpoints (block, unblock, blocked) and works identically to Ryu for demonstration purposes.

---

### Step 6 — Start the Dashboard on Laptop 2

```bash
cd dashboard
npm install
npm run dev

# Open: http://localhost:3000
```

---

### Step 7 — Flash ESP32 Firmware

We provide two separate firmwares. You don't need any external buttons or LEDs—just the ESP32 itself. Both firmwares send simple IoT data (temperature & humidity), but the DDoS firmware floods the network.

**Option A: Normal Device Firmware**
1. Open `esp32/normal_device/normal_device.ino` in Arduino IDE
2. Edit config:
   ```cpp
   const char* WIFI_SSID     = "YOUR_WIFI_SSID";
   const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
   const char* SERVER_URL    = "http://<LAPTOP_1_IP>:3002/api/traffic"; 
   ```
3. Upload to an ESP32. It will send 1 request every 5 seconds.

**Option B: DDoS Attacker Firmware**
1. Open `esp32/ddos_device/ddos_device.ino` in Arduino IDE
2. Edit config to match your Wi-Fi and Laptop 1 IP.
3. Upload to a *second* ESP32 (or reflash the first one). It will send **10 requests per second**.

**Data Payload Structure sent by ESP32:**
The ESP32 constructs and sends a simple JSON payload. The Gateway calculates the rate and builds the complex flow record for the GNN automatically.
```json
{
  "device_id": "ESP32-NORMAL-01",
  "temperature": 24.5,
  "humidity": 60.2
}
```

---

## 🎮 Demo Flow

1. Start the Backend, GNN, Dashboard, and Ryu on Laptop 2.
2. Start the SDN Gateway (`gateway.py`) on Laptop 1.
3. Power on the **Normal** ESP32 → connects to Laptop 1 → dashboard on Laptop 2 shows new device sending traffic.
4. Power on the **DDoS** ESP32 to generate flood traffic.
5. Within ~3 seconds: GNN detects the high request rate as DDoS → backend notifies gateway → IP is blocked via iptables on Laptop 1.
6. Dashboard: The attacking device turns red → then grey (blocked) → alert banner appears.
7. The DDoS ESP32 starts printing `STATUS: BLOCKED BY SDN CONTROLLER!` in the Serial Monitor.
8. Press "Unblock" on dashboard → device recovers.

---

## 🔌 API Reference

### Backend (port 3001)
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/traffic` | ESP8266 sends flow data |
| GET | `/api/devices` | All devices + status |
| GET | `/api/flows?limit=100` | Recent flow records |
| GET | `/api/stats` | Aggregate counts |
| POST | `/api/devices/unblock/:ip` | Unblock a device |

### GNN API (port 5001)
| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Classify a single flow |
| POST | `/predict/batch` | Classify multiple flows |
| GET | `/health` | Model status |
| GET | `/classes` | Class definitions |

### Ryu SDN (port 8080)
| Method | Endpoint | Description |
|---|---|---|
| POST | `/block/<ip>` | Install DROP rule |
| DELETE | `/unblock/<ip>` | Remove DROP rule |
| GET | `/blocked` | List blocked IPs |

---

## 📊 Expected Results

After full training on NF-ToN-IoT-V2:
- **Overall accuracy**: > 92%
- **DDoS F1**: > 0.95
- **Botnet F1**: > 0.88
- **Port Scan F1**: > 0.90
- **Inference latency**: < 50ms per flow

---

*Final Year Project — Real-Time IoT Intrusion Detection using GNN + SDN*
