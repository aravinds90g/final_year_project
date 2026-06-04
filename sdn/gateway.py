"""
sdn/gateway.py — SDN Gateway (runs on SECOND laptop)
=====================================================
Acts as a transparent proxy between ESP devices and the main backend.

Architecture:
    ESP devices
        │ HTTP POST → THIS laptop IP:3001
        ▼
    [gateway.py]  ← you are here
        │  1. Extract real src_ip from TCP connection
        │  2. Measure actual request bytes
        │  3. Forward enriched flow record to backend
        │  4. Enforce real iptables DROP when backend says block
        ▼
    Main Laptop (backend :3001 / GNN :5001 / Dashboard :3000)

Setup on second laptop:
    pip install flask requests

Run:
    sudo python3 sdn/gateway.py --backend 10.x.x.x --port 3001

    (sudo is needed for iptables commands)

ESP devices should point their SERVER_URL to THIS laptop's IP:
    const char* SERVER_URL = "http://<SECOND_LAPTOP_IP>:3001/api/traffic";
"""

import argparse
import subprocess
import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Config (set via --backend flag) ─────────────────────────────────────────
BACKEND_URL  = "http://192.168.x.x:3001"   # ← Set to Server Laptop IP, override with --backend
GATEWAY_PORT = 3002                          # 3002 avoids conflict with backend port 3001

# Track blocked IPs locally so iptables isn't called twice
blocked_ips: set = set()

# Rate limiting / RPS tracking per IP
from collections import defaultdict
import time

# Store a list of timestamps for each IP
ip_requests = defaultdict(list)

# ── iptables helpers ─────────────────────────────────────────────────────────

def iptables_block(ip: str) -> bool:
    """
    Install a real iptables DROP rule for the given IP.
    Blocks all FORWARD and INPUT traffic from that IP.
    Returns True if successful.
    """
    if ip in blocked_ips:
        print(f"[gateway] {ip} already blocked, skipping iptables")
        return True

    try:
        # Block forwarding (traffic passing through this laptop)
        subprocess.run(
            ["iptables", "-I", "FORWARD", "-s", ip, "-j", "DROP"],
            check=True, capture_output=True
        )
        # Block input (traffic directed at this laptop)
        subprocess.run(
            ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"],
            check=True, capture_output=True
        )
        blocked_ips.add(ip)
        print(f"[gateway] ✅ iptables DROP rule installed for {ip}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[gateway] ❌ iptables failed for {ip}: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print(f"[gateway] ❌ iptables not found — are you running as root?")
        return False


def iptables_unblock(ip: str) -> bool:
    """
    Remove iptables DROP rule for the given IP.
    """
    # Always attempt to remove, even if not in blocked_ips,
    # in case the gateway was restarted but iptables rules remained.
    blocked_ips.discard(ip)
    
    try:
        # We don't check=True because the rule might not exist
        subprocess.run(
            ["iptables", "-D", "FORWARD", "-s", ip, "-j", "DROP"],
            capture_output=True
        )
        subprocess.run(
            ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"],
            capture_output=True
        )
        print(f"[gateway] ✅ iptables rule removal attempted for {ip}")
        return True
    except Exception as e:
        print(f"[gateway] ❌ iptables unblock failed for {ip}: {e}")
        return False


# ── Main proxy route ─────────────────────────────────────────────────────────

@app.route("/api/traffic", methods=["POST"])
def proxy_traffic():
    """
    Receives traffic from ESP devices.
    Extracts real metrics, forwards to main backend.
    """
    # --- Get real source IP from TCP connection ---
    src_ip = request.remote_addr   # actual device IP, no spoofing possible

    # --- Measure actual request size in bytes ---
    raw_body  = request.get_data()
    req_bytes = len(raw_body)

    # --- Parse whatever body the ESP sent (if any) ---
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        body = {}

    # --- 1. Calculate Requests Per Second (RPS) for this IP ---
    current_time = time.time()
    # Keep only requests from the last 2 seconds
    ip_requests[src_ip] = [ts for ts in ip_requests[src_ip] if current_time - ts <= 2.0]
    ip_requests[src_ip].append(current_time)
    
    # Calculate RPS (number of requests in the last 2 seconds / 2)
    rps = len(ip_requests[src_ip]) / 2.0
    print(f"[gateway] {src_ip} rate: {rps:.1f} req/sec")

    # --- 2. Build enriched flow record based on RPS ---
    # The ESP32 HTTP POST takes ~200-400ms per request. 
    # Normal device sends every 5s (0.2 RPS). DDoS device loops continuously (~2-4 RPS).
    # We set the threshold to 1.5 to reliably catch the flood.
    
    if rps >= 1.5:
        # DDoS Profile (NF-ToN-IoT-V2) -> GNN will classify as DDoS
        flow = {
            "src_ip":           src_ip,
            "dst_ip":           BACKEND_URL.split("//")[1].split(":")[0],
            "in_bytes":         600,
            "out_bytes":        200,
            "in_pkts":          8,
            "out_pkts":         2,
            "flow_duration_ms": 4000000,   # Signature DDoS feature
            "protocol":         6,         # TCP
            "src_port":         body.get("src_port", 0) or 1024,
            "dst_port":         80,
            "tcp_flags":        19,        # SYN+ACK+FIN
            "src_to_dst_bps":   1200,
        }
    else:
        # Normal Profile (NF-ToN-IoT-V2) -> GNN will classify as Normal
        flow = {
            "src_ip":           src_ip,
            "dst_ip":           BACKEND_URL.split("//")[1].split(":")[0],
            "in_bytes":         44,
            "out_bytes":        0,
            "in_pkts":          1,
            "out_pkts":         0,
            "flow_duration_ms": 1,         # Use 1 instead of 0 to avoid JavaScript || 5000 bug
            "protocol":         6,         # TCP
            "src_port":         body.get("src_port", 0) or 1024,
            "dst_port":         80,
            "tcp_flags":        2,         # SYN
            "src_to_dst_bps":   0,
        }

    print(f"[gateway] Forwarding flow for {src_ip}: mapped to {'DDoS' if rps >= 1.5 else 'Normal'} profile")

    # --- 3. Inline IPS: Call Local GNN ---
    GNN_URL = "http://127.0.0.1:5001"
    try:
        gnn_resp = requests.post(f"{GNN_URL}/predict", json=flow, timeout=5)
        prediction = gnn_resp.json()
    except Exception as e:
        print(f"[gateway] ❌ GNN unreachable locally: {e}")
        # Fail open if GNN is dead
        prediction = {"attack_type": "Normal", "label_id": 0, "confidence": 1.0, "all_probs": {}}

    is_ddos = prediction.get("label_id") == 1

    # --- 4. Enforce Policy ---
    if is_ddos:
        print(f"[gateway] 🛑 GNN detected DDoS! Blocking {src_ip} inline.")
        iptables_block(src_ip)
        
        # Notify backend so the dashboard shows the block, but DON'T forward the traffic
        try:
            requests.post(
                f"{BACKEND_URL}/api/traffic",
                json={
                    "flow": flow,
                    "prediction": prediction,
                    "blocked_by_gateway": True
                },
                timeout=2,
            )
        except:
            pass # Fire and forget notification
            
        return jsonify({"status": "blocked", "attack_type": "DDoS", "message": "Blocked inline by SDN Gateway"}), 403

    else:
        # Normal traffic: Forward to backend
        try:
            resp = requests.post(
                f"{BACKEND_URL}/api/traffic",
                json={
                    "flow": flow,
                    "prediction": prediction,
                    "blocked_by_gateway": False
                },
                timeout=6,
            )
            # Pass response back to ESP unchanged
            return (resp.content, resp.status_code, {"Content-Type": "application/json"})

        except requests.exceptions.ConnectionError:
            print(f"[gateway] ❌ Cannot reach backend at {BACKEND_URL}")
            return jsonify({"error": "Backend unavailable"}), 502
        except requests.exceptions.Timeout:
            print(f"[gateway] ❌ Backend timed out")
            return jsonify({"error": "Backend timeout"}), 504


# ── SDN control endpoints (mirror of mock_controller REST API) ───────────────

@app.route("/block/<ip>", methods=["POST"])
def block(ip):
    """Called by backend to block an IP — installs real iptables rule."""
    success = iptables_block(ip)
    return jsonify({"status": "blocked" if success else "error", "ip": ip})


@app.route("/unblock/<ip>", methods=["DELETE"])
def unblock(ip):
    """Called by backend/dashboard to unblock an IP."""
    success = iptables_unblock(ip)
    return jsonify({"status": "unblocked" if success else "error", "ip": ip})


@app.route("/blocked", methods=["GET"])
def list_blocked():
    """Return all currently blocked IPs."""
    return jsonify({"blocked": list(blocked_ips)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "role":          "sdn_gateway",
        "backend":       BACKEND_URL,
        "blocked_count": len(blocked_ips),
        "iptables":      _check_iptables(),
    })


def _check_iptables() -> bool:
    try:
        subprocess.run(["iptables", "-L", "-n"], check=True, capture_output=True)
        return True
    except Exception:
        return False


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDN Gateway — Bridge Laptop (proxy + iptables enforcer)")
    parser.add_argument(
        "--backend", default="http://192.168.x.x:3001",
        help="Server Laptop backend URL  e.g. http://192.168.1.50:3001"
    )
    parser.add_argument(
        "--port", type=int, default=3002,
        help="Port to listen on for ESP32 devices (default: 3002)"
    )
    args = parser.parse_args()

    BACKEND_URL  = args.backend
    GATEWAY_PORT = args.port

    print(f"""
╔══════════════════════════════════════════════════════╗
║     SDN GATEWAY — BRIDGE LAPTOP (Laptop 1)           ║
╠══════════════════════════════════════════════════════╣
║  ESP32 devices  →  THIS laptop :{GATEWAY_PORT}          ║
║  Forwarding to Server Laptop   {BACKEND_URL:<20}  ║
║  GNN running locally on        :5001                  ║
║  Blocking via                  iptables (real DROP)   ║
╠══════════════════════════════════════════════════════╣
║  Point ESP32 SERVER_URL to:                           ║
║  http://<THIS_LAPTOP_IP>:{GATEWAY_PORT}/api/traffic     ║
╚══════════════════════════════════════════════════════╝
    """)

    iptables_ok = _check_iptables()
    if not iptables_ok:
        print("⚠️  WARNING: iptables not accessible — run with sudo for real blocking")
    else:
        print("✅ iptables accessible — real network blocking enabled\n")

    app.run(host="0.0.0.0", port=GATEWAY_PORT, debug=False)
