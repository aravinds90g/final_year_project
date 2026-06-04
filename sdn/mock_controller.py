"""
mock_controller.py — Lightweight Ryu-compatible OpenFlow REST API
===================================================================
Drop-in replacement for Ryu that works with Python 3.12.
Provides the same REST API endpoints for blocking/unblocking IPs.

Usage:
    python3 mock_controller.py

REST API (port 8080):
    POST   /block/<ip>      → block an IP
    DELETE /unblock/<ip>    → unblock an IP
    GET    /blocked         → list blocked IPs
    GET    /health          → health check
"""

from flask import Flask, jsonify, request
from datetime import datetime
import logging
import json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("mock_sdn_controller")

# In-memory storage
blocked_ips = set()
rules_log = []


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'mock-gnn-sdn-controller',
        'timestamp': datetime.now().isoformat(),
        'blocked_count': len(blocked_ips)
    })


@app.route('/block/<ip>', methods=['POST'])
def block_ip(ip):
    """Block an IP address by adding OpenFlow DROP rule."""
    if not _is_valid_ip(ip):
        return jsonify({'error': f'Invalid IP: {ip}'}), 400
    
    if ip in blocked_ips:
        return jsonify({'status': 'already_blocked', 'ip': ip}), 200
    
    blocked_ips.add(ip)
    rule = {
        'action': 'block',
        'ip': ip,
        'timestamp': datetime.now().isoformat(),
        'reason': request.json.get('reason', 'GNN detection') if request.json else 'GNN detection'
    }
    rules_log.append(rule)
    
    LOG.info(f"🚨 BLOCKED IP: {ip}")
    return jsonify({
        'status': 'blocked',
        'ip': ip,
        'rules_count': len(blocked_ips),
        'timestamp': rule['timestamp']
    }), 200


@app.route('/unblock/<ip>', methods=['DELETE'])
def unblock_ip(ip):
    """Remove OpenFlow DROP rule for an IP."""
    if not _is_valid_ip(ip):
        return jsonify({'error': f'Invalid IP: {ip}'}), 400
    
    if ip not in blocked_ips:
        return jsonify({'status': 'not_blocked', 'ip': ip}), 200
    
    blocked_ips.discard(ip)
    rule = {
        'action': 'unblock',
        'ip': ip,
        'timestamp': datetime.now().isoformat()
    }
    rules_log.append(rule)
    
    LOG.info(f"✅ UNBLOCKED IP: {ip}")
    return jsonify({
        'status': 'unblocked',
        'ip': ip,
        'rules_count': len(blocked_ips),
        'timestamp': rule['timestamp']
    }), 200


@app.route('/blocked', methods=['GET'])
def list_blocked():
    """List all blocked IPs."""
    return jsonify({
        'count': len(blocked_ips),
        'ips': sorted(list(blocked_ips)),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/stats/flows', methods=['GET'])
def flow_stats():
    """Return flow statistics (mock)."""
    return jsonify({
        'total_rules': len(rules_log),
        'blocked_ips': len(blocked_ips),
        'rules': rules_log[-10:] if rules_log else [],
        'timestamp': datetime.now().isoformat()
    }), 200


def _is_valid_ip(ip: str) -> bool:
    """Basic IP validation."""
    parts = ip.split('.')
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(part) <= 255 for part in parts)
    except ValueError:
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("🛡️  Mock SDN OpenFlow Controller (Ryu Alternative)")
    print("=" * 60)
    print(f"Starting on http://127.0.0.1:8080")
    print(f"Use with backend on port 3001")
    print()
    app.run(host='0.0.0.0', port=8080, debug=False)
