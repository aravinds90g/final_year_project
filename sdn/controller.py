"""
controller.py — Ryu SDN OpenFlow Controller
=============================================
Runs as a Ryu application on port 8080.

REST API:
    POST   /block/<ip>      → install DROP rule for IP
    DELETE /unblock/<ip>    → remove DROP rule for IP
    GET    /blocked         → list of currently blocked IPs
    GET    /stats/flows     → all active flow rules

Usage:
    ryu-manager controller.py --observe-links

OpenFlow actions per attack type (set by backend):
    DDoS             → DROP from src_ip
    Data_Exfiltration→ DROP to dst_ip
    Botnet           → DROP from src_ip (isolate)
    Port_Scan        → ALERT only  (no OpenFlow rule)
"""

import json
import logging
from ryu.base         import app_manager
from ryu.controller   import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto      import ofproto_v1_3
from ryu.lib.packet   import packet, ethernet, ipv4
from ryu.app.wsgi     import ControllerBase, WSGIApplication, route
from webob            import Response

LOG = logging.getLogger("ryu.app.gnn_sdn_controller")

# Simple in-memory store for blocked IPs and MAC table
blocked_ips: set = set()    # set of blocked IP strings
mac_table:   dict = {}      # { (dpid, mac): port }
datapaths:   dict = {}      # { dpid: datapath }

APP_NAME = "gnn_sdn_controller"


class GNNSDNController(app_manager.RyuApp):
    """Main Ryu application — handles OpenFlow events."""

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS    = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wsgi = kwargs["wsgi"]
        wsgi.register(SDNRestAPI, {APP_NAME: self})
        LOG.info("[SDN] Controller started. REST API available on port 8080.")

    # ── OpenFlow: Switch handshake ─────────────────────────────────────
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath   = ev.msg.datapath
        ofproto    = datapath.ofproto
        parser     = datapath.ofproto_parser

        datapaths[datapath.id] = datapath
        LOG.info(f"[SDN] Switch connected: dpid={datapath.id}")

        # Install table-miss: send unknown packets to controller
        match  = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions)

    # ── OpenFlow: Packet-in handler ────────────────────────────────────
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg      = ev.msg
        datapath = msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        in_port  = msg.match["in_port"]

        pkt  = packet.Packet(msg.data)
        eth  = pkt.get_protocol(ethernet.ethernet)
        if eth is None:
            return

        dst_mac = eth.dst
        src_mac = eth.src
        dpid    = datapath.id

        # Learn MAC → port mapping
        mac_table.setdefault(dpid, {})
        mac_table[dpid][src_mac] = in_port

        # Check if this packet's IP is blocked
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt and ip_pkt.src in blocked_ips:
            LOG.debug(f"[SDN] Packet from blocked IP {ip_pkt.src} — DROP")
            return   # silently drop

        # Forward normally
        out_port = mac_table[dpid].get(dst_mac, ofproto.OFPP_FLOOD)
        actions  = [parser.OFPActionOutput(out_port)]

        # Install forwarding flow (to reduce controller load)
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac, eth_src=src_mac)
            self._add_flow(datapath, 1, match, actions)

        # Send packet out
        data = None if msg.buffer_id == ofproto.OFP_NO_BUFFER else msg.data
        out  = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data
        )
        datapath.send_msg(out)

    # ── Helpers ──────────────────────────────────────────────────────────
    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0):
        """Install an OpenFlow flow rule."""
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions
        )]
        mod = parser.OFPFlowMod(
            datapath     = datapath,
            priority     = priority,
            match        = match,
            instructions = inst,
            idle_timeout = idle_timeout,
            hard_timeout = hard_timeout,
        )
        datapath.send_msg(mod)

    def block_ip(self, ip: str) -> bool:
        """Install DROP rule for src_ip on all connected switches."""
        if not datapaths:
            LOG.warning("[SDN] No switches connected — cannot install block rule")
            blocked_ips.add(ip)   # still track it
            return False

        blocked_ips.add(ip)

        for dpid, datapath in datapaths.items():
            parser  = datapath.ofproto_parser
            ofproto = datapath.ofproto
            match   = parser.OFPMatch(eth_type=0x0800, ipv4_src=ip)
            # Empty actions = DROP
            self._add_flow(datapath, priority=100, match=match, actions=[],
                           hard_timeout=300)   # auto-expire after 5 min
            LOG.info(f"[SDN] DROP rule installed: src={ip} on switch dpid={dpid}")

        return True

    def unblock_ip(self, ip: str) -> bool:
        """Remove DROP rule for src_ip from all connected switches."""
        blocked_ips.discard(ip)

        for dpid, datapath in datapaths.items():
            parser  = datapath.ofproto_parser
            ofproto = datapath.ofproto

            # Send flow-mod DELETE
            match = parser.OFPMatch(eth_type=0x0800, ipv4_src=ip)
            mod   = parser.OFPFlowMod(
                datapath  = datapath,
                command   = ofproto.OFPFC_DELETE,
                out_port  = ofproto.OFPP_ANY,
                out_group = ofproto.OFPG_ANY,
                match     = match,
            )
            datapath.send_msg(mod)
            LOG.info(f"[SDN] DROP rule removed: src={ip} on switch dpid={dpid}")

        return True


# ── REST API Controller ────────────────────────────────────────────────────

class SDNRestAPI(ControllerBase):
    """REST endpoints exposed by the Ryu WSGI server."""

    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.sdn_app: GNNSDNController = data[APP_NAME]

    @route(APP_NAME, "/block/{ip}", methods=["POST"])
    def block(self, req, ip, **kwargs):
        success = self.sdn_app.block_ip(ip)
        body    = json.dumps({
            "status":  "blocked" if success else "queued",
            "ip":      ip,
            "message": "DROP rule installed" if success else "No switches connected yet",
        })
        return Response(content_type="application/json", body=body)

    @route(APP_NAME, "/unblock/{ip}", methods=["DELETE"])
    def unblock(self, req, ip, **kwargs):
        success = self.sdn_app.unblock_ip(ip)
        body    = json.dumps({
            "status": "unblocked",
            "ip":     ip,
        })
        return Response(content_type="application/json", body=body)

    @route(APP_NAME, "/blocked", methods=["GET"])
    def get_blocked(self, req, **kwargs):
        body = json.dumps({
            "blocked_ips": list(blocked_ips),
            "count":       len(blocked_ips),
        })
        return Response(content_type="application/json", body=body)

    @route(APP_NAME, "/health", methods=["GET"])
    def health(self, req, **kwargs):
        body = json.dumps({
            "status":           "ok",
            "switches_connected": len(datapaths),
            "blocked_count":    len(blocked_ips),
        })
        return Response(content_type="application/json", body=body)
