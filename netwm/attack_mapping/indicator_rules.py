"""Observed behaviour -> CANDIDATE ATT&CK techniques.

These are transparent threshold rules on raw (unscaled) window features. They produce
"behaviour consistent with <technique>" statements with an evidence-strength tag; they never
claim that a technique was used. Thresholds are heuristics chosen to be rare in benign traffic;
re-tune them on your own benign validation data (docs/LABEL_MAPPING.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Rule:
    name: str
    technique: str
    strength: str                 # low | medium
    mode: str                     # host | network
    test: Callable[[dict], bool]
    evidence: list[str]
    note: str = ""


def g(f: dict, k: str) -> float:
    return float(f.get(k, 0.0) or 0.0)


RULES = [
    # ---------------- host mode ----------------
    Rule("port_scan_from_host", "T1046", "medium", "host",
         lambda f: g(f, "uniq_dst_ports_out") >= 100 and g(f, "half_open_ratio") + g(f, "rst_ratio") >= 0.3,
         ["uniq_dst_ports_out", "half_open_ratio", "rst_ratio"],
         "many ports probed, many connections unanswered or reset"),
    Rule("host_sweep", "T1018", "low", "host",
         lambda f: g(f, "out_degree_internal") >= 20 and g(f, "new_peers_out") >= 10,
         ["out_degree_internal", "new_peers_out"], "contacting many internal hosts it did not talk to before"),
    Rule("scanned_from_outside", "T1595", "medium", "host",
         lambda f: g(f, "uniq_dst_ports_in") >= 100 and g(f, "in_degree_external") >= 1,
         ["uniq_dst_ports_in", "in_degree_external"], "many local ports probed by external hosts"),
    Rule("password_guessing", "T1110.001", "medium", "host",
         lambda f: g(f, "svc_ssh") + g(f, "svc_ftp") >= 0.5 and g(f, "n_flows_in") + g(f, "n_flows_out") >= 30,
         ["svc_ssh", "svc_ftp", "n_flows_in", "n_flows_out"], "many repeated SSH/FTP sessions"),
    Rule("remote_service_spread", "T1021", "low", "host",
         lambda f: g(f, "svc_smb") + g(f, "svc_rdp") >= 0.3 and g(f, "new_peers_out") >= 5
         and g(f, "internal_peer_ratio") >= 0.5,
         ["svc_smb", "svc_rdp", "new_peers_out"], "SMB/RDP to several new internal hosts (no ground truth)"),
    Rule("beaconing", "T1071", "low", "host",
         lambda f: g(f, "n_flows_out") >= 5 and 0 < g(f, "flow_start_gap_cv") <= 0.2,
         ["flow_start_gap_cv", "n_flows_out"], "very regular outbound connection timing"),
    Rule("flood_target", "T1499", "medium", "host",
         lambda f: g(f, "z_n_flows_out") >= 4 or (g(f, "n_flows_in") >= 500 and g(f, "short_flow_ratio") >= 0.5),
         ["n_flows_in", "short_flow_ratio"], "connection volume far above the host's baseline"),
    Rule("large_outbound_transfer", "T1048", "low", "host",
         lambda f: g(f, "z_bytes_sent") >= 5 and g(f, "bytes_ratio") >= 2,
         ["bytes_sent", "bytes_ratio", "z_bytes_sent"],
         "unusually large upload (heuristic only; exfiltration has no ground truth here)"),
    # ---------------- network mode ----------------
    Rule("network_port_scan", "T1046", "low", "network",
         lambda f: g(f, "uniq_dst_ports") >= 300 and g(f, "z_uniq_dst_ports") >= 3,
         ["uniq_dst_ports", "z_uniq_dst_ports", "half_open_ratio"],
         "sudden rise in distinct destination ports network-wide"),
    Rule("network_password_guessing", "T1110.001", "medium", "network",
         lambda f: g(f, "svc_ssh") + g(f, "svc_ftp") >= 0.3 and g(f, "z_n_flows") >= 2,
         ["svc_ssh", "svc_ftp", "z_n_flows"], "surge of SSH/FTP sessions"),
    Rule("network_flood", "T1498", "medium", "network",
         lambda f: g(f, "z_n_flows") >= 4 and g(f, "short_flow_ratio") >= 0.4,
         ["n_flows", "z_n_flows", "short_flow_ratio"], "connection volume far above baseline"),
    Rule("network_web_surge", "T1190", "low", "network",
         lambda f: g(f, "svc_web") >= 0.6 and g(f, "z_n_flows") >= 3,
         ["svc_web", "z_n_flows"], "surge of web requests (exploitation attempts possible)"),
]


def evaluate_rules(features: dict, mode: str) -> list[dict]:
    hits = []
    for r in RULES:
        if r.mode != mode:
            continue
        try:
            ok = r.test(features)
        except (TypeError, ValueError):
            ok = False
        if ok:
            hits.append({"rule": r.name, "technique": r.technique, "strength": r.strength,
                         "evidence": {k: round(g(features, k), 4) for k in r.evidence},
                         "note": r.note})
    return hits
