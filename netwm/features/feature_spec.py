"""Ordered feature lists (the model input contract) and per-feature transforms."""

# ---- host mode (one state per internal host per window) --------------------------------------
HOST_A = ["n_flows_out", "n_flows_in", "bytes_sent", "bytes_recv", "pkts_sent", "pkts_recv",
          "active"]
HOST_B = ["uniq_dst_ips_out", "uniq_src_ips_in", "uniq_dst_ports_out", "uniq_dst_ports_in",
          "dst_port_entropy_out", "dst_ip_entropy_out", "internal_peer_ratio"]
HOST_C = ["syn_flow_ratio", "half_open_ratio", "rst_ratio", "fin_ratio", "psh_ratio",
          "short_flow_ratio", "tcp_ratio", "udp_ratio", "icmp_ratio"]
HOST_D = ["flow_dur_mean", "flow_dur_std", "flow_iat_mean", "flow_start_gap_mean",
          "flow_start_gap_cv"]
HOST_E = ["fwd_pkt_len_mean", "bwd_pkt_len_mean", "pkt_len_std_mean", "bytes_ratio",
          "init_fwd_win_mean"]
SERVICE = ["svc_web", "svc_ssh", "svc_ftp", "svc_smb", "svc_rdp", "svc_dns", "svc_other_low",
           "svc_high"]
HOST_G = ["out_degree_internal", "new_peers_out", "new_ports_out", "in_degree_external",
          "nbr_mean_fanout"]
PACKET_H = ["ttl_mean_in", "ttl_std_in", "n_distinct_ttl_in", "ip_frag_ratio",
            "tcp_retrans_ratio", "payload_len_entropy", "seq_port_score", "tcp_win_std"]
HOST_I = ["d_n_flows_out", "d_uniq_dst_ports_out", "z_n_flows_out", "z_uniq_dst_ips_out",
          "z_bytes_sent"]
GLOBAL_J = ["g_total_flows", "g_active_hosts", "g_syn_ratio", "g_half_open_ratio",
            "g_dst_port_entropy", "g_n_external_src", "g_internal_to_internal_ratio",
            "g_total_bytes"]

# temporal block definitions: (output name, source column)
HOST_DELTAS = [("d_n_flows_out", "n_flows_out"), ("d_uniq_dst_ports_out", "uniq_dst_ports_out")]
HOST_ZSCORES = [("z_n_flows_out", "n_flows_out"), ("z_uniq_dst_ips_out", "uniq_dst_ips_out"),
                ("z_bytes_sent", "bytes_sent")]


def host_features(with_packet: bool) -> list[str]:
    base = HOST_A + HOST_B + HOST_C + HOST_D + HOST_E + SERVICE + HOST_G
    return base + (PACKET_H if with_packet else []) + HOST_I + GLOBAL_J


# ---- network mode (one state per window; works without IP addresses) --------------------------
NET_BASE = ["n_flows", "fwd_bytes", "bwd_bytes", "fwd_pkts", "bwd_pkts", "uniq_dst_ports",
            "dst_port_entropy", "syn_flow_ratio", "half_open_ratio", "rst_ratio", "fin_ratio",
            "psh_ratio", "short_flow_ratio", "tcp_ratio", "udp_ratio", "icmp_ratio",
            "flow_dur_mean", "flow_dur_std", "flow_iat_mean", "fwd_pkt_len_mean",
            "bwd_pkt_len_mean", "pkt_len_std_mean", "bytes_ratio", "init_fwd_win_mean"]
NET_NOVELTY = ["new_ports"]
NET_I = ["d_n_flows", "d_uniq_dst_ports", "z_n_flows", "z_uniq_dst_ports", "z_fwd_bytes"]
NET_DELTAS = [("d_n_flows", "n_flows"), ("d_uniq_dst_ports", "uniq_dst_ports")]
NET_ZSCORES = [("z_n_flows", "n_flows"), ("z_uniq_dst_ports", "uniq_dst_ports"),
               ("z_fwd_bytes", "fwd_bytes")]


def network_features() -> list[str]:
    return NET_BASE + SERVICE + NET_NOVELTY + NET_I


# ---- transforms --------------------------------------------------------------------------------
# heavy-tailed non-negative features get log1p before robust scaling
LOG1P = set(HOST_A + ["uniq_dst_ips_out", "uniq_src_ips_in", "uniq_dst_ports_out",
                      "uniq_dst_ports_in", "flow_dur_mean", "flow_dur_std", "flow_iat_mean",
                      "flow_start_gap_mean", "fwd_pkt_len_mean", "bwd_pkt_len_mean",
                      "pkt_len_std_mean", "init_fwd_win_mean", "out_degree_internal",
                      "new_peers_out", "new_ports_out", "in_degree_external", "nbr_mean_fanout",
                      "n_distinct_ttl_in", "ttl_std_in", "tcp_win_std", "g_total_flows",
                      "g_active_hosts", "g_n_external_src", "g_total_bytes",
                      "n_flows", "fwd_bytes", "bwd_bytes", "fwd_pkts", "bwd_pkts",
                      "uniq_dst_ports", "new_ports"])
LOG1P.discard("active")
# features that are already bounded and informative in absolute terms: no centring/scaling
BOUNDED = set(HOST_C + SERVICE + ["active", "internal_peer_ratio", "ip_frag_ratio",
                                  "tcp_retrans_ratio", "seq_port_score", "g_syn_ratio",
                                  "g_half_open_ratio", "g_internal_to_internal_ratio"])

# human-readable descriptions (used by the evidence templates and the dashboard)
DESCRIPTIONS = {
    "n_flows_out": "connections started by the host",
    "n_flows_in": "connections received by the host",
    "bytes_sent": "bytes sent by the host",
    "bytes_recv": "bytes received by the host",
    "uniq_dst_ips_out": "distinct hosts contacted",
    "uniq_src_ips_in": "distinct hosts connecting to it",
    "uniq_dst_ports_out": "distinct destination ports contacted",
    "uniq_dst_ports_in": "distinct local ports contacted by others",
    "dst_port_entropy_out": "destination-port entropy (spread of ports contacted)",
    "dst_ip_entropy_out": "destination-host entropy",
    "internal_peer_ratio": "share of traffic with internal hosts",
    "syn_flow_ratio": "share of connections with SYN",
    "half_open_ratio": "share of SYN connections that got no reply",
    "rst_ratio": "share of connections reset (RST)",
    "fin_ratio": "share of connections closed normally (FIN)",
    "psh_ratio": "share of connections carrying data (PSH)",
    "short_flow_ratio": "share of very short connections (<=3 packets)",
    "tcp_ratio": "TCP share", "udp_ratio": "UDP share", "icmp_ratio": "ICMP share",
    "flow_dur_mean": "mean connection duration (s)",
    "flow_dur_std": "variation of connection duration",
    "flow_iat_mean": "mean packet inter-arrival time (s)",
    "flow_start_gap_mean": "mean gap between new connections (s)",
    "flow_start_gap_cv": "irregularity of connection timing (low = machine-like/beaconing)",
    "fwd_pkt_len_mean": "mean payload size sent by initiators",
    "bwd_pkt_len_mean": "mean payload size of responses",
    "pkt_len_std_mean": "payload-size variation",
    "bytes_ratio": "upload/download balance (log ratio)",
    "init_fwd_win_mean": "initial TCP window size",
    "svc_web": "share of web (80/443/8080) connections",
    "svc_ssh": "share of SSH (22) connections",
    "svc_ftp": "share of FTP (20/21) connections",
    "svc_smb": "share of SMB/NetBIOS (139/445) connections",
    "svc_rdp": "share of RDP (3389) connections",
    "svc_dns": "share of DNS (53) connections",
    "svc_other_low": "share of other well-known ports (<1024)",
    "svc_high": "share of high ports (>=1024)",
    "out_degree_internal": "internal hosts contacted",
    "new_peers_out": "hosts contacted that were not contacted in the previous 30 min",
    "new_ports_out": "ports used that were not used in the previous 30 min",
    "in_degree_external": "external hosts connecting to it",
    "nbr_mean_fanout": "port fan-out of the hosts it talks to",
    "ttl_mean_in": "mean TTL of received packets",
    "ttl_std_in": "TTL variation of received packets",
    "n_distinct_ttl_in": "distinct TTL values received",
    "ip_frag_ratio": "share of fragmented IP packets",
    "tcp_retrans_ratio": "share of TCP retransmissions",
    "payload_len_entropy": "payload-size diversity",
    "seq_port_score": "sequential port probing score",
    "tcp_win_std": "TCP window-size variation",
    "d_n_flows_out": "change in connections started vs previous minute",
    "d_uniq_dst_ports_out": "change in distinct ports contacted vs previous minute",
    "z_n_flows_out": "connections started vs the host's own baseline (z-score)",
    "z_uniq_dst_ips_out": "distinct hosts contacted vs own baseline (z-score)",
    "z_bytes_sent": "bytes sent vs own baseline (z-score)",
    "g_total_flows": "network-wide connections",
    "g_active_hosts": "active internal hosts",
    "g_syn_ratio": "network-wide SYN share",
    "g_half_open_ratio": "network-wide unanswered SYN share",
    "g_dst_port_entropy": "network-wide destination-port entropy",
    "g_n_external_src": "distinct external sources",
    "g_internal_to_internal_ratio": "share of internal-to-internal connections",
    "g_total_bytes": "network-wide bytes",
    "n_flows": "connections in the network",
    "fwd_bytes": "bytes sent by initiators", "bwd_bytes": "bytes sent by responders",
    "fwd_pkts": "packets sent by initiators", "bwd_pkts": "packets sent by responders",
    "uniq_dst_ports": "distinct destination ports in use",
    "dst_port_entropy": "destination-port entropy",
    "new_ports": "destination ports not seen in the previous 30 min",
    "d_n_flows": "change in connections vs previous minute",
    "d_uniq_dst_ports": "change in distinct ports vs previous minute",
    "z_n_flows": "connections vs recent baseline (z-score)",
    "z_uniq_dst_ports": "distinct ports vs recent baseline (z-score)",
    "z_fwd_bytes": "initiator bytes vs recent baseline (z-score)",
}
