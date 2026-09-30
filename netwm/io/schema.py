"""Canonical flow schema shared by every loader (contract C1).

Every data source (CICFlowMeter CSV, NFStream, our own PCAP extractor) is converted to these
columns so that the rest of the pipeline never needs to know where the flows came from.
Columns that a source cannot provide are filled with NaN / <NA>.
"""
import numpy as np
import pandas as pd

# column -> dtype
CANONICAL = {
    "ts": "datetime64[ns, UTC]",   # flow start time (UTC)
    "src_ip": "string",
    "dst_ip": "string",
    "src_port": "float32",         # float so that missing values can be NaN
    "dst_port": "float32",
    "proto": "float32",            # 6 = TCP, 17 = UDP, 1 = ICMP
    "duration_s": "float32",
    "fwd_pkts": "float32",
    "bwd_pkts": "float32",
    "fwd_bytes": "float32",        # payload bytes sent by the initiator
    "bwd_bytes": "float32",
    "flow_iat_mean": "float32",    # seconds
    "fwd_pkt_len_mean": "float32",
    "bwd_pkt_len_mean": "float32",
    "pkt_len_std": "float32",
    "syn_cnt": "float32",
    "ack_cnt": "float32",
    "fin_cnt": "float32",
    "rst_cnt": "float32",
    "psh_cnt": "float32",
    "urg_cnt": "float32",
    "init_fwd_win": "float32",
    "label_raw": "string",
    "day": "string",
}

# Optional packet-level per-flow accumulators (only produced by the PCAP extractor).
PACKET_COLUMNS = [
    "fwd_ttl_sum", "fwd_ttl_sq", "bwd_ttl_sum", "bwd_ttl_sq",
    "fwd_ttl_distinct", "bwd_ttl_distinct",
    "frag_pkts", "retrans_pkts",
    "plen_b0", "plen_b1", "plen_b2", "plen_b3", "plen_b4",   # payload-size histogram bins
    "win_sum", "win_sq", "win_n",
]

# CICFlowMeter (CSE-CIC-IDS2018 "Processed Traffic Data for ML Algorithms")
CIC2018_MAP = {
    "Timestamp": "ts",
    "Src IP": "src_ip",
    "Dst IP": "dst_ip",
    "Src Port": "src_port",
    "Dst Port": "dst_port",
    "Protocol": "proto",
    "Flow Duration": "duration_s",          # microseconds -> converted
    "Tot Fwd Pkts": "fwd_pkts",
    "Tot Bwd Pkts": "bwd_pkts",
    "TotLen Fwd Pkts": "fwd_bytes",
    "TotLen Bwd Pkts": "bwd_bytes",
    "Flow IAT Mean": "flow_iat_mean",       # microseconds -> converted
    "Fwd Pkt Len Mean": "fwd_pkt_len_mean",
    "Bwd Pkt Len Mean": "bwd_pkt_len_mean",
    "Pkt Len Std": "pkt_len_std",
    "SYN Flag Cnt": "syn_cnt",
    "ACK Flag Cnt": "ack_cnt",
    "FIN Flag Cnt": "fin_cnt",
    "RST Flag Cnt": "rst_cnt",
    "PSH Flag Cnt": "psh_cnt",
    "URG Flag Cnt": "urg_cnt",
    "Init Fwd Win Byts": "init_fwd_win",
    "Label": "label_raw",
}

# CICFlowMeter (CIC-IDS2017 "TrafficLabelling" CSVs); header names carry leading spaces,
# which the loader strips before mapping.
CIC2017_MAP = {
    "Timestamp": "ts",
    "Source IP": "src_ip",
    "Destination IP": "dst_ip",
    "Source Port": "src_port",
    "Destination Port": "dst_port",
    "Protocol": "proto",
    "Flow Duration": "duration_s",
    "Total Fwd Packets": "fwd_pkts",
    "Total Backward Packets": "bwd_pkts",
    "Total Length of Fwd Packets": "fwd_bytes",
    "Total Length of Bwd Packets": "bwd_bytes",
    "Flow IAT Mean": "flow_iat_mean",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "SYN Flag Count": "syn_cnt",
    "ACK Flag Count": "ack_cnt",
    "FIN Flag Count": "fin_cnt",
    "RST Flag Count": "rst_cnt",
    "PSH Flag Count": "psh_cnt",
    "URG Flag Count": "urg_cnt",
    "Init_Win_bytes_forward": "init_fwd_win",
    "Label": "label_raw",
}

MICROSECOND_COLUMNS = ["duration_s", "flow_iat_mean"]


def empty_frame() -> pd.DataFrame:
    return conform(pd.DataFrame({c: [] for c in CANONICAL}))


def conform(df: pd.DataFrame, keep_extra: bool = True) -> pd.DataFrame:
    """Add missing canonical columns, cast dtypes, order columns (extras kept at the end)."""
    df = df.copy()
    for col, dtype in CANONICAL.items():
        if col not in df.columns:
            df[col] = pd.NA if dtype == "string" else np.nan
        if col == "ts":
            if not isinstance(df[col].dtype, pd.DatetimeTZDtype):
                df[col] = pd.to_datetime(df[col], utc=True)
            df[col] = df[col].astype("datetime64[ns, UTC]")
        else:
            df[col] = df[col].astype(dtype)
    extras = [c for c in df.columns if c not in CANONICAL] if keep_extra else []
    return df[list(CANONICAL) + extras]


def has_ips(df: pd.DataFrame) -> bool:
    return bool(df["src_ip"].notna().any() and df["dst_ip"].notna().any())
