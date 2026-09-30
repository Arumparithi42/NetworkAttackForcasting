"""Canonical flows -> per-window states.

Two entity modes:
  * host    : one state per (window, internal host)   - needs IP addresses
  * network : one state per window                     - works on CSVs without IP columns

Windows are absolute: window = floor(unix_seconds / window_s), so features and labels computed
by different modules always line up.
"""
from __future__ import annotations

import ipaddress
from functools import lru_cache

import numpy as np
import pandas as pd

from netwm.features import feature_spec as fs

WEB = {80, 443, 8080, 8443}
SSH = {22}
FTP = {20, 21}
SMB = {137, 138, 139, 445}
RDP = {3389}
DNS = {53}


def add_window(flows: pd.DataFrame, window_s: int) -> pd.DataFrame:
    sec = flows["ts"].astype("int64") // 10**9
    return flows.assign(window=(sec // window_s).astype(np.int64))


def make_internal_checker(cidrs):
    nets = [ipaddress.ip_network(c) for c in cidrs]

    @lru_cache(maxsize=None)
    def is_internal(ip) -> bool:
        try:
            addr = ipaddress.ip_address(str(ip))
        except ValueError:
            return False
        return any(addr in n for n in nets)

    return is_internal


def internal_set(flows: pd.DataFrame, cidrs) -> set:
    chk = make_internal_checker(cidrs)
    ips = pd.unique(pd.concat([flows["src_ip"].dropna(), flows["dst_ip"].dropna()]).astype(str))
    return {ip for ip in ips if chk(ip)}


def _service_columns(dst_port: pd.Series) -> dict:
    p = dst_port.fillna(-1).astype(np.int64)
    web, ssh, ftp = p.isin(WEB), p.isin(SSH), p.isin(FTP)
    smb, rdp, dns = p.isin(SMB), p.isin(RDP), p.isin(DNS)
    known = web | ssh | ftp | smb | rdp | dns
    return {
        "svc_web": web, "svc_ssh": ssh, "svc_ftp": ftp, "svc_smb": smb, "svc_rdp": rdp,
        "svc_dns": dns, "svc_other_low": (~known) & (p >= 0) & (p < 1024),
        "svc_high": (~known) & (p >= 1024),
    }


def _flow_flags(df: pd.DataFrame) -> pd.DataFrame:
    total_pkts = df["fwd_pkts"].fillna(0) + df["bwd_pkts"].fillna(0)
    cols = {
        "syn_flow": df["syn_cnt"] > 0,
        "half_open": (df["syn_cnt"] > 0) & (df["bwd_pkts"] == 0),
        "rst_flow": df["rst_cnt"] > 0,
        "fin_flow": df["fin_cnt"] > 0,
        "psh_flow": df["psh_cnt"] > 0,
        "short_flow": total_pkts <= 3,
        "is_tcp": df["proto"] == 6,
        "is_udp": df["proto"] == 17,
        "is_icmp": df["proto"] == 1,
    }
    cols.update(_service_columns(df["dst_port"]))
    return df.assign(**{k: v.astype(np.float32) for k, v in cols.items()})


def _grouped_entropy(df: pd.DataFrame, keys: list[str], col: str) -> pd.Series:
    """Shannon entropy (bits) of `col` values within each group, fully vectorised."""
    c = df.groupby(keys + [col], observed=True).size().rename("c").reset_index()
    c["tot"] = c.groupby(keys, observed=True)["c"].transform("sum")
    p = c["c"] / c["tot"]
    c["h"] = -p * np.log2(p)
    return c.groupby(keys, observed=True)["h"].sum()


def _novelty(pairs: pd.DataFrame, keys: list[str], item: str, lookback: int) -> pd.Series:
    """Count, per (window, *keys), distinct `item`s not seen for this key in the previous
    `lookback` windows (strictly causal)."""
    p = pairs[["window"] + keys + [item]].drop_duplicates().sort_values("window")
    prev = p.groupby(keys + [item], observed=True)["window"].shift()
    p = p.assign(new=(prev.isna() | (p["window"] - prev > lookback)).astype(np.float32))
    return p.groupby(["window"] + keys, observed=True)["new"].sum()


# ------------------------------------------------------------------------------------------------
# host mode
# ------------------------------------------------------------------------------------------------
def host_view(flows: pd.DataFrame, internal: set) -> pd.DataFrame:
    """One row per (flow, internal endpoint). If the flows come from per-host captures
    (column `capture_host`), each flow is only used from the capturing host's perspective, which
    avoids double counting internal-to-internal flows seen in two captures (and halves memory)."""
    base = _flow_flags(flows)
    if "capture_host" in base.columns:
        cap = base["capture_host"].astype(str).to_numpy()
        is_out = base["src_ip"].astype(str).to_numpy() == cap
        is_in = ~is_out & (base["dst_ip"].astype(str).to_numpy() == cap)
        base = base[is_out | is_in]
        o = is_out[is_out | is_in]
        v = base.assign(
            host=base["capture_host"], peer=base["dst_ip"].where(o, base["src_ip"]),
            is_out=o.astype(np.float32),
            bytes_sent=np.where(o, base["fwd_bytes"], base["bwd_bytes"]).astype(np.float32),
            bytes_recv=np.where(o, base["bwd_bytes"], base["fwd_bytes"]).astype(np.float32),
            pkts_sent=np.where(o, base["fwd_pkts"], base["bwd_pkts"]).astype(np.float32),
            pkts_recv=np.where(o, base["bwd_pkts"], base["fwd_pkts"]).astype(np.float32))
        if "fwd_ttl_sum" in base.columns:
            v = v.assign(ttl_in_sum=np.where(o, base["bwd_ttl_sum"], base["fwd_ttl_sum"]).astype(np.float32),
                         ttl_in_sq=np.where(o, base["bwd_ttl_sq"], base["fwd_ttl_sq"]).astype(np.float32),
                         ttl_in_n=np.where(o, base["bwd_pkts"], base["fwd_pkts"]).astype(np.float32),
                         ttl_in_distinct=np.where(o, base["bwd_ttl_distinct"], base["fwd_ttl_distinct"]).astype(np.float32))
        v = v[v["host"].isin(internal)]
    else:
        out = base.assign(host=base["src_ip"], peer=base["dst_ip"], is_out=np.float32(1),
                          bytes_sent=base["fwd_bytes"], bytes_recv=base["bwd_bytes"],
                          pkts_sent=base["fwd_pkts"], pkts_recv=base["bwd_pkts"])
        inn = base.assign(host=base["dst_ip"], peer=base["src_ip"], is_out=np.float32(0),
                          bytes_sent=base["bwd_bytes"], bytes_recv=base["fwd_bytes"],
                          pkts_sent=base["bwd_pkts"], pkts_recv=base["fwd_pkts"])
        if "fwd_ttl_sum" in base.columns:
            out = out.assign(ttl_in_sum=out["bwd_ttl_sum"], ttl_in_sq=out["bwd_ttl_sq"],
                             ttl_in_n=out["bwd_pkts"], ttl_in_distinct=out["bwd_ttl_distinct"])
            inn = inn.assign(ttl_in_sum=inn["fwd_ttl_sum"], ttl_in_sq=inn["fwd_ttl_sq"],
                             ttl_in_n=inn["fwd_pkts"], ttl_in_distinct=inn["fwd_ttl_distinct"])
        v = pd.concat([out, inn], ignore_index=True)
        v = v[v["host"].isin(internal)]
        v["host"] = v["host"].astype(str)
        v["peer"] = v["peer"].astype(str)
    v = v.assign(peer_internal=v["peer"].isin(internal).astype(np.float32))
    return v.reset_index(drop=True)


def _packet_block(v: pd.DataFrame) -> pd.DataFrame:
    g = v.groupby(["window", "host"], observed=True)
    s = g[["ttl_in_sum", "ttl_in_sq", "ttl_in_n", "frag_pkts", "retrans_pkts", "win_sum",
           "win_sq", "win_n", "plen_b0", "plen_b1", "plen_b2", "plen_b3", "plen_b4",
           "fwd_pkts", "bwd_pkts"]].sum()
    n_in = s["ttl_in_n"].replace(0, np.nan)
    ttl_mean = s["ttl_in_sum"] / n_in
    ttl_var = (s["ttl_in_sq"] / n_in - ttl_mean ** 2).clip(lower=0)
    pk = (s["fwd_pkts"] + s["bwd_pkts"]).replace(0, np.nan)
    wn = s["win_n"].replace(0, np.nan)
    win_mean = s["win_sum"] / wn
    win_var = (s["win_sq"] / wn - win_mean ** 2).clip(lower=0)
    hist = s[["plen_b0", "plen_b1", "plen_b2", "plen_b3", "plen_b4"]].to_numpy(dtype=np.float64)
    tot = hist.sum(1, keepdims=True)
    p = np.divide(hist, tot, out=np.zeros_like(hist), where=tot > 0)
    ent = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1)
    out = pd.DataFrame({
        "ttl_mean_in": ttl_mean, "ttl_std_in": np.sqrt(ttl_var),
        "n_distinct_ttl_in": g["ttl_in_distinct"].max(),
        "ip_frag_ratio": s["frag_pkts"] / pk, "tcp_retrans_ratio": s["retrans_pkts"] / pk,
        "payload_len_entropy": ent, "tcp_win_std": np.sqrt(win_var),
    }, index=s.index)
    # sequential port probing: outbound SYN flows ordered by time, share of +1 port steps
    syn = v[(v["is_out"] == 1) & (v["syn_flow"] == 1)].sort_values(["host", "ts"])
    if len(syn):
        step = syn.groupby(["window", "host"], observed=True)["dst_port"].diff()
        seq = (step == 1).astype(np.float32).where(step.notna())
        out["seq_port_score"] = seq.groupby([syn["window"], syn["host"]], observed=True).mean()
    else:
        out["seq_port_score"] = 0.0
    return out


def host_window_features(flows: pd.DataFrame, cidrs, window_s: int = 60,
                         novelty_lookback: int = 30) -> pd.DataFrame:
    """Blocks A-G (+H if packet accumulators exist) + J, one row per active (window, host)."""
    if "window" not in flows.columns:
        flows = add_window(flows, window_s)
    internal = internal_set(flows, cidrs)
    v = host_view(flows, internal)
    v["peer_out"] = v["peer"].where(v["is_out"] == 1)
    v["peer_in"] = v["peer"].where(v["is_out"] == 0)
    v["dport_out"] = v["dst_port"].where(v["is_out"] == 1)
    v["dport_in"] = v["dst_port"].where(v["is_out"] == 0)
    v["int_peer_out"] = v["peer"].where((v["is_out"] == 1) & (v["peer_internal"] == 1))
    v["ext_peer_in"] = v["peer"].where((v["is_out"] == 0) & (v["peer_internal"] == 0))

    keys = ["window", "host"]
    g = v.groupby(keys, observed=True)
    agg = {
        "n_flows_out": ("is_out", "sum"), "n_rows": ("is_out", "size"),
        "bytes_sent": ("bytes_sent", "sum"), "bytes_recv": ("bytes_recv", "sum"),
        "pkts_sent": ("pkts_sent", "sum"), "pkts_recv": ("pkts_recv", "sum"),
        "uniq_dst_ips_out": ("peer_out", "nunique"), "uniq_src_ips_in": ("peer_in", "nunique"),
        "uniq_dst_ports_out": ("dport_out", "nunique"),
        "uniq_dst_ports_in": ("dport_in", "nunique"),
        "internal_peer_ratio": ("peer_internal", "mean"),
        "syn_flow_ratio": ("syn_flow", "mean"), "half_open_ratio": ("half_open", "mean"),
        "rst_ratio": ("rst_flow", "mean"), "fin_ratio": ("fin_flow", "mean"),
        "psh_ratio": ("psh_flow", "mean"), "short_flow_ratio": ("short_flow", "mean"),
        "tcp_ratio": ("is_tcp", "mean"), "udp_ratio": ("is_udp", "mean"),
        "icmp_ratio": ("is_icmp", "mean"),
        "flow_dur_mean": ("duration_s", "mean"), "flow_dur_std": ("duration_s", "std"),
        "flow_iat_mean": ("flow_iat_mean", "mean"),
        "fwd_pkt_len_mean": ("fwd_pkt_len_mean", "mean"),
        "bwd_pkt_len_mean": ("bwd_pkt_len_mean", "mean"),
        "pkt_len_std_mean": ("pkt_len_std", "mean"),
        "init_fwd_win_mean": ("init_fwd_win", "mean"),
        "out_degree_internal": ("int_peer_out", "nunique"),
        "in_degree_external": ("ext_peer_in", "nunique"),
    }
    for c in fs.SERVICE:
        agg[c] = (c, "mean")
    feats = g.agg(**agg)
    feats["n_flows_in"] = feats.pop("n_rows") - feats["n_flows_out"]
    feats["active"] = 1.0
    feats["bytes_ratio"] = np.log((feats["bytes_sent"] + 1) / (feats["bytes_recv"] + 1))

    out_rows = v[v["is_out"] == 1]
    feats["dst_port_entropy_out"] = _grouped_entropy(out_rows.dropna(subset=["dst_port"]), keys,
                                                     "dst_port")
    feats["dst_ip_entropy_out"] = _grouped_entropy(out_rows, keys, "peer")

    # timing regularity of new outbound connections (beaconing -> low coefficient of variation)
    o = out_rows[["window", "host", "ts"]].sort_values(["host", "ts"])
    gap = o.groupby(keys, observed=True)["ts"].diff().dt.total_seconds()
    gstat = gap.groupby([o["window"], o["host"]], observed=True).agg(["mean", "std"])
    feats["flow_start_gap_mean"] = gstat["mean"]
    feats["flow_start_gap_cv"] = gstat["std"] / gstat["mean"].replace(0, np.nan)

    # graph / novelty block
    feats["new_peers_out"] = _novelty(out_rows, ["host"], "peer", novelty_lookback)
    feats["new_ports_out"] = _novelty(out_rows.dropna(subset=["dst_port"]), ["host"], "dst_port",
                                      novelty_lookback)
    src_fan = flows.groupby(["window", "src_ip"], observed=True)["dst_port"].nunique()
    src_fan.index = src_fan.index.set_names(["window", "peer"])
    pairs = v[["window", "host", "peer"]].drop_duplicates()
    pairs = pairs.join(src_fan.rename("fan"), on=["window", "peer"])
    feats["nbr_mean_fanout"] = pairs.fillna({"fan": 0}).groupby(keys, observed=True)["fan"].mean()

    if "ttl_in_sum" in v.columns:
        feats = feats.join(_packet_block(v))

    glob = _global_block(v, out_rows)
    feats = feats.join(glob, on="window")
    feats = feats.fillna(0.0).astype(np.float32).reset_index()
    feats["host"] = feats["host"].astype(str)
    return feats


def _global_block(v: pd.DataFrame, out_rows: pd.DataFrame) -> pd.DataFrame:
    gw = v.groupby("window")
    ext_in = v[(v["is_out"] == 0) & (v["peer_internal"] == 0)]
    g = pd.DataFrame({
        "g_total_flows": gw.size(),
        "g_active_hosts": gw["host"].nunique(),
        "g_syn_ratio": gw["syn_flow"].mean(),
        "g_half_open_ratio": gw["half_open"].mean(),
        "g_internal_to_internal_ratio": gw["peer_internal"].mean(),
        "g_total_bytes": gw["bytes_sent"].sum() + gw["bytes_recv"].sum(),
    })
    g["g_n_external_src"] = ext_in.groupby("window")["peer"].nunique()
    g["g_dst_port_entropy"] = _grouped_entropy(out_rows.dropna(subset=["dst_port"]), ["window"],
                                               "dst_port")
    return g.fillna(0.0)


# ------------------------------------------------------------------------------------------------
# network mode
# ------------------------------------------------------------------------------------------------
def network_window_features(flows: pd.DataFrame, window_s: int = 60,
                            novelty_lookback: int = 30) -> pd.DataFrame:
    if "window" not in flows.columns:
        flows = add_window(flows, window_s)
    f = _flow_flags(flows)
    g = f.groupby("window")
    agg = {
        "n_flows": ("is_tcp", "size"),
        "fwd_bytes": ("fwd_bytes", "sum"), "bwd_bytes": ("bwd_bytes", "sum"),
        "fwd_pkts": ("fwd_pkts", "sum"), "bwd_pkts": ("bwd_pkts", "sum"),
        "uniq_dst_ports": ("dst_port", "nunique"),
        "syn_flow_ratio": ("syn_flow", "mean"), "half_open_ratio": ("half_open", "mean"),
        "rst_ratio": ("rst_flow", "mean"), "fin_ratio": ("fin_flow", "mean"),
        "psh_ratio": ("psh_flow", "mean"), "short_flow_ratio": ("short_flow", "mean"),
        "tcp_ratio": ("is_tcp", "mean"), "udp_ratio": ("is_udp", "mean"),
        "icmp_ratio": ("is_icmp", "mean"),
        "flow_dur_mean": ("duration_s", "mean"), "flow_dur_std": ("duration_s", "std"),
        "flow_iat_mean": ("flow_iat_mean", "mean"),
        "fwd_pkt_len_mean": ("fwd_pkt_len_mean", "mean"),
        "bwd_pkt_len_mean": ("bwd_pkt_len_mean", "mean"),
        "pkt_len_std_mean": ("pkt_len_std", "mean"),
        "init_fwd_win_mean": ("init_fwd_win", "mean"),
    }
    for c in fs.SERVICE:
        agg[c] = (c, "mean")
    feats = g.agg(**agg)
    feats["bytes_ratio"] = np.log((feats["fwd_bytes"] + 1) / (feats["bwd_bytes"] + 1))
    feats["dst_port_entropy"] = _grouped_entropy(f.dropna(subset=["dst_port"]), ["window"],
                                                 "dst_port")
    tmp = f.dropna(subset=["dst_port"]).assign(_all="net")
    nov = _novelty(tmp, ["_all"], "dst_port", novelty_lookback)
    feats["new_ports"] = nov.droplevel("_all")
    return feats.fillna(0.0).astype(np.float32).reset_index()
