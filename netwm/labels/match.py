"""Transfer the published CSV labels onto flows re-extracted from the raw per-host PCAPs.

Problem: the CSE-CIC-IDS2018 daily CSVs have no IP columns and contain only a subset of the
traffic, so labels cannot be joined by 5-tuple. The dataset authors labelled attack traffic by
(victim host, time interval): e.g. "Infilteration" rows are the traffic of the compromised
host(s) during the attack periods.

Procedure (deterministic; statistics are returned and reported in docs/DATA_NOTES.md):
  1. Victim identification with a STRICT key (start second, dst port, fwd packets, bwd packets),
     restricted to destination ports that are NOT among the most common benign ports (DNS, HTTPS,
     RDP... collide across hosts within the same second); flows seen in two captures count once.
     * attack SOURCE = internal source IP with >= source_outlier_factor x the median of the top-10
       source counts (the compromised host that scans/probes others);
     * attack TARGET = internal destination whose matched incoming flows are >= min_target_share
       attack (non-victims sit at the key-collision base rate of ~0.3).
  2. Flow labelling with a LOOSE key (start second, dst port, protocol): a flow gets the attack
     label if a victim is its source or destination and its key occurs only in attack rows.
     This mirrors the dataset's own IP+time labelling, including hosts probed by a victim.
  3. Everything else is benign. Window stages then follow the usual >= min_attack_flows rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _strict_key(df: pd.DataFrame, shift: int = 0) -> np.ndarray:
    s = (df["ts"].astype("int64").to_numpy() // 10**9 + shift) % 86400     # seconds of day: no int64 overflow
    return ((s * 10**6 + df["dst_port"].fillna(0).to_numpy(np.int64)) * 10**4
            + np.clip(df["fwd_pkts"].fillna(0).to_numpy(np.int64), 0, 99) * 100
            + np.clip(df["bwd_pkts"].fillna(0).to_numpy(np.int64), 0, 99))


def _loose_key(df: pd.DataFrame, shift: int = 0) -> np.ndarray:
    s = (df["ts"].astype("int64").to_numpy() // 10**9 + shift) % 86400     # seconds of day: no int64 overflow
    return (s * 10**6 + df["dst_port"].fillna(0).to_numpy(np.int64)) * 10 \
        + (df["proto"].fillna(0).to_numpy(np.int64) % 10)


def _match(keys_fn, flows, csv, attack_mask, tol, flow_mask=None, csv_mask=None):
    ck = keys_fn(csv)
    cm = np.ones(len(csv), bool) if csv_mask is None else csv_mask
    a = np.unique(ck[attack_mask & cm])
    b = np.unique(ck[~attack_mask & cm])
    a_only, b_only = np.setdiff1d(a, b), np.setdiff1d(b, a)
    is_a = np.zeros(len(flows), bool)
    is_b = np.zeros(len(flows), bool)
    matched = np.zeros(len(flows), np.int64)
    fm = np.ones(len(flows), bool) if flow_mask is None else flow_mask
    for sh in range(-tol, tol + 1):
        k = keys_fn(flows, sh)
        hit = np.isin(k, a_only) & fm & ~is_a
        matched[hit] = k[hit]
        is_a |= hit
        is_b |= np.isin(k, b_only) & fm
    return is_a, is_b & ~is_a, matched, a_only, b_only


def transfer_labels(pcap_flows: pd.DataFrame, csv_flows: pd.DataFrame,
                    min_victim_matches: int = 100, min_target_share: float = 0.75,
                    source_outlier_factor: float = 5.0, tolerance_s: int = 1,
                    n_common_ports: int = 15, internal_prefix: str = "172.31."):
    csv_attack = (csv_flows["label_raw"].str.lower() != "benign").to_numpy()
    common = set(csv_flows.loc[~csv_attack, "dst_port"].value_counts().head(n_common_ports).index)
    rare_csv = ~csv_flows["dst_port"].isin(common).to_numpy()
    rare_flow = ~pcap_flows["dst_port"].isin(common).to_numpy()

    # 1) strict-key matches on rare ports; a flow seen in two captures is counted once
    sa, sb, _, a_keys, _ = _match(_strict_key, pcap_flows, csv_flows, csv_attack, tolerance_s,
                                  rare_flow, rare_csv)
    d = pd.DataFrame({"src": pcap_flows["src_ip"].astype(str).to_numpy(),
                      "dst": pcap_flows["dst_ip"].astype(str).to_numpy(),
                      "att": sa, "ben": sb, "k": _strict_key(pcap_flows)})
    d = d.drop_duplicates(["src", "dst", "k"])

    def role_stats(col):
        st = d.groupby(col)[["att", "ben"]].sum()
        st = st[st.index.str.startswith(internal_prefix)]
        st["share"] = st["att"] / (st["att"] + st["ben"]).replace(0, np.nan)
        return st.sort_values("att", ascending=False)

    src, dst = role_stats("src"), role_stats("dst")
    # attack SOURCE (compromised host): far more attack flows originate from it than from anyone else
    ref = float(np.median(src["att"].head(10))) if len(src) else 0.0
    src["victim"] = (src["att"] >= min_victim_matches) & (src["att"] >= source_outlier_factor * max(ref, 1.0))
    # attack TARGET: the traffic it receives is predominantly attack traffic (non-victims ~0.3)
    dst["victim"] = (dst["att"] >= min_victim_matches) & (dst["share"] >= min_target_share)
    victims = sorted(set(src.index[src["victim"]]) | set(dst.index[dst["victim"]]))

    # 2) flow labels (loose key, flows touching a victim), i.e. the dataset's IP + time rule
    touches = (pcap_flows["src_ip"].isin(victims) | pcap_flows["dst_ip"].isin(victims)).to_numpy()
    la, _, matched, _, _ = _match(_loose_key, pcap_flows, csv_flows, csv_attack, tolerance_s, touches)
    ck = _loose_key(csv_flows)
    label_of_key = (pd.Series(csv_flows["label_raw"].to_numpy()[csv_attack], index=ck[csv_attack])
                    .groupby(level=0).agg(lambda s: s.value_counts().index[0]))
    lab = np.full(len(pcap_flows), "Benign", dtype=object)
    idx = np.nonzero(la)[0]
    lab[idx] = label_of_key.reindex(matched[idx]).fillna("Benign").to_numpy()
    rnd = lambda df: df.head(8).reset_index().round(3).to_dict("records")  # noqa: E731
    summary = {
        "csv_rows": int(len(csv_flows)), "csv_attack_rows": int(csv_attack.sum()),
        "common_ports_excluded_for_victim_id": sorted(int(p) for p in common if p == p),
        "strict_attack_only_keys": int(len(a_keys)), "pcap_flows": int(len(pcap_flows)),
        "attack_sources": sorted(src.index[src["victim"]]),
        "attack_targets": sorted(dst.index[dst["victim"]]),
        "victims": victims,
        "source_reference_median_top10": ref,
        "top_sources": rnd(src), "top_destinations": rnd(dst),
        "pcap_flows_labelled_attack": int((lab != "Benign").sum()),
    }
    stats = pd.concat({"as_source": src, "as_destination": dst}, axis=1)
    return pd.Series(lab, index=pcap_flows.index, dtype="string"), stats, summary
