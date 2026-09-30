"""Orchestrates raw data -> canonical flows -> per-day states + labels (contracts C1-C3).

Outputs (per day) in cfg['processed_dir']:
  states_<day>.parquet : day, window, host, <features>      (unscaled; scaling happens at training)
  labels_<day>.parquet : day, window, host, stage, n_mal, n_<stage>...
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from netwm.config import resolve
from netwm.features import feature_spec as fs
from netwm.features.temporal import build_states
from netwm.features.windowing import (add_window, host_view, host_window_features,
                                      internal_set, network_window_features)
from netwm.io.load_cic import load_cic_csv
from netwm.labels.build_labels import map_flow_stage, window_stage


def feature_list(cfg: dict) -> list[str]:
    if cfg["entity"] == "network":
        return fs.network_features()
    return fs.host_features(with_packet=bool(cfg.get("with_packet", False)))


def csv_flows(cfg: dict, day: str, force: bool = False) -> pd.DataFrame:
    """Cached canonical flows for one day from the CICFlowMeter CSV."""
    out = resolve(cfg["interim_dir"]) / f"flows_{day}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = load_cic_csv(resolve(cfg["raw_dir"]) / cfg["files"][day], flavour=cfg.get("flavour", "cic2018"),
                      utc_offset_hours=cfg["utc_offset_hours"],
                      clock_12h_threshold=cfg.get("clock_12h_threshold"), day=day)
    df.to_parquet(out, index=False)
    return df


def _capture_filter(flows: pd.DataFrame, cfg: dict, day: str) -> pd.DataFrame:
    start, end = cfg.get("capture_window_utc", ["12:00", "22:00"])
    lo = pd.Timestamp(f"{day} {start}", tz="UTC")
    hi = pd.Timestamp(f"{day} {end}", tz="UTC")
    return flows[(flows["ts"] >= lo) & (flows["ts"] < hi)]


def _windows(flows: pd.DataFrame) -> np.ndarray:
    return np.arange(flows["window"].min(), flows["window"].max() + 1)


def prepare_network_day(cfg: dict, day: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ws = cfg["window_s"]
    flows = _capture_filter(csv_flows(cfg, day), cfg, day)
    flows = add_window(flows, ws)
    flows = flows.assign(flow_stage=map_flow_stage(flows["label_raw"]))
    feats = network_window_features(flows, ws, cfg["novelty_lookback"])
    windows = _windows(flows)
    states = build_states(feats, "network", day, windows, feature_list(cfg), cfg["ewma_alpha"])
    labels = window_stage(flows, ["window"], cfg["min_attack_flows"])
    labels = labels.set_index("window").reindex(windows).fillna(0).reset_index()
    labels["day"], labels["host"] = day, "network"
    info = {"day": day, "flows": int(len(flows)), "windows": int(len(windows)),
            "label_counts": flows["label_raw"].value_counts().to_dict()}
    return states, labels, info


def pcap_day_flows(cfg: dict, day: str) -> pd.DataFrame:
    files = sorted((resolve(cfg["flows_dir"]) / day).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no extracted PCAP flows for {day}; run "
                                "scripts/extract_cic2018_pcaps.py first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # one shared categorical dtype for every IP column: ~10x less memory, fast group-bys
    ips = pd.unique(pd.concat([df["src_ip"], df["dst_ip"], df["capture_host"]]).astype(str))
    ipdtype = pd.CategoricalDtype(sorted(ips))
    for c in ("src_ip", "dst_ip", "capture_host"):
        df[c] = df[c].astype(str).astype(ipdtype)
    return df.drop(columns=["label_raw", "day"], errors="ignore")


def prepare_host_day(cfg: dict, day: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from netwm.labels.match import transfer_labels

    ws = cfg["window_s"]
    flows = _capture_filter(pcap_day_flows(cfg, day), cfg, day)
    lm = cfg.get("label_matching", {})
    flows = flows.reset_index(drop=True)
    labels_raw, host_stats, summary = transfer_labels(
        flows, _capture_filter(csv_flows(cfg, day), cfg, day),
        lm.get("min_victim_matches", 100), lm.get("min_target_share", 0.75),
        lm.get("source_outlier_factor", 5.0))
    flows = flows.assign(label_raw=labels_raw)
    flows = add_window(flows, ws)
    internal = internal_set(flows, cfg["internal_cidrs"])
    src_int = flows["src_ip"].isin(internal)
    flows = flows.assign(flow_stage=map_flow_stage(flows["label_raw"], src_int))

    feats = host_window_features(flows, cfg["internal_cidrs"], ws, cfg["novelty_lookback"])
    active = feats.groupby("host")["window"].nunique()
    hosts = sorted(active.index[active >= cfg.get("min_active_windows", 1)])
    feats = feats[feats["host"].isin(hosts)]
    windows = _windows(flows)
    states = build_states(feats, "host", day, windows, feature_list(cfg), cfg["ewma_alpha"],
                          entities=hosts)

    v = host_view(flows, internal)
    v = v[v["host"].isin(hosts)]
    lab = window_stage(v, ["window", "host"], cfg["min_attack_flows"])
    lab["host"] = lab["host"].astype(str)
    lab = lab[lab["host"].isin(hosts)]
    del v
    grid = states[["window", "host"]]
    labels = grid.merge(lab, on=["window", "host"], how="left")
    labels = labels.fillna({c: 0 for c in labels.columns if c not in ("window", "host")})
    labels["day"] = day
    info = {"day": day, "flows": int(len(flows)), "windows": int(len(windows)),
            "hosts": len(hosts), "label_transfer": summary,
            }
    return states, labels, info


def prepare_all(cfg: dict, days: list[str] | None = None) -> dict:
    out_dir = resolve(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    days = days or [d for d in cfg["files"] if d not in cfg.get("excluded_days", {})]
    infos = {}
    for day in days:
        fn = prepare_network_day if cfg["entity"] == "network" else prepare_host_day
        states, labels, info = fn(cfg, day)
        states.to_parquet(out_dir / f"states_{day}.parquet", index=False)
        labels.to_parquet(out_dir / f"labels_{day}.parquet", index=False)
        infos[day] = info
        print(f"[prepare] {day}: {json.dumps({k: v for k, v in info.items() if k in ('flows', 'windows', 'hosts')})}",
              flush=True)
    with open(out_dir / "prepare_info.json", "w") as fh:
        json.dump(infos, fh, indent=1, default=str)
    with open(out_dir / "feature_list.json", "w") as fh:
        json.dump(feature_list(cfg), fh, indent=1)
    return infos


def load_processed(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = resolve(cfg["processed_dir"])
    days = [p.stem.replace("states_", "") for p in sorted(d.glob("states_*.parquet"))]
    days = [x for x in days if x not in cfg.get("excluded_days", {})]
    states = pd.concat([pd.read_parquet(d / f"states_{x}.parquet") for x in days], ignore_index=True)
    labels = pd.concat([pd.read_parquet(d / f"labels_{x}.parquet") for x in days], ignore_index=True)
    return states, labels
