"""Scenario discovery and loading for the dashboard.

A scenario = a trained artifact directory + a replay day (states, labels, optional flows).
Committed demo scenarios live in demo/<name>/{artifact,replay}; locally trained experiments in
artifacts/<dataset>/<tag> are also offered when their processed data exists.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IP_RE = re.compile(r"(\d+\.\d+\.\d+\.\d+)")


def discover() -> dict:
    """name -> {'artifact': Path, 'days': {day: {'states','labels','flows'?}}}"""
    out = {}
    for d in sorted((ROOT / "demo").glob("*/artifact")):
        name = d.parent.name
        days = {}
        for r in sorted((d.parent / "replay").glob("*")):
            days[r.name] = {"states": r / "states.parquet", "labels": r / "labels.parquet",
                            "flows": r / "flows.parquet"}
        if days:
            out[f"demo · {name}"] = {"artifact": d, "days": days, "flows_dir": None}
    for art in sorted((ROOT / "artifacts").glob("*/main")):
        cfg_path = art / "config.json"
        if not cfg_path.exists() or not (art / "model.pt").exists():
            continue
        cfg = json.load(open(cfg_path))
        proc = ROOT / cfg["processed_dir"]
        proto = cfg["protocols"][cfg.get("protocol", "A")]
        test_days = list(proto.get("test", [])) + list((proto.get("split_days") or {}).keys())
        days = {}
        for day in test_days:
            st, lb = proc / f"states_{day}.parquet", proc / f"labels_{day}.parquet"
            if st.exists() and lb.exists():
                flows = ROOT / cfg.get("interim_dir", "") / f"flows_{day}.parquet"
                days[day] = {"states": st, "labels": lb,
                             "flows": flows if cfg["entity"] == "network" else None}
        if days:
            out[f"local · {art.parent.name}"] = {
                "artifact": art, "days": days,
                "flows_dir": ROOT / cfg["flows_dir"] if cfg.get("source") == "pcap_flows" else None}
    return out


def load_day(entry: dict, day: str):
    d = entry["days"][day]
    states = pd.read_parquet(d["states"])
    labels = pd.read_parquet(d["labels"])
    flows = pd.read_parquet(d["flows"]) if d.get("flows") and Path(d["flows"]).exists() else None
    return states, labels, flows


def host_flows(entry: dict, day: str, host: str) -> pd.DataFrame | None:
    """Per-host capture flows (host mode): from the demo replay or the extracted PCAP flows."""
    d = entry["days"][day]
    if d.get("flows") and Path(d["flows"]).exists():
        f = pd.read_parquet(d["flows"])
        return f[(f["src_ip"] == host) | (f["dst_ip"] == host)]
    if entry.get("flows_dir") is None:
        return None
    files = [p for p in (entry["flows_dir"] / day).glob("*.parquet")
             if IP_RE.findall(p.stem) and IP_RE.findall(p.stem)[-1] == host]
    if not files:
        return None
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
