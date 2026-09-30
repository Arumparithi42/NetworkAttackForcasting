"""Copy trained artifacts + a small replay subset into demo/ so the dashboard runs on a fresh clone
(no dataset download, no training).

    python scripts/make_demo_bundle.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netwm.config import resolve  # noqa: E402

ART_FILES = ["model.pt", "scaler.json", "calibrator.pkl", "bundle.json", "config.json", "metrics.json"]
IP_RE = re.compile(r"(\d+\.\d+\.\d+\.\d+)")
FLOW_COLS = ["ts", "src_ip", "dst_ip", "src_port", "dst_port", "proto", "fwd_pkts", "bwd_pkts",
             "fwd_bytes", "bwd_bytes", "syn_cnt", "rst_cnt", "label_raw"]


def copy_artifact(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in ART_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)


def network(name="network", art="artifacts/cic2018_network/main", flows_day="2018-03-01"):
    src = resolve(art)
    cfg = json.load(open(src / "config.json"))
    out = resolve("demo") / name
    copy_artifact(src, out / "artifact")
    proto = cfg["protocols"][cfg["protocol"]]
    days = list(proto["test"]) + list((proto.get("split_days") or {}).keys())
    for day in days:
        d = out / "replay" / day
        d.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(resolve(cfg["processed_dir"]) / f"states_{day}.parquet").to_parquet(d / "states.parquet")
        pd.read_parquet(resolve(cfg["processed_dir"]) / f"labels_{day}.parquet").to_parquet(d / "labels.parquet")
        if day == flows_day:
            f = pd.read_parquet(resolve(cfg["interim_dir"]) / f"flows_{day}.parquet")
            f[[c for c in FLOW_COLS if c in f]].to_parquet(d / "flows.parquet", index=False)


def host(name="host", art="artifacts/cic2018_host/main", day="2018-03-01", n_hosts=40):
    src = resolve(art)
    if not (src / "model.pt").exists():
        print("no host model yet - skipped")
        return
    cfg = json.load(open(src / "config.json"))
    out = resolve("demo") / name
    copy_artifact(src, out / "artifact")
    states = pd.read_parquet(resolve(cfg["processed_dir"]) / f"states_{day}.parquet")
    labels = pd.read_parquet(resolve(cfg["processed_dir"]) / f"labels_{day}.parquet")
    fc = pd.read_parquet(src / "test_forecasts.parquet")
    fc = fc[fc["day"] == day]
    # hosts with attack labels + highest peak risk + a fixed sample of ordinary hosts
    att = labels[labels["stage"] > 0].groupby("host").size().sort_values(ascending=False).index.tolist()
    risky = fc.groupby("host")["p_within"].max().sort_values(ascending=False).index.tolist()
    rest = sorted(set(states["host"]) - set(att) - set(risky[:15]))[:: max(1, len(states["host"].unique()) // 15)]
    keep = list(dict.fromkeys(att[:15] + risky[:15] + rest))[:n_hosts]
    d = out / "replay" / day
    d.mkdir(parents=True, exist_ok=True)
    states[states["host"].isin(keep)].to_parquet(d / "states.parquet", index=False)
    labels[labels["host"].isin(keep)].to_parquet(d / "labels.parquet", index=False)
    frames = []
    for p in (resolve(cfg["flows_dir"]) / day).glob("*.parquet"):
        ips = IP_RE.findall(p.stem)
        if ips and ips[-1] in keep:
            f = pd.read_parquet(p, columns=[c for c in FLOW_COLS if c != "label_raw"])
            frames.append(f)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(d / "flows.parquet", index=False,
                                                        compression="zstd")
    print("host demo hosts:", len(keep))


if __name__ == "__main__":
    network()
    host()
    for p in sorted(resolve("demo").rglob("*")):
        if p.is_file():
            print(f"{p.stat().st_size / 1e6:7.2f} MB  {p.relative_to(resolve('.'))}")
