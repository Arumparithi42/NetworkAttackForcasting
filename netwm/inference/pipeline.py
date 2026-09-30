"""Offline inference: CSV / PCAP / canonical flows -> forecasts, explanations, ATT&CK mapping.

    fc = Forecaster.load("artifacts/cic2018_network/main")
    flows = fc.load_flows("capture.pcap")          # or a CICFlowMeter / canonical CSV
    timeline = fc.timeline(flows)                  # P(attack within K) for every entity & window
    record = fc.record(timeline_ctx, entity, t)    # full observed / inferred / predicted JSON
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from netwm.attack_mapping.mapper import build_record
from netwm.explain.evidence import evidence_sentences
from netwm.explain.ig import explain
from netwm.features.scaler import RobustScaler
from netwm.features.temporal import build_states
from netwm.features.windowing import (add_window, host_window_features,
                                      network_window_features)
from netwm.forecasting.risk import summarise, surprise_percentile
from netwm.forecasting.rollout import forecast_entity
from netwm.models.world_model import build_model, p_attack_within


@dataclass
class Context:
    states: pd.DataFrame          # raw states (day, window, host, features)
    X: dict                       # entity -> [T, D] normalised
    raw: dict                     # entity -> [T, D] raw
    windows: dict                 # entity -> [T] window ids
    flows: pd.DataFrame


class Forecaster:
    def __init__(self, art_dir: str | Path):
        self.art = Path(art_dir)
        self.bundle = json.load(open(self.art / "bundle.json"))
        self.cfg = json.load(open(self.art / "config.json"))
        self.features = self.bundle["features"]
        self.classes = self.bundle["classes"]
        self.K, self.L = self.bundle["K"], self.bundle["L"]
        self.T = self.bundle["temperature"]
        self.threshold = self.bundle["threshold"]
        self.entity = self.bundle["entity"]
        self.scaler = RobustScaler.load(self.art / "scaler.json")
        self.calibrator = pickle.load(open(self.art / "calibrator.pkl", "rb"))
        self.model = build_model(len(self.features), len(self.classes), self.bundle["model_cfg"])
        self.model.load_state_dict(torch.load(self.art / "model.pt", map_location="cpu"))
        self.model.eval()
        self.baseline = np.asarray(self.bundle["benign_baseline"], dtype=np.float32)
        self.raw_baseline = self.scaler.inverse_raw(self.baseline[None])[0]

    @classmethod
    def load(cls, art_dir):
        return cls(art_dir)

    # ------------------------------------------------------------------ inputs
    def load_flows(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix.lower() in (".pcap", ".pcapng", ".cap"):
            from netwm.io.pcap_packets import pcap_to_flows
            return pcap_to_flows(path)
        if path.suffix.lower() == ".parquet":
            from netwm.io.schema import conform
            return conform(pd.read_parquet(path))
        header = open(path, encoding="utf-8", errors="replace").readline()
        if "Flow Duration" in header:
            from netwm.io.load_cic import load_cic_csv
            flavour = "cic2018" if "Dst Port" in header else "cic2017"
            return load_cic_csv(path, flavour=flavour,
                                utc_offset_hours=self.cfg.get("utc_offset_hours", 0),
                                clock_12h_threshold=self.cfg.get("clock_12h_threshold"))
        from netwm.io.schema import conform
        return conform(pd.read_csv(path))            # already canonical

    # ------------------------------------------------------------------ states
    def build_context(self, flows: pd.DataFrame) -> Context:
        ws = self.cfg["window_s"]
        flows = add_window(flows, ws)
        day = str(flows["ts"].min().date())
        windows = np.arange(flows["window"].min(), flows["window"].max() + 1)
        if self.entity == "network":
            feats = network_window_features(flows, ws, self.cfg["novelty_lookback"])
            states = build_states(feats, "network", day, windows, self.features, self.cfg["ewma_alpha"])
        else:
            feats = host_window_features(flows, self.cfg["internal_cidrs"], ws, self.cfg["novelty_lookback"])
            active = feats.groupby("host")["window"].nunique()
            hosts = sorted(active.index[active >= min(self.cfg.get("min_active_windows", 1), 3)])
            feats = feats[feats["host"].isin(hosts)]
            states = build_states(feats, "host", day, windows, self.features, self.cfg["ewma_alpha"],
                                  entities=hosts)
        X, raw, win = {}, {}, {}
        for host, g in states.groupby("host", sort=False):
            g = g.sort_values("window")
            X[host] = self.scaler.transform(g)
            raw[host] = g[self.features].to_numpy(np.float32)
            win[host] = g["window"].to_numpy()
        return Context(states, X, raw, win, flows)

    def context_from_states(self, states: pd.DataFrame, flows: pd.DataFrame | None = None) -> Context:
        """Build a context from an already-processed state table (replay of a prepared day)."""
        X, raw, win = {}, {}, {}
        for host, g in states.groupby("host", sort=False):
            g = g.sort_values("window")
            X[host] = self.scaler.transform(g)
            raw[host] = g[self.features].to_numpy(np.float32)
            win[host] = g["window"].to_numpy()
        if flows is None:
            flows = pd.DataFrame(columns=["window", "src_ip", "dst_ip", "dst_port", "proto",
                                          "fwd_pkts", "bwd_pkts"])
        elif "window" not in flows.columns:
            flows = add_window(flows, self.cfg["window_s"])
        return Context(states, X, raw, win, flows)

    # ------------------------------------------------------------------ forecasts
    @torch.no_grad()
    def timeline(self, ctx: Context) -> pd.DataFrame:
        """Deterministic calibrated P(attack within K) + stage probabilities for every window."""
        rows = []
        for host, X in ctx.X.items():
            T = len(X)
            ts = np.arange(self.L - 1, T)
            if not len(ts):
                continue
            xh = torch.from_numpy(np.stack([X[t - self.L + 1:t + 1] for t in ts]))
            out = self.model(xh, self.K, return_surprise=True)
            p = self.calibrator.predict(p_attack_within(out["stage_future"], temperature=self.T).numpy())
            now = (out["stage_now"] / self.T).softmax(-1).numpy()
            fut = (out["stage_future"] / self.T).softmax(-1).numpy()
            ref = self.bundle.get("surprise_ref", [])
            for j, t in enumerate(ts):
                r = {"host": host, "t": int(t), "window": int(ctx.windows[host][t]),
                     "p_within": float(p[j]),
                     "surprise_pct": surprise_percentile(float(out["surprise"][j]), ref) if ref else np.nan}
                for c, name in enumerate(self.classes):
                    r[f"now_{name}"] = float(now[j, c])
                    for k in range(self.K):
                        r[f"k{k + 1}_{name}"] = float(fut[j, k, c])
                rows.append(r)
        df = pd.DataFrame(rows)
        if len(df):
            df["time"] = pd.to_datetime(df["window"] * self.cfg["window_s"], unit="s", utc=True)
            df["alert"] = df["p_within"] >= self.threshold
        return df

    def record(self, ctx: Context, host: str, t: int, n_samples: int = 32,
               with_explanation: bool = True, intervention=None) -> dict:
        X = ctx.X[host]
        xh = X[t - self.L + 1:t + 1]
        fc = forecast_entity(self.model, xh, self.K, n_samples, self.T, self.calibrator,
                             intervention=intervention)
        pred = summarise(fc, self.classes, self.threshold, "host" if self.entity == "host" else "network",
                         calibrate=self.calibrator.predict)
        ref = self.bundle.get("surprise_ref", [])
        pred["surprise_percentile"] = round(surprise_percentile(fc["surprise"], ref), 4) if ref else None
        pred["horizon_windows"] = self.K
        pred["window_s"] = self.cfg["window_s"]
        raw_hist = ctx.raw[host][t - self.L + 1:t + 1]
        explanation = None
        if with_explanation:
            e = explain(self.model, xh, self.baseline, self.features, self.K, self.T)
            e["evidence"] = evidence_sentences(e["top_features"], raw_hist, self.raw_baseline, self.features)
            e.pop("heatmap_raw", None)
            explanation = e
        current = {f: float(v) for f, v in zip(self.features, raw_hist[-1])}
        observed = {"window_features_last3": {f: [round(float(v), 4) for v in raw_hist[-3:, i]]
                                              for i, f in enumerate(self.features)},
                    "top_flows": self.top_flows(ctx, host, int(ctx.windows[host][t]))}
        window_end = pd.Timestamp((int(ctx.windows[host][t]) + 1) * self.cfg["window_s"], unit="s", tz="UTC")
        return build_record(
            entity=host, window_end=window_end.isoformat(), observed=observed,
            current_stage={c: float(v) for c, v in zip(self.classes, fc["stage_now"])},
            current_features=current, mode=self.entity, predicted=pred, explanation=explanation,
            model_info={"dataset": self.bundle["dataset"], "sha256": self.bundle["model_sha256"][:16],
                        "config_sha256": self.bundle["config_sha256"][:16],
                        "threshold": self.threshold, "target_fpr": self.bundle["target_fpr"]})

    def top_flows(self, ctx: Context, host: str, window: int, n: int = 10) -> list[dict]:
        f = ctx.flows[(ctx.flows["window"] >= window - 2) & (ctx.flows["window"] <= window)]
        if self.entity == "host":
            f = f[(f["src_ip"] == host) | (f["dst_ip"] == host)]
        if not len(f):
            return []
        cols = [c for c in ["src_ip", "dst_ip", "dst_port", "proto"] if c in f.columns]
        g = (f.assign(pk=f["fwd_pkts"].fillna(0) + f["bwd_pkts"].fillna(0))
             .groupby(cols, dropna=False).agg(flows=("pk", "size"), packets=("pk", "sum"))
             .sort_values("flows", ascending=False).head(n).reset_index())
        return json.loads(g.to_json(orient="records"))
