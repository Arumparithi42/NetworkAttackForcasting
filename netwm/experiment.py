"""End-to-end experiment: scaler -> sequences -> world model + baselines -> calibration ->
test metrics, lead time, state-forecast error -> reports and deployable artifacts.

Everything (features, samples, split, threshold rule) is identical across models.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from netwm.config import config_hash, resolve
from netwm.data.prepare import load_processed
from netwm.data.sequences import CLASS_NAMES, SequenceDataset, build_series
from netwm.data.splits import window_roles
from netwm.evaluation.bootstrap import block_bootstrap_ci
from netwm.evaluation.lead_time import early_warning
from netwm.evaluation.metrics import (binary_report, pr_auc, stage_macro_f1_per_step,
                                      threshold_at_fpr)
from netwm.features.scaler import RobustScaler
from netwm.models.baselines import all_baselines
from netwm.training.calibrate import ProbCalibrator, fit_temperature
from netwm.training.train_world_model import predict, train_world_model


def _log(fh):
    def log(msg):
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()
    return log


def fit_scaler(states: pd.DataFrame, labels: pd.DataFrame, cfg: dict, features: list[str]):
    df = states.merge(labels[["day", "window", "host", "stage"]], on=["day", "window", "host"],
                      how="left")
    roles = pd.Series("", index=df.index, dtype=object)
    for day, g in df.groupby("day"):
        roles.loc[g.index] = window_roles(day, g["window"].to_numpy(), cfg, cfg["window_s"])
    roles = roles.to_numpy()
    mask = (roles == "train") & (df["stage"].fillna(0) == 0)
    if "active" in df.columns:
        mask &= df["active"] > 0
    return RobustScaler(features).fit(df[mask])


def _meta_with_series(ds, series_ids=None) -> pd.DataFrame:
    m = ds.meta()
    m["series"] = ds.index[:, 0] if len(ds.index) else []
    return m


def run_experiment(cfg: dict, tag: str = "main", run_baselines: bool = True,
                   export_forecasts: bool = True, log_path: str | None = None) -> dict:
    t_start = time.time()
    art = resolve(cfg["artifact_dir"]) / tag
    art.mkdir(parents=True, exist_ok=True)
    fh = open(log_path or art / "run.log", "w")
    log = _log(fh)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    states, labels = load_processed(cfg)
    features = json.load(open(resolve(cfg["processed_dir"]) / "feature_list.json"))
    drop = set(cfg.get("drop_features", []))
    features = [f for f in features if f not in drop and not any(f.startswith(p) for p in cfg.get("drop_prefixes", []))]
    scaler = fit_scaler(states, labels, cfg, features)
    series = build_series(states, labels, scaler, cfg)
    L, K = cfg["L"], cfg["K"]
    warm = cfg["warmup_windows"]
    train_ds = SequenceDataset(series, "train", L, K, warm, cfg["sampling"]["benign_ratio"], cfg["seed"])
    val_ds = SequenceDataset(series, "val", L, K, warm)
    test_ds = SequenceDataset(series, "test", L, K, warm)
    log(f"[{tag}] features={len(features)} series={len(series)} train={len(train_ds)} "
        f"val={len(val_ds)} test={len(test_ds)} "
        f"train_pos={int(train_ds.y_within.sum())} val_pos={int(val_ds.y_within.sum())} "
        f"test_pos={int(test_ds.y_within.sum())}")

    # ---------------- world model ----------------
    C = len(CLASS_NAMES)
    model, history = train_world_model(train_ds, val_ds, cfg, len(features), C, art, log)
    pv = predict(model, val_ds, K)
    # temperature on the future-stage probabilities (log-probs are valid logits up to a constant)
    T = fit_temperature(np.log(np.clip(pv["stage_future"], 1e-9, 1)).reshape(-1, C),
                        val_ds.y_future.reshape(-1))
    pv = predict(model, val_ds, K, temperature=T, with_states=True)
    vmask = val_ds.valid > 0
    calib = ProbCalibrator().fit(pv["p_within"][vmask], val_ds.y_within[vmask])
    pt = predict(model, test_ds, K, temperature=T, with_states=True)
    p_val = calib.predict(pv["p_within"])
    p_test = calib.predict(pt["p_within"])

    preds_val = {"WM_world_model": p_val}
    preds_test = {"WM_world_model": p_test}

    # ---------------- baselines (same samples) ----------------
    if run_baselines:
        Xtr, Xv, Xte = train_ds.history_array(), val_ds.history_array(), test_ds.history_array()
        tmask = train_ds.valid > 0
        for b in all_baselines(cfg["seed"]):
            t0 = time.time()
            b.fit(Xtr[tmask], train_ds.y_within[tmask], train_ds.y_now[tmask])
            bv = b.predict_proba(Xv, val_ds.y_now)
            bt = b.predict_proba(Xte, test_ds.y_now)
            if b.name != "B0_persistence":
                bc = ProbCalibrator().fit(bv[vmask], val_ds.y_within[vmask])
                bv, bt = bc.predict(bv), bc.predict(bt)
            preds_val[b.name], preds_test[b.name] = bv, bt
            log(f"[{tag}] baseline {b.name} fitted in {time.time() - t0:.1f}s")
            if b.name == "B3_xgboost_lagged":
                with open(art / "xgb_baseline.pkl", "wb") as f:
                    pickle.dump(b, f)

    # ---------------- evaluation ----------------
    tm = test_ds.valid > 0
    yt = test_ds.y_within
    quiet_val = vmask & (val_ds.y_now == 0) & (val_ds.y_within == 0)
    meta = _meta_with_series(test_ds)
    series_y = {i: s.y for i, s in enumerate(series) if (s.roles == "test").any()}
    groups = (meta["series"].astype(str) + "_" + (meta["t"] // 60).astype(str)).to_numpy()
    onset_mask = tm & (test_ds.y_now == 0)
    ongoing_mask = tm & (test_ds.y_now > 0)
    results = {}
    for name in preds_test:
        pvn, ptn = preds_val[name], preds_test[name]
        thr = threshold_at_fpr(pvn[quiet_val], cfg["eval"]["target_fpr"]) if name != "B0_persistence" else 0.5
        rep = binary_report(ptn[tm], yt[tm], thr)
        rep["pr_auc_onset"] = pr_auc(ptn[onset_mask], yt[onset_mask])
        rep["pr_auc_ongoing"] = pr_auc(ptn[ongoing_mask], yt[ongoing_mask])
        rep["onset_positives"] = int(yt[onset_mask].sum())
        rep["pr_auc_ci"] = block_bootstrap_ci(pr_auc, groups[tm], ptn[tm], yt[tm],
                                              n=cfg["eval"]["bootstrap"], seed=cfg["seed"])
        rep["pr_auc_onset_ci"] = block_bootstrap_ci(pr_auc, groups[onset_mask], ptn[onset_mask],
                                                    yt[onset_mask], n=cfg["eval"]["bootstrap"],
                                                    seed=cfg["seed"])
        lead, lead_df = early_warning(meta, ptn, series_y, thr, cfg["eval"]["leads"],
                                      cfg["quiet_windows"])
        rep["early_warning"] = lead
        results[name] = rep
        if name == "WM_world_model":
            lead_df.to_csv(art / "onsets_world_model.csv", index=False)
        log(f"[{tag}] {name:22s} PR-AUC={rep['pr_auc']:.3f} onset PR-AUC={rep['pr_auc_onset']:.3f} "
            f"F1={rep['f1']:.3f} FPR={rep['fpr']:.4f} lead1={lead.get(1, {}).get('recall')}")

    # state forecasting vs persistence, stage F1 per step
    results["state_forecast"] = {
        "wm_nll_per_k": pt["nll_k"][tm].mean(0).tolist() if len(pt["nll_k"]) else [],
        "wm_mse_per_k": pt["se_k"].mean(0).tolist() if len(pt["se_k"]) else [],
        "persistence_mse_per_k": pt["se_persist_k"].mean(0).tolist() if len(pt["se_k"]) else [],
    }
    results["stage_macro_f1_per_k"] = stage_macro_f1_per_step(pt["stage_future"], test_ds.y_future)
    results["validation"] = {   # used for model selection - never the test numbers
        "pr_auc": pr_auc(pv["p_within"][vmask], val_ds.y_within[vmask]),
        "state_nll": float(pv["nll_k"].mean()) if len(pv["nll_k"]) else None,
        "state_mse_per_k": pv["se_k"].mean(0).tolist() if len(pv["se_k"]) else [],
        "persistence_mse_per_k": pv["se_persist_k"].mean(0).tolist() if len(pv["se_k"]) else [],
    }
    results["temperature"] = T
    results["sizes"] = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds),
                        "test_positive": int(yt[tm].sum()), "test_valid": int(tm.sum())}

    # ---------------- artifacts ----------------
    surprise_ref = np.sort(pv["surprise"][quiet_val]) if quiet_val.any() else np.sort(pv["surprise"])
    benign_train = np.concatenate([s.X[(s.roles == "train") & (s.y == 0)] for s in series])
    if "active" in features:
        benign_train = benign_train[benign_train[:, features.index("active")] > 0]
    benign_baseline = np.median(benign_train, axis=0).astype(np.float32)
    scaler.save(art / "scaler.json")
    with open(art / "calibrator.pkl", "wb") as f:
        pickle.dump(calib, f)
    wm_thr = threshold_at_fpr(p_val[quiet_val], cfg["eval"]["target_fpr"])
    bundle = {
        "features": features, "classes": CLASS_NAMES, "temperature": T, "threshold": wm_thr,
        "target_fpr": cfg["eval"]["target_fpr"], "L": L, "K": K, "entity": cfg["entity"],
        "model_cfg": dict(cfg["model"], K=K), "benign_baseline": benign_baseline.tolist(),
        "surprise_ref": surprise_ref[:: max(1, len(surprise_ref) // 2000)].tolist(),
        "config_sha256": config_hash(cfg),
        "model_sha256": hashlib.sha256(open(art / "model.pt", "rb").read()).hexdigest(),
        "dataset": cfg["name"], "tag": tag,
    }
    with open(art / "bundle.json", "w") as f:
        json.dump(bundle, f, indent=1)
    with open(art / "config.json", "w") as f:
        json.dump(cfg, f, indent=1, default=str)

    if export_forecasts:
        out = meta.copy()
        out["p_within"] = p_test
        for name, p in preds_test.items():
            out[f"p_{name}"] = p
        out["surprise_pct"] = np.searchsorted(surprise_ref, pt["surprise"]) / max(len(surprise_ref), 1)
        for c, cname in enumerate(CLASS_NAMES):
            out[f"now_{cname}"] = pt["stage_now"][:, c]
            for k in range(K):
                out[f"k{k + 1}_{cname}"] = pt["stage_future"][:, k, c]
        out["threshold"] = wm_thr
        out.to_parquet(art / "test_forecasts.parquet", index=False)

    results["runtime_s"] = round(time.time() - t_start, 1)
    with open(art / "metrics.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    fh.close()
    return results
