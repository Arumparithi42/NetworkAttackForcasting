"""Build the results report (markdown tables + figures) from trained experiment artifacts.

    python scripts/evaluate.py --config configs/cic2018_network.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netwm.config import load_config, resolve  # noqa: E402
from netwm.evaluation import plots  # noqa: E402
from netwm.evaluation.plots import MODEL_STYLE  # noqa: E402

MODELS = ["WM_world_model", "B3_xgboost_lagged", "B2_logreg_lagged", "B1_logreg_now", "B0_persistence"]


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.{nd}f}"


def ci(v):
    return "–" if not v or any(x is None or np.isnan(x) for x in v) else f"[{v[0]:.2f}, {v[1]:.2f}]"


def main_table(m: dict) -> str:
    rows = ["| Model | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC (95% CI) | PR-AUC onset (95% CI) | PR-AUC ongoing | Brier | ECE |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in MODELS:
        if name not in m:
            continue
        r = m[name]
        rows.append(f"| {MODEL_STYLE[name][0]} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | "
                    f"{fmt(r['fpr'], 4)} | {fmt(r['roc_auc'])} | {fmt(r['pr_auc'])} {ci(r['pr_auc_ci'])} | "
                    f"{fmt(r['pr_auc_onset'])} {ci(r['pr_auc_onset_ci'])} | {fmt(r['pr_auc_ongoing'])} | "
                    f"{fmt(r['brier'])} | {fmt(r['ece'])} |")
    return "\n".join(rows)


def lead_table(m: dict) -> str:
    rows = ["| Model | lead 0 (onset window) | T−1 | T−2 | T−3 | # onsets |", "|---|---|---|---|---|---|"]
    for name in MODELS:
        if name not in m:
            continue
        ew = m[name]["early_warning"]
        g = lambda k: ew.get(str(k), ew.get(k, {})).get("recall")  # noqa: E731
        n = ew.get("0", ew.get(0, {})).get("n_onsets")
        rows.append(f"| {MODEL_STYLE[name][0]} | {fmt(g(0), 2)} | {fmt(g(1), 2)} | {fmt(g(2), 2)} | {fmt(g(3), 2)} | {n} |")
    return "\n".join(rows)


def ablation_table(art: Path) -> str:
    rows = ["| Run | What changed | PR-AUC | PR-AUC onset | F1 | FPR | recall T−1 | recall T−2 | state MSE k=1 / k=5 |",
            "|---|---|---|---|---|---|---|---|---|"]
    desc = json.load(open(art / "ablations.json")) if (art / "ablations.json").exists() else {}
    for d in sorted(art.iterdir()):
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        m = json.load(open(mf))
        r = m["WM_world_model"]
        ew = r["early_warning"]
        sf = m.get("state_forecast", {}).get("wm_mse_per_k", [])
        rows.append(f"| {d.name} | {desc.get(d.name, '')} | {fmt(r['pr_auc'])} | {fmt(r['pr_auc_onset'])} | "
                    f"{fmt(r['f1'])} | {fmt(r['fpr'], 4)} | {fmt(ew.get('1', {}).get('recall'), 2)} | "
                    f"{fmt(ew.get('2', {}).get('recall'), 2)} | "
                    f"{fmt(sf[0]) if sf else '–'} / {fmt(sf[-1]) if sf else '–'} |")
    return "\n".join(rows)


def seed_summary(art_root: Path) -> str:
    vals = {}
    for tag in ("main", "seed1", "seed2"):
        f = art_root / tag / "metrics.json"
        if f.exists():
            r = json.load(open(f))["WM_world_model"]
            for k in ("pr_auc", "pr_auc_onset", "f1", "fpr"):
                vals.setdefault(k, []).append(r[k])
    if len(vals.get("pr_auc", [])) < 2:
        return "(only one seed available)"
    n = len(vals["pr_auc"])
    return (f"World model over {n} seeds: PR-AUC {np.mean(vals['pr_auc']):.3f} ± {np.std(vals['pr_auc']):.3f}, "
            f"onset PR-AUC {np.mean(vals['pr_auc_onset']):.3f} ± {np.std(vals['pr_auc_onset']):.3f}, "
            f"F1 {np.mean(vals['f1']):.3f} ± {np.std(vals['f1']):.3f}, FPR {np.mean(vals['fpr']):.3f} ± {np.std(vals['fpr']):.3f}. "
            "Differences between ablations smaller than this seed spread are not meaningful.")


def explain_examples(cfg, art: Path, out: Path, n: int = 40) -> dict:
    """IG heatmap for the highest-risk test forecast + deletion test on the top-n forecasts."""
    from netwm.explain.faithfulness import deletion_test
    from netwm.explain.ig import explain
    from netwm.inference.pipeline import Forecaster
    from netwm.data.prepare import load_processed
    from netwm.data.sequences import SequenceDataset, build_series

    fc = Forecaster(art)
    f = pd.read_parquet(art / "test_forecasts.parquet")
    states, labels = load_processed(cfg)
    series = build_series(states, labels, fc.scaler, cfg)
    ds = SequenceDataset(series, "test", cfg["L"], cfg["K"], cfg["warmup_windows"])
    top = np.argsort(-f["p_within"].to_numpy())[:n]
    samples = [series[ds.index[j][0]].X[ds.index[j][1] - cfg["L"] + 1: ds.index[j][1] + 1] for j in top]
    e = explain(fc.model, samples[0], fc.baseline, fc.features, cfg["K"], fc.T, top_k=8)
    row = f.iloc[top[0]]
    plots.ig_heatmap(np.array(e["heatmap"]), fc.features, str(out / "ig_heatmap_example.png"),
                     f"Integrated Gradients – {row['host']} @ window {int(row['window'])} "
                     f"(P={row['p_within']:.2f})")
    faith = deletion_test(fc.model, samples, fc.baseline, fc.features, cfg["K"], fc.T, k=5)
    return {"example": {"host": row["host"], "window": int(row["window"]),
                        "p": float(row["p_within"]), "top_features": e["top_features"],
                        "time_importance": e["time_importance"],
                        "convergence_delta": e["convergence_delta"]},
            "deletion_test": faith}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--max-timelines", type=int, default=6)
    a = ap.parse_args()
    cfg = load_config(a.config)
    art_root = resolve(cfg["artifact_dir"])
    art = art_root / a.tag
    out = resolve("reports") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    m = json.load(open(art / "metrics.json"))
    f = pd.read_parquet(art / "test_forecasts.parquet")
    labels = pd.concat([pd.read_parquet(p) for p in resolve(cfg["processed_dir"]).glob("labels_*.parquet")])
    thr = float(f["threshold"].iloc[0])

    # ---- figures
    v = f[f["valid"] > 0]
    preds = {n: v[f"p_{n}"].to_numpy() for n in MODELS if f"p_{n}" in v}
    plots.pr_curves(v["y_within"].to_numpy(), preds, str(out / "pr_curves.png"),
                    f"Attack within K={cfg['K']} windows – all test windows")
    on = v[v["y_now"] == 0]
    plots.pr_curves(on["y_within"].to_numpy(), {n: on[f"p_{n}"].to_numpy() for n in preds},
                    str(out / "pr_curves_onset.png"), "Onset forecasting – currently benign windows only")
    plots.reliability(v["y_within"].to_numpy(), v["p_WM_world_model"].to_numpy(),
                      str(out / "reliability_world_model.png"), "Calibration – world model (test)")
    plots.lead_curves({n: m[n]["early_warning"] for n in MODELS if n in m and n != "B0_persistence"},
                      str(out / "lead_time.png"), "Early warning before attack onset")
    sf = m["state_forecast"]
    if sf.get("wm_mse_per_k"):
        plots.state_error(sf["wm_mse_per_k"], sf["persistence_mse_per_k"], str(out / "state_forecast_error.png"),
                          "Next-state forecasting error (test)")
    # timelines: entities with the most attack windows in test (+ for network: every test day)
    lab = labels.merge(f[["day", "host", "window"]].drop_duplicates(), on=["day", "window", "host"])
    att = lab[lab["stage"] > 0].groupby(["day", "host"]).size().sort_values(ascending=False)
    ents = list(att.index[: a.max_timelines])
    for day, host in ents:
        g = f[(f["day"] == day) & (f["host"] == host)].sort_values("window")
        lg = labels[(labels["day"] == day) & (labels["host"] == host)].set_index("window")["stage"]
        times = pd.to_datetime(g["window"] * cfg["window_s"], unit="s", utc=True).dt.tz_localize(None)
        stage = lg.reindex(g["window"] + cfg["K"] * 0).fillna(0).to_numpy()
        probs = {n: g[f"p_{n}"].to_numpy() for n in ["WM_world_model", "B3_xgboost_lagged", "B1_logreg_now"] if f"p_{n}" in g}
        safe = f"{day}_{host}".replace(".", "-")
        plots.timeline(times.to_numpy(), stage, probs, thr,
                       f"{day} · {host} · forecast made at each window vs dataset label",
                       str(out / f"timeline_{safe}.png"))

    expl = explain_examples(cfg, art, out)
    with open(out / "explainability.json", "w") as fh:
        json.dump(expl, fh, indent=1, default=float)

    # ---- markdown
    s = m["sizes"]
    md = [f"# Results – {cfg['name']} (Protocol {cfg.get('protocol', 'A')})", "",
          "Generated by `scripts/evaluate.py` from `" + str(art.relative_to(resolve('.'))) + "`. "
          "All numbers are on the held-out **test** split; thresholds were chosen on validation "
          f"(FPR {cfg['eval']['target_fpr']:.0%} on quiet validation windows).", "",
          f"Test samples: {s['test_valid']} (positives: {s['test_positive']}); train {s['train']}, val {s['val']}.", "",
          "## Horizon forecast: P(attack within K windows)", "", main_table(m), "",
          "*Persistence uses the TRUE current label (an oracle detector) – it is a reference, not a deployable model.* "
          "*Onset = windows whose own label is benign; ongoing = windows already inside an attack.*", "",
          seed_summary(art_root), "",
          "## Early warning (recall at the validation threshold)", "", lead_table(m), "",
          "## Next-state forecasting (does the model learn dynamics?)", "",
          "| step k | " + " | ".join(str(i + 1) for i in range(len(sf.get('wm_mse_per_k', [])))) + " |",
          "|---|" + "---|" * len(sf.get("wm_mse_per_k", [])),
          "| World model MSE | " + " | ".join(fmt(x) for x in sf.get("wm_mse_per_k", [])) + " |",
          "| Persistence MSE | " + " | ".join(fmt(x) for x in sf.get("persistence_mse_per_k", [])) + " |", "",
          "## Future-stage macro-F1 per step (world model)", "",
          " | ".join(f"k={i + 1}: {fmt(x)}" for i, x in enumerate(m.get("stage_macro_f1_per_k", []))), "",
          "## Explainability sanity check (deletion test)", "",
          f"Replacing the top-5 IG features by benign values lowers P(attack) by "
          f"{fmt(expl['deletion_test']['mean_drop_top_k'])} on average vs {fmt(expl['deletion_test']['mean_drop_random_k'])} "
          f"for 5 random features (n={expl['deletion_test']['n']} highest-risk test forecasts).", "",
          "## Ablations (world model only)", "", ablation_table(art_root), "",
          "## Figures", ""]
    for p in sorted(out.glob("*.png")):
        md.append(f"![{p.stem}]({p.name})")
    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print((out / "RESULTS.md").read_text()[:3000])


if __name__ == "__main__":
    with torch.no_grad():
        pass
    main()
