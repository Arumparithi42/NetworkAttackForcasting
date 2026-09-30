"""Copy headline numbers from artifacts/*/metrics.json into README.md and docs/SLIDES.md
(between RESULTS / SLIDE4 markers) so the documents never drift from the actual runs."""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"WM_world_model": "World model", "B3_xgboost_lagged": "XGBoost (lagged history)",
         "B2_logreg_lagged": "LogReg (lagged history)", "B1_logreg_now": "LogReg (current window)",
         "B0_persistence": "Persistence (oracle current label)"}


def load(ds, tag):
    p = ROOT / "artifacts" / ds / tag / "metrics.json"
    return json.load(open(p)) if p.exists() else None


def f(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def table(m, title):
    ew = lambda r, k: r["early_warning"].get(str(k), {}).get("recall")  # noqa: E731
    rows = [f"**{title}** – test windows: {m['sizes']['test_valid']:,} (positives {m['sizes']['test_positive']:,}); "
            f"onsets for early warning: {m['WM_world_model']['early_warning']['1']['n_onsets']}", "",
            "| Model | PR-AUC [95% CI] | PR-AUC onset | F1 | FPR | Recall T−1 | Recall T−2 |",
            "|---|---|---|---|---|---|---|"]
    for k, name in NAMES.items():
        r = m.get(k)
        if not r:
            continue
        ci = r["pr_auc_ci"]
        rows.append(f"| {name} | {f(r['pr_auc'])} [{f(ci[0], 2)}, {f(ci[1], 2)}] | {f(r['pr_auc_onset'])} | "
                    f"{f(r['f1'])} | {f(r['fpr'], 4)} | {f(ew(r, 1), 2)} | {f(ew(r, 2), 2)} |")
    return "\n".join(rows)


def seeds(ds):
    v = [load(ds, t)["WM_world_model"]["pr_auc"] for t in ("main", "seed1", "seed2") if load(ds, t)]
    return f"{np.mean(v):.3f} ± {np.std(v):.3f} (n={len(v)} seeds)"


def main():
    net, host, pb = load("cic2018_network", "main"), load("cic2018_host", "main"), load("cic2018_network", "protocolB")
    a1n, a1h = load("cic2018_network", "A1_no_dynamics_loss"), load("cic2018_host", "A1_no_dynamics_loss")
    parts = []
    if host:
        parts += [table(host, "Host level – infiltration, raw PCAPs (train 28-02 → test 01-03), 433 hosts, 67 features"), "",
                  f"World-model PR-AUC across seeds: {seeds('cic2018_host')}.", ""]
    if net:
        parts += [table(net, "Network level – 8 CSV days, Protocol A (per-family chronological)"), "",
                  f"World-model PR-AUC across seeds: {seeds('cic2018_network')}.", ""]
    if pb:
        parts += [f"Protocol B (strict global chronology, attack families unseen in training): world model PR-AUC "
                  f"{f(pb['WM_world_model']['pr_auc'])}, XGBoost {f(pb['B3_xgboost_lagged']['pr_auc'])}, "
                  f"LogReg (current) {f(pb['B1_logreg_now']['pr_auc'])}.", ""]
    sf_n, sf_h = net["state_forecast"], host["state_forecast"] if host else None
    parts += ["**What the numbers say**", "",
              "- *Continuation vs onset.* Inside an ongoing attack every model scores high (network PR-AUC on ongoing "
              f"windows ≈ {f(net['WM_world_model']['pr_auc_ongoing'], 2)}), but the base rate there is already very "
              "high – continuation is easy. On currently-benign windows (onset) PR-AUC stays near the base rate for "
              "every model: no reliable pre-onset warning on this data.",
              "- *World model vs baselines.* The world model is on par with the current-window logistic regression "
              "and XGBoost at host level and below XGBoost at network level; seed-to-seed spread is large, and "
              "ablation differences within that spread are not meaningful.",
              f"- *Dynamics.* Removing the dynamics loss (A1) gives PR-AUC {f(a1n['WM_world_model']['pr_auc']) if a1n else '–'} "
              f"(network) / {f(a1h['WM_world_model']['pr_auc']) if a1h else '–'} (host) vs "
              f"{f(net['WM_world_model']['pr_auc'])} / {f(host['WM_world_model']['pr_auc']) if host else '–'}: the "
              "learned dynamics do not measurably help the attack forecast here. The mean next-state prediction is "
              f"better than persistence on validation days but not on the test days (network MSE k=1: "
              f"{f(sf_n['wm_mse_per_k'][0])} vs {f(sf_n['persistence_mse_per_k'][0])}) – a day-to-day distribution shift.",
              "- *Calibration and explanations work.* Host-level ECE ≈ "
              f"{f(host['WM_world_model']['ece']) if host else '–'}; the IG deletion test shows the attributed features "
              "drive the forecast (see reports).",
              "- *Synthetic sanity check* (`tests/test_model.py::test_toy_world…`): when a 3-window precursor trend "
              "exists, the world model reaches onset PR-AUC ≈ 0.50 vs 0.10 for a current-window classifier (chance "
              "0.09) over 3 seeds – the model can exploit precursors when the data contains them.", "",
              "Full tables, confidence intervals, ablations and figures: "
              "[`reports/cic2018_host/RESULTS.md`](reports/cic2018_host/RESULTS.md), "
              "[`reports/cic2018_network/RESULTS.md`](reports/cic2018_network/RESULTS.md).", "",
              "![dashboard](docs/img/dashboard_overview.png)"]
    block = "\n".join(parts)
    readme = (ROOT / "README.md").read_text()
    readme = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                    lambda _: f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", readme, flags=re.S)
    (ROOT / "README.md").write_text(readme)
    slides = (ROOT / "docs" / "SLIDES.md").read_text()
    short = [table(host, "Host level (infiltration, PCAP)") if host else "", "",
             table(net, "Network level (8 days, CSV)"), "",
             "Message: calibrated, explainable forecasts; continuation forecast strong; **onset forecasting not "
             "demonstrated on CIC-IDS2018** (few precursors) – world model ≈ baselines; synthetic check shows the "
             "mechanism works when precursors exist."]
    slides = re.sub(r"<!-- SLIDE4:START -->.*<!-- SLIDE4:END -->",
                    lambda _: "<!-- SLIDE4:START -->\n" + "\n".join(short) + "\n<!-- SLIDE4:END -->", slides, flags=re.S)
    (ROOT / "docs" / "SLIDES.md").write_text(slides)
    print(block)


if __name__ == "__main__":
    sys.exit(main())
