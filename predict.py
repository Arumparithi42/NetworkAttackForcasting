"""Offline inference on a CSV or PCAP file.

    python predict.py --artifact artifacts/cic2018_network/main --csv data/sample/demo_flows.csv
    python predict.py --artifact artifacts/cic2018_host/main --pcap capture.pcap --top 5

Writes a JSON report (top-risk entities at the latest window, with observed / inferred /
predicted blocks and explanations) and a timeline CSV (P(attack within K) per entity & window).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="trained experiment directory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--pcap")
    src.add_argument("--flows", help="canonical flows parquet")
    ap.add_argument("--out", default="forecast.json")
    ap.add_argument("--timeline", default=None, help="timeline CSV path (default: <out>_timeline.csv)")
    ap.add_argument("--top", type=int, default=5, help="number of entities to report in detail")
    ap.add_argument("--at", default="latest", help="'latest' or a window index t")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--ledger", default=None, help="append records to this evidence ledger")
    a = ap.parse_args()

    from netwm.inference.pipeline import Forecaster

    fc = Forecaster(a.artifact)
    flows = fc.load_flows(a.csv or a.pcap or a.flows)
    if a.pcap and fc.cfg.get("source") != "pcap_flows":
        print("WARNING: this model was trained on CICFlowMeter flows; PCAP flows come from a "
              "different extractor, so expect some distribution shift.")
    ctx = fc.build_context(flows)
    tl = fc.timeline(ctx)
    if tl.empty:
        raise SystemExit(f"Not enough windows: need at least L={fc.L} windows of traffic.")
    tl_path = a.timeline or str(Path(a.out).with_suffix("")) + "_timeline.csv"
    tl.to_csv(tl_path, index=False)

    t_sel = int(tl["t"].max()) if a.at == "latest" else int(a.at)
    now = tl[tl["t"] == t_sel].sort_values("p_within", ascending=False)
    records = [fc.record(ctx, h, t_sel, n_samples=a.samples) for h in now["host"].head(a.top)]
    report = {
        "input": a.csv or a.pcap or a.flows, "entity_mode": fc.entity,
        "window_s": fc.cfg["window_s"], "horizon_windows": fc.K,
        "n_entities": int(tl["host"].nunique()), "n_windows": int(tl["t"].nunique()),
        "alerts_in_timeline": int(tl["alert"].sum()),
        "records": records,
    }
    if a.ledger:
        from netwm.ledger.hashchain import EvidenceLedger
        led = EvidenceLedger(a.ledger)
        for r in records:
            r["ledger"] = led.append(r)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    for r in records:
        p = r["predicted"]
        print(f"{r['entity']:>16s}  P(attack within {fc.K})={p['p_attack_within_K']:.2f} "
              f"[{p['band_5_95'][0]:.2f}-{p['band_5_95'][1]:.2f}]  risk={p['risk_level']:8s} "
              f"stage={p['most_likely_future_stage']}  now={r['inferred']['current_stage']}")
    print(f"report -> {a.out}; timeline -> {tl_path}")


if __name__ == "__main__":
    main()
