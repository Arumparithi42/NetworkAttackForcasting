"""Train + evaluate the world model and the baselines on one dataset config.

    python train.py --config configs/cic2018_network.yaml                 # Protocol A, tag 'main'
    python train.py --config configs/cic2018_network.yaml --protocol B --tag protocolB
    python train.py --config configs/cic2018_network.yaml --tag A1_no_dyn --no-baselines \
        --set train.lambdas.dyn=0
"""
import argparse
import json

import yaml

from netwm.config import load_config


def parse_set(items):
    out = {}
    for it in items or []:
        key, val = it.split("=", 1)
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--set", nargs="*", help="override config keys, e.g. train.epochs=5 L=5")
    a = ap.parse_args()
    over = parse_set(a.set)
    if a.protocol:
        over["protocol"] = a.protocol
    if a.seed is not None:
        over["seed"] = a.seed
    cfg = load_config(a.config, overrides=over)

    from netwm.experiment import run_experiment
    res = run_experiment(cfg, tag=a.tag, run_baselines=not a.no_baselines)
    keep = {k: v for k, v in res.items() if k not in ("state_forecast",)}
    print(json.dumps(keep, indent=1, default=float)[:4000])
