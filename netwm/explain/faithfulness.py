"""Deletion test: do the features that IG calls important actually drive the forecast?

For each sample: replace the top-k attributed features (all time steps) by their benign-baseline
values and record the drop in P(attack within K); compare with replacing k random features.
"""
from __future__ import annotations

import numpy as np
import torch

from netwm.explain.ig import explain
from netwm.models.world_model import p_attack_within


@torch.no_grad()
def _p(model, x, K, T):
    return float(p_attack_within(model(torch.as_tensor(x[None], dtype=torch.float32), K)["stage_future"],
                                 temperature=T)[0])


def deletion_test(model, samples: list[np.ndarray], baseline: np.ndarray, feature_names, K: int,
                  temperature: float = 1.0, k: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    drops_top, drops_rand = [], []
    for x in samples:
        p0 = _p(model, x, K, temperature)
        e = explain(model, x, baseline, feature_names, K, temperature, top_k=k)
        top = [feature_names.index(n) for n, _ in e["top_features"]][:k]
        if not top:
            continue
        xt = x.copy()
        xt[:, top] = baseline[top]
        rand = rng.choice(len(feature_names), len(top), replace=False)
        xr = x.copy()
        xr[:, rand] = baseline[rand]
        drops_top.append(p0 - _p(model, xt, K, temperature))
        drops_rand.append(p0 - _p(model, xr, K, temperature))
    return {"n": len(drops_top), "k": k,
            "mean_drop_top_k": float(np.mean(drops_top)) if drops_top else float("nan"),
            "mean_drop_random_k": float(np.mean(drops_rand)) if drops_rand else float("nan")}
