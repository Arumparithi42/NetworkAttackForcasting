"""Integrated Gradients on the world model's forecast (through the K-step rollout)."""
from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients

from netwm.models.world_model import p_attack_within


def _target_fn(model, K: int, temperature: float, stage_idx: int | None = None, step: int | None = None):
    def f(x):
        out = model(x, K)
        if stage_idx is None:
            return p_attack_within(out["stage_future"], temperature=temperature)
        probs = (out["stage_future"] / temperature).softmax(-1)
        return probs[:, step, stage_idx]
    return f


def explain(model, x_hist: np.ndarray, benign_baseline: np.ndarray, feature_names: list[str],
            K: int, temperature: float = 1.0, top_k: int = 5, n_steps: int = 32,
            stage_idx: int | None = None, step: int | None = None) -> dict:
    """x_hist [L, D]; benign_baseline [D] (normalised median benign state).
    Target: P(attack within K) (default) or P(stage=stage_idx at t+step+1)."""
    model.eval()
    x = torch.as_tensor(x_hist, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    base = torch.as_tensor(benign_baseline, dtype=torch.float32).expand_as(x[0]).unsqueeze(0)
    ig = IntegratedGradients(_target_fn(model, K, temperature, stage_idx, step))
    attr, delta = ig.attribute(x, baselines=base, n_steps=n_steps, return_convergence_delta=True)
    attr = attr[0].detach().numpy()                               # [L, D]
    per_feature = attr.sum(0)
    per_time = np.abs(attr).sum(1)
    order = np.argsort(-per_feature)
    top = [(feature_names[i], float(per_feature[i])) for i in order[:top_k] if per_feature[i] > 0]
    against = [(feature_names[i], float(per_feature[i])) for i in np.argsort(per_feature)[:3]
               if per_feature[i] < 0]
    return {
        "method": "integrated_gradients",
        "target": "P(attack within K)" if stage_idx is None else f"P(stage {stage_idx} at t+{step + 1})",
        "baseline": "median benign state",
        "top_features": top,
        "features_against": against,
        "time_importance": (per_time / max(per_time.sum(), 1e-12)).tolist(),
        "heatmap": attr.tolist(),
        "convergence_delta": float(delta),
    }
