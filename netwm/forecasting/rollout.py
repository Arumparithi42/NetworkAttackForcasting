"""K-step forward simulation of one entity (host or network) with Monte-Carlo uncertainty."""
from __future__ import annotations

import numpy as np
import torch

from netwm.models.world_model import p_attack_within


@torch.no_grad()
def forecast_entity(model, x_hist: np.ndarray, K: int, n_samples: int = 32,
                    temperature: float = 1.0, calibrator=None, intervention=None,
                    seed: int | None = 0) -> dict:
    """x_hist: [L, D] normalised history. Returns calibrated horizon probability (mean rollout),
    a Monte-Carlo 5-95 % band, per-step stage probabilities and imagined state trajectories."""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.as_tensor(x_hist, dtype=torch.float32).unsqueeze(0)
    det = model(x, K, intervention=intervention, return_surprise=True)
    p_mean_raw = float(p_attack_within(det["stage_future"], temperature=temperature)[0])
    xs = x.expand(n_samples, -1, -1).contiguous()
    mc = model(xs, K, sample=True, intervention=intervention)
    p_mc = p_attack_within(mc["stage_future"], temperature=temperature).numpy()
    lo, hi = np.percentile(p_mc, [5, 95])
    cal = (lambda v: calibrator.predict(np.atleast_1d(v))) if calibrator is not None else (lambda v: np.atleast_1d(v))
    p_cal, lo_cal, hi_cal = (float(cal(p_mean_raw)[0]), float(cal(lo)[0]), float(cal(hi)[0]))
    lo_cal, hi_cal = min(lo_cal, p_cal), max(hi_cal, p_cal)
    stage_steps = (det["stage_future"][0] / temperature).softmax(-1).numpy()          # [K, C]
    stage_steps_mc = (mc["stage_future"] / temperature).softmax(-1).numpy()          # [N, K, C]
    imagined = mc["imagined"].numpy()                                                 # [N, K, D]
    return {
        "p_attack_within_K": p_cal,
        "p_raw": p_mean_raw,
        "band_5_95": (lo_cal, hi_cal),
        "stage_now": (det["stage_now"][0] / temperature).softmax(-1).numpy(),
        "stage_probs_per_step": stage_steps,
        "stage_probs_band": np.percentile(stage_steps_mc, [5, 95], axis=0),          # [2, K, C]
        "state_mean": det["mu"][0].numpy(),                                           # [K, D]
        "state_band": np.percentile(imagined, [5, 95], axis=0),                       # [2, K, D]
        "surprise": float(det["surprise"][0]),
    }


def feature_intervention(feature_idx: list[int], values: list[float]):
    """What-if: clamp selected (normalised) features of every imagined future state."""
    idx = torch.tensor(feature_idx, dtype=torch.long)
    vals = torch.tensor(values, dtype=torch.float32)

    def fn(s: torch.Tensor, k: int) -> torch.Tensor:
        s = s.clone()
        s[:, idx] = vals
        return s

    return fn
