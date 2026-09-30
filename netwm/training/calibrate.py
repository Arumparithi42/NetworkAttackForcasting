"""Probability calibration fitted on the validation split (never on test)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> float:
    """Single temperature T minimising NLL of softmax(logits / T) on non-ignored labels."""
    keep = y >= 0
    if keep.sum() < 10:
        return 1.0
    lg = torch.tensor(logits[keep], dtype=torch.float64)
    yt = torch.tensor(y[keep], dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))


class ProbCalibrator:
    """Isotonic regression p_raw -> p_calibrated (monotone, so quantile bands stay ordered)."""

    def __init__(self):
        self.iso = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "ProbCalibrator":
        if len(np.unique(y)) < 2:
            return self                              # cannot calibrate without both classes
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return p if self.iso is None else np.asarray(self.iso.predict(p), dtype=np.float64)
