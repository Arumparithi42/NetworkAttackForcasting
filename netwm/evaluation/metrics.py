"""Forecast metrics. Every function tolerates degenerate inputs (single class) by returning NaN."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             roc_auc_score)


def threshold_at_fpr(p_neg: np.ndarray, target_fpr: float = 0.01) -> float:
    """Smallest threshold whose false-positive rate on `p_neg` is <= target_fpr."""
    p_neg = np.asarray(p_neg, dtype=np.float64)
    if len(p_neg) == 0:
        return 0.5
    thr = float(np.quantile(p_neg, 1 - target_fpr, method="higher"))
    return float(np.nextafter(thr, 1.0)) if (p_neg >= thr).mean() > target_fpr else thr


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def binary_report(p: np.ndarray, y: np.ndarray, thr: float) -> dict:
    # float64 everywhere: calibrated scores have many exact ties, and a float32 comparison would
    # silently undo the tie-breaking in threshold_at_fpr
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(int)
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "positives": int(y.sum()), "threshold": float(thr),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))) if len(y) else float("nan"),
        "ece": ece(np.clip(p, 0, 1), y),
    }


def pr_auc(p, y) -> float:
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def stage_macro_f1_per_step(probs: np.ndarray, y_future: np.ndarray) -> list[float]:
    """probs [N, K, C], y_future [N, K] (-1 ignored) -> macro-F1 per step k (classes present)."""
    out = []
    for k in range(y_future.shape[1]):
        m = y_future[:, k] >= 0
        if m.sum() == 0:
            out.append(float("nan"))
            continue
        pred = probs[m, k].argmax(-1)
        labels = np.unique(y_future[m, k])
        out.append(float(f1_score(y_future[m, k], pred, labels=labels, average="macro",
                                  zero_division=0)))
    return out
