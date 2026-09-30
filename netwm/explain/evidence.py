"""Attribution -> sentences an analyst can check against the raw traffic numbers."""
from __future__ import annotations

import numpy as np

from netwm.features.feature_spec import DESCRIPTIONS

RATIO_WORDS = ("ratio", "svc_", "share")


def _fmt(name: str, v: float) -> str:
    if abs(v) < 5e-3:
        return "0"
    if any(w in name for w in RATIO_WORDS) and abs(v) <= 1.0:
        return f"{100 * v:.0f}%"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f}M"
    if abs(v) >= 1e4:
        return f"{v / 1e3:.0f}k"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def evidence_sentences(top_features, raw_hist: np.ndarray, raw_baseline: np.ndarray,
                       feature_names: list[str], last: int = 3) -> list[dict]:
    """top_features: [(name, attribution)], raw_hist [L, D] unscaled, raw_baseline [D]."""
    out = []
    for name, score in top_features:
        j = feature_names.index(name)
        recent = raw_hist[-last:, j]
        desc = DESCRIPTIONS.get(name, name)
        trend = " → ".join(_fmt(name, v) for v in recent)
        normal = raw_baseline[j]
        cur = recent[-1]
        tol = 0.1 * max(abs(normal), 1e-3)
        direction = "above normal" if cur > normal + tol else ("below normal" if cur < normal - tol else "near normal")
        text = (f"{desc[0].upper() + desc[1:]}: {trend} over the last {last} windows "
                f"(normal: {_fmt(name, normal)}; now {direction})")
        out.append({"feature": name, "attribution": round(score, 4), "text": text, "direction": direction,
                    "recent": [float(v) for v in recent], "normal": float(raw_baseline[j])})
    return out
