"""Block bootstrap: resample whole blocks (e.g. an entity-day, or 1-hour chunks of a series),
because consecutive windows are strongly correlated - resampling windows would give falsely
narrow confidence intervals."""
from __future__ import annotations

import numpy as np


def block_bootstrap_ci(metric, groups: np.ndarray, *arrays, n: int = 1000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.nonzero(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        v = metric(*[a[idx] for a in arrays])
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < n // 10:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
