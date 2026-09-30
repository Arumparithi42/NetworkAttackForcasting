"""Early-warning evaluation: can a forecast made BEFORE an attack onset see it coming?

Onset T (per entity time series): first attack window after >= `quiet` benign windows.
A forecast made at T - l (using data <= T - l) is a hit if P(attack within K) >= threshold.
Lead 0 = the onset window itself (detection); leads >= 1 are genuine early warnings (need K >= l).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def onsets(y: np.ndarray, quiet: int = 5) -> list[int]:
    out = []
    for T in np.nonzero(y > 0)[0]:
        if T >= quiet and (y[T - quiet:T] == 0).all():
            out.append(int(T))
    return out


def early_warning(meta: pd.DataFrame, p: np.ndarray, series_y: dict, thr: float,
                  leads=(0, 1, 2, 3), quiet: int = 5) -> tuple[dict, pd.DataFrame]:
    """meta: rows with columns series (id), t; p aligned with meta.
    series_y: series id -> full class array (so onsets are found on the whole series)."""
    lookup = {(s, t): v for s, t, v in zip(meta["series"].to_numpy(), meta["t"].to_numpy(), p)}
    rows = []
    for sid, y in series_y.items():
        for T in onsets(y, quiet):
            rec = {"series": sid, "onset_t": T, "stage": int(y[T])}
            for lead in leads:
                v = lookup.get((sid, T - lead))
                rec[f"p_lead{lead}"] = v
                rec[f"hit_lead{lead}"] = None if v is None else bool(float(v) >= float(thr))
            rows.append(rec)
    df = pd.DataFrame(rows)
    summary = {}
    for lead in leads:
        col = f"hit_lead{lead}"
        h = df[col].dropna() if len(df) else pd.Series(dtype=bool)
        summary[lead] = {"recall": float(h.mean()) if len(h) else float("nan"),
                         "n_onsets": int(len(h))}
    return summary, df
