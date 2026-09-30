"""Full time grid + causal temporal features (block I).

Every feature at window t is computed from windows <= t only (tested in tests/test_causality.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from netwm.features import feature_spec as fs


def _ewma_z(x: pd.Series, alpha: float) -> pd.Series:
    """z-score of x[t] against an exponentially weighted mean/variance of x[<t]."""
    lx = np.log1p(x.clip(lower=0))
    m = lx.ewm(alpha=alpha, adjust=False).mean().shift(1)
    v = lx.ewm(alpha=alpha, adjust=False).var(bias=True).shift(1)
    z = (lx - m) / np.sqrt(v.fillna(0) + 0.05)
    return z.fillna(0.0).clip(-10, 10)


def _delta(x: pd.Series) -> pd.Series:
    lx = np.log1p(x.clip(lower=0))
    return lx.diff().fillna(0.0)


def complete_grid(feats: pd.DataFrame, mode: str, windows: np.ndarray, entities=None,
                  global_cols=()) -> pd.DataFrame:
    """Reindex to every window of the day (and every selected host). Idle host-windows get zeros
    (active = 0) but keep the window's network-wide (global) values."""
    if mode == "network":
        out = feats.set_index("window").reindex(windows).fillna(0.0)
        out.index.name = "window"
        return out.reset_index()
    entities = entities if entities is not None else feats["host"].unique()
    idx = pd.MultiIndex.from_product([entities, windows], names=["host", "window"])
    local_cols = [c for c in feats.columns if c not in ("window", "host", *global_cols)]
    loc = feats.set_index(["host", "window"])[local_cols].reindex(idx).fillna(0.0)
    if global_cols:
        glob = feats.groupby("window")[list(global_cols)].first().reindex(windows).fillna(0.0)
        loc = loc.join(glob, on="window")
    return loc.reset_index()


def add_temporal(df: pd.DataFrame, mode: str, alpha: float = 0.1) -> pd.DataFrame:
    """Add deltas and EWMA z-scores, computed per entity in time order (causal)."""
    deltas = fs.HOST_DELTAS if mode == "host" else fs.NET_DELTAS
    zs = fs.HOST_ZSCORES if mode == "host" else fs.NET_ZSCORES
    df = df.sort_values((["host"] if mode == "host" else []) + ["window"]).reset_index(drop=True)
    if mode == "host":
        g = df.groupby("host", sort=False)
        for name, src in deltas:
            df[name] = g[src].transform(_delta)
        for name, src in zs:
            df[name] = g[src].transform(lambda s: _ewma_z(s, alpha))
    else:
        for name, src in deltas:
            df[name] = _delta(df[src])
        for name, src in zs:
            df[name] = _ewma_z(df[src], alpha)
    return df


def build_states(feats: pd.DataFrame, mode: str, day: str, windows: np.ndarray,
                 feature_list: list[str], alpha: float = 0.1, entities=None) -> pd.DataFrame:
    """feats (active windows only) -> complete, ordered state table for one day."""
    global_cols = tuple(fs.GLOBAL_J) if mode == "host" else ()
    grid = complete_grid(feats, mode, windows, entities, global_cols)
    grid = add_temporal(grid, mode, alpha)
    for c in feature_list:                       # e.g. packet block missing for CSV sources
        if c not in grid.columns:
            grid[c] = 0.0
    grid["day"] = day
    if mode == "network":
        grid["host"] = "network"
    cols = ["day", "window", "host"] + list(feature_list)
    return grid[cols].astype({c: np.float32 for c in feature_list})
