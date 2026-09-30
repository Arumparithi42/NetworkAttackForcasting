"""Leakage-free split roles per (day, window).

roles: 'train', 'val', 'test', or '' (unused). A sequence (history t-L+1..t plus future t+1..t+K)
is only used if EVERY window it touches has the same role -> automatic purge gaps between
roles (no sequence straddles a boundary).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN, VAL, TEST, UNUSED = "train", "val", "test", ""


def window_roles(day: str, windows: np.ndarray, cfg: dict, window_s: int) -> np.ndarray:
    proto = cfg["protocols"][cfg.get("protocol", "A")]
    roles = np.full(len(windows), UNUSED, dtype=object)
    if day in cfg.get("excluded_days", {}):
        return roles
    split_at = (proto.get("split_days") or {}).get(day)
    if day in proto.get("train", []):
        roles[:] = TRAIN
    elif day in proto.get("test", []):
        roles[:] = TEST
    elif split_at is not None:
        cut = pd.Timestamp(split_at).value // 10**9 // window_s
        roles[windows < cut] = TRAIN
        roles[windows >= cut] = TEST
    else:
        return roles
    v = cfg.get("validation") or {}
    if v:
        first = windows.min()
        block = (windows - first) // int(v.get("block_windows", 60))
        is_val = (block % int(v.get("every", 4))) == int(v.get("offset", 3))
        roles[(roles == TRAIN) & is_val] = VAL
    return roles


def sequence_starts(roles: np.ndarray, role: str, L: int, K: int, first_t: int = 0,
                    stride: int = 1) -> np.ndarray:
    """Indices t such that windows t-L+1 .. t+K all have `role` (and t >= first_t)."""
    T = len(roles)
    ok = (roles == role).astype(np.int32)
    span = L + K
    if T < span:
        return np.array([], dtype=np.int64)
    run = np.convolve(ok, np.ones(span, dtype=np.int32), mode="valid")   # run[i] = sum ok[i:i+span]
    starts = np.nonzero(run == span)[0] + (L - 1)                        # t = i + L - 1
    starts = starts[starts >= max(first_t, L - 1)]
    return starts[::stride] if stride > 1 else starts
