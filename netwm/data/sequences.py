"""State tables -> per-entity time series -> (history, future) training samples."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from netwm.data.splits import sequence_starts, window_roles
from netwm.labels.build_labels import STAGE_ID, TRAINED_STAGES

# global stage id -> model class index (only stages with ground truth are model classes)
STAGE_TO_CLASS = {STAGE_ID[s]: i for i, s in enumerate(TRAINED_STAGES)}
CLASS_NAMES = list(TRAINED_STAGES)


def to_class(stage: np.ndarray) -> np.ndarray:
    out = np.full(stage.shape, -1, dtype=np.int64)          # -1 = ignored (ambiguous/excluded)
    for sid, cls in STAGE_TO_CLASS.items():
        out[stage == sid] = cls
    return out


@dataclass
class Series:
    day: str
    host: str
    windows: np.ndarray        # [T] absolute window ids
    X: np.ndarray              # [T, D] normalised states
    y: np.ndarray              # [T] model class (-1 ignored)
    roles: np.ndarray          # [T] train/val/test/''
    raw: np.ndarray | None = field(default=None, repr=False)   # [T, D] unscaled (for evidence)


def build_series(states: pd.DataFrame, labels: pd.DataFrame, scaler, cfg: dict,
                 keep_raw: bool = False) -> list[Series]:
    """states: day, window, host, features...; labels: day, window, host, stage."""
    feats = scaler.features
    df = states.merge(labels[["day", "window", "host", "stage"]], on=["day", "window", "host"],
                      how="left")
    df["stage"] = df["stage"].fillna(0).astype(np.int16)
    df = df.sort_values(["day", "host", "window"])
    Xall = scaler.transform(df)
    rawall = df[feats].to_numpy(np.float32) if keep_raw else None
    out = []
    pos = 0
    for (day, host), g in df.groupby(["day", "host"], sort=False):
        n = len(g)
        w = g["window"].to_numpy()
        roles = window_roles(day, w, cfg, cfg["window_s"])
        out.append(Series(day, host, w, Xall[pos:pos + n], to_class(g["stage"].to_numpy()),
                          roles, rawall[pos:pos + n] if keep_raw else None))
        pos += n
    return out


class SequenceDataset(Dataset):
    def __init__(self, series: list[Series], role: str, L: int, K: int, warmup: int = 0,
                 benign_ratio: float | None = None, seed: int = 0):
        self.series, self.L, self.K = series, L, K
        index = []
        for i, s in enumerate(series):
            for t in sequence_starts(s.roles, role, L, K, first_t=warmup):
                index.append((i, int(t)))
        self.index = np.array(index, dtype=np.int64).reshape(-1, 2)
        if benign_ratio is not None and len(self.index):
            touch = np.array([(self.series[i].y[t - L + 1:t + K + 1] > 0).any()
                              for i, t in self.index])
            pos, neg = np.nonzero(touch)[0], np.nonzero(~touch)[0]
            rng = np.random.default_rng(seed)
            n_neg = min(len(neg), max(int(benign_ratio * max(len(pos), 1)), 1))
            keep = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
            self.index = self.index[np.sort(keep)]

        self._cache_targets()

    def _cache_targets(self):
        K = self.K
        n = len(self.index)
        self.y_now = np.zeros(n, np.int64)
        self.y_future = np.zeros((n, K), np.int64)
        for j, (i, t) in enumerate(self.index):
            s = self.series[i]
            self.y_now[j] = s.y[t]
            self.y_future[j] = s.y[t + 1:t + 1 + K]
        self.y_within = (self.y_future > 0).any(1).astype(np.float32)
        self.valid = ((self.y_within == 1) | (self.y_future == 0).all(1)).astype(np.float32)

    def __len__(self):
        return len(self.index)

    def batch(self, idx) -> dict:
        """Tensors for sample indices `idx` (fast path used by the training loop)."""
        L, K = self.L, self.K
        xh = np.stack([self.series[i].X[t - L + 1:t + 1] for i, t in self.index[idx]])
        xf = np.stack([self.series[i].X[t + 1:t + 1 + K] for i, t in self.index[idx]])
        return {
            "x_hist": torch.from_numpy(xh),
            "x_future": torch.from_numpy(xf),
            "y_now": torch.from_numpy(self.y_now[idx]),
            "y_future": torch.from_numpy(self.y_future[idx]),
            "y_within": torch.from_numpy(self.y_within[idx]),
            "within_valid": torch.from_numpy(self.valid[idx]),
        }

    def __getitem__(self, j):
        b = self.batch(np.array([j]))
        return {k: v[0] for k, v in b.items()}

    def iterate(self, batch_size: int, shuffle: bool = False, rng=None):
        order = np.arange(len(self))
        if shuffle:
            (rng or np.random.default_rng()).shuffle(order)
        for s in range(0, len(order), batch_size):
            yield self.batch(order[s:s + batch_size])

    def meta(self) -> pd.DataFrame:
        """One row per sample: day, host, window, t, current class, y_within, valid."""
        return pd.DataFrame({
            "day": [self.series[i].day for i, _ in self.index],
            "host": [self.series[i].host for i, _ in self.index],
            "window": [int(self.series[i].windows[t]) for i, t in self.index],
            "t": self.index[:, 1] if len(self.index) else [],
            "y_now": self.y_now, "y_within": self.y_within, "valid": self.valid,
        })

    def history_array(self) -> np.ndarray:
        L = self.L
        if not len(self.index):
            return np.zeros((0, L, 0), np.float32)
        return np.stack([self.series[i].X[t - L + 1:t + 1] for i, t in self.index])

    def future_array(self) -> np.ndarray:
        K = self.K
        return np.stack([self.series[i].X[t + 1:t + 1 + K] for i, t in self.index])
