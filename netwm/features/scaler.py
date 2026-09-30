"""log1p + robust scaling fitted on benign TRAIN windows only; clip to [-clip, clip]."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from netwm.features.feature_spec import BOUNDED, LOG1P


class RobustScaler:
    def __init__(self, features: list[str], clip: float = 5.0):
        self.features = list(features)
        self.clip = clip
        self.median: dict[str, float] = {}
        self.iqr: dict[str, float] = {}

    def _pre(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].to_numpy(dtype=np.float64, copy=True)
        for j, f in enumerate(self.features):
            if f in LOG1P:
                x[:, j] = np.log1p(np.clip(x[:, j], 0, None))
        return x

    def fit(self, df: pd.DataFrame) -> "RobustScaler":
        x = self._pre(df)
        med = np.median(x, axis=0)
        q75, q25 = np.percentile(x, [75, 25], axis=0)
        iqr = q75 - q25
        std = x.std(axis=0)
        for j, f in enumerate(self.features):
            if f in BOUNDED:
                self.median[f], self.iqr[f] = 0.0, 1.0
            else:
                scale = iqr[j] if iqr[j] > 1e-6 else (std[j] if std[j] > 1e-6 else 1.0)
                self.median[f], self.iqr[f] = float(med[j]), float(scale)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._pre(df)
        med = np.array([self.median[f] for f in self.features])
        iqr = np.array([self.iqr[f] for f in self.features])
        return np.clip((x - med) / iqr, -self.clip, self.clip).astype(np.float32)

    def inverse_raw(self, z: np.ndarray) -> np.ndarray:
        """Normalised values -> original units (approximate where clipping happened)."""
        med = np.array([self.median[f] for f in self.features])
        iqr = np.array([self.iqr[f] for f in self.features])
        x = z * iqr + med
        for j, f in enumerate(self.features):
            if f in LOG1P:
                x[..., j] = np.expm1(x[..., j])
        return x

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"features": self.features, "clip": self.clip, "median": self.median,
                       "iqr": self.iqr}, fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "RobustScaler":
        with open(path) as fh:
            d = json.load(fh)
        s = cls(d["features"], d["clip"])
        s.median, s.iqr = d["median"], d["iqr"]
        return s
