"""Dataset labels -> attack stages (contract C2).

Three kinds of "label" are kept strictly apart:
  * label_raw   : the dataset's own per-flow label (ground truth as published)
  * flow_stage  : our documented, deterministic re-grouping of label_raw (this module)
  * predictions : model outputs (never written back into these tables)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STAGES = ["BENIGN", "RECON_DISCOVERY", "INITIAL_ACCESS_ATTEMPT", "LATERAL_MOVEMENT",
          "COMMAND_AND_CONTROL", "IMPACT", "EXFILTRATION"]
STAGE_ID = {s: i for i, s in enumerate(STAGES)}
BENIGN = 0
AMBIGUOUS = -1   # a few malicious flows only: masked in the loss
EXCLUDED = -2    # unknown traffic (e.g. CTU-13 background): masked
# kill-chain order used to pick "the most advanced stage" of a window
STAGE_ORDER = [STAGE_ID[s] for s in ["RECON_DISCOVERY", "INITIAL_ACCESS_ATTEMPT",
                                      "LATERAL_MOVEMENT", "COMMAND_AND_CONTROL", "IMPACT",
                                      "EXFILTRATION"]]
# stages that have flow-level ground truth in the datasets we use -> trained model classes
TRAINED_STAGES = ["BENIGN", "RECON_DISCOVERY", "INITIAL_ACCESS_ATTEMPT", "COMMAND_AND_CONTROL",
                  "IMPACT"]
NOT_SUPPORTED = ["LATERAL_MOVEMENT", "EXFILTRATION"]

DEFAULT_MAPPING = Path(__file__).with_name("label_to_stage.yaml")


def normalise_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def load_mapping(path: str | Path = DEFAULT_MAPPING) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def map_flow_stage(label_raw: pd.Series, src_internal: pd.Series | None = None,
                   mapping: dict | None = None) -> pd.Series:
    """Per-flow stage id. Raises on labels that are not in the mapping (never guess silently)."""
    mapping = mapping or load_mapping()
    table = mapping["labels"]
    norm = label_raw.astype(str).map(normalise_label)
    unknown = sorted(set(norm.unique()) - set(table))
    if unknown:
        raise ValueError(f"Unmapped dataset labels: {unknown}. Add them to label_to_stage.yaml")
    name = norm.map(table)
    infil = (name == "INFILTRATION_RULE").to_numpy()
    if infil.any():
        if src_internal is None or src_internal.isna().all():
            fill = np.full(len(name), mapping.get("infiltration_unknown_src", "RECON_DISCOVERY"),
                           dtype=object)
        else:
            fill = np.where(src_internal.fillna(False).to_numpy(), "RECON_DISCOVERY",
                            "INITIAL_ACCESS_ATTEMPT")
        name = pd.Series(np.where(infil, fill, name.to_numpy()), index=name.index)
    ids = name.map(lambda s: EXCLUDED if s == "EXCLUDED" else STAGE_ID[s])
    return ids.astype(np.int8)


def window_stage(df: pd.DataFrame, keys: list[str], min_flows: int = 3,
                 stage_col: str = "flow_stage") -> pd.DataFrame:
    """Aggregate per-flow stages to one stage per group (e.g. (window) or (window, host)).

    Rule: the most advanced stage (kill-chain order) with >= min_flows flows; windows with some
    malicious flows but none reaching the threshold are AMBIGUOUS (-1); windows containing only
    EXCLUDED traffic are EXCLUDED (-2); otherwise BENIGN.
    Also returns n_mal (malicious flow count) and one count column per stage for inspection.
    """
    s = df[stage_col].astype(np.int16)
    counts = pd.crosstab([df[k] for k in keys], s)
    out = pd.DataFrame(index=counts.index)
    mal_cols = [c for c in counts.columns if c > 0]
    n_mal = counts[mal_cols].sum(axis=1) if mal_cols else pd.Series(0, index=counts.index)
    n_benign = counts[0] if 0 in counts.columns else pd.Series(0, index=counts.index)
    n_excl = counts[EXCLUDED] if EXCLUDED in counts.columns else pd.Series(0, index=counts.index)
    stage = pd.Series(BENIGN, index=counts.index, dtype=np.int8)
    stage[(n_benign == 0) & (n_excl > 0) & (n_mal == 0)] = EXCLUDED
    stage[n_mal > 0] = AMBIGUOUS
    for sid in STAGE_ORDER:
        if sid in counts.columns:
            stage[counts[sid] >= min_flows] = sid
    out["stage"] = stage
    out["n_mal"] = n_mal.astype(np.int32)
    for sid in mal_cols:
        out[f"n_{STAGES[sid].lower()}"] = counts[sid].astype(np.int32)
    return out.reset_index()


def future_within(stage: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """For a single time series of stages: y_within[t] = any attack in (t, t+K], and a validity
    mask (unknown if the future contains only benign + masked windows, or runs past the end)."""
    T = len(stage)
    y = np.zeros(T, dtype=np.float32)
    valid = np.zeros(T, dtype=np.float32)
    for t in range(T - K):
        fut = stage[t + 1:t + 1 + K]
        if (fut > 0).any():
            y[t], valid[t] = 1.0, 1.0
        elif (fut == BENIGN).all():
            valid[t] = 1.0
    return y, valid
