"""CICFlowMeter CSV (CSE-CIC-IDS2018 / CIC-IDS2017) -> canonical flows.

Data-quality fixes applied here (all verified on the real files, see docs/DATA_NOTES.md):
  * repeated header rows inside the CSV are dropped;
  * the 2018 CSV timestamps use a 12-hour clock WITHOUT AM/PM: the capture runs from about 08:00
    to 17:45 local time, so hours < `clock_12h_threshold` are afternoon hours and get +12 h;
  * timestamps are local time (UTC-4, Atlantic Standard Time) -> converted to UTC;
  * garbage rows with 1970 timestamps are dropped;
  * +/-inf -> NaN, negative "Init Fwd Win Byts" (-1 = not observed) -> NaN;
  * microsecond durations/IATs -> seconds.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from netwm.io.schema import CIC2017_MAP, CIC2018_MAP, MICROSECOND_COLUMNS, conform

DAY_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")


def day_from_filename(path: str | Path) -> str:
    """'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv' -> '2018-02-28'."""
    m = DAY_RE.search(Path(path).name)
    if not m:
        return Path(path).stem
    d, mth, y = m.groups()
    return f"{y}-{mth}-{d}"


def _header(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [c.strip() for c in f.readline().strip().split(",")]


def _clean_chunk(raw: pd.DataFrame, colmap: dict, day: str, utc_offset_hours: float,
                 clock_12h_threshold: int | None, ts_format: str | None) -> pd.DataFrame:
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw[raw["Label"] != "Label"]                                   # repeated headers
    df = raw.rename(columns=colmap)[[v for k, v in colmap.items() if k in raw.columns]]

    ts = pd.to_datetime(df["ts"], format=ts_format, dayfirst=True, errors="coerce")
    expected = pd.Timestamp(day) if re.match(r"\d{4}-\d{2}-\d{2}", day) else None
    ok = ts.notna()
    if expected is not None:
        ok &= ts.dt.normalize() == expected
    df, ts = df[ok], ts[ok]
    if clock_12h_threshold is not None:
        ts = ts + pd.to_timedelta((ts.dt.hour < clock_12h_threshold).astype(int) * 12, unit="h")
    ts = ts - pd.to_timedelta(utc_offset_hours, unit="h")                 # local -> UTC
    df = df.assign(ts=ts.dt.tz_localize("UTC"))

    for col in df.columns:
        if col in ("ts", "src_ip", "dst_ip", "label_raw"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in MICROSECOND_COLUMNS:
        if col in df:
            df[col] = df[col] / 1e6
    if "init_fwd_win" in df:
        df.loc[df["init_fwd_win"] < 0, "init_fwd_win"] = np.nan
    df["day"] = day
    df["label_raw"] = df["label_raw"].astype(str).str.strip()
    return conform(df, keep_extra=False)


def resolve_conflicting_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Rows identical in every field except the label, carrying both 'Benign' and an attack label
    (67,973 such pairs on 28-02-2018): keep a single row with the attack label. Keeping both would
    double-count those flows exactly during attack periods (a shortcut a model could learn)."""
    cols = [c for c in df.columns if c != "label_raw"]
    dup = df.duplicated(cols, keep=False)
    if not dup.any():
        return df
    is_benign = df["label_raw"].str.lower() == "benign"
    drop = dup & is_benign & df[cols].duplicated(keep=False)
    # only drop the benign copy when an attack copy of the same flow exists
    attack_keys = pd.MultiIndex.from_frame(df.loc[dup & ~is_benign, cols])
    cand = df.loc[drop, cols]
    has_attack_twin = pd.MultiIndex.from_frame(cand).isin(attack_keys)
    to_drop = cand.index[has_attack_twin]
    df = df.drop(index=to_drop).reset_index(drop=True)
    df.attrs["conflicting_duplicates_resolved"] = int(len(to_drop))
    return df


def load_cic_csv(path: str | Path, flavour: str = "cic2018", utc_offset_hours: float = -4.0,
                 clock_12h_threshold: int | None = 8, chunksize: int = 1_000_000,
                 drop_duplicates: bool = True, day: str | None = None) -> pd.DataFrame:
    """Load one CICFlowMeter CSV into the canonical schema (chunked, works for multi-GB files)."""
    colmap = CIC2018_MAP if flavour == "cic2018" else CIC2017_MAP
    header = _header(path)
    usecols = [c for c in header if c in colmap]
    day = day or day_from_filename(path)
    ts_format = "%d/%m/%Y %H:%M:%S" if flavour == "cic2018" else None
    chunks = []
    reader = pd.read_csv(path, usecols=lambda c: c.strip() in usecols, dtype=str,
                         chunksize=chunksize, encoding_errors="replace")
    for raw in reader:
        chunks.append(_clean_chunk(raw, colmap, day, utc_offset_hours, clock_12h_threshold,
                                   ts_format))
    df = pd.concat(chunks, ignore_index=True)
    if drop_duplicates:
        df = df.drop_duplicates(ignore_index=True)
        df = resolve_conflicting_duplicates(df)
    return df.sort_values("ts", kind="stable", ignore_index=True)
