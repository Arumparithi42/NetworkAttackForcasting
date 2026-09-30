"""EDA: labels over time per day, raw vs after the clock/timezone fixes (run as a script or with
'# %%' cells in VS Code / Jupytext).

    python notebooks/01_eda_label_timeline.py
"""
# %%
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from netwm.io.load_cic import load_cic_csv  # noqa: E402

RAW = "data/raw/cic2018"

# %% raw hour histogram -> reveals the 12-hour clock (only hours 1-5 and 8-12 exist)
for f in sorted(glob.glob(f"{RAW}/*.csv")):
    if "20-02" in f:
        continue
    d = pd.read_csv(f, usecols=["Timestamp", "Label"], dtype=str)
    d = d[d["Label"] != "Label"]
    ts = pd.to_datetime(d["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    print(Path(f).name[:22], "rows", len(d), "hours present:", sorted(ts.dt.hour.dropna().unique().astype(int)))

# %% after the fixes: benign / attack flows per 20 minutes (UTC)
for f in sorted(glob.glob(f"{RAW}/*.csv")):
    if "20-02" in f:
        continue
    df = load_cic_csv(f)
    tab = pd.crosstab(df["ts"].dt.floor("20min"), df["label_raw"] != "Benign")
    print("\n==", Path(f).name[:22])
    print(" ".join(f"{i:%H:%M}:{int(r.get(False, 0))}/{int(r.get(True, 0))}" for i, r in tab.iterrows()))
