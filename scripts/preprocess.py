"""Raw data -> canonical flows -> per-day states + labels.

    python scripts/preprocess.py --config configs/cic2018_network.yaml
    python scripts/preprocess.py --config configs/cic2018_host.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netwm.config import load_config  # noqa: E402
from netwm.data.prepare import prepare_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--days", nargs="*", default=None)
    a = ap.parse_args()
    prepare_all(load_config(a.config), a.days)
