"""Verify (and optionally anchor) an evidence ledger.

    python scripts/verify_ledger.py artifacts/ledger.sqlite
    python scripts/verify_ledger.py artifacts/ledger.sqlite --anchor http://127.0.0.1:8545
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netwm.ledger.hashchain import EvidenceLedger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ledger")
ap.add_argument("--anchor", default=None, help="RPC URL of a local EVM node (optional)")
a = ap.parse_args()
led = EvidenceLedger(a.ledger)
v = led.verify()
print(("OK" if v["ok"] else "TAMPERED"), v)
print("merkle root:", led.merkle())
if a.anchor:
    from netwm.ledger.anchor_evm import anchor_root
    print("anchored in tx", anchor_root(led, a.anchor))
sys.exit(0 if v["ok"] else 1)
