"""Tamper-evident evidence ledger for forecasts/alerts (optional component).

* append-only SQLite table; each record stores SHA-256(payload), the previous record hash and
  its own hash -> editing or deleting any past record breaks verification from that point on;
* optional Ed25519 signature of each record hash (sensor identity) via `cryptography`;
* Merkle root over a range of records, for periodic anchoring to an external party
  (netwm/ledger/anchor_evm.py). A hash chain alone cannot stop someone who rewrites the WHOLE
  database; signatures + an externally held anchor close that gap.

The forecasting system works identically with the ledger disabled.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

GENESIS = "0" * 64


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def record_hash(prev_hash: str, payload_hash: str, ts: float, seq: int) -> str:
    return sha256(f"{seq}|{prev_hash}|{payload_hash}|{ts!r}".encode())


def merkle_root(hashes: list[str]) -> str:
    if not hashes:
        return GENESIS
    level = [bytes.fromhex(h) for h in hashes]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


class Signer:
    """Ed25519 signer; key stored next to the ledger (demo). Use an HSM/KMS in production."""

    def __init__(self, key_path: str | Path):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key_path = Path(key_path)
        if key_path.exists():
            self.key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        else:
            self.key = Ed25519PrivateKey.generate()
            key_path.write_bytes(self.key.private_bytes(serialization.Encoding.PEM,
                                                        serialization.PrivateFormat.PKCS8,
                                                        serialization.NoEncryption()))
        self.public = self.key.public_key()

    def sign(self, msg: str) -> str:
        return self.key.sign(msg.encode()).hex()

    def verify(self, msg: str, sig: str) -> bool:
        from cryptography.exceptions import InvalidSignature
        try:
            self.public.verify(bytes.fromhex(sig), msg.encode())
            return True
        except (InvalidSignature, ValueError):
            return False


class EvidenceLedger:
    def __init__(self, path: str | Path = "artifacts/ledger.sqlite", sign: bool = True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS ledger (
            seq INTEGER PRIMARY KEY, ts REAL, kind TEXT, payload TEXT, payload_hash TEXT,
            prev_hash TEXT, record_hash TEXT, signature TEXT)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS anchors (
            id INTEGER PRIMARY KEY AUTOINCREMENT, from_seq INTEGER, to_seq INTEGER,
            merkle_root TEXT, target TEXT, receipt TEXT, ts REAL)""")
        self.db.commit()
        self.signer = None
        if sign:
            try:
                self.signer = Signer(path.with_suffix(".key.pem"))
            except ImportError:
                self.signer = None

    def head(self) -> tuple[int, str]:
        row = self.db.execute("SELECT seq, record_hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        return (row[0], row[1]) if row else (0, GENESIS)

    def append(self, payload: dict, kind: str = "forecast") -> dict:
        body = canonical(payload)
        ph = sha256(body)
        last_seq, prev = self.head()
        seq, ts = last_seq + 1, time.time()
        rh = record_hash(prev, ph, ts, seq)
        sig = self.signer.sign(rh) if self.signer else ""
        self.db.execute("INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (seq, ts, kind, body.decode(), ph, prev, rh, sig))
        self.db.commit()
        return {"seq": seq, "record_hash": rh, "prev_hash": prev, "signed": bool(sig)}

    def records(self, limit: int = 100) -> list[dict]:
        cur = self.db.execute("SELECT seq, ts, kind, payload_hash, prev_hash, record_hash, signature "
                              "FROM ledger ORDER BY seq DESC LIMIT ?", (limit,))
        cols = ["seq", "ts", "kind", "payload_hash", "prev_hash", "record_hash", "signature"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def verify(self) -> dict:
        """Recompute the chain. Returns {'ok': bool, 'first_bad_seq': int|None, 'n': int, 'reason'}."""
        prev, n = GENESIS, 0
        for seq, ts, payload, ph, prev_hash, rh, sig in self.db.execute(
                "SELECT seq, ts, payload, payload_hash, prev_hash, record_hash, signature "
                "FROM ledger ORDER BY seq"):
            n += 1
            if seq != n:
                return {"ok": False, "first_bad_seq": seq, "n": n, "reason": "missing record"}
            if sha256(payload.encode()) != ph:
                return {"ok": False, "first_bad_seq": seq, "n": n, "reason": "payload modified"}
            if prev_hash != prev or record_hash(prev, ph, ts, seq) != rh:
                return {"ok": False, "first_bad_seq": seq, "n": n, "reason": "chain broken"}
            if sig and self.signer and not self.signer.verify(rh, sig):
                return {"ok": False, "first_bad_seq": seq, "n": n, "reason": "bad signature"}
            prev = rh
        return {"ok": True, "first_bad_seq": None, "n": n, "reason": ""}

    def merkle(self, from_seq: int = 1, to_seq: int | None = None) -> str:
        to_seq = to_seq or self.head()[0]
        hs = [r[0] for r in self.db.execute(
            "SELECT record_hash FROM ledger WHERE seq BETWEEN ? AND ? ORDER BY seq", (from_seq, to_seq))]
        return merkle_root(hs)

    def record_anchor(self, from_seq: int, to_seq: int, root: str, target: str, receipt: str):
        self.db.execute("INSERT INTO anchors (from_seq, to_seq, merkle_root, target, receipt, ts) "
                        "VALUES (?, ?, ?, ?, ?, ?)", (from_seq, to_seq, root, target, receipt, time.time()))
        self.db.commit()
