"""YAML config loading with deep-merge (default.yaml <- dataset/experiment config <- CLI overrides)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str | Path, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            p = ROOT / p
        cfg = deep_merge(cfg, yaml.safe_load(open(p)))
    cfg = deep_merge(cfg, overrides or {})
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
