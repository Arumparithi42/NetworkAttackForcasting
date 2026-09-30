"""Builds the OBSERVED / INFERRED / PREDICTED blocks of a forecast record (contract C5)."""
from __future__ import annotations

from pathlib import Path

import yaml

from netwm.attack_mapping.indicator_rules import evaluate_rules

_SUBSET = yaml.safe_load(open(Path(__file__).with_name("attack_subset.yaml")))
TACTICS = _SUBSET["tactics"]
TECHNIQUES = _SUBSET["techniques"]
ATTACK_VERSION = _SUBSET["attack_version"]

# model stage -> ATT&CK tactic(s). RECON_DISCOVERY is Reconnaissance when the activity comes from
# outside and Discovery when an internal host performs it.
STAGE_TACTICS = {
    "RECON_DISCOVERY": ["TA0043", "TA0007"],
    "INITIAL_ACCESS_ATTEMPT": ["TA0001", "TA0006"],
    "LATERAL_MOVEMENT": ["TA0008"],
    "COMMAND_AND_CONTROL": ["TA0011"],
    "IMPACT": ["TA0040"],
    "EXFILTRATION": ["TA0010"],
}
STAGE_LABEL = {
    "BENIGN": "Benign",
    "RECON_DISCOVERY": "Reconnaissance / Discovery",
    "INITIAL_ACCESS_ATTEMPT": "Initial-access / credential-access attempt",
    "LATERAL_MOVEMENT": "Lateral movement",
    "COMMAND_AND_CONTROL": "Command and control",
    "IMPACT": "Impact (denial of service)",
    "EXFILTRATION": "Exfiltration",
}


def tactic_refs(stage: str | None) -> list[dict]:
    if not stage:
        return []
    return [{"id": t, "name": TACTICS[t]} for t in STAGE_TACTICS.get(stage, [])]


def candidate_techniques(features: dict, mode: str) -> list[dict]:
    out = []
    for hit in evaluate_rules(features, mode):
        t = TECHNIQUES.get(hit["technique"], {})
        out.append({
            "id": hit["technique"], "name": t.get("name", "?"),
            "tactics": [{"id": x, "name": TACTICS.get(x, x)} for x in t.get("tactics", [])],
            "strength": hit["strength"], "wording": "behaviour consistent with",
            "evidence": hit["evidence"], "note": hit["note"],
        })
    return out


def build_record(*, entity: str, window_end: str, observed: dict, current_stage: dict,
                 current_features: dict, mode: str, predicted: dict, explanation: dict | None,
                 model_info: dict) -> dict:
    stage = predicted.get("most_likely_future_stage")
    predicted = dict(predicted)
    predicted["attack_tactics"] = tactic_refs(stage)
    predicted["stage_label"] = STAGE_LABEL.get(stage) if stage else None
    now_stage = max(current_stage, key=current_stage.get)
    return {
        "window_end": window_end,
        "entity": entity,
        "model": model_info,
        "attack_version": ATTACK_VERSION,
        "observed": observed,
        "inferred": {
            "current_stage_probs": {k: round(float(v), 4) for k, v in current_stage.items()},
            "current_stage": now_stage,
            "current_stage_tactics": tactic_refs(now_stage) if now_stage != "BENIGN" else [],
            "candidate_techniques": candidate_techniques(current_features, mode),
            "wording": "interpretation of the current state, not proof",
        },
        "predicted": predicted,
        "explanation": explanation,
        "not_supported": ["LATERAL_MOVEMENT", "EXFILTRATION"],
        "not_supported_reason": "no flow-level ground truth for these stages in the training data",
    }
