"""Turn a rollout into analyst-facing risk numbers (never 'an attack WILL happen')."""
from __future__ import annotations

import numpy as np

STATEMENT = ("The model estimates an increased probability of a future attack trajectory for "
             "this {entity} based on observed network behaviour.")
LOW_STATEMENT = ("No elevated probability of an attack trajectory is estimated for this {entity} "
                 "within the forecast horizon.")


def cumulative_attack(stage_probs_per_step: np.ndarray, benign_idx: int = 0, calibrate=None):
    """P(attack within the first k windows), k = 1..K; optionally mapped through the same
    calibrator as the headline probability (the k = K value then equals the headline)."""
    cum = 1 - np.cumprod(stage_probs_per_step[:, benign_idx])
    return np.asarray(calibrate(cum)) if calibrate is not None else cum


def horizon_step(stage_probs_per_step: np.ndarray, threshold: float, benign_idx: int = 0,
                 calibrate=None):
    """First step k (1-based) at which the cumulative attack probability reaches the threshold."""
    cum = cumulative_attack(stage_probs_per_step, benign_idx, calibrate)
    crossed = np.nonzero(cum >= threshold)[0]
    return int(crossed[0]) + 1 if crossed.size else None


def risk_level(p: float, threshold: float) -> str:
    if p >= max(threshold, 0.5):
        return "HIGH"
    if p >= threshold:
        return "ELEVATED"
    if p >= threshold / 2:
        return "GUARDED"
    return "LOW"


def confidence_label(band: tuple[float, float]) -> str:
    width = band[1] - band[0]
    return "HIGH" if width < 0.15 else ("MEDIUM" if width < 0.35 else "LOW")


def surprise_percentile(surprise: float, ref: list[float]) -> float:
    ref = np.asarray(ref)
    return float(np.searchsorted(np.sort(ref), surprise) / max(len(ref), 1))


def summarise(fc: dict, classes: list[str], threshold: float, entity: str = "host",
              calibrate=None) -> dict:
    steps = fc["stage_probs_per_step"]
    p = fc["p_attack_within_K"]
    attack_cls = [i for i, c in enumerate(classes) if c != "BENIGN"]
    best_k, best_c = np.unravel_index(np.argmax(steps[:, attack_cls]), (steps.shape[0], len(attack_cls)))
    predicted_stage = classes[attack_cls[best_c]] if p >= threshold else None
    return {
        "p_attack_within_K": round(p, 4),
        "band_5_95": [round(fc["band_5_95"][0], 4), round(fc["band_5_95"][1], 4)],
        "risk_level": risk_level(p, threshold),
        "confidence": confidence_label(fc["band_5_95"]),
        "threshold": round(threshold, 4),
        "first_crossing_step": horizon_step(steps, threshold, calibrate=calibrate),
        "cumulative_by_step": [round(float(v), 4) for v in cumulative_attack(steps, calibrate=calibrate)],
        "most_likely_future_stage": predicted_stage,
        "most_likely_step": int(best_k) + 1 if predicted_stage else None,
        "stage_probs": {f"t+{k + 1}": {c: round(float(steps[k, i]), 4) for i, c in enumerate(classes)}
                        for k in range(steps.shape[0])},
        "statement": (STATEMENT if p >= threshold else LOW_STATEMENT).format(entity=entity),
    }
