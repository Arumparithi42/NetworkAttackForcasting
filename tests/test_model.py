"""Model shapes, loss, overfitting, rollout, calibration and a synthetic 'toy world' check that
temporal dynamics let the world model see a precursor that a current-window classifier cannot."""
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from netwm.data.sequences import SequenceDataset, Series
from netwm.forecasting.rollout import feature_intervention, forecast_entity
from netwm.forecasting.risk import horizon_step, summarise
from netwm.models.world_model import NetworkWorldModel, build_model, p_attack_within
from netwm.training.calibrate import ProbCalibrator, fit_temperature
from netwm.training.losses import world_model_loss
from netwm.training.train_world_model import predict, train_world_model

LAM = {"dyn": 1.0, "now": 0.5, "fut": 1.0, "hor": 1.0}


@pytest.mark.parametrize("cell", ["gru", "lstm"])
@pytest.mark.parametrize("rollout", ["recursive", "direct"])
def test_shapes(cell, rollout):
    m = NetworkWorldModel(59, 5, cell=cell, rollout=rollout, K=5)
    out = m(torch.randn(4, 10, 59), 5, return_surprise=True)
    assert out["stage_now"].shape == (4, 5)
    assert out["mu"].shape == out["logvar"].shape == out["imagined"].shape == (4, 5, 59)
    assert out["stage_future"].shape == (4, 5, 5)
    p = p_attack_within(out["stage_future"])
    assert p.shape == (4,) and ((p >= 0) & (p <= 1)).all()


def test_p_attack_within_matches_definition():
    logits = torch.randn(3, 4, 5)
    probs = logits.softmax(-1)
    expected = 1 - probs[..., 0].prod(1)
    assert torch.allclose(p_attack_within(logits), expected, atol=1e-6)


def _batch(B=32, L=10, K=5, D=8, C=5):
    g = torch.Generator().manual_seed(0)
    return {"x_hist": torch.randn(B, L, D, generator=g), "x_future": torch.randn(B, K, D, generator=g),
            "y_now": torch.randint(0, C, (B,), generator=g), "y_future": torch.randint(0, C, (B, K), generator=g),
            "y_within": torch.randint(0, 2, (B,), generator=g).float(), "within_valid": torch.ones(B)}


def test_loss_handles_all_ignored_labels():
    m = NetworkWorldModel(8, 5)
    b = _batch()
    b["y_now"][:] = -1
    b["y_future"][:] = -1
    loss, parts = world_model_loss(m(b["x_hist"], 5), b, None, LAM)
    assert torch.isfinite(loss)


def test_overfits_one_batch():
    torch.manual_seed(0)
    m = NetworkWorldModel(8, 5, dropout=0.0)
    b = _batch()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        loss, parts = world_model_loss(m(b["x_hist"], 5, x_future=b["x_future"], teacher_prob=1.0), b, None, LAM)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else parts
    assert parts["now"] < 0.2 * first["now"] and parts["fut"] < 0.5 * first["fut"]


def test_rollout_is_deterministic_with_seed_and_band_is_ordered():
    m = build_model(8, 5, {"hidden": 32})
    x = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)
    a = forecast_entity(m, x, 5, n_samples=16, seed=3)
    b = forecast_entity(m, x, 5, n_samples=16, seed=3)
    assert a["p_attack_within_K"] == b["p_attack_within_K"]
    assert a["band_5_95"][0] <= a["p_attack_within_K"] <= a["band_5_95"][1]
    assert a["stage_probs_per_step"].shape == (5, 5)
    s = summarise(a, ["BENIGN", "RECON_DISCOVERY", "INITIAL_ACCESS_ATTEMPT", "COMMAND_AND_CONTROL", "IMPACT"], 0.5)
    assert "WILL" not in s["statement"]
    iv = forecast_entity(m, x, 5, n_samples=4, intervention=feature_intervention([0, 1], [0.0, 0.0]))
    assert iv["state_band"].shape == (2, 5, 8)


def test_horizon_step():
    steps = np.array([[0.9, 0.1], [0.5, 0.5], [0.5, 0.5]])
    assert horizon_step(steps, 0.5) == 2          # 1 - 0.9*0.5 = 0.55
    assert horizon_step(steps, 0.99) is None


def test_calibration_helpers():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p ** 2).astype(float)
    cal = ProbCalibrator().fit(p, y)
    q = cal.predict(np.linspace(0, 1, 11))
    assert (np.diff(q) >= -1e-12).all() and q.dtype == np.float64
    logits = rng.normal(size=(500, 3)) * 5
    labels = logits.argmax(1)
    assert fit_temperature(logits, labels) < 1.0    # confident & correct -> sharpen


def _toy_world(n_series=60, T=120, D=4, seed=0):
    """Feature 0 is noisy with the same marginal distribution before and during precursors, but it
    RAMPS UP steadily for 3 windows before every attack. Only the trend reveals the coming attack."""
    rng = np.random.default_rng(seed)
    series = []
    for i in range(n_series):
        X = rng.normal(size=(T, D)).astype(np.float32)
        y = np.zeros(T, dtype=np.int64)
        for onset in range(25, T - 15, 30):
            onset += int(rng.integers(0, 5))
            X[onset - 3:onset, 0] = np.array([-1.2, 0.0, 1.2]) + rng.normal(0, 0.1, 3)
            y[onset:onset + 4] = 2
            X[onset:onset + 4, 1] += 3.0                          # attack itself is easy to see
        roles = np.where(np.arange(n_series)[i] < 40, "train", np.where(i < 50, "val", "test"))
        series.append(Series("d", f"h{i}", np.arange(T), X, y, np.full(T, roles)))
    return series


def test_toy_world_world_model_sees_precursor_trend():
    torch.manual_seed(0)
    series = _toy_world()
    tr = SequenceDataset(series, "train", 10, 3)
    va = SequenceDataset(series, "val", 10, 3)
    te = SequenceDataset(series, "test", 10, 3)
    cfg = {"K": 3, "seed": 0, "model": {"hidden": 32, "emb": 16, "dropout": 0.0},
           "train": {"epochs": 12, "batch_size": 128, "lr": 3e-3, "weight_decay": 0.0, "patience": 6,
                     "teacher_frac": 0.5, "lambdas": LAM, "num_threads": 2}}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        model, _ = train_world_model(tr, va, cfg, 4, 5, d, log=lambda *_: None)
    p_wm = predict(model, te, 3)["p_within"]
    onset = (te.y_now == 0) & (te.valid > 0)
    ap_wm = average_precision_score(te.y_within[onset], p_wm[onset])
    lr = LogisticRegression(max_iter=1000).fit(tr.history_array()[:, -1], tr.y_within)
    ap_lr = average_precision_score(te.y_within[onset], lr.predict_proba(te.history_array()[:, -1])[onset, 1])
    assert ap_wm > ap_lr + 0.1, (ap_wm, ap_lr)
