"""Training loop for the world model (scheduled sampling, early stopping on val PR-AUC)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from netwm.models.world_model import build_model, gaussian_nll, p_attack_within
from netwm.training.losses import world_model_loss


def class_weights(ds, n_classes: int) -> torch.Tensor:
    y = np.concatenate([ds.y_now, ds.y_future.reshape(-1)])
    y = y[y >= 0]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    w = w / w[counts > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model, ds, K: int, batch_size: int = 2048, n_samples: int = 0,
            temperature: float = 1.0, with_states: bool = False) -> dict:
    """Run the model on every sample of a SequenceDataset.

    n_samples = 0: deterministic mean rollout; > 0: Monte-Carlo rollouts, returns mean and
    5/95 % quantiles of P(attack within K)."""
    model.eval()
    outs = {"p_within": [], "stage_now": [], "stage_future": [], "surprise": []}
    if n_samples:
        outs["p_lo"], outs["p_hi"] = [], []
    if with_states:
        outs["nll_k"], outs["se_k"], outs["se_persist_k"] = [], [], []
    for b in ds.iterate(batch_size):
        xh = b["x_hist"]
        out = model(xh, K, return_surprise=True)
        sf = out["stage_future"] / temperature
        outs["stage_now"].append((out["stage_now"] / temperature).softmax(-1).numpy())
        outs["surprise"].append(out["surprise"].numpy())
        if n_samples:
            B = xh.size(0)
            rep = xh.repeat_interleave(n_samples, 0)
            o2 = model(rep, K, sample=True)
            pw = p_attack_within(o2["stage_future"], temperature=temperature).view(B, n_samples)
            outs["p_within"].append(pw.mean(1).numpy())
            outs["p_lo"].append(torch.quantile(pw, 0.05, dim=1).numpy())
            outs["p_hi"].append(torch.quantile(pw, 0.95, dim=1).numpy())
            probs = (o2["stage_future"] / temperature).softmax(-1).view(B, n_samples, K, -1)
            outs["stage_future"].append(probs.mean(1).numpy())
        else:
            outs["p_within"].append(p_attack_within(out["stage_future"], temperature=temperature).numpy())
            outs["stage_future"].append(sf.softmax(-1).numpy())
        if with_states:
            xf = b["x_future"]
            outs["nll_k"].append(gaussian_nll(xf, out["mu"], out["logvar"]).mean(-1).numpy())
            outs["se_k"].append(((xf - out["mu"]) ** 2).mean(-1).numpy())
            outs["se_persist_k"].append(((xf - xh[:, -1:, :]) ** 2).mean(-1).numpy())
    return {k: np.concatenate(v) if v else np.array([]) for k, v in outs.items()}


def train_world_model(train_ds, val_ds, cfg: dict, n_features: int, n_classes: int,
                      out_dir: str | Path, log=print) -> tuple:
    tcfg, K = cfg["train"], cfg["K"]
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    torch.set_num_threads(tcfg.get("num_threads", 4))
    mcfg = dict(cfg["model"], K=K)
    model = build_model(n_features, n_classes, mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    cw = class_weights(train_ds, n_classes)
    lam = tcfg["lambdas"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.pt"
    best, patience, history = -1.0, 0, []
    epochs = tcfg["epochs"]
    for epoch in range(epochs):
        t0 = time.time()
        ss = tcfg.get("scheduled_sampling", True)
        teacher = max(0.0, 1.0 - epoch / max(1.0, tcfg["teacher_frac"] * epochs)) if ss else 1.0
        model.train()
        sums, n = {}, 0
        for b in train_ds.iterate(tcfg["batch_size"], shuffle=True, rng=rng):
            out = model(b["x_hist"], K, x_future=b["x_future"], teacher_prob=teacher)
            loss, parts = world_model_loss(out, b, cw, lam)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        pv = predict(model, val_ds, K)
        keep = val_ds.valid > 0
        yv = val_ds.y_within[keep]
        val_ap = float(average_precision_score(yv, pv["p_within"][keep])) if yv.min() != yv.max() else float("nan")
        rec = {"epoch": epoch, "teacher_prob": round(teacher, 3), "val_pr_auc": val_ap,
               **{k: round(v / max(n, 1), 4) for k, v in sums.items()},
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        log(json.dumps(rec))
        score = val_ap if not np.isnan(val_ap) else -sums.get("dyn", 0)
        # only start early-stopping once teacher forcing has fully decayed (inference conditions)
        if teacher == 0.0 or epoch == 0 or not ss:
            if score > best:
                best, patience = score, 0
                torch.save(model.state_dict(), ckpt)
            elif teacher == 0.0 or not ss:
                patience += 1
                if patience >= tcfg["patience"]:
                    break
    model.load_state_dict(torch.load(ckpt))
    with open(out_dir / "history.json", "w") as fh:
        json.dump(history, fh, indent=1)
    return model, history
