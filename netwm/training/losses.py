import torch
import torch.nn.functional as F


def _ce(logits, target, weight):
    """Cross-entropy that returns 0 (not NaN) when every target in the batch is ignored."""
    if (target >= 0).any():
        return F.cross_entropy(logits, target, weight=weight, ignore_index=-1)
    return logits.sum() * 0.0


def world_model_loss(out, batch, class_weights, lambdas):
    C = out["stage_now"].size(-1)
    dyn = F.gaussian_nll_loss(out["mu"], batch["x_future"], out["logvar"].exp())
    now = _ce(out["stage_now"], batch["y_now"], class_weights)
    fut = _ce(out["stage_future"].reshape(-1, C), batch["y_future"].reshape(-1), class_weights)

    # horizon: p = 1 - prod_k P(benign at t+k), BCE computed stably in log space
    log_no_attack = out["stage_future"].log_softmax(-1)[..., 0].sum(1).clamp(max=-1e-6)
    log_attack = torch.log(-torch.expm1(log_no_attack))
    y = batch["y_within"]
    bce = -(y * log_attack + (1 - y) * log_no_attack)
    valid = batch["within_valid"]
    hor = (bce * valid).sum() / valid.sum().clamp(min=1)

    total = (lambdas["dyn"] * dyn + lambdas["now"] * now + lambdas["fut"] * fut
             + lambdas["hor"] * hor)
    return total, {"dyn": dyn.item(), "now": now.item(), "fut": fut.item(), "hor": hor.item()}
