"""Static report figures (matplotlib, offline). Colours follow a fixed role mapping:
the world model is always slot 1 (blue); baselines keep their own fixed colours."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_STYLE = {
    "WM_world_model": ("World model", "#2a78d6", "-"),
    "B3_xgboost_lagged": ("XGBoost (lagged)", "#eb6834", "-"),
    "B2_logreg_lagged": ("LogReg (lagged)", "#1baf7a", "-"),
    "B1_logreg_now": ("LogReg (current window)", "#4a3aa7", "--"),
    "B0_persistence": ("Persistence (oracle label)", "#8a8984", ":"),
}
STAGE_COLOURS = {1: "#eda100", 2: "#e87ba4", 3: "#4a3aa7", 4: "#008300", 5: "#e34948"}
STAGE_NAMES = {1: "Recon/Discovery", 2: "Initial-access attempt", 3: "Lateral movement",
               4: "Command & control", 5: "Impact (DoS)"}


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def timeline(times, stage, probs: dict, threshold: float, title: str, path: str):
    """Forecast P(attack within K) over time, with the dataset's ground-truth stage band."""
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(10, 4.2), dpi=130, sharex=True,
                                  gridspec_kw={"height_ratios": [5, 1]})
    fig.patch.set_facecolor(SURFACE)
    for name, p in probs.items():
        label, colour, ls = MODEL_STYLE.get(name, (name, INK2, "-"))
        ax.plot(times, p, color=colour, lw=2 if name == "WM_world_model" else 1.2, ls=ls, label=label)
    ax.axhline(threshold, color=INK2, lw=0.8, ls="--")
    ax.text(times[0], threshold, " alert threshold (val FPR 1%)", color=INK2, fontsize=8, va="bottom")
    ax.set_ylim(-0.02, 1.02)
    _style(ax, title, None, "P(attack within K)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    stage = np.asarray(stage)
    for s, c in STAGE_COLOURS.items():
        m = stage == s
        if m.any():
            axb.fill_between(times, 0, 1, where=m, color=c, step="mid", lw=0, label=STAGE_NAMES[s])
    axb.set_yticks([])
    _style(axb, None, "time (UTC)", None)
    axb.grid(False)
    axb.set_ylabel("dataset\nlabel", color=INK2, fontsize=8, rotation=0, ha="right", va="center")
    handles, labels = axb.get_legend_handles_labels()
    if handles:
        axb.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.6), ncol=5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def lead_curves(lead_results: dict, path: str, title: str):
    fig, ax = _fig(6, 3.6)
    for name, res in lead_results.items():
        label, colour, ls = MODEL_STYLE.get(name, (name, INK2, "-"))
        leads = sorted(int(k) for k in res)
        rec = [res[str(k)]["recall"] if str(k) in res else res[k]["recall"] for k in leads]
        ax.plot(leads, rec, marker="o", ms=5, color=colour, ls=ls, lw=2, label=label)
    n = next(iter(lead_results.values()))
    n0 = list(n.values())[0]["n_onsets"]
    _style(ax, title, "lead (windows before onset)", f"recall at fixed FPR (n={n0} onsets)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 1, 2, 3])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def pr_curves(y, preds: dict, path: str, title: str):
    fig, ax = _fig(5.5, 4)
    base = float(np.mean(y))
    for name, p in preds.items():
        if name == "B0_persistence":
            continue
        label, colour, ls = MODEL_STYLE.get(name, (name, INK2, "-"))
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=colour, ls=ls, lw=2 if name == "WM_world_model" else 1.2, label=label)
    ax.axhline(base, color=INK2, lw=0.8, ls=":")
    ax.text(0.99, base, f"prevalence {base:.3f}", color=INK2, fontsize=8, ha="right", va="bottom")
    _style(ax, title, "recall", "precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def reliability(y, p, path: str, title: str, bins: int = 10):
    fig, ax = _fig(4.5, 4)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(m.sum())
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.plot(xs, ys, marker="o", ms=6, color="#2a78d6", lw=2)
    for x, yy, n in zip(xs, ys, ns):
        ax.text(x, yy, f" {n}", fontsize=7, color=INK2)
    _style(ax, title, "forecast probability", "observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def state_error(wm, persist, path: str, title: str):
    fig, ax = _fig(5.5, 3.6)
    k = np.arange(1, len(wm) + 1)
    ax.plot(k, wm, marker="o", ms=5, color="#2a78d6", lw=2, label="World model (mean prediction)")
    ax.plot(k, persist, marker="o", ms=5, color="#8a8984", lw=2, ls=":", label="Persistence (S_t+k = S_t)")
    _style(ax, title, "forecast step k", "MSE of next-state features (normalised)")
    ax.set_xticks(k)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def ig_heatmap(attr: np.ndarray, features: list[str], path: str, title: str, top: int = 12):
    attr = np.asarray(attr)
    order = np.argsort(-np.abs(attr).sum(0))[:top]
    a = attr[:, order].T
    lim = np.abs(a).max() or 1.0
    fig, ax = _fig(7, 0.32 * top + 1.2)
    im = ax.imshow(a, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([features[i] for i in order], fontsize=8, color=INK)
    L = attr.shape[0]
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"t-{L - 1 - i}" if i < L - 1 else "t" for i in range(L)], fontsize=8)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("attribution (red = pushes risk up)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
