"""Host-centric recurrent world model.

    filter        h_t = RNN(S_{t-L+1..t})                        summary of the observed past
    dynamics      P(S_{t+1} | h_t) = N(mu, diag(sigma^2))         transition model
    stage head    P(stage | h)                                    applied to real AND imagined states
    rollout       S^_{t+1} ~ P(.|h_t) -> h_{t+1} = RNN(S^_{t+1}, h_t) -> ... (K steps)

The future-stage and horizon outputs are computed ONLY from imagined states, so the forecast has
to go through the learned dynamics (this is what makes it a world model and not a classifier).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class NetworkWorldModel(nn.Module):
    def __init__(self, n_features: int, n_stages: int, emb: int = 64, hidden: int = 128,
                 cell: str = "gru", dropout: float = 0.1, rollout: str = "recursive",
                 K: int = 5, residual: bool = False):
        super().__init__()
        self.residual = residual          # predict S_{t+1} - S_t instead of S_{t+1}
        self.rollout = rollout            # "recursive" (world model) | "direct" (ablation A2)
        self.n_features, self.n_stages, self.hidden, self.cell_type = n_features, n_stages, hidden, cell
        self.encoder = nn.Sequential(nn.Linear(n_features, emb), nn.LayerNorm(emb), nn.GELU(),
                                     nn.Dropout(dropout))
        self.cell = nn.GRUCell(emb, hidden) if cell == "gru" else nn.LSTMCell(emb, hidden)
        self.dynamics = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, 2 * n_features))
        self.stage_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(64, n_stages))
        if rollout == "direct":           # ablation: K separate heads read h_t directly
            self.direct_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(),
                                             nn.Linear(64, K * n_stages))

    # ---- recurrent state helpers (GRU: h; LSTM: (h, c)) ----
    def init_state(self, batch: int, ref: torch.Tensor):
        h = ref.new_zeros(batch, self.hidden)
        return h if self.cell_type == "gru" else (h, ref.new_zeros(batch, self.hidden))

    @staticmethod
    def _h(state):
        return state if isinstance(state, torch.Tensor) else state[0]

    def step(self, s: torch.Tensor, state):
        return self.cell(self.encoder(s), state)

    def predict_next(self, state, last=None):
        mu, logvar = self.dynamics(self._h(state)).chunk(2, dim=-1)
        if self.residual and last is not None:
            mu = last + mu
        return mu, logvar.clamp(-6.0, 4.0)

    def stage_logits(self, state):
        return self.stage_head(self._h(state))

    def filter(self, x_hist: torch.Tensor, return_surprise: bool = False):
        """x_hist [B, L, D] -> recurrent state after S_t (and optional surprise of S_t)."""
        state = self.init_state(x_hist.size(0), x_hist)
        surprise = None
        for t in range(x_hist.size(1)):
            if return_surprise and t == x_hist.size(1) - 1:
                mu, logvar = self.predict_next(state, x_hist[:, t - 1] if t > 0 else None)
                surprise = gaussian_nll(x_hist[:, t], mu, logvar).sum(-1)   # -log P(S_t | h_{t-1})
            state = self.step(x_hist[:, t], state)
        return (state, surprise) if return_surprise else state

    def forward(self, x_hist, K: int, x_future=None, teacher_prob: float = 0.0,
                sample: bool = False, intervention=None, return_surprise: bool = False):
        if return_surprise:
            state, surprise = self.filter(x_hist, return_surprise=True)
        else:
            state, surprise = self.filter(x_hist), None
        stage_now = self.stage_logits(state)                                   # [B, C]
        first_state = state
        mus, logvars, stage_future, imagined = [], [], [], []
        last = x_hist[:, -1]
        for k in range(K):
            mu, logvar = self.predict_next(state, last)                        # P(S_{t+k+1} | h)
            mus.append(mu)
            logvars.append(logvar)
            s_next = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
            if x_future is not None and teacher_prob > 0:                      # scheduled sampling
                use_true = (torch.rand(mu.size(0), 1, device=mu.device) < teacher_prob).float()
                s_next = use_true * x_future[:, k] + (1 - use_true) * s_next
            if intervention is not None:                                       # what-if analysis
                s_next = intervention(s_next, k)
            imagined.append(s_next)
            last = s_next
            state = self.step(s_next, state)                                   # imagine next window
            stage_future.append(self.stage_logits(state))
        stage_future = torch.stack(stage_future, 1)
        if self.rollout == "direct":
            stage_future = self.direct_head(self._h(first_state)).view(x_hist.size(0), K, -1)
        out = {
            "stage_now": stage_now,
            "mu": torch.stack(mus, 1),                 # [B, K, D]
            "logvar": torch.stack(logvars, 1),         # [B, K, D]
            "stage_future": stage_future,              # [B, K, C]
            "imagined": torch.stack(imagined, 1),      # [B, K, D] states actually fed back
        }
        if surprise is not None:
            out["surprise"] = surprise
        return out


def gaussian_nll(x, mu, logvar):
    return 0.5 * (logvar + (x - mu) ** 2 / logvar.exp() + math.log(2 * math.pi))


def p_attack_within(stage_future_logits: torch.Tensor, benign_idx: int = 0,
                    temperature: float = 1.0) -> torch.Tensor:
    """1 - prod_k P(benign at t+k) computed in log space: [B, K, C] -> [B]."""
    log_p_benign = (stage_future_logits / temperature).log_softmax(-1)[..., benign_idx].sum(1)
    return -torch.expm1(log_p_benign)


def build_model(n_features: int, n_stages: int, mcfg: dict) -> NetworkWorldModel:
    return NetworkWorldModel(n_features, n_stages, emb=mcfg.get("emb", 64),
                             hidden=mcfg.get("hidden", 128), cell=mcfg.get("cell", "gru"),
                             dropout=mcfg.get("dropout", 0.1),
                             rollout=mcfg.get("rollout", "recursive"), K=mcfg.get("K", 5),
                             residual=mcfg.get("residual", False))
