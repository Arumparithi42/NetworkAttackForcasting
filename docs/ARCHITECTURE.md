# SIH26153 – Architecture (2 pages)

**AI-based network attack forecasting with a world model** · Theme: Blockchain & Cybersecurity

## 1. Problem framing

A conventional IDS asks *"is this flow malicious?"* and fires after the attack is visible. We
instead learn how the state of each host evolves, **P(S<sub>t+1</sub> | S<sub>≤t</sub>)**, and
simulate K windows ahead to estimate *"how likely is attack activity involving this host within
the next K minutes, which stage, and why"*. The system runs fully offline.

## 2. Pipeline

```text
CICFlowMeter CSV ──┐                         ┌─► label builder (dataset label → stage, kept separate)
raw PCAP (dpkt) ───┴─► canonical flows ──────┤
                                             └─► 60 s windows × entity (host | network)
                                                 flow + packet + graph + causal-temporal + global features
                                                 → robust scaler (fit on benign TRAIN windows)
                                                 → sequences (L = 10 history, K = 5 future)
World model (GRU) ─► K-step Monte-Carlo rollout ─► calibrated P(attack ≤ K), per-step stage probabilities
                 ─► Integrated Gradients + evidence templates ─► ATT&CK mapper (observed/inferred/predicted)
                 ─► Streamlit dashboard  ─► evidence ledger (hash chain + Ed25519, optional EVM anchor)
```

## 3. Network state S<sub>t</sub>

*Host mode* (one vector per internal host per 60 s window, 67 features): volume (flows, bytes,
packets in/out), diversity (distinct peers/ports, entropies), TCP behaviour (SYN, half-open, RST,
FIN, PSH, short flows, protocol mix), timing (duration, IAT, connection-start regularity), sizes,
service mix (web/SSH/FTP/SMB/RDP/DNS/other), graph/novelty (internal out-degree, **new peers/ports
not seen in 30 min**, external in-degree, neighbours' fan-out), **packet level** (TTL mean/std/
distinct, IP fragmentation, TCP retransmissions, payload-size entropy, sequential-port score, TCP
window variation), causal temporal (Δ vs t−1, z-score vs the host's own EWMA baseline) and global
network context. *Network mode* (38 features) is used when the data has no IP addresses.
Raw IPs/ports, timestamps and time-of-day are never model inputs (shortcut/leakage risk).
Normalisation: log1p for heavy-tailed counts, median/IQR robust scaling on benign training
windows, clip ±5.

## 4. World model

```text
S_t ─► Linear(D→64)+LayerNorm+GELU ─► GRUCell(64→128) = h_t
          ├─► dynamics head  → μ, log σ²   : P(S_{t+1} | h_t) = N(μ, diag σ²)
          └─► stage head     → P(stage_t)  (5 classes: benign, recon/discovery, access attempt, C2, impact)
Rollout k = 1..K:  Ŝ_{t+k} ~ N(μ, σ²) → same encoder + GRUCell → h_{t+k} → stage head → P(stage_{t+k})
```

~0.12 M parameters, trains on a laptop CPU. Loss = Gaussian NLL of the next K states (dynamics) +
cross-entropy of the current stage + cross-entropy of the stage of each **imagined** future state +
BCE on **P(attack within K) = 1 − Π<sub>k</sub> P(benign at t+k)**. Because the future outputs are
computed only from imagined states, the forecast must pass through the learned dynamics. Scheduled
sampling moves training from teacher forcing to free-running rollouts. Calibration: temperature
scaling (stage) + isotonic regression (horizon probability) on validation. Uncertainty: 32
Monte-Carlo rollouts → 5–95 % band. A *surprise score* −log P(S<sub>t</sub> | h<sub>t−1</sub>)
flags unexpected behaviour independent of known labels. What-if: clamp features of imagined
states (e.g. SMB activity) and re-simulate.

## 5. Data and leakage control

CSE-CIC-IDS2018: (a) processed CSVs, 8 usable days, network mode (CSVs lack IPs; 2 days excluded
because their benign background is missing); (b) raw per-host PCAPs of the two infiltration days
(116 GB streamed, flows + packet features extracted), host mode, with labels transferred from the
CSV by victim identification (`docs/DATA_NOTES.md`). Data defects found and fixed: 12-hour clock
without AM/PM, UTC−4 local time, repeated headers, 1970 rows, duplicates, truncated files.
Splits are chronological per attack family (first occurrence → train, repeat → test; infiltration
28-02 → 01-03); validation = hourly blocks inside training days; a sequence is used only if all of
its L + K windows share one role (automatic purge gap). Scaler, calibrators and thresholds are fit
on train/validation only; causality of every feature is unit-tested.

## 6. Explainability and ATT&CK

Integrated Gradients (Captum) on P(attack within K), through the rollout, against the median
benign state → per-(window × feature) attributions; the top features are rendered as sentences
with the real traffic values ("distinct ports contacted 12 → 88 → 301 (normal: 6)"). A deletion
test checks faithfulness. Stages map to ATT&CK v19.2 tactics; transparent indicator rules add
*candidate* techniques with an evidence-strength tag. Output is always split into OBSERVED (facts),
INFERRED (interpretation of the current state) and PREDICTED (future, with probability and band).
Lateral movement and exfiltration have no ground truth in the data and are reported as unsupported.

## 7. Evaluation protocol

Same features, samples, split and threshold rule (FPR 1 % on quiet validation windows) for:
B0 persistence (uses the true current label – oracle reference), B1 logistic regression on S<sub>t</sub>,
B2 logistic regression on the flattened history, B3 XGBoost on the flattened history, and the world
model. Metrics: precision/recall/F1/FPR, ROC-AUC, **PR-AUC split into onset (currently benign) vs
ongoing windows**, early-warning recall at T−1/T−2/T−3 before each attack onset, Brier/ECE,
next-state MSE vs persistence, block-bootstrap CIs. Ablations: no dynamics loss, direct heads,
history length, teacher forcing only, LSTM, residual dynamics, strict global-chronology protocol.
Results: `reports/*/RESULTS.md`.

## 8. Blockchain (where it helps)

Not inside the model. Forecast records are appended to a SHA-256 hash chain with Ed25519
signatures (model hash + config hash in every record) → tamper-evident audit trail; Merkle roots can
be anchored to a local EVM chain (Hardhat/Anvil, `EvidenceAnchor.sol`). Real trust gains need a
second, independent party holding the anchors (e.g. a sector CERT) – designed, not deployed. No raw
traffic or IPs go on-chain (privacy, DPDP Act 2023). The AI system works unchanged with it disabled.
