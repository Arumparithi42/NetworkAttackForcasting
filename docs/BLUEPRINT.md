# SIH26153 — Network Attack Forecasting World Model
## Complete Architecture & Implementation Blueprint

> Status: this was the **design blueprint** written before implementation. Every number in it is a
> configuration choice or explicitly illustrative; real results are in `reports/*/RESULTS.md`.
>
> **Where the implementation deliberately differs from this plan** (reasons in `docs/DATA_NOTES.md`):
> * Only CSE-CIC-IDS2018 is used. The CTU-13, CIC-IDS2017 and corrected-dataset download hosts were
>   not reachable from the build environment; the loaders for CIC-IDS2017 and the CTU-13 label
>   mapping exist, but no CTU-13 / cross-dataset results are reported.
> * The daily CSVs have no IP columns, so the host-level model is built from the **raw per-host
>   PCAPs** of the two infiltration days (116 GB streamed) with labels transferred from the CSV;
>   the network-level model uses all 8 usable CSV days. Days 16-02 and 21-02 are excluded.
> * Protocol A uses 20-02 (DDoS) as the Impact test day instead of 16-02/21-02 (truncated files).
> * Model classes are the 5 stages with ground truth; host windows use 67 features (as planned),
>   network windows 38.
> * Additional data fixes not anticipated here: 12-hour clock, contradictory duplicate labels.

---

## Table of contents

1. [What we actually need to build (plain language)](#1-what-we-actually-need-to-build)
2. [End-to-end architecture](#2-end-to-end-architecture)
3. [Model selection](#3-model-selection)
4. [Network state design (S_t)](#4-network-state-design)
5. [Time windows](#5-time-windows)
6. [Dataset strategy](#6-dataset-strategy)
7. [Attack progression labels](#7-attack-progression-labels)
8. [World Model training](#8-world-model-training)
9. [K-step forward simulation](#9-k-step-forward-simulation)
10. [Attack probability calculation](#10-attack-probability-calculation)
11. [Explainability](#11-explainability)
12. [Dashboard](#12-dashboard)
13. [Offline operation](#13-offline-operation)
14. [Repository structure](#14-repository-structure)
15. [Implementation plan (phases)](#15-implementation-plan)
16. [Team division](#16-team-division)
17. [Baseline experiment design](#17-baseline-experiment-design)
18. [Innovations and trade-offs](#18-innovations-and-trade-offs)
19. [Blockchain: where it helps and where it does not](#19-blockchain)
20. [Cybersecurity correctness & limitations](#20-cybersecurity-correctness--limitations)
21. [Final deliverables](#21-final-deliverables)
22. [Build sheet (exact stack, shapes, commands, code)](#22-build-sheet)

---

## 1. What we actually need to build

### 1.1 The one-sentence version

Build a system that watches network traffic minute by minute, learns **how a network normally
evolves over time**, and — given the last few minutes — **simulates the next few minutes** to
estimate how likely it is that the network is heading into an attack stage (and which one), with
an explanation an analyst can check.

### 1.2 Why an ordinary IDS/classifier is not enough

A normal intrusion classifier looks at **one flow (or one minute)** and asks:

```text
"Is this malicious right now?"   → yes / no
```

It only fires **after** the malicious traffic is already visible. It throws away order: it does not
care that a scan happened 3 minutes *before* a burst of SSH logins.

Concrete example — one internal workstation `172.31.64.40`:

| Minute | What the traffic shows | Classifier says | What an analyst would think |
|---|---|---|---|
| 10:00 | normal web + DNS | benign | normal |
| 10:01 | 1 outbound connection to an unusual external host | benign | hmm |
| 10:02 | host starts touching 40 internal IPs on 445/139/22 | *maybe* "scan" | this machine is exploring the network |
| 10:03 | 300 distinct ports, many SYN with no reply | "scan" | Discovery is under way; access attempts usually come next |
| 10:04 | many short SSH sessions to 172.31.69.x | "brute force" | too late — it has already started |

The classifier only reacts at 10:03–10:04. We want a system that at **10:02** says: *"the trajectory
of this host is moving toward a Discovery → Access pattern; estimated probability of attack activity
involving this host within the next 3 minutes is elevated; here is why."*

### 1.3 What "World Model" means here

A world model is a model that has learned **how the world changes from one moment to the next**,
so it can "imagine" the future by running itself forward.

- Weather analogy: a forecaster does not classify "is it raining now?"; they model how pressure,
  humidity and wind evolve and roll that forward to say "70 % chance of rain in 3 hours".
- Here the "world" is the network, the "weather" is the traffic state of each host, and the
  forecast is "probability that this host enters an attack stage in the next K windows".

Practically, our world model has two jobs:

1. **Dynamics:** given the history up to now, predict the *next network state* (a vector of numbers
   such as "number of distinct ports contacted", "SYN-without-reply ratio", …), as a probability
   distribution, not a single guess.
2. **Interpretation:** for any state — real *or imagined* — estimate which attack stage (if any) it
   looks like.

Because job 1 can be applied to its own output, we can run it forward K steps. That is what makes it
a world model rather than a renamed classifier.

### 1.4 What "network state" means

The state `S_t` is a fixed-length list of numbers that summarises what a host (plus the network
around it) did during time window `t`. Example for one host, one 60-second window:

```text
S_t(host=172.31.64.40, 10:03) = [
  n_flows_out = 412, uniq_dst_ips_out = 38, uniq_dst_ports_out = 301,
  dst_port_entropy = 7.9, half_open_ratio = 0.83, rst_ratio = 0.61,
  bytes_sent = 51 KB, smb_ratio = 0.12, new_peers_out = 35, ...  (≈59 numbers)
]
```

### 1.5 What `P(S(t+1) | S(t))` means

"Given what the network looks like now (and just before now), what is the **probability
distribution** of what it will look like one window later?"

- Not "the next state *is* X", but "the next state is most likely around X, with this much spread".
- We implement it as a Gaussian per feature: for each feature the model outputs a mean `μ` and a
  variance `σ²`. E.g. *"next minute, `uniq_dst_ports_out` ≈ 280 ± 90"*.
- In practice we condition on a short history `S_{t-L+1..t}` (the recurrent hidden state `h_t`
  summarises it), i.e. `P(S_{t+1} | h_t)`. This is the standard, honest relaxation of the Markov
  assumption: one minute alone is not a sufficient state.

### 1.6 What K-step forecasting means

Apply the one-step model repeatedly:

```text
real history S_{t-9..t} → h_t → predict S_{t+1} → feed it back → h_{t+1} → predict S_{t+2} → ... → S_{t+K}
```

At each imagined step we also ask the stage head "what stage does this imagined state look like?".
If K = 5 and the window is 60 s, we are looking 5 minutes ahead.

### 1.7 How attack progression can be predicted from traffic

Attacks leave **precursors** that usually appear before the damaging part:

| Precursor visible in traffic | Often followed by | Why it is predictive |
|---|---|---|
| Rising distinct destination ports, SYN-without-reply | Access attempts on the open services found | attackers scan before they exploit |
| A host suddenly talking to many *internal* peers it never contacted before | Credential attacks / remote service use on those peers | discovery before lateral movement |
| Very regular, small outbound connections (low inter-flow-gap variance) | Instructions from C2, then other activity | beaconing precedes tasking |
| Repeated short SSH/FTP sessions ramping up | Successful login / follow-on activity | brute force progresses over minutes |

**Honest limit:** some attacks have *no* traffic precursor (e.g. a DDoS that starts abruptly from
outside). For those, the best any system can do is detect at onset and forecast *continuation*. Our
evaluation separates these two cases (§17.4) so we never over-claim.

---

## 2. End-to-end architecture

### 2.1 Improved pipeline

The suggested pipeline is right in spirit. We add four things that matter in practice: a
**canonical flow schema** (so different datasets/extractors plug in), a **separate label track**
(labels must never leak into features), **calibration** (so probabilities mean something) and an
**evidence ledger** (for the Blockchain & Cybersecurity theme).

```text
 ┌──────────────────────────────── OFFLINE TRAINING PATH ───────────────────────────────┐
 │                                                                                      │
 │  Dataset CSV (CICFlowMeter / Argus)      PCAP (optional subset)                      │
 │            │                                   │                                     │
 │   [A] Loader + cleaner  ──► canonical flows.parquet   [B] Packet feature extractor   │
 │            │                                   │        (tshark / Scapy)             │
 │            ├──────────────► [L] Label builder  │                                     │
 │            │                (dataset label → stage, per host-window; SEPARATE file)  │
 │            ▼                                   ▼                                     │
 │   [C] Window builder: per (host, 60 s window) flow + packet + graph features         │
 │            ▼                                                                         │
 │   [D] Causal temporal features (deltas, EWMA z-scores) + global context              │
 │            ▼                                                                         │
 │   [E] Scaler (fit on TRAIN only) ──► host_windows.parquet  (S_t for every host, t)   │
 │            ▼                                                                         │
 │   [F] Sequence builder (L history, K future) + chronological split                  │
 │            ▼                                                                         │
 │   [G] World Model training (dynamics + stage heads, multi-step rollout loss)         │
 │   [H] Baselines (persistence, LogReg, RF/XGBoost) on the SAME features/splits        │
 │            ▼                                                                         │
 │   [I] Calibration on validation (temperature / isotonic)                             │
 │            ▼                                                                         │
 │   [J] Evaluation: F1/PR-AUC/FPR, lead-time, calibration, state-forecast error        │
 └──────────────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────── OFFLINE INFERENCE / DEMO PATH ───────────────────────┐
 │  CSV or PCAP upload ─► [A]/[B] (NFStream for PCAP→flows) ─► [C][D][E] with saved      │
 │  scaler ─► [K] K-step Monte-Carlo rollout ─► [M] risk + stage probabilities           │
 │  ─► [N] MITRE ATT&CK mapper (observed / inferred / predicted)                         │
 │  ─► [O] Explainer (Integrated Gradients + evidence templates)                          │
 │  ─► [P] Streamlit dashboard       └─► [Q] Evidence ledger (hash chain, optional anchor) │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component table

| # | Component | Input | Processing | Output | Tech | Why required |
|---|---|---|---|---|---|---|
| A | Loader + cleaner | raw CSV (CIC-IDS2018/2017, CTU-13) | column rename to canonical schema, dtype downcast, drop repeated header rows, `inf`→NaN, timestamp parse + timezone offset, dedupe | `flows.parquet` (canonical schema §22.3) | pandas, pyarrow | every later step depends on one clean schema |
| B | Packet feature extractor | PCAP | per packet: time, IPs, TTL, IP flags/frag offset, TCP window, flags, payload length, retransmission flag → aggregated per host-window | `packet_feats.parquet` | `tshark -T fields` (bulk), Scapy (small files) | TTL variance, fragmentation, retransmissions, sequential-port scans are not in flow CSVs |
| L | Label builder | `flows.parquet` (label column only) | dataset label → stage (§7), aggregate per host-window, ambiguity mask | `labels.parquet` | pandas | keeps ground truth physically separate from features (prevents leakage) |
| C | Window builder | flows (+packet feats) | assign each flow to a 60 s window by start time; per internal host compute ~46 features; 1-hop graph features | per host-window raw features | pandas, NetworkX (light) | turns a flow list into states |
| D | Causal temporal features | per host-window raw features | Δ vs previous window, z-score vs host's own EWMA baseline (past only), global context | enriched features | pandas | makes change visible; still causal |
| E | Scaler | train-split features | `log1p` for heavy-tailed counts, robust scaling (median/IQR) fit on **train benign** windows, clip ±5 | `host_windows.parquet`, `scaler.json` | numpy | stable training, no test leakage |
| F | Sequence builder + split | host_windows + labels | sliding windows `[t-L+1..t]` → targets `[t+1..t+K]`; day-level chronological split with purge gap | PyTorch `Dataset` | PyTorch | defines the forecasting task |
| G | World Model | `x_hist[B,L,D]` | GRU filter → dynamics head (Gaussian) + stage head; recursive rollout | `μ, σ², stage probs` per future step | PyTorch | learns `P(S_{t+1}|h_t)` and stage of imagined states |
| H | Baselines | same features | persistence; LogReg on `S_t`; LogReg/XGBoost on flattened `S_{t-L+1..t}` | same outputs as G's horizon head | scikit-learn, xgboost | required comparison; isolates value of dynamics |
| I | Calibration | validation predictions | temperature scaling (stage), isotonic (horizon prob) | `calibrator.pkl` | scikit-learn | makes "72 %" mean ≈72 % empirically |
| J | Evaluation | test predictions + labels | metrics, lead-time, bootstrap CIs by episode | `reports/*.csv`, plots | scikit-learn, matplotlib | evidence for judges |
| K | Rollout engine | model + history | N Monte-Carlo trajectories of K steps | trajectories + per-step stage probs | PyTorch | the forecast itself, with uncertainty |
| M | Risk calculator | trajectories | `P(attack within K)`, `P(stage = X at t+k)`, earliest crossing step | risk JSON | numpy | analyst-facing numbers |
| N | ATT&CK mapper | stage probs + observed indicators | stage → tactic; indicator rules → *candidate* techniques with evidence strength | observed / inferred / predicted blocks | YAML rules, local ATT&CK subset | required mapping, without over-claiming |
| O | Explainer | model, history, baseline state | Integrated Gradients on the horizon probability → (time × feature) attributions → templates with real values | top-evidence list, heatmap | Captum | "black-box outputs are not acceptable" |
| P | Dashboard | forecasts, explanations, flows | replay a test day with a time slider or upload CSV/PCAP | UI | Streamlit, Plotly | required demo interface |
| Q | Evidence ledger (optional) | forecast/alert JSON | SHA-256 hash chain + Ed25519 signatures; optional Merkle-root anchoring to a local EVM chain | `ledger.sqlite`, verify report | hashlib, cryptography, (Hardhat/Anvil + web3.py optional) | tamper-evident audit trail (theme) |

---

## 3. Model selection

### 3.1 Comparison

Scores are qualitative judgements for *our* setting (≈60 features, short sequences of 10 windows,
a few hundred hosts, few attack episodes, student laptop).

| Model | Accuracy potential here | Training complexity | Dataset fit | Explainability | Dev time | GPU need | Demo ease | K-step rollout |
|---|---|---|---|---|---|---|---|---|
| LSTM | good | low | good | IG works | low | none (CPU ok) | easy | natural (recurrent state) |
| **GRU** | good (≈LSTM) | **lowest** | good | IG works | **low** | **none** | easy | **natural** |
| Transformer (vanilla) | similar at L=10, needs more data | medium | ok | attention ≠ explanation; IG works | medium | helpful | ok | must re-encode whole context each step |
| Temporal Transformer (e.g. TFT) | good for multi-horizon, heavy | high | ok | built-in variable selection | high | helpful | medium | usually direct multi-horizon, not a simulator |
| GNN (static, per window) | captures topology, ignores time | medium | needs IPs every window | GNNExplainer (slow) | medium | helpful | medium | no (needs a temporal part) |
| Temporal GNN (TGN, EvolveGCN, …) | highest ceiling for lateral movement | **high** | needs full host graph; sparse labels | hard | **high** | yes | hard | possible, but complex |
| Hybrid GNN + GRU | high ceiling | high | as above | hard | high | yes | medium | yes |
| Latent SSM (Dreamer-style RSSM) | good | medium–high (KL balancing, 2 latents) | ok | harder (latent) | medium–high | optional | medium | yes (latent imagination) |

### 3.2 Decision: host-centric recurrent world model (GRU) with a Gaussian dynamics head

**Chosen architecture:** *"Host-centric GRU World Model"* — one shared GRU runs over each internal
host's sequence of states (host features + 1-hop graph features + global network context).

```text
S_t (D≈59) ─► Encoder (Linear→LayerNorm→GELU, 64) ─► GRUCell (hidden 128) ─► h_t
                                                                    ├─► Dynamics head → μ_{t+1}, log σ²_{t+1}  (= P(S_{t+1}|h_t))
                                                                    └─► Stage head    → P(stage_t)
Rollout: Ŝ_{t+1} ~ N(μ, σ²) ─► same Encoder+GRUCell ─► h_{t+1} ─► stage head → P(stage_{t+1}) ─► ... (K steps)
```

Why this and not something fancier:

1. **Data size:** CIC-style datasets contain *tens* of attack episodes, not thousands. A ~0.12 M
   parameter GRU will not memorise the way a large Transformer/TGN would.
2. **Rollout is native:** a recurrent cell carries a state, so each imagined step costs one cell
   update. This is exactly the `S_t → S_{t+1} → S_{t+2}` simulation the problem statement asks for.
3. **Graph information without a GNN:** host fan-out, novel internal peers, internal/external
   degree and 1-hop neighbour aggregates (§4) capture most of what a GNN would use for
   Discovery/Lateral-Movement signals, at a fraction of the engineering cost. A Temporal GNN is
   listed as a future extension, not an MVP dependency.
4. **Many training sequences:** modelling per host multiplies the number of sequences by the number
   of hosts, and directly produces the required "relevant hosts" output.
5. **Explainability:** Integrated Gradients works on any differentiable model and gives
   *(time step × feature)* attributions — better than attention weights, which are not reliable
   explanations.
6. **Hardware:** trains on CPU in minutes-to-hours; a laptop GPU is a bonus, not a requirement.

It is honestly related to the original "World Models" idea (Ha & Schmidhuber, 2018: a recurrent
model predicting a distribution over the next observation). Our upgrade path, if time permits, is a
**Mixture Density** dynamics head (2–3 Gaussian components), because future network states are
multi-modal ("stays benign" vs "attack starts"), and a single Gaussian averages them.

LSTM is kept as a config switch (`model.cell: gru|lstm`) for an ablation; nothing else changes.

---

## 4. Network state design

### 4.1 Entity choice

- **Primary state:** one vector per **(internal host, window)**. Internal = inside the dataset's
  configured internal CIDRs (e.g. `172.31.0.0/16` for CSE-CIC-IDS2018 — verify from data; CTU-13 uses
  `147.32.0.0/16`).
- Each flow contributes to **both** endpoints' states if they are internal (outbound view for the
  source, inbound view for the destination).
- **Global context** (same for every host in a window) is appended so each host "sees" the network.
- **Fallback** if a dataset has no IP columns: a single network-level state per window (global block
  only). This loses host attribution, so it is only a fallback.

### 4.2 Final feature set (flow-only D = 59; with packet block D = 67)

All counts/bytes are computed per 60 s window. "out" = host is the flow source; "in" = host is the
destination. Log-transform is applied in the scaler (§4.4), not here.

| Block | # | Feature | Definition | Why it matters |
|---|---|---|---|---|
| **A. Volume** | 1 | `n_flows_out` | flows started by host | activity level |
| | 2 | `n_flows_in` | flows targeting host | being targeted |
| | 3 | `bytes_sent` | bytes sent by host (fwd bytes if src, bwd if dst) | exfil/DoS indicators |
| | 4 | `bytes_recv` | bytes received by host | download, floods |
| | 5 | `pkts_sent` | packets sent by host | |
| | 6 | `pkts_recv` | packets received by host | |
| | 7 | `active` | 1 if any flow involves host | distinguishes "idle" from "zero-valued" |
| **B. Diversity** | 8 | `uniq_dst_ips_out` | distinct peers contacted | scanning / spreading |
| | 9 | `uniq_src_ips_in` | distinct peers contacting host | inbound scan / DDoS |
| | 10 | `uniq_dst_ports_out` | distinct destination ports contacted | port scan |
| | 11 | `uniq_dst_ports_in` | distinct ports on host that were contacted | being scanned |
| | 12 | `dst_port_entropy_out` | Shannon entropy of dst ports (out) | scans → high entropy |
| | 13 | `dst_ip_entropy_out` | entropy of peers (out) | sweeps |
| | 14 | `internal_peer_ratio` | share of flows whose peer is internal | internal spread |
| **C. TCP/protocol behaviour** | 15 | `syn_flow_ratio` | flows with ≥1 SYN | connection attempts |
| | 16 | `half_open_ratio` | SYN and zero backward packets | scan / unanswered attempts |
| | 17 | `rst_ratio` | flows with RST | closed ports, rejections |
| | 18 | `fin_ratio` | flows with FIN | normal completions |
| | 19 | `psh_ratio` | flows with PSH | data-carrying sessions |
| | 20 | `short_flow_ratio` | flows with ≤3 packets total | probes, brute force |
| | 21 | `tcp_ratio` | protocol share | |
| | 22 | `udp_ratio` | | UDP floods, DNS |
| | 23 | `icmp_ratio` | | ping sweeps |
| **D. Timing** | 24 | `flow_dur_mean` | mean flow duration | slow attacks (slowloris) |
| | 25 | `flow_dur_std` | | |
| | 26 | `flow_iat_mean` | mean of per-flow packet IAT means | automated vs human |
| | 27 | `flow_start_gap_mean` | mean gap between successive flow starts by host | rate |
| | 28 | `flow_start_gap_cv` | std/mean of those gaps | **beaconing** (C2) → very regular → low CV |
| **E. Size** | 29 | `fwd_pkt_len_mean` | | payload profile |
| | 30 | `bwd_pkt_len_mean` | | |
| | 31 | `pkt_len_std_mean` | | |
| | 32 | `bytes_ratio` | `log((sent+1)/(recv+1))` | upload-heavy vs download-heavy |
| | 33 | `init_fwd_win_mean` | CICFlowMeter initial TCP window (fwd) | packet-level proxy (TCP window) |
| **F. Service mix** (share of host's flows by dst port bucket) | 34–41 | `svc_web` (80/443/8080), `svc_ssh` (22), `svc_ftp` (20/21), `svc_smb` (139/445/137/138), `svc_rdp` (3389), `svc_dns` (53), `svc_other_low` (<1024), `svc_high` (≥1024) | categorical port → bag-of-buckets | SMB/RDP/SSH shifts are classic lateral-movement/access signals |
| **G. Graph / novelty** | 42 | `out_degree_internal` | distinct internal peers contacted | spread inside the network |
| | 43 | `new_peers_out` | peers not contacted by host in its previous 30 windows | **novel** relationships (lateral movement) |
| | 44 | `new_ports_out` | dst ports not used by host in previous 30 windows | novel services |
| | 45 | `in_degree_external` | distinct external sources contacting host | external targeting |
| | 46 | `nbr_mean_fanout` | mean `uniq_dst_ports_out` of the host's peers this window | 1-hop graph aggregation ("am I talking to a scanner?") |
| **H. Packet-level** (optional, from PCAP) | 47 | `ttl_mean_in` | mean TTL of packets received | |
| | 48 | `ttl_std_in` | TTL variance | spoofing / multiple sources |
| | 49 | `n_distinct_ttl_in` | | |
| | 50 | `ip_frag_ratio` | packets with MF flag or frag offset > 0 | fragmentation evasion |
| | 51 | `tcp_retrans_ratio` | tshark `tcp.analysis.retransmission` share | congestion / floods |
| | 52 | `payload_len_entropy` | entropy of payload-size histogram | scripted vs varied traffic |
| | 53 | `seq_port_score` | share of consecutive probes with dst port +1 | sequential scan signature |
| | 54 | `tcp_win_std` | TCP window size std | tool/OS fingerprint changes |
| **I. Temporal (causal)** | 55 | `d_n_flows_out` | `log1p` diff vs t-1 | sudden change |
| | 56 | `d_uniq_dst_ports_out` | diff vs t-1 | |
| | 57 | `z_n_flows_out` | z-score vs host's EWMA mean/var of *past* windows (α=0.1) | deviation from own baseline |
| | 58 | `z_uniq_dst_ips_out` | same | |
| | 59 | `z_bytes_sent` | same | |
| **J. Global context** | 60 | `g_total_flows` | all flows in window | |
| | 61 | `g_active_hosts` | internal hosts active | |
| | 62 | `g_syn_ratio` | | |
| | 63 | `g_half_open_ratio` | | network-wide scanning |
| | 64 | `g_dst_port_entropy` | | |
| | 65 | `g_n_external_src` | distinct external sources | campaign from outside |
| | 66 | `g_internal_to_internal_ratio` | | internal spread |
| | 67 | `g_total_bytes` | | |

(Flow-only configuration = blocks A–G, I, J = 59 features; the packet block H adds 8.)

### 4.3 Feature types

- **Numerical:** everything above.
- **Categorical:** protocol and destination port are *not* one-hot encoded as identities. They are
  turned into **window-level distributions** (ratios in blocks C and F) and entropies. Raw IPs and
  ports are **never** model inputs (they would let the model memorise attacker addresses).
- **Temporal:** block I (deltas, EWMA z-scores) plus the GRU's own memory. All are **causal** —
  computed only from windows ≤ t.
- **Graph:** block G. For each window we build a directed host graph (edge = at least one flow);
  features are degrees, novelty and a 1-hop aggregate. NetworkX is only needed for the dashboard
  graph view; the features themselves are pandas group-bys.
- **Deliberately excluded:** time of day / day of week (datasets schedule attacks at fixed hours →
  a model would learn "10:30 = attack"), Flow ID, raw IPs, raw timestamps, and dataset-specific
  ID columns.

### 4.4 Normalisation

1. `log1p` on heavy-tailed non-negative features (counts, bytes, packets, durations, IATs, degrees).
2. **Robust scaling** `(x − median) / IQR`, with statistics fitted on **benign windows of the
   training days only**, stored in `artifacts/scaler.json`.
3. Clip to `[−5, 5]` (avoids a single flood window dominating the loss).
4. Ratios/entropies in `[0, 1]` (entropy divided by `log2(#ports observed)` or a fixed cap) are left
   as-is after step 2.
5. Missing values (e.g. packet block unavailable, IAT undefined for 1-packet flows) → 0 after
   scaling **plus** the block is either always present or always absent for a given model
   configuration (no silent mixing).

---

## 5. Time windows

### 5.1 Converting a dataset into sequences

```text
flows ──(floor(start_time / 60 s))──► window index w
          per (host, w) features ──► matrix X_host[T, D]   (T = windows in the capture day)

For every host and every t with L-1 ≤ t ≤ T-K-1:
  input   x_hist   = X_host[t-L+1 : t+1]        # S(t-9) … S(t)
  targets x_future = X_host[t+1 : t+K+1]        # S(t+1) … S(t+5)       (dynamics)
          y_now    = stage_host[t]              # stage at t            (current, inferred)
          y_future = stage_host[t+1 : t+K+1]    # stage at t+1..t+K     (predicted)
          y_within = any(y_future ≠ benign)     # infiltration within K
```

Sequences never cross day boundaries (each day is a separate capture).

### 5.2 Recommended values

| Parameter | Value | How to choose / validate |
|---|---|---|
| Window duration | **60 s** (ablate 30 s, 120 s) | Must be short relative to the attack phases (CIC attacks last from a few minutes to about an hour) so each phase spans several windows, yet long enough that each window has enough flows for stable ratios. Check: median flows per active host-window should be ≥ ~5. |
| History length L | **10 windows** (10 min) | Enough to see a ramp-up; ablate {1, 5, 10, 20}. L = 1 is the "no temporal context" control. |
| Forecast horizon K | **5 windows** (5 min) | Must be ≥ the largest lead time we evaluate (3). Longer horizons give lower confidence and a larger rollout error. |
| Stride | **1 window** for train/val/test | Overlapping sequences are fine *within* a split; the split itself is by day (§6.3), so overlaps cannot leak across splits. |
| Onset quiet period M | **5 windows** | An "onset" requires ≥5 benign windows before the first attack window (§17.4). |
| Label minimum | **≥3 malicious flows** in a host-window to label it with a stage; 1–2 → "ambiguous" (masked) | Reduces label noise from mislabeled stray flows. |

Choose by validation only (never by looking at test results).

---

## 6. Dataset strategy

### 6.1 Evaluation of candidate datasets

| Dataset | What it contains | Temporal progression? | Flows / PCAP | Preprocessing effort | Suitable attacks | Fit for our world model | Main limitations |
|---|---|---|---|---|---|---|---|
| **CSE-CIC-IDS2018** | 10 capture days (Feb–Mar 2018) on AWS, a victim network of several hundred machines, CICFlowMeter flows (~80 features) + PCAPs + logs | **Partial.** Each attack family runs on a scheduled day; several families **repeat on a second day** (DoS, DDoS, web, infiltration). Infiltration days contain a real multi-step sequence (malicious document → compromised internal host → internal Nmap scanning). | Flow CSVs (~several GB); PCAPs very large (hundreds of GB) | Medium: large, repeated header rows, `inf` values; **most original daily CSVs lack Src/Dst IP columns** (verify on download) | Brute force (FTP/SSH), DoS, DDoS, web attacks, infiltration, botnet | **Best primary.** Repeated families allow honest chronological train/test; named in the PS | Label noise (esp. "Infilteration"), CICFlowMeter bugs, attacks launched by few attacker IPs, synthetic benign behaviour. Corrected re-extractions with full flow keys exist (Engelen et al. 2021; Liu et al. 2022, Distrinet) — use them if available. |
| **CIC-IDS2017** | 5 days (Mon–Fri, July 2017), CICFlowMeter flows with IPs/timestamps (the "TrafficLabelling" CSVs), PCAPs (~10 GB/day) | **Partial.** Scheduled attacks, each family **once**; Thursday infiltration includes "victim scans internal network" (few labelled flows) | Flows + PCAP | Easy–medium; known timestamp ambiguities and label errors (corrected versions exist) | Same families as 2018 + Heartbleed, PortScan, Botnet ARES | **Development + cross-dataset test** (same CICFlowMeter feature space as 2018) | Each attack once ⇒ a chronological split puts unseen families in test; small infiltration class |
| **UNSW-NB15** | 2015, IXIA PerfectStorm-generated traffic, Argus/Bro features (49), 9 attack categories, PCAPs | **Weak.** Attacks are generated independently, not as campaigns | Flows (full CSVs have start/end times) + PCAP | Medium; the popular train/test CSVs drop timestamps | Recon, exploits, DoS, backdoors, worms… | Optional stage-diversity data only | Synthetic generator, no kill-chain sequences, different feature space |
| **CTU-13** | 13 scenarios (2011), real botnets (Neris, Rbot, Virut, Menti, Sogou, Murlo, NSIS.ay) in a university network, bidirectional NetFlow (Argus) with descriptive labels (e.g. `From-Botnet-…-CC…`) | **Yes, for botnet life-cycle:** infected host → C&C → spam/DDoS/click-fraud/scan over hours | Bidirectional flows; PCAPs of botnet traffic | Easy (small files per scenario); flags encoded in `State` string | **C&C**, scanning, DDoS, spam | **Secondary:** C&C stage + leave-scenario-out generalisation | 2011 era; most traffic is unlabeled "Background"; few infected hosts per scenario; different flow extractor (needs common feature subset) |
| **CICIoT2023** | 105 IoT devices, 33 attacks in 7 categories | **No.** Attacks executed separately; released CSVs are windowed packet statistics without flow keys/timestamps (verify) | CSV + PCAP | Easy for classification, hard for temporal use | IoT DDoS/DoS/recon/Mirai | Low for a world model | No campaign structure; IoT-specific |
| **LANL (2015 multi-source)** | 58 days of auth, process, DNS and flow events from a real enterprise; `redteam.txt` lists red-team compromise events | **Yes for lateral movement** (red-team authentications over time) | Events (not PCAP); flows unlabeled | **Hard** (billions of events, anonymised) | **Lateral movement / credential use** | Stretch goal only (auth-graph model) | Huge; only ~hundreds of red-team events; flows not labelled; anonymised time |
| **DARPA 1998/99, DARPA 2000 (LLDOS 1.0/2.0.2)** | MIT-LL tcpdump + attack lists; LLDOS is a scripted multi-stage scenario (IP sweep → probe → exploit → install DDoS agent → launch DDoS) | **Yes (textbook multi-stage)**, but a single scenario | PCAP | Medium (old formats) | Multi-stage walk-through | **Qualitative case study / demo only** | Very old, known artefacts (e.g. TTL), tiny |

### 6.2 Recommended strategy

```text
PRIMARY   : CSE-CIC-IDS2018 flows (corrected version with IPs if available)   → train / val / test
SECONDARY : CTU-13 (3–5 scenarios)                                            → C&C stage + leave-scenario-out test
CROSS-DATA: CIC-IDS2017 (test only)                                           → "does it transfer?" (same feature space)
PACKET    : PCAPs of the two CIC-IDS2018 infiltration days only (subset hosts) → flow vs flow+packet ablation
OPTIONAL  : DARPA 2000 LLDOS                                                  → qualitative progression demo
STRETCH   : LANL auth + redteam                                               → lateral-movement stage
```

Development tip: build the whole pipeline first on **one** CIC-IDS2017 day (small, has IPs), then
switch the config to CIC-IDS2018.

### 6.3 Splitting without leakage

**Protocol A — per-family chronological (primary).** Train on the *first* day of each repeated
family, test on the *later* day. Validation = last 20 % (by time) of each training day, separated by
a purge gap of `L + K` windows.

| Family | Train day(s) | Test day(s) |
|---|---|---|
| DoS | 15-02-2018 (GoldenEye, Slowloris) | 16-02-2018 (SlowHTTPTest, Hulk) — *different tools, same stage* |
| DDoS | 20-02-2018 (LOIC-HTTP, LOIC-UDP) | 21-02-2018 (HOIC, LOIC-UDP) |
| Web | 22-02-2018 | 23-02-2018 |
| Infiltration | 28-02-2018 | 01-03-2018 |
| Brute force | 14-02-2018 | CIC-IDS2017 Tuesday (cross-dataset) |
| Botnet | 02-03-2018 first 60 % of the day | 02-03-2018 last 40 % (after purge gap) |

(Dates from the dataset's published schedule — confirm against the downloaded files.)

Within a family every test sample is later in time than every training sample. Across families the
days are independent captures and the model has no calendar features, so this ordering is
acceptable — **state it explicitly in the report**.

**Protocol B — strict global chronological (stress test).** Train on all days up to 22-02, test on
23-02 → 02-03. Test then contains families never seen in training (infiltration, bot): this measures
generalisation to unseen attacks. Expect a drop; report it honestly.

**Protocol C — CTU-13 leave-scenario-out.** Train on some scenarios, test on others (different
botnets).

**Leakage checklist (automated in tests):**
- scaler/EWMA/calibrator fitted on train (or val for calibrator) only;
- no sequence spans two splits;
- features at window t unchanged when rows after t are modified (causality test);
- no IP/port/timestamp/Flow-ID columns in the model input;
- benign downsampling applied to **train only** (val/test keep natural prevalence);
- thresholds chosen on validation, never on test.

---

## 7. Attack progression labels

### 7.1 Three kinds of "label" — never mix them

| Kind | Source | Where it lives | Example |
|---|---|---|---|
| **Ground-truth dataset label** | the dataset's own per-flow label | `flows.parquet: label_raw` | `"DoS attacks-Hulk"`, `"Infilteration"`, `flow=From-Botnet-V42-TCP-CC6-...` |
| **Derived stage label** (our mapping, documented) | deterministic function of the dataset label (+ flow direction) | `labels.parquet: stage` | `IMPACT`, `RECON_DISCOVERY` |
| **Model prediction** | world model output | forecast JSON | `P(stage=C2 at t+2) = …` |

The derived stage is a **defensible re-grouping** of what the dataset documents, not new ground
truth. The mapping table is versioned (`netwm/attack_mapping/label_to_stage.yaml`) and cited in the
README.

### 7.2 Stage set

| ID | Stage | ATT&CK tactic(s) | Trained in MVP? | Ground-truth source |
|---|---|---|---|---|
| 0 | `BENIGN` | — | yes | dataset "Benign"/"Normal" |
| 1 | `RECON_DISCOVERY` | TA0043 Reconnaissance (scan source external) / TA0007 Discovery (scan source internal) | yes | PortScan (2017), infiltration internal scanning, CTU-13 scan flows |
| 2 | `INITIAL_ACCESS_ATTEMPT` | TA0001 Initial Access / TA0006 Credential Access | yes | FTP/SSH brute force, web brute force, SQL injection, XSS (low-confidence mapping), Heartbleed |
| 3 | `LATERAL_MOVEMENT` | TA0008 | **no** (no flow-level ground truth in primary data) | LANL red-team events (stretch) |
| 4 | `COMMAND_AND_CONTROL` | TA0011 | yes | CIC Bot/ARES, CTU-13 `CC` flows |
| 5 | `IMPACT` | TA0040 (DoS/DDoS) | yes | DoS/DDoS families |
| 6 | `EXFILTRATION` | TA0010 | **no** (no ground truth) | — shown only as an *inferred heuristic indicator*, never as a model prediction |

Two honest deviations from the suggested list: **Impact** is added because DoS/DDoS dominate the
datasets and are an ATT&CK tactic; **Lateral Movement** and **Exfiltration** stay in the taxonomy but
are *not* trained or predicted in the MVP because none of the primary datasets labels them at flow
level. The dashboard shows them as "insufficient ground truth".

### 7.3 Label → stage mapping (initial table)

| Dataset label (raw) | Stage | Candidate ATT&CK technique(s) — "consistent with", not proof |
|---|---|---|
| `Benign`, `BENIGN`, CTU `Normal` | BENIGN | — |
| CTU `Background` | **excluded** (unknown, masked) | — |
| `PortScan`, CTU `…Scan…` | RECON_DISCOVERY | T1595 Active Scanning (external src) / T1046 Network Service Discovery (internal src) |
| `Infilteration`/`Infiltration` | RECON_DISCOVERY if flow src is internal (post-compromise scanning); otherwise INITIAL_ACCESS_ATTEMPT | T1046; initial vector (T1566 Phishing / T1204 User Execution) is **not observable** in flows |
| `FTP-BruteForce`, `SSH-Bruteforce`, `FTP-Patator`, `SSH-Patator` | INITIAL_ACCESS_ATTEMPT | T1110.001 Password Guessing |
| `Brute Force -Web` | INITIAL_ACCESS_ATTEMPT | T1110 |
| `SQL Injection` | INITIAL_ACCESS_ATTEMPT | T1190 Exploit Public-Facing Application |
| `Brute Force -XSS` | INITIAL_ACCESS_ATTEMPT (low confidence) | T1190 (weak fit; flag as low confidence) |
| `Heartbleed` | INITIAL_ACCESS_ATTEMPT | T1190 |
| `DoS attacks-*`, `DoS *` | IMPACT | T1499 Endpoint DoS (e.g. .002 Service Exhaustion Flood) |
| `DDoS attacks-LOIC-HTTP`, `DDOS attack-HOIC`, `DDoS` | IMPACT | T1499.002 / T1498 Network DoS |
| `DDOS attack-LOIC-UDP` | IMPACT | T1498.001 Direct Network Flood |
| `Bot`, `Bot` (ARES) | COMMAND_AND_CONTROL | T1071.001 Web Protocols (if HTTP) |
| CTU `From-Botnet-…-CC…` | COMMAND_AND_CONTROL | T1071 / T1095 / T1571 depending on protocol/port |
| CTU `…DNS…`, `…Spam…`, `…ClickFraud…` botnet flows | **excluded** from stage training (no clean mapping) or `IMPACT` for DDoS | — |

Exact raw strings differ across versions — the loader prints `value_counts()` of the raw labels and
the unit test fails on any unmapped label.

### 7.4 From flow labels to host-window stages

```text
for each window w, internal host h:
    M = malicious flows in w where h is src or dst, grouped by stage
    if Σ|M| == 0                           → stage = BENIGN
    elif max_stage_count < 3 (min flows)   → stage = AMBIGUOUS (-1, masked in the loss)
    else                                   → stage = most-advanced stage with ≥3 flows
         (order: RECON < INITIAL_ACCESS < LATERAL < C2 < IMPACT; EXFIL not used)
also store: role (attacker-side / victim-side), multi-hot of all stages, n_malicious_flows
```

Known caveat: CIC labels were assigned by attacker IP + time interval, so some benign traffic
between those IPs is labelled malicious (and vice-versa). The ambiguity mask and "≥3 flows" rule
reduce but do not remove this noise. Record it as a limitation.

---

## 8. World Model training

### 8.1 Task definition

| | Tensor | Shape |
|---|---|---|
| Input | `x_hist` — normalised states `S_{t-L+1..t}` | `[B, 10, D]` |
| Dynamics target | `x_future` — `S_{t+1..t+K}` | `[B, 5, D]` |
| Current stage target | `y_now` | `[B]` (int, −1 = ambiguous) |
| Future stage target | `y_future` | `[B, 5]` |
| Horizon target | `y_within` = 1 if any future stage ∉ {BENIGN} | `[B]` |

### 8.2 Loss

```text
L = λ_dyn · Σ_k GaussianNLL(S_{t+k} | μ_k, σ²_k)          # learns P(S_{t+1} | h_t), multi-step
  + λ_now · CE(stage_t | h_t)                              # interprets the current state
  + λ_fut · Σ_k CE(stage_{t+k} | ĥ_{t+k})                   # interprets IMAGINED states
  + λ_hor · BCE(y_within | p_within)                       # p_within composed from the rollout (§10)
defaults: λ_dyn = 1.0, λ_now = 0.5, λ_fut = 1.0, λ_hor = 1.0
CE uses class weights ∝ 1/√freq and ignore_index = -1
```

The future-stage and horizon losses are computed **on states the model itself imagined**. This
forces the stage head to read the dynamics' predictions — the forecast cannot bypass the world
model.

### 8.3 Teacher forcing → scheduled sampling

- **History (filtering) phase:** always real observations `S_{t-L+1..t}`.
- **Rollout phase during training:** at step k feed either the true `S_{t+k}` (teacher forcing) or
  the model's own `μ_k`. Probability of feeding the truth decays linearly from 1.0 → 0.0 over the
  first 60 % of epochs, then stays 0 (pure free-running, identical to inference).
- **Inference:** only model predictions (the future is unknown).

This reduces the "compounding error" of recursive forecasting.

### 8.4 Class imbalance

- Train only: keep all sequences with any non-benign label in `[t-L+1, t+K]`, plus a random 10:1
  sample of all-benign sequences (ratio is a config value).
- Class weights in CE.
- Calibrate on the untouched validation set (natural prevalence), so downsampling does not distort
  the final probabilities.

### 8.5 Training loop (pseudocode)

```text
model = NetworkWorldModel(D, C)
opt   = AdamW(lr=1e-3, weight_decay=1e-4)
for epoch in 1..30:
    teacher_prob = max(0, 1 - epoch / (0.6 * 30))
    for batch in train_loader:
        out  = model(batch.x_hist, K, x_future=batch.x_future, teacher_prob=teacher_prob)
        loss = world_model_loss(out, batch)
        loss.backward(); clip_grad_norm(1.0); opt.step(); opt.zero_grad()
    val = evaluate(model, val_loader, teacher_prob=0)      # free-running, like inference
    early-stop on val PR-AUC(y_within) (patience 5); keep best checkpoint
fit temperature on stage logits (val); fit isotonic on p_within (val)
save model.pt, scaler.json, feature_list.json, calibrator.pkl, config.yaml, git commit hash
```

Real code: §22.8.

### 8.6 Probability calibration

- **Stage probabilities:** temperature scaling (one scalar T fitted by minimising NLL on val).
- **Horizon probability:** isotonic regression (or Platt) on val, fitted on exactly the quantity the
  system outputs (Monte-Carlo mean, §10).
- **Check:** reliability diagram, Expected Calibration Error (ECE), Brier score on **test**.

---

## 9. K-step forward simulation

### 9.1 Recursive vs direct

| Option | How | Pros | Cons |
|---|---|---|---|
| **Recursive (autoregressive)** | predict S_{t+1}, feed it back, repeat | a genuine simulator; any K; supports what-if interventions and uncertainty rollouts; one model for all horizons | errors compound |
| Direct multi-horizon | separate head for each k | no compounding | not a simulator, can't do counterfactuals; K fixed; closer to "a classifier with K outputs" |

**Decision: recursive rollout**, with the compounding-error problem handled by (1) multi-step
training loss, (2) scheduled sampling, (3) Monte-Carlo sampling to *show* the growing uncertainty.
Direct multi-horizon heads are kept as ablation A2 (§17.6).

### 9.2 Algorithm

```text
Input: history S_{t-L+1..t} for one host, N samples, horizon K
h_t  = GRU-filter(history)                       # summary of the past
current_stage = softmax(stage_head(h_t) / T)     # inferred current stage
repeat N times (in parallel, as a batch):
    h = h_t
    for k = 1..K:
        μ, logσ² = dynamics_head(h)
        Ŝ = μ + σ ⊙ ε,  ε ~ N(0, I)              # sample one plausible next state
        h = GRUCell(Encoder(Ŝ), h)                # imagine living through it
        p_k = softmax(stage_head(h) / T)          # what stage does this imagined state look like?
    store trajectory (Ŝ_1..Ŝ_K, p_1..p_K)
aggregate over N trajectories → mean and 5–95 % band for states and probabilities
```

Real code: §22.9.

### 9.3 Uncertainty honesty

MC sampling of `σ` captures *aleatoric* (inherent) variability. Model (*epistemic*) uncertainty is
not captured unless we add a small deep ensemble (3–5 models with different seeds) — a cheap optional
extension. Bands widen with k; the dashboard shows that explicitly.

---

## 10. Attack probability calculation

### 10.1 Per-step stage probability

For trajectory n and step k: `p_{n,k}(c) = P(stage_{t+k} = c | imagined state)`.
Report: `P(stage at t+k = c) = (1/N) Σ_n p_{n,k}(c)`.

### 10.2 Probability of attack within K windows

Per trajectory, treating per-step attack probabilities as conditional hazards:

```text
P_n(no attack in t+1..t+K) = Π_k p_{n,k}(BENIGN)
P_n(attack within K)       = 1 − Π_k p_{n,k}(BENIGN)
P_raw(attack within K)     = (1/N) Σ_n P_n(attack within K)
P(attack within K)         = isotonic_calibrator(P_raw)        # fitted on validation
```

The product form assumes the per-step judgements are conditionally independent given a trajectory
— a modelling assumption, not a fact. That is exactly why (a) the same composition is trained
end-to-end with BCE on `y_within`, and (b) it is calibrated on held-out data. The **published number
is judged by its calibration curve**, not by the elegance of the formula.

### 10.3 Other outputs

| Output | Definition |
|---|---|
| Forecast horizon | smallest k where the cumulative `1 − Π_{j≤k} p(BENIGN)` exceeds the alert threshold (else "none within K") |
| Predicted stage | `argmax_{c ≠ BENIGN} max_k P(stage at t+k = c)`, shown only if `P(attack within K) ≥ threshold` |
| Confidence | the 5–95 % MC band + calibration bin reliability; displayed as LOW/MEDIUM/HIGH based on band width |
| Network risk | `max_h P_h(attack within K)` over hosts, and top-5 hosts. (Noisy-OR across hundreds of correlated hosts would overstate risk, so we avoid it.) |
| Surprise score | `−log P(S_{t} | h_{t-1})` from the dynamics head, as a percentile of benign validation values — a stage-agnostic "this network is behaving unexpectedly" signal for unseen attacks |
| Alert threshold | chosen on validation to hit a target FPR on quiet windows (default 1 %) or an alert budget (e.g. ≤ N alerts/hour) |

### 10.4 Evaluation of probabilities

Reliability diagram, ECE (15 bins), Brier score, PR-AUC, and precision/recall/FPR at the chosen
threshold — all on the test split.

---

## 11. Explainability

### 11.1 Method choice

| Method | Works on GRU rollout? | Speed | Output | Verdict |
|---|---|---|---|---|
| Attention weights | no attention in GRU; even in Transformers attention ≠ attribution | fast | per-time weights | not used |
| SHAP (Kernel) | model-agnostic | **slow** (thousands of evaluations × 590 inputs) | per-feature | not for the world model |
| SHAP (Deep/Gradient) | RNN support is fragile | medium | | not used |
| Permutation importance | yes | medium | **global** only | used for global feature ranking in the report |
| **Integrated Gradients (Captum)** | **yes** (differentiable through the rollout) | fast (32–50 steps) | **(time × feature)** attribution for **each** prediction | **chosen** for per-prediction explanations |
| TreeSHAP | trees only | fast | per-feature | used for the XGBoost/RF baseline explanations |

### 11.2 Mechanism

1. **Target:** the horizon probability `P(attack within K)` (deterministic mean rollout, so the
   gradient is well defined). Optionally also `P(stage = c at t+k)` for the predicted stage.
2. **Baseline input:** the host's *benign median state* (per-feature median over benign training
   windows, repeated over L), not zeros — "compared to normal behaviour, what pushed the risk up?".
3. **Attributions:** `A[L, D]`. Temporal importance = `Σ_features |A|`; feature importance =
   `Σ_time A` (signed; positive = pushes risk up).
4. **Evidence translation:** for the top-5 positive features, render a template with the **raw
   (unscaled)** values, e.g.
   `uniq_dst_ports_out → "Host contacted {cur} distinct destination ports in the last window (its usual level: {base})"`.
5. **Faithfulness check (in the report):** deletion test — replace the top-5 attributed features with
   baseline values and measure the drop in probability; compare with deleting 5 random features.

### 11.3 Example output (format illustration — values are made up)

```text
PREDICTED (model estimate): elevated probability of attack activity involving 172.31.64.40 within 5 min
  P(attack within 5 windows) = 0.74   [band 0.61–0.83]
  Most likely future stage   = INITIAL_ACCESS_ATTEMPT (t+2)   → ATT&CK TA0006 / TA0001 (candidate)

OBSERVED (facts from traffic, last 3 windows):
  1. Distinct destination ports contacted: 12 → 88 → 301        (usual: ~6)
  2. Share of SYN with no reply: 0.05 → 0.41 → 0.83
  3. New internal peers never contacted before: 0 → 14 → 35
  4. Share of flows to SSH (22): 0.01 → 0.06 → 0.19

INFERRED (interpretation, not proof):
  Behaviour consistent with network service discovery (T1046) from an internal host.

Most influential time steps: t-1, t (last 2 minutes)
```

---

## 12. Dashboard

### 12.1 Modes

- **Replay mode (primary demo):** load a pre-processed test day; a time slider sets "now"; everything
  updates as if it were live. Ground-truth labels can be toggled on as a *separate* band ("dataset
  label — not available in real deployment").
- **Upload mode:** upload a CSV (CICFlowMeter/NFStream format) or a PCAP → run the full pipeline
  locally.

### 12.2 Pages / layout

```text
┌ Sidebar ──────────────┐ ┌ OVERVIEW ───────────────────────────────────────────────────────┐
│ Data: [test day ▾]    │ │ [Network risk: HIGH] [P(attack ≤5 min): 0.74 (0.61–0.83)]       │
│ Upload CSV / PCAP     │ │ [Predicted stage: Initial-access attempt @ t+2] [Horizon: 2 win] │
│ Now: ──●────── 10:03  │ │ Top hosts at risk (table: host, P, stage, trend sparkline)       │
│ Horizon K: 5          │ ├ TIMELINE ───────────────────────────────────────────────────────┤
│ Threshold: FPR 1%     │ │ past (observed/inferred)   │ now │  future (predicted, fan band)  │
│ Show ground truth ☐   │ │ Benign → Recon ─────────── ▲ ───► Initial access? ─► …           │
│ Ledger: ON/OFF        │ ├ NETWORK ────────────────────────────────────────────────────────┤
└───────────────────────┘ │ Plotly host graph (node colour = risk, edge width = flows);     │
                          │ table: src IP, dst IP, dst ports, suspicious flag               │
                          ├ EXPLAIN ────────────────────────────────────────────────────────┤
                          │ top evidence cards + IG heatmap (time × feature)                │
                          ├ EVIDENCE ───────────────────────────────────────────────────────┤
                          │ suspicious flows from the last windows (sortable)               │
                          ├ FORECAST ───────────────────────────────────────────────────────┤
                          │ stacked per-stage probabilities for t+1..t+K; state-feature      │
                          │ forecasts with bands vs actual (replay mode)                     │
                          ├ WHAT-IF (stretch) ──────────────────────────────────────────────┤
                          │ "block SMB/SSH for host X" → re-simulate → new probabilities     │
                          ├ AUDIT ──────────────────────────────────────────────────────────┤
                          │ ledger records, verify chain button, tamper demo                 │
                          └─────────────────────────────────────────────────────────────────┘
```

Consistent badges everywhere: **OBSERVED** (grey), **INFERRED** (amber), **PREDICTED** (blue). Risk
levels come from calibrated thresholds, not a hard-coded 0.5.

### 12.3 Implementation notes

- `st.cache_resource` for the model; `st.cache_data` for parquet loading.
- Plotly for all charts (bundled with the Python package → offline). The network graph is drawn with
  NetworkX layout + Plotly scatter (avoid pyvis defaults that pull JS from a CDN).
- Replay mode reads precomputed `forecasts.parquet` for instant scrubbing; live inference is used for
  uploads and the what-if panel.

---

## 13. Offline operation

| Component | Needs internet? | Core demo? |
|---|---|---|
| Python, PyTorch, scikit-learn, pandas, NumPy, pyarrow, Captum, Streamlit, Plotly, NetworkX, Scapy, NFStream, tshark | only for **installation** | yes |
| Dataset download (AWS CLI / website) | one-time | done before demo |
| MITRE ATT&CK data | one-time: we ship a small curated YAML (+ optional pinned copy of `enterprise-attack.json`) | yes (local file) |
| Streamlit usage stats | disabled via `.streamlit/config.toml` (`gatherUsageStats = false`) | — |
| LLM APIs, cloud dashboards, online threat intel feeds, CVE/NVD live queries | **not used** | removed from core |
| Local EVM chain (Hardhat/Anvil) | runs locally; install needs internet once | optional |

Demo rule: record the demo video with Wi-Fi **off** to prove the requirement.

---

## 14. Repository structure

```text
NetworkAttackForcasting/
├── README.md                     # problem, architecture, install, data prep, train, eval, dashboard
├── LICENSE
├── requirements.txt              # core deps (CPU)
├── requirements-optional.txt     # xgboost, nfstream, web3, cryptography, pyshark
├── Makefile                      # make data / windows / train / eval / app / test
├── .streamlit/config.toml        # offline, no telemetry
├── configs/
│   ├── default.yaml              # window=60s, L=10, K=5, model sizes, loss weights, seeds
│   ├── cic2018.yaml              # paths, internal CIDRs, timezone offset, split days
│   ├── cic2017.yaml              # cross-dataset test config
│   └── ctu13.yaml                # scenarios, label parsing
├── data/                         # gitignored except README + tiny demo sample
│   ├── raw/  interim/  processed/
│   └── sample/demo_day_small.parquet
├── docs/
│   ├── BLUEPRINT.md              # this document
│   ├── ARCHITECTURE.md           # 2-page architecture document
│   ├── SLIDES.md                 # 5-slide outline
│   ├── LABEL_MAPPING.md          # label→stage table + justification + citations
│   └── problem_statement_SIH26153.md
├── netwm/                        # the Python package
│   ├── config.py                 # load/merge YAML configs
│   ├── io/
│   │   ├── schema.py             # canonical flow schema + column maps per dataset
│   │   ├── load_cic.py           # CIC-IDS2017/2018 CSV → canonical flows
│   │   ├── load_ctu13.py         # binetflow → canonical flows
│   │   ├── pcap_flows.py         # PCAP → flows (NFStream)
│   │   └── pcap_packets.py       # PCAP → packet features (tshark / Scapy)
│   ├── labels/
│   │   ├── label_to_stage.yaml   # versioned mapping table
│   │   └── build_labels.py       # flow labels → host-window stages
│   ├── features/
│   │   ├── feature_spec.py       # ordered feature list, which get log1p
│   │   ├── windowing.py          # flows → per host-window features (blocks A–G)
│   │   ├── packet_features.py    # block H aggregation
│   │   ├── temporal.py           # block I (causal deltas, EWMA z)
│   │   ├── global_context.py     # block J
│   │   └── scaler.py             # robust scaler fit/transform/save
│   ├── data/
│   │   ├── splits.py             # Protocol A/B/C splits + purge gaps
│   │   └── sequences.py          # HostSequenceDataset
│   ├── models/
│   │   ├── world_model.py        # NetworkWorldModel
│   │   └── baselines.py          # persistence, LogReg, RF/XGBoost wrappers
│   ├── training/
│   │   ├── losses.py
│   │   ├── train_world_model.py
│   │   ├── train_baselines.py
│   │   └── calibrate.py
│   ├── forecasting/
│   │   ├── rollout.py            # Monte-Carlo K-step simulation
│   │   └── risk.py               # P(within K), stage probs, horizon, surprise
│   ├── attack_mapping/
│   │   ├── attack_subset.yaml    # tactic/technique IDs + names used (offline)
│   │   ├── indicator_rules.py    # observed indicators → candidate techniques
│   │   └── mapper.py             # builds observed / inferred / predicted blocks
│   ├── explain/
│   │   ├── ig.py                 # Captum Integrated Gradients
│   │   ├── evidence.py           # templates → analyst text
│   │   └── faithfulness.py       # deletion test
│   ├── evaluation/
│   │   ├── metrics.py            # P/R/F1/FPR/ROC/PR/Brier/ECE
│   │   ├── lead_time.py          # onset detection, recall@lead
│   │   ├── bootstrap.py          # CIs by episode
│   │   └── plots.py
│   ├── ledger/
│   │   ├── hashchain.py          # SQLite hash chain + Ed25519 signatures
│   │   ├── anchor_evm.py         # optional: Merkle root → local chain
│   │   └── contracts/EvidenceAnchor.sol
│   └── inference/
│       └── pipeline.py           # CSV/PCAP → forecasts + explanations (used by CLI + dashboard)
├── scripts/
│   ├── download_cic2018.sh
│   ├── preprocess.py             # raw → flows.parquet + labels.parquet
│   ├── build_windows.py          # flows → host_windows.parquet (+ scaler)
│   ├── extract_packets.py        # PCAP → packet_feats.parquet
│   ├── evaluate.py               # all models → reports/
│   └── verify_ledger.py
├── train.py                      # thin wrapper: world model (+ --baselines)
├── predict.py                    # thin wrapper: CSV/PCAP → forecast JSON
├── app.py                        # Streamlit entry point
├── dashboard/
│   ├── pages/ (overview, timeline, network, explain, evidence, forecast, whatif, audit)
│   └── components/ (charts.py, badges.py, graph.py)
├── notebooks/
│   ├── 01_eda_label_timeline.ipynb   # align labels vs time, verify timezone offset
│   ├── 02_feature_sanity.ipynb
│   └── 03_results.ipynb
├── tests/
│   ├── test_schema.py  test_windowing.py  test_causality.py  test_labels.py
│   ├── test_splits_no_leakage.py  test_model_shapes.py  test_overfit_batch.py
│   ├── test_rollout.py  test_toy_world.py  test_ledger.py  test_dashboard_smoke.py
└── artifacts/                    # gitignored: model.pt, scaler.json, calibrator.pkl, reports/
```

---

## 15. Implementation plan

Assumes roughly 5–6 weeks before submission; compress proportionally. **MVP = Phases 1–6 + a basic
9 + 10.** Phases 7–8 are required by the PS but can start in parallel.

| Phase | Exact task | Expected output | Files | Tech | Difficulty | Depends on |
|---|---|---|---|---|---|---|
| **1. Dataset** | Download CIC-IDS2018 flow CSVs (+ one CIC-IDS2017 day for dev, CTU-13 scenarios); plot per-minute label counts per day; determine timezone offset and internal CIDRs | data on disk; `01_eda` notebook with label timelines; config values filled | `scripts/download_cic2018.sh`, `configs/*.yaml`, notebook 01 | AWS CLI, pandas | Easy | — |
| **2. Preprocessing** | Canonical schema, cleaning, label→stage mapping, unit tests | `flows.parquet`, `labels.parquet` per day; mapping YAML | `netwm/io/*`, `netwm/labels/*`, `scripts/preprocess.py`, `tests/test_schema.py`, `test_labels.py` | pandas, pyarrow | Medium | 1 |
| **3. Temporal states** | Windowing (blocks A–G, I, J), scaler, splits, sequence dataset, causality + leakage tests | `host_windows.parquet`, `scaler.json`, split manifest | `netwm/features/*`, `netwm/data/*`, `scripts/build_windows.py`, tests | pandas, numpy, PyTorch | Medium–Hard | 2 |
| **4. Baselines** | Persistence, LogReg on S_t, LogReg + XGBoost on flattened history; threshold at val FPR | baseline predictions + metrics CSV | `netwm/models/baselines.py`, `training/train_baselines.py` | scikit-learn, xgboost | Easy | 3 |
| **5. World Model** | Model, losses, training loop with scheduled sampling, early stopping, overfit-one-batch test, toy-world test | `model.pt`, training curves | `netwm/models/world_model.py`, `training/*`, `train.py` | PyTorch | Hard | 3 |
| **6. K-step forecasting** | MC rollout, risk calculation, calibration, lead-time evaluation | `forecasts.parquet` for test days; calibration plots; lead-time table | `netwm/forecasting/*`, `training/calibrate.py`, `evaluation/lead_time.py` | PyTorch, scikit-learn | Medium | 5 |
| **7. MITRE mapping** | Curated ATT&CK subset, indicator rules, observed/inferred/predicted builder, docs | mapping YAML, `LABEL_MAPPING.md`, mapper output in JSON | `netwm/attack_mapping/*`, docs | YAML | Easy–Medium | 2 (rules), 6 (predicted part) |
| **8. Explainability** | IG wrapper, benign baseline, evidence templates, deletion-test faithfulness | explanation JSON per forecast; faithfulness table | `netwm/explain/*` | Captum | Medium | 5 |
| **9. Dashboard** | Replay mode first, then upload mode; pages as §12 | running `streamlit run app.py` | `app.py`, `dashboard/*`, `netwm/inference/pipeline.py` | Streamlit, Plotly, NetworkX | Medium | 6, 7, 8 (can start with mock JSON from day 1) |
| **10. Evaluation + demo** | Full experiment grid (§17), ablations, bootstrap CIs, packet ablation, ledger, docs, video | `reports/`, figures, README, 2-page doc, slides, video | `scripts/evaluate.py`, `docs/*` | matplotlib | Medium | all |

Optional after MVP: packet block (H) on infiltration days, CTU-13 C&C, what-if panel, ensemble,
ledger anchoring, MDN head.

---

## 16. Team division

Integration works through **file contracts** (fixed schemas), so members can work in parallel with
mock data from day 1.

| Contract | Producer → Consumer | Content |
|---|---|---|
| C1 `flows.parquet` | Data → Features | canonical flow schema (§22.3) |
| C2 `labels.parquet` | Data/Security → Training/Eval | `day, window, host, stage, n_mal_flows, ambiguous, role` |
| C3 `host_windows.parquet` + `feature_list.json` | Features → ML | `day, window, host, f_1..f_D` |
| C4 model bundle | ML → Backend | `model.pt, scaler.json, calibrator.pkl, config.yaml` |
| C5 forecast JSON (§22.14) | Backend → Dashboard/Ledger | observed / inferred / predicted / explanation |

| Member | Role | Owns | Integrates with |
|---|---|---|---|
| M1 | **Data engineering** | Phases 1–3: loaders, cleaning, windowing, scaler, packet extraction | produces C1, C3 |
| M2 | **ML / world model** | Phase 5–6: model, training, rollout, calibration | consumes C2, C3; produces C4 |
| M3 | **Cybersecurity / ATT&CK** | label→stage mapping (C2), indicator rules, evidence templates, limitations section, verifying labels in EDA | reviews every "inferred" claim |
| M4 | **Backend / inference + ledger** | `pipeline.py`, `predict.py`, forecast JSON (C5), explainability wrapper, hash-chain ledger | consumes C4, produces C5 |
| M5 | **Frontend / dashboard** | Streamlit app against mock C5 from week 1, replay mode, graph view | consumes C5 |
| M6 (or shared M2+M3) | **Evaluation / documentation** | baselines (Phase 4), metrics, lead-time, ablations, README, 2-page doc, slides, video | consumes all |

With 4 members: merge M3+M6 and M4+M5.

---

## 17. Baseline experiment design

### 17.1 Fixed across all models

Same `host_windows.parquet`, same feature list, same scaler, same Protocol-A split, same sequences
(identical sample IDs), same val-chosen threshold rule (FPR = 1 % on quiet validation windows), same
seeds (report mean ± std over 3 seeds for neural models).

### 17.2 Models

| ID | Model | Input | Output | What it tests |
|---|---|---|---|---|
| B0 | Persistence | current label/state | `y_within = (stage_t ≠ benign)`; `Ŝ_{t+k} = S_t` | "the future equals now" — the minimum a forecaster must beat |
| B1 | **Logistic Regression** (required) | `S_t` only (D) | `y_within` | static classifier on the same features |
| B2 | Logistic Regression, lagged | flattened `S_{t-L+1..t}` (L·D) | `y_within` | temporal *features* without learned dynamics |
| B3 | XGBoost (or RF) lagged | flattened history | `y_within` | strong non-linear non-dynamic baseline |
| B4 (opt.) | Detector + Markov chain | B1 current-stage + empirical stage transition matrix from train labels | `P(stage_{t+k})` | interpretable "classic" forecast |
| **WM** | Host-centric GRU World Model | `S_{t-L+1..t}` | everything | the proposal |

### 17.3 Metrics

- **Horizon forecasting (`y_within`):** Precision, Recall, F1, FPR at the val threshold; ROC-AUC;
  **PR-AUC** (primary — imbalanced); Brier; ECE.
- **Stage forecasting:** macro-F1 of predicted stage at each k (on windows with non-ambiguous labels).
- **State forecasting (dynamics):** NLL and MSE of `Ŝ_{t+k}` vs truth for k = 1..K, compared with
  persistence. *This is the direct evidence that the model learned transition dynamics.*
- **Lead time:** for each attack **onset** (first attack window after ≥5 benign windows on a host):
  recall of `P(within K) ≥ threshold` at `T−1`, `T−2`, `T−3`, at the same FPR.
- **Onset vs ongoing split:** PR-AUC separately on windows where the host is currently benign
  ("onset forecasting") and currently under attack ("continuation forecasting").

### 17.4 Why the onset/ongoing split matters

When the host is *already* under attack, "attack in the next 5 minutes" is easy for any model (B0/B1
will look excellent). The interesting question — and the whole point of SIH26153 — is performance on
**currently benign** windows that precede an attack. Report both; lead with onset.

### 17.5 Expected tables (templates — fill with real results)

**Table 1 — Horizon forecast, Protocol A test, K = 5, threshold at val FPR 1 %**

| Model | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC | PR-AUC (onset) | PR-AUC (ongoing) | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 Persistence | | | | | | | | | | |
| B1 LogReg S_t | | | | | | | | | | |
| B2 LogReg lagged | | | | | | | | | | |
| B3 XGBoost lagged | | | | | | | | | | |
| WM (mean ± std, 3 seeds) | | | | | | | | | | |

**Table 2 — Early-warning recall at fixed FPR** (columns: lead 0 / T−1 / T−2 / T−3, #onsets)

**Table 3 — State forecasting NLL/MSE per k** (WM vs persistence)

**Table 4 — Stage macro-F1 per k**

**Table 5 — Generalisation:** Protocol B (unseen families), CTU-13 leave-scenario-out, CIC-IDS2017
cross-dataset.

### 17.6 Ablations

| ID | Change | Question answered |
|---|---|---|
| A1 | λ_dyn = 0 (no dynamics loss) | Does learning dynamics help, or is it just a sequence classifier? |
| A2 | direct multi-horizon heads instead of recursive rollout | value of simulation vs direct prediction |
| A3 | L ∈ {1, 5, 10, 20} | how much history matters |
| A4 | window ∈ {30, 60, 120} s | temporal resolution |
| A5 | − graph block G / − global block J / + packet block H (infiltration days) | value of each feature family (flow vs flow+packet requirement) |
| A6 | teacher forcing only vs scheduled sampling | compounding-error handling |
| A7 | Gaussian vs MDN dynamics head (stretch) | multi-modal futures |

### 17.7 Plots

1. **The key plot:** for one test host, `P(attack within 5)` over time with the dataset label band
   underneath (WM vs B1), showing whether the WM rises *before* onset.
2. Recall-vs-lead-time curves (WM vs baselines).
3. PR curves (overall, onset subset).
4. Reliability diagrams.
5. State-forecast error vs k (WM vs persistence).
6. Stage confusion matrices at k = 1 and k = 3.
7. IG heatmap examples + deletion-test bars.

### 17.8 What counts as meaningful evidence

- WM beats **B2/B3** (not just B1) on **onset PR-AUC** and **recall at T−1/T−2**, at equal FPR, with
  bootstrap 95 % CIs computed by resampling **attack episodes/host-days** (windows are correlated —
  resampling windows gives falsely narrow intervals).
- WM beats persistence on state-forecast NLL for k ≥ 1 (it learned dynamics).
- A1 shows a drop when the dynamics loss is removed (the world-model component matters).
- Calibration: ECE low and reliability curve near the diagonal on test.
- **If WM does not beat B2/B3, report it.** That result is still informative (it means the temporal
  signal lives in lagged features, not in learned dynamics) and judges respect honesty more than a
  suspicious 99.9 % F1. Very high scores on CIC data usually signal leakage or shortcut features —
  investigate before celebrating.

---

## 18. Innovations and trade-offs

Presented without ranking.

**(a) Uncertainty-aware Monte-Carlo rollouts (+ optional small ensemble)**
- Idea: sample N trajectories from the Gaussian dynamics head; show a fan chart of future risk; an
  ensemble of 3–5 seeds adds model uncertainty.
- Difficulty: low (sampling is ~10 lines; ensemble = train 3× and average).
- Demo value: high — analysts see "0.74, but anywhere from 0.61 to 0.83".
- Research value: moderate; uncertainty in cyber forecasting is under-reported.
- Trade-off: aleatoric-only without ensemble; ensemble multiplies training/inference time.

**(b) Counterfactual "what-if" simulation**
- Idea: intervene on the imagined states during rollout (e.g. force `svc_smb = 0` and
  `new_peers_out = 0` for a host = "isolate SMB") and compare risk trajectories.
- Difficulty: medium (rollout already exists; need a clean intervention API and UI).
- Demo value: very high — only a simulator can do this; a classifier cannot.
- Research value: high, but must be framed carefully: the model is **correlational**, so this is a
  *model-based what-if*, not a causal guarantee that blocking SMB prevents the attack.

**(c) World-model surprise score for unseen attacks**
- Idea: `−log P(S_t | h_{t-1})`, calibrated as a percentile of benign validation values, flags
  behaviour the dynamics did not expect — independent of known attack labels.
- Difficulty: low (the dynamics head already provides it).
- Demo value: medium — "unusual trajectory" alert on Protocol-B unseen families.
- Research value: high — directly addresses "generalise to unseen attack patterns".
- Trade-off: also fires on benign novelty (new software rollout) → more false positives; must be
  shown as a separate signal, never merged silently into the attack probability.

**(d) Host-centric temporal-graph-lite state**
- Idea: per-host sequences with graph-derived features (degree, novelty of peers, 1-hop aggregates)
  instead of a full Temporal GNN.
- Difficulty: medium (feature engineering; no new model class).
- Demo value: high — gives "relevant hosts" and a network graph view for free.
- Research value: medium; a clean ablation (A5) vs a real TGN would be a good future-work paper.
- Trade-off: misses multi-hop structure a GNN could learn.

**(e) Multi-resolution evidence: flow + packet features, multi-scale windows**
- Idea: add packet-level block H (TTL variance, fragmentation, retransmissions, sequential-port
  score) and optionally a second, coarser window scale (e.g. 5 min) to catch slow scans.
- Difficulty: medium–high (PCAP processing is heavy; multi-scale needs a second encoder).
- Demo value: medium; directly satisfies the PS "both levels" requirement.
- Research value: medium–high — slow, low-rate scans are exactly what flow thresholds miss.
- Trade-off: PCAP storage/time; must restrict to a subset of days → smaller evaluation set.

**Realistic for the SIH prototype without over-engineering:** (a) and (c) come almost free with the
chosen model; (d) is already part of the core feature design; (e) is feasible on the two infiltration
days only (packet block) with multi-scale as stretch; (b) is a moderate add-on once the rollout is
stable.

---

## 19. Blockchain

### 19.1 Where it does **not** make sense

- **Not in the model** (training, inference, features). Consensus adds latency and nothing to accuracy.
- **Not for raw traffic/PCAP** — size, and privacy (IP addresses are personal data under India's
  DPDP Act 2023); store hashes only.
- **Not as a "decentralised AI"/federated-learning-on-chain gimmick** for this prototype.
- **Not a single-node "blockchain" claimed to add trust:** one party running one chain gives no more
  guarantee than a signed hash chain. The real value appears only when **independent parties** hold
  copies (e.g. an organisation's SOC and a sector CERT / NCIIPC).

### 19.2 Where it does make sense (and what we build)

| Use | Mechanism | Built in prototype? |
|---|---|---|
| Tamper-evident forecast/alert log | append-only SHA-256 hash chain (each record contains the previous hash) + Ed25519 signature by the sensor | **yes (core of the ledger)** |
| Model provenance / audit trail | at startup record hash of `model.pt`, scaler, config, git commit; every forecast references it | **yes** |
| Incident evidence integrity | hash of the exact input CSV/PCAP slice and feature vector behind an alert | **yes** |
| External anchoring | every N records (or hourly) compute a Merkle root and write it to a local EVM chain (Hardhat or Anvil — Ganache is discontinued) via a 10-line Solidity contract | optional demo |
| Multi-organisation threat-intel sharing | permissioned ledger (Hyperledger Fabric) shared by CII organisations exchanging *hashed/abstracted* forecast summaries | design only (future scope) |

Tamper demo for judges: edit one past record in SQLite → "Verify chain" turns red at that record.

Honest note: a hash chain alone cannot stop someone who can rewrite the *entire* database and
recompute all hashes — signatures and external anchoring (a copy held by another party) are what
close that gap.

The core system runs with `ledger.enabled: false`; the ledger only consumes forecast JSON (C5).

---

## 20. Cybersecurity correctness & limitations

**Language rules for all UI text, docs, slides:**
- Never: "an attack WILL happen", "the model detected APT X", "proves lateral movement".
- Always: "the model estimates an increased probability of a future attack trajectory based on
  observed network behaviour"; "behaviour consistent with …"; "candidate technique".

**Observed / inferred / predicted** — enforced by the forecast JSON structure (§22.14):
- *Observed* = computed directly from traffic (counts, ratios, flows).
- *Inferred* = interpretation of the current state (current-stage head, indicator rules, ATT&CK
  candidates).
- *Predicted* = world-model outputs about future windows, with probability and band.

| Issue | Impact | Mitigation in this design |
|---|---|---|
| False positives | alert fatigue | threshold chosen by FPR/alert budget; show band; top hosts only |
| False negatives | missed attacks, esp. without precursors | report onset recall honestly; surprise score as second signal |
| Concept drift | benign behaviour changes over weeks | per-host EWMA baselines; monitor surprise distribution; periodic recalibration |
| Unseen attacks | stage head only knows trained families | Protocol B evaluation; surprise score; "unknown/novel" label in UI |
| Adversarial traffic | slow scans, mimicry, padding | packet block (timing/sequential ports), multi-scale windows; acknowledge a determined attacker can evade |
| Encrypted traffic | payload invisible | we only use metadata (sizes, timing, flags, ports), which still works on TLS — but port-based service buckets are weaker when services run on 443 |
| Dataset bias | synthetic benign traffic, few attacker IPs, attacker OS fingerprints (TTL/window) can become shortcuts | no IP features; check whether TTL/init-window features dominate attributions; cross-dataset tests |
| Temporal leakage | inflated scores | day-level splits, purge gaps, causal features, automated tests (§6.3) |
| Class imbalance | trivial "all benign" models | PR-AUC, class weights, onset metrics, natural-prevalence calibration |
| Label noise | wrong stages | ≥3-flow rule, ambiguity mask, documented caveats |
| Model uncertainty | over-confident numbers | calibration + MC band + optional ensemble |
| Extractor mismatch | PCAP→NFStream features differ from CICFlowMeter | same canonical subset; show a distribution-shift warning on uploads; replay mode for the demo |

---

## 21. Final deliverables

### 21.1 GitHub
Complete source (`netwm/`, scripts, dashboard), configs, trained weights for the demo configuration
(small — ~0.5 MB), `scaler.json`, `calibrator.pkl`, tests, a tiny demo data sample, `reports/` with
metric CSVs and figures.

### 21.2 README outline
1. Problem (3 lines) and what makes it a world model
2. Architecture diagram
3. Installation (CPU/GPU)
4. Dataset preparation (download commands, expected folder layout, timezone/CIDR config)
5. Preprocessing & windowing commands
6. Training (world model + baselines)
7. Inference (`predict.py` on CSV/PCAP) with sample output
8. Dashboard (`streamlit run app.py`)
9. Evaluation (commands, tables, figures)
10. Limitations & responsible use
11. Reproducibility (seeds, config hashes, versions)

### 21.3 2-page architecture document (`docs/ARCHITECTURE.md`)
- Page 1: problem framing (classifier vs world model), architecture diagram, state definition,
  model diagram with shapes, rollout + probability formulas.
- Page 2: datasets & leakage-free split, stage mapping principle (observed/inferred/predicted),
  explainability, evaluation protocol (lead time, onset vs ongoing, baselines), ledger, limitations.

### 21.4 5-slide presentation
1. **Problem** — "detection after the fact" vs "forecast before completion"; the timeline figure.
2. **Proposed solution** — host-centric world model: learns `P(S_{t+1}|h_t)`, simulates K steps,
   maps to ATT&CK, explains; offline.
3. **Architecture** — pipeline + model diagram + dashboard screenshot.
4. **Results** — Table 1 onset columns, recall-vs-lead curve, key host timeline plot, calibration.
   Only real numbers.
5. **Impact / future scope** — enterprise & CII SOC early warning; ledger for audit and
   inter-agency sharing; Temporal GNN, LANL lateral movement, live NetFlow/IPFIX ingestion.

### 21.5 Demo video (≤ 2 min) — script
| Time | Scene |
|---|---|
| 0:00–0:15 | problem in one sentence; Wi-Fi off icon visible |
| 0:15–0:50 | replay of an infiltration test day: scrub the slider; host risk rises; timeline shows predicted stage with band; ground-truth band toggled on to show where the attack began |
| 0:50–1:15 | explain panel: top evidence with real values, IG heatmap; observed/inferred/predicted badges |
| 1:15–1:30 | forecast panel (per-stage probabilities, state forecast vs actual); what-if if ready |
| 1:30–1:45 | baseline comparison figure (real results) |
| 1:45–2:00 | ledger: tamper a record → verification fails; closing line |

Pre-compute all forecasts; keep screenshots as a fallback.

---

## 22. Build sheet

### 22.1 Exact technology stack

| Purpose | Tool | Notes |
|---|---|---|
| Language | Python 3.11 | |
| Data | pandas ≥ 2.1, numpy, pyarrow | parquet everywhere |
| Deep learning | PyTorch ≥ 2.2 (CPU build fine) | |
| Classic ML | scikit-learn ≥ 1.4, xgboost ≥ 2.0 (optional) | |
| Explainability | captum ≥ 0.7; shap (for tree baseline only) | |
| PCAP → flows | nfstream (Linux/macOS; WSL2 on Windows) | |
| PCAP → packet features | tshark (Wireshark CLI) for bulk; scapy for small files/tests | pyshark optional (wraps tshark) |
| Graph | networkx | |
| Dashboard | streamlit ≥ 1.33, plotly | |
| Config | pyyaml | |
| Ledger | hashlib (stdlib), sqlite3 (stdlib), cryptography (Ed25519); optional web3.py + Hardhat/Anvil | |
| Testing | pytest | |
| Download | awscli (CIC-IDS2018 public S3) | one-time |

### 22.2 Exact dataset recommendation

- **Primary:** CSE-CIC-IDS2018 "Processed Traffic Data for ML Algorithms" (10 daily CSVs), or the
  corrected re-extraction with full flow keys if obtainable. Check the IP columns first thing in
  Phase 1; if they are missing, switch to the corrected version or use the network-level fallback.
- **Secondary:** CTU-13 scenarios (e.g. 1, 2, 9 for Neris; 3 for Rbot; plus one other family) —
  choose after checking label counts.
- **Cross-dataset:** CIC-IDS2017 "TrafficLabelling" CSVs (Tuesday, Thursday, Friday).
- **Packet subset:** CIC-IDS2018 PCAPs of 28-02 and 01-03 (victim hosts involved in infiltration +
  a sample of benign hosts).

### 22.3 Canonical flow schema (contract C1)

| Column | Type | CIC-IDS2018 source | CIC-IDS2017 source | CTU-13 source |
|---|---|---|---|---|
| `ts` | datetime64[ns, UTC] | `Timestamp` (dayfirst) + offset | ` Timestamp` + offset | `StartTime` |
| `src_ip`, `dst_ip` | string | `Src IP`, `Dst IP` (if present) | `Source IP`, `Destination IP` | `SrcAddr`, `DstAddr` |
| `src_port`, `dst_port` | int32 | `Src Port`, `Dst Port` | `Source Port`, `Destination Port` | `Sport`, `Dport` (hex possible) |
| `proto` | int8 (6/17/1) | `Protocol` | `Protocol` | `Proto` (text → number) |
| `duration_s` | float32 | `Flow Duration` / 1e6 | `Flow Duration` / 1e6 | `Dur` |
| `fwd_pkts`, `bwd_pkts` | int32 | `Tot Fwd Pkts`, `Tot Bwd Pkts` | `Total Fwd Packets`, `Total Backward Packets` | NaN / `TotPkts` only |
| `fwd_bytes`, `bwd_bytes` | int64 | `TotLen Fwd Pkts`, `TotLen Bwd Pkts` | `Total Length of Fwd Packets`, `… Bwd …` | `SrcBytes`, `TotBytes − SrcBytes` |
| `flow_iat_mean` | float32 | `Flow IAT Mean` | `Flow IAT Mean` | NaN |
| `fwd_pkt_len_mean`, `bwd_pkt_len_mean`, `pkt_len_std` | float32 | `Fwd Pkt Len Mean`, `Bwd Pkt Len Mean`, `Pkt Len Std` | equivalents | NaN |
| `syn_cnt`, `ack_cnt`, `fin_cnt`, `rst_cnt`, `psh_cnt`, `urg_cnt` | int16 | `* Flag Cnt` | `* Flag Count` | parsed from `State` |
| `init_fwd_win` | int32 | `Init Fwd Win Byts` | `Init_Win_bytes_forward` | NaN |
| `label_raw` | string | `Label` | ` Label` | `Label` |
| `day` | string | from filename | from filename | scenario id |

(Column names vary slightly between releases — strip whitespace and keep the maps in
`netwm/io/schema.py`. CICFlowMeter flag counters are known to be imperfect; we only use them as
"flag present" indicators.)

### 22.4 Exact feature pipeline

```text
raw CSV ─► load_cic.py ─► clean (strip cols, drop header rows, inf→NaN, dedupe, parse ts, apply tz offset)
        ─► flows.parquet (C1)                     ─► build_labels.py ─► labels.parquet (C2)
        ─► windowing.py   : window = floor((ts - day_start)/60 s); host view (src & dst); blocks A–G
        ─► packet_features.py (optional) : block H joined on (day, window, host)
        ─► temporal.py    : block I, causal, per host sorted by window
        ─► global_context.py : block J joined on (day, window)
        ─► reindex each host to the full window grid (idle windows → zeros, active=0)
        ─► scaler.py      : log1p + robust scale (fit: train-day benign windows) + clip
        ─► host_windows.parquet (C3) + feature_list.json + scaler.json
```

### 22.5 Windowing code (blocks A–C shown; the rest follow the same pattern)

```python
# netwm/features/windowing.py
import ipaddress
import numpy as np
import pandas as pd

WINDOW_S = 60


def make_is_internal(cidrs):
    nets = [ipaddress.ip_network(c) for c in cidrs]

    def is_internal(ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(addr in n for n in nets)

    return is_internal


def add_window_index(flows: pd.DataFrame, day_start: pd.Timestamp) -> pd.DataFrame:
    seconds = (flows["ts"] - day_start).dt.total_seconds()
    return flows.assign(window=(seconds // WINDOW_S).astype(np.int32))


def host_view(flows: pd.DataFrame, cidrs) -> pd.DataFrame:
    """Each flow appears once for its source (direction out) and once for its destination (in),
    keeping only internal hosts."""
    is_internal = make_is_internal(cidrs)
    uniq_ips = pd.unique(pd.concat([flows["src_ip"], flows["dst_ip"]]))
    internal = {ip for ip in uniq_ips if is_internal(ip)}   # evaluate each IP once (fast)

    out = flows.assign(host=flows["src_ip"], peer=flows["dst_ip"], is_out=1,
                       bytes_sent=flows["fwd_bytes"], bytes_recv=flows["bwd_bytes"],
                       pkts_sent=flows["fwd_pkts"], pkts_recv=flows["bwd_pkts"])
    inn = flows.assign(host=flows["dst_ip"], peer=flows["src_ip"], is_out=0,
                       bytes_sent=flows["bwd_bytes"], bytes_recv=flows["fwd_bytes"],
                       pkts_sent=flows["bwd_pkts"], pkts_recv=flows["fwd_pkts"])
    v = pd.concat([out, inn], ignore_index=True)
    v = v[v["host"].isin(internal)].copy()
    v["peer_internal"] = v["peer"].isin(internal).astype(np.int8)
    return v


def entropy_of_counts(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def host_window_features(v: pd.DataFrame) -> pd.DataFrame:
    v = v.copy()
    total_pkts = v["fwd_pkts"] + v["bwd_pkts"]
    v["peer_out"] = v["peer"].where(v["is_out"] == 1)
    v["peer_in"] = v["peer"].where(v["is_out"] == 0)
    v["dport_out"] = v["dst_port"].where(v["is_out"] == 1)
    v["dport_in"] = v["dst_port"].where(v["is_out"] == 0)
    v["syn_flow"] = (v["syn_cnt"] > 0).astype(np.int8)
    v["half_open"] = ((v["syn_cnt"] > 0) & (v["bwd_pkts"] == 0)).astype(np.int8)
    v["rst_flow"] = (v["rst_cnt"] > 0).astype(np.int8)
    v["fin_flow"] = (v["fin_cnt"] > 0).astype(np.int8)
    v["psh_flow"] = (v["psh_cnt"] > 0).astype(np.int8)
    v["short_flow"] = (total_pkts <= 3).astype(np.int8)
    v["is_tcp"] = (v["proto"] == 6).astype(np.int8)
    v["is_udp"] = (v["proto"] == 17).astype(np.int8)
    v["is_icmp"] = (v["proto"] == 1).astype(np.int8)

    g = v.groupby(["window", "host"])
    feats = g.agg(
        n_flows_out=("is_out", "sum"),
        n_flows=("is_out", "size"),
        bytes_sent=("bytes_sent", "sum"),
        bytes_recv=("bytes_recv", "sum"),
        pkts_sent=("pkts_sent", "sum"),
        pkts_recv=("pkts_recv", "sum"),
        uniq_dst_ips_out=("peer_out", "nunique"),
        uniq_src_ips_in=("peer_in", "nunique"),
        uniq_dst_ports_out=("dport_out", "nunique"),
        uniq_dst_ports_in=("dport_in", "nunique"),
        internal_peer_ratio=("peer_internal", "mean"),
        syn_flow_ratio=("syn_flow", "mean"),
        half_open_ratio=("half_open", "mean"),
        rst_ratio=("rst_flow", "mean"),
        fin_ratio=("fin_flow", "mean"),
        psh_ratio=("psh_flow", "mean"),
        short_flow_ratio=("short_flow", "mean"),
        tcp_ratio=("is_tcp", "mean"),
        udp_ratio=("is_udp", "mean"),
        icmp_ratio=("is_icmp", "mean"),
        flow_dur_mean=("duration_s", "mean"),
        flow_dur_std=("duration_s", "std"),
        flow_iat_mean=("flow_iat_mean", "mean"),
    )
    feats["n_flows_in"] = feats.pop("n_flows") - feats["n_flows_out"]
    feats["active"] = 1

    # port entropy of outbound flows (block B, feature 12)
    out_ports = v[v["is_out"] == 1].groupby(["window", "host", "dst_port"]).size()
    feats["dst_port_entropy_out"] = (
        out_ports.groupby(level=["window", "host"]).apply(lambda c: entropy_of_counts(c.to_numpy()))
    )
    return feats.fillna(0.0).reset_index()
```

### 22.6 Label → stage (contract C2)

```python
# netwm/labels/build_labels.py
import numpy as np
import pandas as pd
import yaml

STAGES = ["BENIGN", "RECON_DISCOVERY", "INITIAL_ACCESS_ATTEMPT", "LATERAL_MOVEMENT",
          "COMMAND_AND_CONTROL", "IMPACT", "EXFILTRATION"]
STAGE_ID = {s: i for i, s in enumerate(STAGES)}
EXCLUDED = -2      # e.g. CTU-13 "Background": unknown, never used as benign or attack
AMBIGUOUS = -1


def map_flow_stage(label_raw: pd.Series, src_internal: pd.Series, mapping_yaml: str) -> pd.Series:
    """mapping YAML: {raw_label: STAGE_NAME | 'EXCLUDED' | 'INFILTRATION_RULE'}"""
    table = yaml.safe_load(open(mapping_yaml))
    unknown = set(label_raw.unique()) - set(table)
    if unknown:
        raise ValueError(f"Unmapped labels: {sorted(unknown)}")      # never guess silently
    name = label_raw.map(table)
    # documented rule: infiltration flows from an internal source = post-compromise discovery
    infil = name == "INFILTRATION_RULE"
    name = name.where(~infil, np.where(src_internal, "RECON_DISCOVERY", "INITIAL_ACCESS_ATTEMPT"))
    return name.map(lambda s: EXCLUDED if s == "EXCLUDED" else STAGE_ID[s]).astype(np.int8)


def host_window_stage(v: pd.DataFrame, min_flows: int = 3) -> pd.DataFrame:
    """v: host view with columns window, host, flow_stage. Most advanced stage with >= min_flows."""
    order = [1, 2, 3, 4, 5]                       # RECON < ACCESS < LATERAL < C2 < IMPACT
    mal = v[v["flow_stage"] > 0]
    counts = mal.groupby(["window", "host", "flow_stage"]).size().unstack(fill_value=0)
    stage = pd.Series(0, index=counts.index, dtype=np.int8)
    stage[counts.sum(axis=1) > 0] = AMBIGUOUS
    for s in order:
        if s in counts.columns:
            stage[counts[s] >= min_flows] = s     # later (more advanced) stages overwrite earlier
    all_hw = v.groupby(["window", "host"]).size().index
    return stage.reindex(all_hw, fill_value=0).rename("stage").reset_index()
```

(Windows where a host has *only* EXCLUDED/background flows are treated as benign-by-absence only if
the config says so; for CTU-13 we mask them.)

### 22.7 Exact model architecture and shapes

| Symbol | Value |
|---|---|
| B (batch) | 256 |
| L (history) | 10 |
| K (horizon) | 5 |
| D (features) | 59 (flow-only) / 67 (flow+packet) |
| C (trained stages) | 5: BENIGN, RECON_DISCOVERY, INITIAL_ACCESS_ATTEMPT, COMMAND_AND_CONTROL, IMPACT (model-local indices 0–4) |
| Encoder | Linear(D, 64) → LayerNorm(64) → GELU |
| Recurrent core | GRUCell(64, 128) |
| Dynamics head | Linear(128,128) → GELU → Linear(128, 2D) → (μ, log σ² clamped to [−6, 4]) |
| Stage head | Linear(128, 64) → GELU → Linear(64, C) |
| Parameters | ≈ 0.12 M (D = 59) |
| N (MC samples at inference) | 32 |

```python
# netwm/models/world_model.py
import torch
import torch.nn as nn


class NetworkWorldModel(nn.Module):
    """Host-centric recurrent world model.
    filter():       h_t = f(S_{t-L+1..t})                 (summary of the observed past)
    predict_next(): P(S_{t+1} | h_t) = N(mu, diag(sigma^2)) (transition dynamics)
    stage_head:     P(stage | h)                            (works on real AND imagined states)
    """

    def __init__(self, n_features: int, n_stages: int, emb: int = 64, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(n_features, emb), nn.LayerNorm(emb), nn.GELU())
        self.cell = nn.GRUCell(emb, hidden)
        self.dynamics = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, 2 * n_features))
        self.stage_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, n_stages))

    def step(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.cell(self.encoder(s), h)

    def predict_next(self, h: torch.Tensor):
        mu, logvar = self.dynamics(h).chunk(2, dim=-1)
        return mu, logvar.clamp(-6.0, 4.0)

    def filter(self, x_hist: torch.Tensor) -> torch.Tensor:          # [B, L, D] -> [B, H]
        h = x_hist.new_zeros(x_hist.size(0), self.hidden)
        for t in range(x_hist.size(1)):
            h = self.step(x_hist[:, t], h)
        return h

    def forward(self, x_hist, K, x_future=None, teacher_prob=0.0, sample=False,
                intervention=None):
        h = self.filter(x_hist)
        stage_now = self.stage_head(h)                                   # [B, C]
        mus, logvars, stage_future = [], [], []
        for k in range(K):
            mu, logvar = self.predict_next(h)                            # P(S_{t+k+1} | h)
            mus.append(mu)
            logvars.append(logvar)
            s_next = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
            if x_future is not None and teacher_prob > 0:                # scheduled sampling
                use_true = (torch.rand(mu.size(0), 1, device=mu.device) < teacher_prob).float()
                s_next = use_true * x_future[:, k] + (1 - use_true) * s_next
            if intervention is not None:                                 # what-if (stretch)
                s_next = intervention(s_next)
            h = self.step(s_next, h)                                     # imagine the next window
            stage_future.append(self.stage_head(h))
        return {
            "stage_now": stage_now,                                      # [B, C]
            "mu": torch.stack(mus, 1),                                   # [B, K, D]
            "logvar": torch.stack(logvars, 1),                           # [B, K, D]
            "stage_future": torch.stack(stage_future, 1),                # [B, K, C]
        }


def p_attack_within(stage_future_logits: torch.Tensor, benign_idx: int = 0) -> torch.Tensor:
    """1 - prod_k P(benign at t+k), computed in log space. [B, K, C] -> [B]"""
    log_p_benign = stage_future_logits.log_softmax(-1)[..., benign_idx].sum(1)
    return -torch.expm1(log_p_benign)
```

### 22.8 Exact training procedure

```python
# netwm/data/sequences.py
import numpy as np
import torch
from torch.utils.data import Dataset


class HostSequenceDataset(Dataset):
    """series: list of (X [T, D] float32, y [T] int64 with -1 = ambiguous) — one per (day, host)."""

    def __init__(self, series, L=10, K=5, starts=None):
        self.series, self.L, self.K = series, L, K
        self.index = starts if starts is not None else [
            (i, t) for i, (X, _) in enumerate(series) for t in range(L - 1, len(X) - K)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, t = self.index[j]
        X, y = self.series[i]
        y_future = y[t + 1:t + 1 + self.K]
        known = y_future >= 0
        y_within = float((y_future > 0).any())
        within_valid = float(y_within == 1.0 or known.all())   # unknown if only ambiguous futures
        return {
            "x_hist": torch.from_numpy(X[t - self.L + 1:t + 1]),
            "x_future": torch.from_numpy(X[t + 1:t + 1 + self.K]),
            "y_now": torch.tensor(y[t], dtype=torch.long),
            "y_future": torch.from_numpy(y_future.astype(np.int64)),
            "y_within": torch.tensor(y_within),
            "within_valid": torch.tensor(within_valid),
        }
```

```python
# netwm/training/losses.py
import torch
import torch.nn.functional as F


def world_model_loss(out, batch, class_weights, lam=(1.0, 0.5, 1.0, 1.0)):
    lam_dyn, lam_now, lam_fut, lam_hor = lam
    C = out["stage_now"].size(-1)

    dyn = F.gaussian_nll_loss(out["mu"], batch["x_future"], out["logvar"].exp())
    now = F.cross_entropy(out["stage_now"], batch["y_now"], weight=class_weights, ignore_index=-1)
    fut = F.cross_entropy(out["stage_future"].reshape(-1, C), batch["y_future"].reshape(-1),
                          weight=class_weights, ignore_index=-1)

    # horizon BCE with p = 1 - prod_k P(benign), stable in log space
    log_no_attack = out["stage_future"].log_softmax(-1)[..., 0].sum(1).clamp(max=-1e-6)
    log_attack = torch.log(-torch.expm1(log_no_attack))
    y = batch["y_within"]
    bce = -(y * log_attack + (1 - y) * log_no_attack)
    hor = (bce * batch["within_valid"]).sum() / batch["within_valid"].sum().clamp(min=1)

    total = lam_dyn * dyn + lam_now * now + lam_fut * fut + lam_hor * hor
    return total, {"dyn": dyn.item(), "now": now.item(), "fut": fut.item(), "hor": hor.item()}
```

(If a batch contains only ignored stage labels, `cross_entropy` returns NaN — the sampler guarantees
at least some non-ambiguous labels per batch; add a guard in the real code.)

```python
# netwm/training/train_world_model.py (core loop)
import torch
from sklearn.metrics import average_precision_score

from netwm.models.world_model import NetworkWorldModel, p_attack_within
from netwm.training.losses import world_model_loss


def train(train_loader, val_loader, n_features, n_stages, class_weights, K=5, epochs=30,
          device="cpu", ckpt="artifacts/model.pt"):
    torch.manual_seed(42)
    model = NetworkWorldModel(n_features, n_stages).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, patience = -1.0, 0
    for epoch in range(epochs):
        teacher_prob = max(0.0, 1.0 - epoch / (0.6 * epochs))
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x_hist"], K, x_future=batch["x_future"], teacher_prob=teacher_prob)
            loss, parts = world_model_loss(out, batch, class_weights.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()                                   # validation = free-running, like inference
        ps, ys = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["x_hist"], K)
                keep = batch["within_valid"] > 0
                ps.append(p_attack_within(out["stage_future"])[keep].cpu())
                ys.append(batch["y_within"][keep].cpu())
        val_ap = average_precision_score(torch.cat(ys).numpy(), torch.cat(ps).numpy())
        print(f"epoch {epoch:02d} teacher={teacher_prob:.2f} val PR-AUC={val_ap:.4f} {parts}")
        if val_ap > best:
            best, patience = val_ap, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= 5:
                break
    return best
```

### 22.9 Exact forecasting algorithm

```python
# netwm/forecasting/rollout.py
import numpy as np
import torch

from netwm.models.world_model import p_attack_within


@torch.no_grad()
def forecast_host(model, x_hist, K=5, n_samples=32, temperature=1.0, calibrator=None):
    """x_hist: [L, D] normalised history of ONE host. Returns a dict of numpy arrays."""
    model.eval()
    x = x_hist.unsqueeze(0).expand(n_samples, -1, -1).contiguous()     # [N, L, D]
    out = model(x, K, sample=True)
    logits = out["stage_future"] / temperature                            # [N, K, C]
    stage_probs = logits.softmax(-1)
    p_within = p_attack_within(logits).cpu().numpy()                      # [N]
    p_mean = float(p_within.mean())
    lo, hi = np.percentile(p_within, [5, 95])
    if calibrator is not None:                                            # isotonic: monotone
        p_mean, lo, hi = calibrator.predict([p_mean, lo, hi])
    return {
        "p_attack_within_K": float(p_mean),
        "band_5_95": (float(lo), float(hi)),
        "stage_probs_per_step": stage_probs.mean(0).cpu().numpy(),        # [K, C]
        "stage_now": (out["stage_now"][0] / temperature).softmax(-1).cpu().numpy(),
        "state_mean": out["mu"].mean(0).cpu().numpy(),                    # [K, D]
        "state_band": np.percentile(out["mu"].cpu().numpy(), [5, 95], axis=0),
    }


def horizon_step(stage_probs_per_step, threshold, benign_idx=0):
    """First k (1-based) where cumulative attack probability crosses the threshold, else None."""
    cum_no_attack = np.cumprod(stage_probs_per_step[:, benign_idx])
    crossed = np.nonzero(1 - cum_no_attack >= threshold)[0]
    return int(crossed[0]) + 1 if crossed.size else None
```

Note: the stage heads read the hidden state *after* each sampled step, so `stage_now` is the same
across samples (history is identical) — we read sample 0.

### 22.10 Exact attack-stage mapping strategy

1. **Training labels:** `label_to_stage.yaml` (§7.3), unit-tested, documented in `LABEL_MAPPING.md`.
2. **Predicted stage → tactic:** fixed table stage → ATT&CK tactic ID(s) (§7.2).
3. **Observed indicators → candidate techniques** (`indicator_rules.py`), each rule with a
   threshold on *raw* values and an evidence-strength tag:

| Rule | Condition (example thresholds, tune on val benign data) | Candidate | Strength |
|---|---|---|---|
| Port scan (internal src) | `uniq_dst_ports_out ≥ 100` and `half_open_ratio ≥ 0.5` and host internal | T1046 Network Service Discovery (TA0007) | medium |
| Port scan (external src → host) | `uniq_dst_ports_in ≥ 100` from one external source | T1595 Active Scanning (TA0043) | medium |
| Password guessing | `svc_ssh`/`svc_ftp` ≥ 0.5 and `short_flow_ratio` ≥ 0.7 and `n_flows ≥ 30` | T1110.001 (TA0006) | medium |
| Internal SMB/RDP spread | `svc_smb + svc_rdp ≥ 0.3` and `new_peers_out ≥ 5` (internal) | T1021 Remote Services (TA0008) | **low** (no ground truth) |
| Beaconing | `flow_start_gap_cv ≤ 0.1` over ≥ 5 windows, small flows, same external peer | T1071 Application Layer Protocol (TA0011) | low–medium |
| Flood | `n_flows_in` or `bytes_recv` z-score ≥ 5 with high `short_flow_ratio` | T1498 / T1499 (TA0040) | medium |
| Large outbound transfer | `z_bytes_sent ≥ 5` and `bytes_ratio ≫ 0` to external peer | T1048 / T1041 (TA0010) | **low** (heuristic, untrained) |

4. The mapper outputs three separate blocks (observed / inferred / predicted); techniques only ever
   appear as "candidate … consistent with …" with the strength tag.

### 22.11 Exact explainability implementation

```python
# netwm/explain/ig.py
import torch
from captum.attr import IntegratedGradients

from netwm.models.world_model import p_attack_within


def explain_forecast(model, x_hist, benign_baseline, feature_names, K=5, top_k=5):
    """x_hist: [L, D]; benign_baseline: [D] (median benign state, normalised)."""
    model.eval()

    def f(x):                                     # deterministic mean rollout -> [B]
        return p_attack_within(model(x, K)["stage_future"])

    ig = IntegratedGradients(f)
    x = x_hist.unsqueeze(0)
    base = benign_baseline.expand_as(x_hist).unsqueeze(0)
    attr, delta = ig.attribute(x, baselines=base, n_steps=32, return_convergence_delta=True)
    attr = attr[0].detach()                       # [L, D]
    per_feature = attr.sum(0)                     # signed contribution per feature
    per_time = attr.abs().sum(1)                  # which windows mattered
    order = torch.argsort(per_feature, descending=True)[:top_k]
    return {
        "top_features": [(feature_names[i], float(per_feature[i])) for i in order
                         if per_feature[i] > 0],
        "time_importance": per_time.tolist(),     # index 0 = t-L+1 ... last = t
        "heatmap": attr.tolist(),
        "convergence_delta": float(delta),        # sanity check: should be small
    }
```

`netwm/explain/evidence.py` turns `top_features` into text using the raw (unscaled) values from the
last three windows and the host's benign median — see §11.3.

### 22.12 Exact Python files/modules to create

See §14. Creation order: `schema.py` → `load_cic.py` → `build_labels.py` → `windowing.py` →
`temporal.py` → `global_context.py` → `scaler.py` → `splits.py` → `sequences.py` → `baselines.py` →
`world_model.py` → `losses.py` → `train_world_model.py` → `calibrate.py` → `rollout.py` →
`risk.py` → `lead_time.py` → `metrics.py` → `mapper.py` → `ig.py` → `evidence.py` →
`pipeline.py` → `app.py` → `hashchain.py`.

### 22.13 Commands

```bash
# ---------- install (Python 3.11) ----------
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                              # torch CPU build is fine
pip install -r requirements-optional.txt                     # xgboost, nfstream, cryptography, web3
sudo apt install tshark                                      # packet features (optional)

# ---------- download (one-time, needs internet) ----------
pip install awscli
aws s3 sync --no-sign-request --region ca-central-1 \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" data/raw/cic2018/

# ---------- preprocess ----------
python scripts/preprocess.py     --config configs/cic2018.yaml     # -> data/interim/flows_*.parquet, labels_*.parquet
python scripts/build_windows.py  --config configs/cic2018.yaml     # -> data/processed/host_windows.parquet, artifacts/scaler.json
python scripts/extract_packets.py --pcap-dir data/raw/cic2018_pcap/28-02 --config configs/cic2018.yaml   # optional

# ---------- train ----------
python train.py --config configs/default.yaml --data configs/cic2018.yaml --baselines
python train.py --config configs/default.yaml --data configs/cic2018.yaml --seed 1   # repeat for seeds 1..3

# ---------- evaluate ----------
python scripts/evaluate.py --config configs/default.yaml --protocol A    # -> artifacts/reports/
python scripts/evaluate.py --config configs/default.yaml --protocol B
python scripts/evaluate.py --config configs/default.yaml --cross configs/cic2017.yaml

# ---------- inference ----------
python predict.py --csv  data/sample/demo_day_small.csv --out forecast.json
python predict.py --pcap capture.pcap --out forecast.json

# ---------- dashboard ----------
streamlit run app.py

# ---------- tests / ledger ----------
pytest -q
python scripts/verify_ledger.py artifacts/ledger.sqlite
```

The bulk packet-feature extraction uses tshark, e.g.:

```bash
tshark -r in.pcap -T fields -E separator=, -E header=y \
  -e frame.time_epoch -e ip.src -e ip.dst -e ip.ttl -e ip.flags.mf -e ip.frag_offset \
  -e tcp.dstport -e udp.dstport -e tcp.window_size_value -e tcp.flags \
  -e tcp.analysis.retransmission -e tcp.len -e udp.length > packets.csv
```

### 22.14 Sample input / output

**Input** (one canonical flow row):

```text
ts=2018-03-01 14:03:12Z src_ip=172.31.69.13 src_port=51422 dst_ip=172.31.69.24 dst_port=445
proto=6 duration_s=0.0021 fwd_pkts=2 bwd_pkts=0 fwd_bytes=0 bwd_bytes=0 syn_cnt=1 rst_cnt=0 ...
```

**Output** (`predict.py` / contract C5). *Format illustration only — values are invented.*

```json
{
  "window_end": "2018-03-01T14:04:00Z",
  "host": "172.31.69.13",
  "model": {"version": "wm-gru-0.1", "sha256": "…", "config_sha256": "…"},
  "observed": {
    "uniq_dst_ports_out": [12, 88, 301],
    "half_open_ratio": [0.05, 0.41, 0.83],
    "new_peers_out": [0, 14, 35],
    "top_flows": [{"dst": "172.31.69.24", "dst_port": 445, "flows": 37}]
  },
  "inferred": {
    "current_stage": {"RECON_DISCOVERY": 0.81, "BENIGN": 0.15, "…": 0.04},
    "candidate_techniques": [{"id": "T1046", "name": "Network Service Discovery",
                              "tactic": "TA0007", "strength": "medium",
                              "wording": "behaviour consistent with"}]
  },
  "predicted": {
    "horizon_windows": 5,
    "p_attack_within_K": 0.74,
    "band_5_95": [0.61, 0.83],
    "first_crossing_step": 2,
    "stage_probs": {"t+1": {"BENIGN": 0.55, "RECON_DISCOVERY": 0.30, "INITIAL_ACCESS_ATTEMPT": 0.12},
                    "t+2": {"BENIGN": 0.41, "RECON_DISCOVERY": 0.21, "INITIAL_ACCESS_ATTEMPT": 0.33}},
    "most_likely_future_stage": "INITIAL_ACCESS_ATTEMPT",
    "surprise_percentile": 0.97,
    "risk_level": "HIGH",
    "statement": "The model estimates an increased probability of a future attack trajectory for this host based on observed network behaviour."
  },
  "explanation": {
    "top_features": [["uniq_dst_ports_out", 0.21], ["half_open_ratio", 0.14], ["new_peers_out", 0.09]],
    "time_importance": [0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.06, 0.10, 0.28, 0.43],
    "evidence_text": ["Host contacted 301 distinct destination ports in the last window (usual: ~6)"]
  },
  "not_supported": ["LATERAL_MOVEMENT", "EXFILTRATION"],
  "ledger": {"record_hash": "…", "prev_hash": "…"}
}
```

### 22.15 Ledger core

```python
# netwm/ledger/hashchain.py
import hashlib
import json
import sqlite3
import time

GENESIS = "0" * 64


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def record_hash(prev_hash: str, payload_hash: str, ts: float) -> str:
    return hashlib.sha256(f"{prev_hash}|{payload_hash}|{ts!r}".encode()).hexdigest()


class EvidenceLedger:
    def __init__(self, path="artifacts/ledger.sqlite"):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS ledger (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, payload TEXT,
            payload_hash TEXT, prev_hash TEXT, record_hash TEXT)""")

    def _head(self) -> str:
        row = self.db.execute("SELECT record_hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS

    def append(self, payload: dict) -> str:
        body = canonical(payload)
        payload_hash = hashlib.sha256(body).hexdigest()
        prev, ts = self._head(), time.time()
        rh = record_hash(prev, payload_hash, ts)
        self.db.execute("INSERT INTO ledger (ts, payload, payload_hash, prev_hash, record_hash) "
                        "VALUES (?, ?, ?, ?, ?)", (ts, body.decode(), payload_hash, prev, rh))
        self.db.commit()
        return rh

    def verify(self):
        """Returns (True, None) or (False, first_bad_seq)."""
        prev = GENESIS
        rows = self.db.execute("SELECT seq, ts, payload, payload_hash, prev_hash, record_hash "
                               "FROM ledger ORDER BY seq")
        for seq, ts, payload, ph, prev_hash, rh in rows:
            if (hashlib.sha256(payload.encode()).hexdigest() != ph or prev_hash != prev
                    or record_hash(prev, ph, ts) != rh):
                return False, seq
            prev = rh
        return True, None
```

(Add Ed25519 signing of `record_hash` with `cryptography` and periodic Merkle-root anchoring as
optional layers.)

### 22.16 Lead-time evaluation

```python
# netwm/evaluation/lead_time.py
import numpy as np
import pandas as pd


def threshold_at_fpr(p_quiet_val: np.ndarray, target_fpr: float = 0.01) -> float:
    """p_quiet_val: forecasts on validation windows whose current AND next K stages are benign."""
    return float(np.quantile(p_quiet_val, 1 - target_fpr))


def early_warning_recall(df: pd.DataFrame, thr: float, leads=(0, 1, 2, 3), quiet=5):
    """df columns: day, host, t (window index), stage (0 benign, >0 attack, -1 ambiguous),
    p (forecast P(attack within K) made at window t using data <= t)."""
    hits = {l: [] for l in leads}
    for _, g in df.groupby(["day", "host"]):
        g = g.set_index("t").sort_index()
        stage, p = g["stage"], g["p"]
        for T in stage.index[stage > 0]:
            before = stage.reindex(range(T - quiet, T))
            if before.isna().any() or (before != 0).any():
                continue                              # not an onset
            for l in leads:
                if (T - l) in p.index:
                    hits[l].append(bool(p[T - l] >= thr))
    return {l: {"recall": float(np.mean(h)) if h else float("nan"), "n_onsets": len(h)}
            for l, h in hits.items()}
```

Lead 0 means "forecast made in the onset window itself" (the ongoing detection case); leads 1–3 are
true early warnings (require K ≥ 3).

### 22.17 Testing strategy

| Test | What it checks |
|---|---|
| `test_schema.py` | each loader outputs exactly the canonical columns/dtypes on a 20-row fixture |
| `test_labels.py` | every raw label in the fixtures is mapped; unknown label raises; ≥3-flow and ambiguity rules |
| `test_windowing.py` | synthetic flows (one host hitting 500 ports) → `uniq_dst_ports_out = 500`, `half_open_ratio = 1`, correct in/out bytes swap |
| `test_causality.py` | modify all flows after window t → features at ≤ t unchanged (incl. EWMA/deltas) |
| `test_splits_no_leakage.py` | train/val/test day sets disjoint; purge gap respected; scaler stats equal to train-only recomputation |
| `test_model_shapes.py` | output shapes for B=4, L=10, K=5, D=59, C=5; `p_attack_within` ∈ [0, 1] |
| `test_overfit_batch.py` | model drives loss near zero on 32 sequences (catches wiring bugs) |
| `test_rollout.py` | seeded rollout is deterministic; band widens (non-decreasing on average) with k |
| `test_toy_world.py` | synthetic Markov world where "recon" precedes "access" by 2 windows: WM recall at lead 2 is clearly above the lead-2 recall of B1 on the same data |
| `test_ledger.py` | append 10 → verify ok; edit record 5 → verify returns (False, 5) |
| `test_dashboard_smoke.py` | `streamlit.testing.v1.AppTest` loads `app.py` with the demo sample without exceptions |

Run `pytest -q` in CI (GitHub Actions, CPU only, demo sample data).

### 22.18 Demo strategy

1. **Precompute** forecasts/explanations for the test infiltration day and one DoS/DDoS test day into
   `forecasts.parquet`; the dashboard scrubs instantly.
2. **Pick the showcase host(s) from the test split by a rule decided in advance** (e.g. the host with
   the most onsets), not by cherry-picking the best-looking curve; show one failure case as well —
   judges trust that.
3. Show the ground-truth band only as a toggle labelled "dataset label (not available in deployment)".
4. Upload a small PCAP live to prove the PCAP path works offline (warn about extractor shift).
5. Tamper-evidence moment with the ledger.
6. Record with Wi-Fi off; keep a screenshot deck as fallback.

---

### References to cite (verify exact titles/links when writing the README)
- Sharafaldin, Lashkari, Ghorbani — CIC-IDS2017 (ICISSP 2018); CSE-CIC-IDS2018 dataset page (UNB CIC).
- Engelen, Rimmer, Joosen — "Troubleshooting an intrusion detection dataset: the CICIDS2017 case study" (2021).
- Liu, Engelen, et al. — "Error prevalence in NIDS datasets: a case study on CIC-IDS-2017 and CSE-CIC-IDS-2018" (IEEE CNS 2022).
- García, Grill, Stiborek, Zunino — CTU-13 (Computers & Security, 2014).
- Moustafa & Slay — UNSW-NB15 (2015).
- Kent — LANL comprehensive multi-source cyber-security events (2015).
- Ha & Schmidhuber — "World Models" (2018).
- Sundararajan, Taly, Yan — Integrated Gradients (ICML 2017).
- MITRE ATT&CK Enterprise matrix (record the version used).
