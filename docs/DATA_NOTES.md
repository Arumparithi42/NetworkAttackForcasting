# Data notes – CSE-CIC-IDS2018 as actually used

Every statement here was checked on the real files (the scripts that check them are referenced).
If you re-download the data, re-run `notebooks/01_eda_label_timeline.py` before training.

## 1. Sources

| What | Where | Size | Used for |
|---|---|---|---|
| Processed flow CSVs (CICFlowMeter), 10 days | `s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/` (public, no sign-in) | 6.4 GB | network-level model (all usable days) |
| Raw per-host PCAPs, 2 days (28-02, 01-03) | `s3://cse-cic-ids2018/Original Network Traffic and Log data/<day>/pcap.zip` | 53 + 63 GB (streamed, never stored) | host-level model with packet-level features |

`scripts/download_cic2018.sh` fetches the CSVs; `scripts/extract_cic2018_pcaps.py` streams single
members of the remote ZIPs through HTTP range requests and keeps only the extracted flows
(~0.4 GB per day as parquet).

## 2. Problems found in the published CSVs, and what the loader does about them

| # | Problem | Evidence | Fix (`netwm/io/load_cic.py`) |
|---|---|---|---|
| 1 | **12-hour clock without AM/PM.** Afternoon hours appear as 01–05. | Per-hour counts contain only hours 1–5 and 8–12; SSH brute force on 14-02 appears at "02:01–03:32" while the published schedule says 14:01–15:31. | Hours < 8 get +12 h (`clock_12h_threshold: 8`). |
| 2 | **Local time (UTC−4).** | Matching CSV rows against the raw PCAP (UTC epoch timestamps) maximises at an offset of −4 h (283 exact matches vs 212/201 for −5/−3 h on the probe host). | `utc_offset_hours: -4`. |
| 3 | **Repeated header rows** inside files. | Rows whose `Label` equals "Label". | Dropped. |
| 4 | **Garbage rows dated 1970.** | 14-02 and 22-02 contain a handful. | Rows whose date ≠ file date are dropped. |
| 5 | **Exact duplicate rows.** | 14-02: 1,048,575 rows → 822,799 after de-duplication. | `drop_duplicates()`. |
| 6 | **Truncated files (1,048,575 rows = spreadsheet row limit)** for 7 of 10 days. | For 16-02 and 21-02 the benign background practically vanishes (≈20–40 benign flows per 20 min, vs ≈30,000–50,000 on other days) while hundreds of thousands of attack flows remain. | 16-02 and 21-02 are **excluded** (`excluded_days`). Other truncated days keep a continuous benign background and are used. |
| 7 | **No IP columns** except in the 20-02 file (Flow ID, Src IP, Src Port, Dst IP). | Header inspection. | Network-level entity for the CSV model; host-level model built from PCAPs instead. |
| 8 | **Contradictory duplicate rows (28-02).** 67,973 of the 68,856 `Infilteration` flows of 28-02 also appear as byte-identical rows labelled `Benign` (01-03: 7 pairs; other days: none). | Group-by over all columns except the label. | One row kept, with the attack label (`resolve_conflicting_duplicates`). Keeping both would double-count flows precisely during attack periods – a shortcut feature. |
| 9 | `Infinity` / `-1` sentinel values. | `Flow Byts/s`, `Init Fwd Win Byts` = −1. | → NaN. |

## 3. Attack periods after the fixes (UTC)

| Day | Label(s) | Approx. period (UTC) | Split role (Protocol A) |
|---|---|---|---|
| 14-02 | FTP-BruteForce, SSH-Bruteforce | 14:20–16:10, 18:00–19:30 | train |
| 15-02 | DoS GoldenEye, DoS Slowloris | 13:20–14:00, 15:00–15:45 | train |
| 20-02 | DDoS LOIC-HTTP, LOIC-UDP | (IP-bearing file) | test |
| 22-02 | Web brute force, XSS, SQLi (few flows) | scattered | train |
| 23-02 | Web brute force, XSS, SQLi (few flows) | scattered | test |
| 28-02 | Infilteration | 14:40–16:00, 17:40–18:20 | train |
| 01-03 | Infilteration | 13:40–14:40, 18:00–19:20 | test |
| 02-03 | Bot | 14:00–19:40 (continuous) | train before 16:30, test after |

## 4. Host-level labels for the PCAP-derived flows (`netwm/labels/match.py`)

The CSV has no IPs and only a subset of the flows, so labels are transferred:

1. **Matching key** – strict key (start second ±1, dst port, fwd packets, bwd packets), using only
   destination ports outside the 15 most common benign ports (DNS/HTTPS/RDP/SMB flows collide
   across hosts within the same second). A flow seen in two captures (internal↔internal) counts once.
2. **Attack source** (the compromised host) – internal source IP whose number of attack-matched
   flows is ≥ 5× the median of the top-10 sources.
3. **Attack target** – internal destination whose matched incoming flows are ≥ 75 % attack
   (hosts that merely share the network sit at the key-collision base rate of ~30 %).
4. **Flow labelling** – loose key (start second ±1, dst port, protocol) on flows whose source or
   destination is a source/target, i.e. the dataset's own "victim IP + time interval" rule.
   Infiltration flows from an internal source become RECON_DISCOVERY, from an external source
   INITIAL_ACCESS_ATTEMPT (`label_to_stage.yaml`).

| Day | Attack source (compromised host) | Attack targets | Next-largest source count | PCAP flows labelled | Attack host-windows (hosts) |
|---|---|---|---|---|---|
| 28-02 (train) | 172.31.69.24 (3,648 matched flows) | 172.31.69.7 (97 % attack) | 294 | 9,747 | 263 (14 hosts) |
| 01-03 (test) | 172.31.69.13 (9,991 matched flows) | 172.31.69.7, .15, .22 | 329 | 34,798 | 354 (16 hosts) |

The per-host match times of the compromised host follow the CSV's attack periods exactly.
Detailed statistics: `data/processed/cic2018_host/prepare_info.json`.

Development history (kept for transparency): a first version keyed victims on the share of
matched flows per *capture file*; it missed 172.31.69.24 (a busy host, share diluted by all-day
benign traffic) and exposed the contradictory duplicate rows of 28-02 (problem #8), which left
zero attack-only keys for that day. Both were fixed before any host-level model was trained.

Caveats: (a) the transferred labels inherit the dataset's coarse IP+time labelling (benign traffic
of a victim during the attack period is "attack"); (b) flows missing from the CSV subset stay
benign, lowering label recall; (c) the ≥ 3-flows-per-window rule filters isolated matches.

## 5. PCAP extraction

442 / 436 per-host capture files for 01-03 / 28-02. The extractor (`netwm/io/pcap_packets.py`,
dpkt) follows CICFlowMeter-like semantics (bidirectional 5-tuple, FIN/RST closes a flow,
120 s active / 60 s idle timeout, payload byte counts) and keeps packet-level accumulators (TTL
statistics, IP fragmentation, TCP window, retransmissions, payload-size histogram).
AWS metadata traffic (169.254.169.254) is ignored. A truncated last record in a capture ends the
file gracefully; failed network reads are retried.
