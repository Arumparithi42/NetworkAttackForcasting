# Attack-stage labels and MITRE ATT&CK mapping

**ATT&CK version:** 19.2 (tactic/technique names and IDs extracted from the official STIX bundle,
`mitre-attack/attack-stix-data`, into `netwm/attack_mapping/attack_subset.yaml`; note that v19
renamed TA0005 "Defense Evasion" to "Stealth").

## 1. Three kinds of label – never mixed

| Kind | Source | Stored in | Example |
|---|---|---|---|
| Ground-truth dataset label | CSE-CIC-IDS2018 CSV `Label` column | `flows_*.parquet: label_raw` | `DoS attacks-Hulk`, `Infilteration` |
| Derived stage (this document) | deterministic function of the dataset label (+ flow direction) | `labels_*.parquet: stage` | `IMPACT`, `RECON_DISCOVERY` |
| Model output | world model | forecast JSON (`inferred` / `predicted`) | `P(stage=IMPACT at t+2)=0.41` |

Derived stages are a **documented re-grouping** of what the dataset states – not new ground truth.

## 2. Stages

| ID | Stage | ATT&CK tactic(s) | Model class? | Why |
|---|---|---|---|---|
| 0 | BENIGN | – | yes | |
| 1 | RECON_DISCOVERY | TA0043 Reconnaissance (activity from outside) / TA0007 Discovery (from an internal host) | yes | port scans; the network-observable part of the infiltration scenario is the compromised host exploring the internal network |
| 2 | INITIAL_ACCESS_ATTEMPT | TA0001 Initial Access / TA0006 Credential Access | yes | brute force (FTP/SSH/web), SQL injection, XSS |
| 3 | LATERAL_MOVEMENT | TA0008 | **no** | no flow-level ground truth in CIC-IDS2018 – shown as "not supported" |
| 4 | COMMAND_AND_CONTROL | TA0011 | yes | Bot (Ares/Zeus) traffic |
| 5 | IMPACT | TA0040 | yes | DoS / DDoS families dominate the dataset and are an ATT&CK tactic |
| 6 | EXFILTRATION | TA0010 | **no** | no ground truth; only a low-strength heuristic indicator |

## 3. Dataset label → stage (`netwm/labels/label_to_stage.yaml`)

| Dataset label | Stage | Candidate technique ("consistent with") | Confidence of the fit |
|---|---|---|---|
| FTP-BruteForce, SSH-Bruteforce (2018); FTP/SSH-Patator (2017) | INITIAL_ACCESS_ATTEMPT | T1110.001 Password Guessing | good |
| Brute Force -Web | INITIAL_ACCESS_ATTEMPT | T1110 Brute Force | good |
| SQL Injection | INITIAL_ACCESS_ATTEMPT | T1190 Exploit Public-Facing Application | good |
| Brute Force -XSS | INITIAL_ACCESS_ATTEMPT | T1190 | **weak** (XSS targets users, not the server) |
| DoS attacks-GoldenEye / Slowloris / SlowHTTPTest / Hulk | IMPACT | T1499 Endpoint DoS (.002 Service Exhaustion Flood) | good |
| DDoS attacks-LOIC-HTTP, DDOS attack-HOIC | IMPACT | T1499.002 / T1498 Network DoS | good |
| DDOS attack-LOIC-UDP | IMPACT | T1498.001 Direct Network Flood | good |
| Bot | COMMAND_AND_CONTROL | T1071.001 Web Protocols (HTTP-based bots) | medium |
| Infilteration (sic) | RECON_DISCOVERY if the flow source is internal, else INITIAL_ACCESS_ATTEMPT; unknown source (no IPs) → RECON_DISCOVERY | T1046 Network Service Discovery; the initial vector (T1566 Phishing / T1204 User Execution) is **not observable** in flows | medium; labels are coarse (all victim traffic during the attack period) |
| PortScan (2017) | RECON_DISCOVERY | T1595 Active Scanning / T1046 | good |

The loader raises an error on any label that is not in the table – nothing is guessed silently.

## 4. From flows to windows

A (host, window) – or (network, window) – gets the **most advanced** stage (kill-chain order
RECON < ACCESS < LATERAL < C2 < IMPACT) that has **≥ 3 flows**; 1–2 malicious flows make the window
AMBIGUOUS (ignored by the loss and the metrics). This reduces the effect of stray mislabelled flows.

## 5. Observed indicators → candidate techniques (`netwm/attack_mapping/indicator_rules.py`)

Transparent threshold rules on raw window features, each with an evidence-strength tag
(`low` / `medium`). They are displayed as "behaviour consistent with …", never as proof. The
thresholds are heuristics chosen to be rare in benign traffic; re-tune them on your own benign
validation data before operational use.
