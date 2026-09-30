"""PCAP extractor, label transfer, ATT&CK mapping, explanations, metrics, lead time, ledger."""
import sqlite3

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import make_flows
from netwm.attack_mapping.mapper import TACTICS, TECHNIQUES, build_record, candidate_techniques
from netwm.attack_mapping.indicator_rules import RULES
from netwm.evaluation.lead_time import early_warning, onsets
from netwm.evaluation.metrics import binary_report, threshold_at_fpr
from netwm.explain.evidence import evidence_sentences
from netwm.explain.ig import explain
from netwm.labels.match import transfer_labels
from netwm.ledger.hashchain import EvidenceLedger, merkle_root
from netwm.models.world_model import NetworkWorldModel


# ------------------------------------------------------------------ PCAP extractor
def test_pcap_extractor(tmp_path):
    scapy = pytest.importorskip("scapy.all")
    from netwm.io.pcap_packets import pcap_to_flows
    E, IP, TCP, UDP = scapy.Ether, scapy.IP, scapy.TCP, scapy.UDP
    pkts = []
    t = 1519905600.0
    for i, port in enumerate(range(1000, 1020)):                 # SYN scan, no replies
        p = E() / IP(src="172.31.1.1", dst="172.31.1.2", ttl=64) / TCP(sport=5555, dport=port, flags="S")
        p.time = t + i * 0.01
        pkts.append(p)
    s = [E() / IP(src="172.31.1.1", dst="10.0.0.9", ttl=64) / TCP(sport=4000, dport=80, flags="S", window=1234),
         E() / IP(src="10.0.0.9", dst="172.31.1.1", ttl=50) / TCP(sport=80, dport=4000, flags="SA"),
         E() / IP(src="172.31.1.1", dst="10.0.0.9", ttl=64) / TCP(sport=4000, dport=80, flags="PA", seq=1) / (b"x" * 100),
         E() / IP(src="172.31.1.1", dst="10.0.0.9", ttl=64) / TCP(sport=4000, dport=80, flags="PA", seq=1) / (b"x" * 100),
         E() / IP(src="10.0.0.9", dst="172.31.1.1", ttl=50, flags="MF") / UDP(sport=53, dport=53) / b"frag",
         E() / IP(src="172.31.1.1", dst="10.0.0.9", ttl=64) / TCP(sport=4000, dport=80, flags="FA")]
    for i, p in enumerate(s):
        p.time = t + 1 + i * 0.1
        pkts.append(p)
    path = tmp_path / "x.pcap"
    scapy.wrpcap(str(path), pkts)
    f = pcap_to_flows(path)
    scan = f[(f["dst_ip"] == "172.31.1.2")]
    assert len(scan) == 20 and (scan["bwd_pkts"] == 0).all() and (scan["syn_cnt"] == 1).all()
    web = f[(f["dst_port"] == 80)].iloc[0]
    assert web["fwd_pkts"] == 4 and web["bwd_pkts"] == 1
    assert web["fwd_bytes"] == 200 and web["retrans_pkts"] == 1 and web["init_fwd_win"] == 1234
    assert web["bwd_ttl_sum"] == 50
    assert f["frag_pkts"].sum() == 1


# ------------------------------------------------------------------ label transfer
def test_transfer_labels_finds_victim():
    rng = np.random.default_rng(0)
    rows, csv = [], []
    for i in range(3000):
        host = f"172.31.1.{i % 6 + 1}"
        port = int(rng.integers(2000, 60000))
        rows.append(dict(ts=i * 0.5, src_ip=host, dst_ip="8.8.8.8", dst_port=port, capture_host=host))
        lab = "Infilteration" if host == "172.31.1.3" else "Benign"
        if i % 2 == 0:                                            # CSV contains a subset
            csv.append(dict(ts=i * 0.5, src_ip=None, dst_ip=None, dst_port=port, label_raw=lab))
    pf = make_flows(rows)
    pf["capture_host"] = [r["capture_host"] for r in rows]
    cf = make_flows(csv)
    cf["label_raw"] = [c["label_raw"] for c in csv]
    lab, stats, summary = transfer_labels(pf, cf, min_victim_matches=50, n_common_ports=0)
    assert summary["victims"] == ["172.31.1.3"]
    labelled = pf[lab != "Benign"]
    assert set(labelled["capture_host"]) == {"172.31.1.3"} and len(labelled) >= 200


# ------------------------------------------------------------------ ATT&CK mapping
def test_attack_subset_is_consistent():
    for r in RULES:
        assert r.technique in TECHNIQUES, r.technique
    for t in TECHNIQUES.values():
        assert all(x in TACTICS for x in t["tactics"])


def test_rules_and_record_structure():
    feats = {"uniq_dst_ports_out": 300, "half_open_ratio": 0.8, "rst_ratio": 0.1}
    ct = candidate_techniques(feats, "host")
    assert ct[0]["id"] == "T1046" and ct[0]["wording"] == "behaviour consistent with"
    rec = build_record(entity="h", window_end="x", observed={}, current_stage={"BENIGN": 0.2, "RECON_DISCOVERY": 0.8},
                       current_features=feats, mode="host",
                       predicted={"most_likely_future_stage": "INITIAL_ACCESS_ATTEMPT"}, explanation=None,
                       model_info={})
    assert set(rec) >= {"observed", "inferred", "predicted", "not_supported"}
    assert [t["id"] for t in rec["predicted"]["attack_tactics"]] == ["TA0001", "TA0006"]
    assert rec["inferred"]["current_stage"] == "RECON_DISCOVERY"


# ------------------------------------------------------------------ explainability
def test_integrated_gradients_completeness():
    torch.manual_seed(0)
    m = NetworkWorldModel(6, 5, dropout=0.0)
    x = np.random.default_rng(0).normal(size=(10, 6)).astype(np.float32)
    e = explain(m, x, np.zeros(6, np.float32), [f"f{i}" for i in range(6)], K=5, n_steps=64)
    assert abs(e["convergence_delta"]) < 1e-2
    assert np.array(e["heatmap"]).shape == (10, 6)
    assert abs(sum(e["time_importance"]) - 1) < 1e-6


def test_evidence_sentence():
    raw = np.array([[1, 0.1], [5, 0.5], [300, 0.8]], dtype=float)
    ev = evidence_sentences([("uniq_dst_ports_out", 0.3), ("half_open_ratio", 0.1)], raw,
                            np.array([6, 0.05]), ["uniq_dst_ports_out", "half_open_ratio"])
    assert "300" in ev[0]["text"] and "normal: 6.0" in ev[0]["text"]
    assert "80%" in ev[1]["text"]


# ------------------------------------------------------------------ metrics
def test_threshold_ties_and_float_precision():
    p_neg = np.array([0.1] * 90 + [0.7777777910232545] * 10, dtype=np.float32)
    thr = threshold_at_fpr(p_neg, 0.05)
    rep = binary_report(p_neg, np.zeros(100), thr)
    assert rep["fpr"] <= 0.05


def test_onsets_and_early_warning():
    y = np.array([0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 1])
    assert onsets(y, quiet=5) == [6, 14]
    meta = pd.DataFrame({"series": [0] * 15, "t": range(15)})
    p = np.zeros(15)
    p[5] = 0.9                                                # warned 1 window before the first onset
    summ, df = early_warning(meta, p, {0: y}, thr=0.5, leads=(0, 1, 2))
    assert summ[1]["recall"] == 0.5 and summ[0]["recall"] == 0.0 and summ[1]["n_onsets"] == 2


# ------------------------------------------------------------------ ledger
def test_ledger_detects_tampering(tmp_path):
    led = EvidenceLedger(tmp_path / "l.sqlite")
    for i in range(10):
        led.append({"i": i, "p": 0.1 * i})
    assert led.verify()["ok"]
    root = led.merkle()
    con = sqlite3.connect(tmp_path / "l.sqlite")
    con.execute("UPDATE ledger SET payload='{\"i\":4,\"p\":0.0}' WHERE seq=5")
    con.commit()
    v = led.verify()
    assert not v["ok"] and v["first_bad_seq"] == 5 and v["reason"] == "payload modified"
    assert led.merkle() == root                            # stored hashes unchanged, payload differs


def test_ledger_detects_deleted_record(tmp_path):
    led = EvidenceLedger(tmp_path / "l.sqlite")
    for i in range(5):
        led.append({"i": i})
    con = sqlite3.connect(tmp_path / "l.sqlite")
    con.execute("DELETE FROM ledger WHERE seq=3")
    con.commit()
    assert not led.verify()["ok"]


def test_merkle_root():
    assert merkle_root([]) == "0" * 64
    h = ["aa" * 32, "bb" * 32, "cc" * 32]
    assert merkle_root(h) != merkle_root(h[:2]) and len(merkle_root(h)) == 64
