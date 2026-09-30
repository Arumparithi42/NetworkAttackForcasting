"""Schema, CSV loader fixes, label mapping, windowing, causality and split leakage."""
import numpy as np
import pandas as pd
import pytest

from conftest import T0, make_flows
from netwm.data.splits import sequence_starts, window_roles
from netwm.features import feature_spec as fs
from netwm.features.temporal import build_states
from netwm.features.windowing import add_window, host_window_features, network_window_features
from netwm.io.load_cic import load_cic_csv
from netwm.io.schema import CANONICAL
from netwm.labels.build_labels import (AMBIGUOUS, EXCLUDED, STAGE_ID, map_flow_stage,
                                       window_stage)

CIDRS = ["172.31.0.0/16"]


# ------------------------------------------------------------------ CSV loader
def test_cic2018_loader_fixes_clock_timezone_and_headers(tmp_path):
    header = ("Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,"
              "TotLen Bwd Pkts,Flow IAT Mean,SYN Flag Cnt,Init Fwd Win Byts,Label")
    rows = [
        "443,6,01/03/2018 08:30:00,1000000,3,2,100,200,500000,1,8192,Benign",   # 08:30 local
        "22,6,01/03/2018 02:15:00,2000000,5,5,10,20,Infinity,1,-1,Infilteration",  # 14:15 local
        header,                                                                   # repeated header
        "53,17,01/01/1970 03:00:00,10,1,1,1,1,1,0,0,Benign",                      # garbage row
    ]
    p = tmp_path / "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n")
    df = load_cic_csv(p, utc_offset_hours=-4, clock_12h_threshold=8)
    assert list(df.columns) == list(CANONICAL)
    assert len(df) == 2
    assert df["ts"].iloc[0] == pd.Timestamp("2018-03-01 12:30", tz="UTC")     # 08:30 AST -> UTC
    assert df["ts"].iloc[1] == pd.Timestamp("2018-03-01 18:15", tz="UTC")     # 02:15 -> 14:15 AST
    assert df["duration_s"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(df["flow_iat_mean"].iloc[1]) and np.isnan(df["init_fwd_win"].iloc[1])
    assert df["day"].iloc[0] == "2018-03-01"


# ------------------------------------------------------------------ labels
def test_label_mapping_and_unknown_label():
    s = map_flow_stage(pd.Series(["Benign", "DoS attacks-Hulk", "Bot", "SSH-Bruteforce",
                                  "DDOS attack-LOIC-UDP", "Brute Force -XSS"]))
    assert s.tolist() == [0, STAGE_ID["IMPACT"], STAGE_ID["COMMAND_AND_CONTROL"],
                          STAGE_ID["INITIAL_ACCESS_ATTEMPT"], STAGE_ID["IMPACT"],
                          STAGE_ID["INITIAL_ACCESS_ATTEMPT"]]
    with pytest.raises(ValueError):
        map_flow_stage(pd.Series(["Benign", "Totally-New-Attack"]))


def test_infiltration_rule_uses_source_side():
    s = map_flow_stage(pd.Series(["Infilteration", "Infilteration"]), pd.Series([True, False]))
    assert s.tolist() == [STAGE_ID["RECON_DISCOVERY"], STAGE_ID["INITIAL_ACCESS_ATTEMPT"]]


def test_window_stage_rules():
    df = pd.DataFrame({"window": [1] * 5 + [2] * 2 + [3] * 4 + [4],
                       "flow_stage": [0, 1, 1, 1, 5] + [0, 1] + [1, 1, 1, 4] + [EXCLUDED]})
    df.loc[df.index[8:11], "flow_stage"] = [4, 4, 4]
    out = window_stage(df, ["window"], min_flows=3).set_index("window")["stage"]
    assert out[1] == STAGE_ID["RECON_DISCOVERY"]          # 3 recon flows; 1 impact flow ignored
    assert out[2] == AMBIGUOUS                             # a single malicious flow
    assert out[3] == STAGE_ID["COMMAND_AND_CONTROL"]      # most advanced stage with >= 3 flows
    assert out[4] == EXCLUDED


# ------------------------------------------------------------------ windowing
def test_host_features_port_scan():
    rows = [dict(ts=5 + i * 0.01, src_ip="172.31.1.1", dst_ip="172.31.1.2", dst_port=1000 + i,
                 fwd_pkts=1, bwd_pkts=0, fwd_bytes=0, bwd_bytes=0, fin_cnt=0) for i in range(500)]
    f = host_window_features(make_flows(rows), CIDRS)
    a = f[f["host"] == "172.31.1.1"].iloc[0]
    b = f[f["host"] == "172.31.1.2"].iloc[0]
    assert a["uniq_dst_ports_out"] == 500 and a["half_open_ratio"] == 1.0
    assert a["n_flows_out"] == 500 and a["n_flows_in"] == 0
    assert a["dst_port_entropy_out"] == pytest.approx(np.log2(500), rel=1e-4)
    assert b["uniq_dst_ports_in"] == 500 and b["n_flows_in"] == 500
    assert a["new_ports_out"] == 500 and a["out_degree_internal"] == 1


def test_bytes_are_swapped_for_the_receiving_host():
    f = host_window_features(make_flows([dict(ts=1, src_ip="172.31.1.1", dst_ip="172.31.1.2",
                                              dst_port=445, fwd_bytes=10, bwd_bytes=999)]), CIDRS)
    a = f.set_index("host").loc["172.31.1.1"]
    b = f.set_index("host").loc["172.31.1.2"]
    assert (a["bytes_sent"], a["bytes_recv"]) == (10, 999)
    assert (b["bytes_sent"], b["bytes_recv"]) == (999, 10)
    assert a["svc_smb"] == 1.0


@pytest.mark.parametrize("mode", ["host", "network"])
def test_features_are_causal(benign_flows, mode):
    """Changing traffic after window t must not change any feature at windows <= t."""
    def states(flows):
        flows = add_window(flows, 60)
        windows = np.arange(flows["window"].min(), flows["window"].max() + 1)
        if mode == "host":
            feats = host_window_features(flows, CIDRS)
            return build_states(feats, "host", "d", windows, fs.host_features(False), 0.1)
        feats = network_window_features(flows)
        return build_states(feats, "network", "d", windows, fs.network_features(), 0.1)

    a = states(benign_flows)
    cut = T0 + pd.Timedelta(minutes=20)
    later = benign_flows["ts"] >= cut
    changed = benign_flows.copy()
    changed.loc[later, "dst_port"] = 31337
    changed.loc[later, "fwd_bytes"] *= 50
    b = states(changed)
    w_cut = cut.value // 10**9 // 60
    cols = [c for c in a.columns if c not in ("day", "host", "window")]
    ea = a[a["window"] < w_cut].sort_values(["host", "window"])[cols].to_numpy()
    eb = b[b["window"] < w_cut].sort_values(["host", "window"])[cols].to_numpy()
    np.testing.assert_allclose(ea, eb, rtol=1e-6, atol=1e-6)
    assert not np.allclose(a[a["window"] >= w_cut][cols].to_numpy(), b[b["window"] >= w_cut][cols].to_numpy())


# ------------------------------------------------------------------ splits
def test_no_sequence_straddles_split_roles():
    cfg = {"protocols": {"A": {"train": ["d"], "test": [], "split_days": {}}}, "protocol": "A",
           "validation": {"block_windows": 60, "every": 4, "offset": 2}}
    w = np.arange(1000, 1400)
    roles = window_roles("d", w, cfg, 60)
    assert set(roles) == {"train", "val"}
    L, K = 10, 5
    for role in ("train", "val"):
        for t in sequence_starts(roles, role, L, K):
            assert (roles[t - L + 1:t + K + 1] == role).all()


def test_split_day_cut():
    cfg = {"protocols": {"A": {"train": [], "test": [], "split_days": {"d": "2018-03-02T16:30:00Z"}}},
           "protocol": "A"}
    cut = pd.Timestamp("2018-03-02T16:30:00Z").value // 10**9 // 60
    w = np.arange(cut - 5, cut + 5)
    roles = window_roles("d", w, cfg, 60)
    assert (roles[w < cut] == "train").all() and (roles[w >= cut] == "test").all()


def test_excluded_day_is_unused():
    cfg = {"protocols": {"A": {"train": ["d"], "test": []}}, "protocol": "A", "excluded_days": {"d": "x"}}
    assert (window_roles("d", np.arange(10), cfg, 60) == "").all()
