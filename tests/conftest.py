import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netwm.io.schema import conform  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)   # tiny models: extra threads only add contention

DAY = "2018-03-01"
T0 = pd.Timestamp(f"{DAY} 12:00", tz="UTC")


def make_flows(rows):
    """rows: list of dicts with at least ts (seconds after T0), src_ip, dst_ip, dst_port."""
    base = dict(src_port=40000, proto=6, duration_s=0.5, fwd_pkts=3, bwd_pkts=3, fwd_bytes=100,
                bwd_bytes=200, flow_iat_mean=0.1, fwd_pkt_len_mean=33, bwd_pkt_len_mean=66,
                pkt_len_std=10, syn_cnt=1, ack_cnt=4, fin_cnt=1, rst_cnt=0, psh_cnt=1, urg_cnt=0,
                init_fwd_win=8192, label_raw="Benign", day=DAY)
    out = []
    for r in rows:
        d = dict(base)
        d.update(r)
        d["ts"] = T0 + pd.to_timedelta(d["ts"], unit="s")
        out.append(d)
    return conform(pd.DataFrame(out))


@pytest.fixture
def benign_flows():
    rng = np.random.default_rng(0)
    rows = []
    for w in range(40):
        for _ in range(20):
            h = f"172.31.1.{rng.integers(1, 6)}"
            rows.append(dict(ts=w * 60 + rng.uniform(0, 59), src_ip=h,
                             dst_ip=rng.choice(["8.8.8.8", "172.31.1.9", "1.1.1.1"]),
                             dst_port=int(rng.choice([53, 443, 80]))))
    return make_flows(rows)
