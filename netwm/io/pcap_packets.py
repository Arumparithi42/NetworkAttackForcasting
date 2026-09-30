"""PCAP -> bidirectional flows with packet-level accumulators (pure Python, dpkt).

Why our own extractor: it runs everywhere (no Java/CICFlowMeter, no tshark), streams large files
with constant memory, and keeps packet-level information that flow CSVs lose (TTL, IP
fragmentation, TCP window, retransmissions, payload-size distribution).

Flow semantics are close to CICFlowMeter so that dataset labels can be transferred by matching
(netwm/labels/match.py):
  * key = bidirectional 5-tuple; the initiator (first packet's sender) is the "forward" direction;
  * a flow is closed when a FIN or RST packet is seen (the packet is included), or when it has
    been active longer than `active_timeout` seconds, or idle longer than `idle_timeout`;
  * byte counts are transport payload bytes (as in CICFlowMeter "TotLen Fwd Pkts").
"""
from __future__ import annotations

import socket
from typing import BinaryIO, Iterator

import numpy as np
import pandas as pd

try:
    import dpkt
except ImportError:  # pragma: no cover - dpkt is a core requirement, but keep import errors clear
    dpkt = None

TCP, UDP, ICMP = 6, 17, 1
FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
PLEN_EDGES = (0, 64, 256, 1024)             # payload histogram: 0 | 1-64 | 65-256 | 257-1024 | >1024
IGNORED_IPS = {"169.254.169.254"}          # cloud metadata service (AWS) - infrastructure noise


class _Flow:
    __slots__ = ("key", "src", "dst", "sport", "dport", "proto", "first", "last", "fwd_pkts",
                 "bwd_pkts", "fwd_bytes", "bwd_bytes", "len_sum", "len_sq", "fwd_len_sum",
                 "bwd_len_sum", "iat_sum", "syn", "ack", "fin", "rst", "psh", "urg",
                 "init_fwd_win", "fwd_ttl_sum", "fwd_ttl_sq", "bwd_ttl_sum", "bwd_ttl_sq",
                 "fwd_ttls", "bwd_ttls", "frag", "retrans", "plen", "win_sum", "win_sq",
                 "win_n", "seen_seq")

    def __init__(self, key, src, dst, sport, dport, proto, ts):
        self.key, self.src, self.dst, self.sport, self.dport, self.proto = key, src, dst, sport, dport, proto
        self.first = self.last = ts
        self.fwd_pkts = self.bwd_pkts = self.fwd_bytes = self.bwd_bytes = 0
        self.len_sum = self.len_sq = self.fwd_len_sum = self.bwd_len_sum = self.iat_sum = 0.0
        self.syn = self.ack = self.fin = self.rst = self.psh = self.urg = 0
        self.init_fwd_win = np.nan
        self.fwd_ttl_sum = self.fwd_ttl_sq = self.bwd_ttl_sum = self.bwd_ttl_sq = 0.0
        self.fwd_ttls, self.bwd_ttls = set(), set()
        self.frag = self.retrans = 0
        self.plen = [0, 0, 0, 0, 0]
        self.win_sum = self.win_sq = 0.0
        self.win_n = 0
        self.seen_seq = set()

    def record(self) -> dict:
        n = self.fwd_pkts + self.bwd_pkts
        mean = self.len_sum / n if n else 0.0
        return {
            "ts": self.first, "src_ip": self.src, "dst_ip": self.dst, "src_port": self.sport,
            "dst_port": self.dport, "proto": self.proto, "duration_s": self.last - self.first,
            "fwd_pkts": self.fwd_pkts, "bwd_pkts": self.bwd_pkts,
            "fwd_bytes": self.fwd_bytes, "bwd_bytes": self.bwd_bytes,
            "flow_iat_mean": self.iat_sum / (n - 1) if n > 1 else 0.0,
            "fwd_pkt_len_mean": self.fwd_len_sum / self.fwd_pkts if self.fwd_pkts else 0.0,
            "bwd_pkt_len_mean": self.bwd_len_sum / self.bwd_pkts if self.bwd_pkts else 0.0,
            "pkt_len_std": float(np.sqrt(max(self.len_sq / n - mean * mean, 0.0))) if n else 0.0,
            "syn_cnt": self.syn, "ack_cnt": self.ack, "fin_cnt": self.fin, "rst_cnt": self.rst,
            "psh_cnt": self.psh, "urg_cnt": self.urg, "init_fwd_win": self.init_fwd_win,
            "fwd_ttl_sum": self.fwd_ttl_sum, "fwd_ttl_sq": self.fwd_ttl_sq,
            "bwd_ttl_sum": self.bwd_ttl_sum, "bwd_ttl_sq": self.bwd_ttl_sq,
            "fwd_ttl_distinct": len(self.fwd_ttls), "bwd_ttl_distinct": len(self.bwd_ttls),
            "frag_pkts": self.frag, "retrans_pkts": self.retrans,
            "plen_b0": self.plen[0], "plen_b1": self.plen[1], "plen_b2": self.plen[2],
            "plen_b3": self.plen[3], "plen_b4": self.plen[4],
            "win_sum": self.win_sum, "win_sq": self.win_sq, "win_n": self.win_n,
        }


def _plen_bin(n: int) -> int:
    if n == 0:
        return 0
    if n <= 64:
        return 1
    if n <= 256:
        return 2
    if n <= 1024:
        return 3
    return 4


def _open_reader(fh: BinaryIO):
    magic = fh.read(4)
    if hasattr(fh, "seekable") and fh.seekable():
        fh.seek(0)
    else:
        raise ValueError("PCAP stream must be seekable")
    if magic == b"\x0a\x0d\x0d\x0a":
        return dpkt.pcapng.Reader(fh)
    return dpkt.pcap.Reader(fh)


def _safe_packets(reader):
    """Yield packets until the end of the capture; a truncated last record (common in large
    captures that were cut off) ends the stream instead of raising."""
    it = iter(reader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except (dpkt.NeedData, dpkt.UnpackError, ValueError):
            return


def iter_flows(fh: BinaryIO, active_timeout: float = 120.0, idle_timeout: float = 60.0,
               close_on_fin: bool = True, max_seq_track: int = 4096) -> Iterator[dict]:
    """Stream flow records (dicts) from a PCAP/PCAPNG file handle."""
    if dpkt is None:
        raise ImportError("dpkt is required for PCAP parsing: pip install dpkt")
    reader = _open_reader(fh)
    link = reader.datalink()
    active: dict[tuple, _Flow] = {}
    last_sweep = None
    inet_ntoa = socket.inet_ntoa

    for ts, buf in _safe_packets(reader):
        try:
            if link == 1:                         # Ethernet
                ip = dpkt.ethernet.Ethernet(buf).data
            elif link in (101, 228):              # raw IPv4
                ip = dpkt.ip.IP(buf)
            elif link == 113:                     # Linux cooked capture
                ip = dpkt.sll.SLL(buf).data
            else:
                continue
        except (dpkt.UnpackError, IndexError, ValueError):
            continue
        if not isinstance(ip, dpkt.ip.IP):
            continue
        src, dst = inet_ntoa(ip.src), inet_ntoa(ip.dst)
        if src in IGNORED_IPS or dst in IGNORED_IPS:
            continue
        proto = ip.p
        l4 = ip.data
        flags = 0
        win = None
        seq = None
        if proto == TCP and isinstance(l4, dpkt.tcp.TCP):
            sport, dport, flags, win, seq = l4.sport, l4.dport, l4.flags, l4.win, l4.seq
            payload = len(l4.data)
        elif proto == UDP and isinstance(l4, dpkt.udp.UDP):
            sport, dport = l4.sport, l4.dport
            payload = len(l4.data)
        elif proto == ICMP:
            sport = dport = 0
            payload = max(len(bytes(ip.data)) - 8, 0)
        else:
            continue
        if (src, sport) <= (dst, dport):
            key = (src, sport, dst, dport, proto)
        else:
            key = (dst, dport, src, sport, proto)

        f = active.get(key)
        if f is not None and (ts - f.first > active_timeout or ts - f.last > idle_timeout):
            yield f.record()
            del active[key]
            f = None
        if f is None:
            f = _Flow(key, src, dst, sport, dport, proto, ts)
            active[key] = f
        fwd = src == f.src and sport == f.sport
        n_before = f.fwd_pkts + f.bwd_pkts
        if n_before:
            f.iat_sum += ts - f.last
        f.last = ts
        f.len_sum += payload
        f.len_sq += payload * payload
        ttl = ip.ttl
        if fwd:
            f.fwd_pkts += 1
            f.fwd_bytes += payload
            f.fwd_len_sum += payload
            f.fwd_ttl_sum += ttl
            f.fwd_ttl_sq += ttl * ttl
            if len(f.fwd_ttls) < 16:
                f.fwd_ttls.add(ttl)
            if win is not None and f.fwd_pkts == 1:
                f.init_fwd_win = win
        else:
            f.bwd_pkts += 1
            f.bwd_bytes += payload
            f.bwd_len_sum += payload
            f.bwd_ttl_sum += ttl
            f.bwd_ttl_sq += ttl * ttl
            if len(f.bwd_ttls) < 16:
                f.bwd_ttls.add(ttl)
        if ip.mf or ip.offset:
            f.frag += 1
        f.plen[_plen_bin(payload)] += 1
        if win is not None:
            f.win_sum += win
            f.win_sq += win * win
            f.win_n += 1
        if flags:
            f.syn += bool(flags & SYN)
            f.ack += bool(flags & ACK)
            f.fin += bool(flags & FIN)
            f.rst += bool(flags & RST)
            f.psh += bool(flags & PSH)
            f.urg += bool(flags & URG)
        if seq is not None and payload > 0:
            sk = (fwd, seq, payload)
            if sk in f.seen_seq:
                f.retrans += 1
            elif len(f.seen_seq) < max_seq_track:
                f.seen_seq.add(sk)

        if close_on_fin and flags & (FIN | RST):
            yield f.record()
            del active[key]

        # periodic sweep of idle flows keeps memory bounded on long captures
        if last_sweep is None:
            last_sweep = ts
        elif ts - last_sweep > 30:
            last_sweep = ts
            stale = [k for k, fl in active.items() if ts - fl.last > idle_timeout
                     or ts - fl.first > active_timeout]
            for k in stale:
                yield active.pop(k).record()

    for fl in active.values():
        yield fl.record()


def pcap_to_flows(path_or_fh, day: str | None = None, **kwargs) -> pd.DataFrame:
    """Parse a PCAP into a canonical flow DataFrame (+ packet accumulator columns)."""
    from netwm.io.schema import conform

    if isinstance(path_or_fh, (str, bytes)) or hasattr(path_or_fh, "__fspath__"):
        with open(path_or_fh, "rb") as fh:
            rows = list(iter_flows(fh, **kwargs))
    else:
        rows = list(iter_flows(path_or_fh, **kwargs))
    df = pd.DataFrame(rows)
    if df.empty:
        from netwm.io.schema import PACKET_COLUMNS, empty_frame
        out = empty_frame()
        for c in PACKET_COLUMNS:
            out[c] = pd.Series(dtype="float32")
        return out
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["label_raw"] = pd.NA
    df["day"] = day if day is not None else df["ts"].dt.strftime("%Y-%m-%d").iloc[0]
    out = conform(df)
    for c in out.columns:
        if out[c].dtype == "float64" or out[c].dtype == "int64":
            out[c] = out[c].astype("float32")
    return out.sort_values("ts", ignore_index=True)
