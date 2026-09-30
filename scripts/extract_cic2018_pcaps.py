"""Stream per-host PCAPs of one CSE-CIC-IDS2018 day from S3 and convert them to flow parquet files.

Only flows are stored (a few hundred MB per day) - the ~50-60 GB of PCAP is never written to disk.

    python scripts/extract_cic2018_pcaps.py --day Thursday-01-03-2018 --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netwm.io.pcap_packets import pcap_to_flows  # noqa: E402
from netwm.io.remote_zip import open_remote_zip  # noqa: E402

BASE = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Original%20Network%20Traffic%20and%20Log%20data"
IP_RE = re.compile(r"(\d+\.\d+\.\d+\.\d+)")
_zip = None


def _init(url):
    global _zip
    warnings.filterwarnings("ignore")
    _zip = open_remote_zip(url)


def _work(args):
    member, out_path, day = args
    if Path(out_path).exists():
        return member, "skip", 0, 0.0
    t = time.time()
    try:
        with _zip.open(member) as fh:
            import io
            df = pcap_to_flows(io.BufferedReader(fh, buffer_size=4 << 20), day=day)
        df["capture_host"] = IP_RE.findall(member)[-1]
        tmp = str(out_path) + ".tmp"
        df.to_parquet(tmp, index=False)
        Path(tmp).rename(out_path)
        return member, "ok", len(df), time.time() - t
    except Exception as exc:  # keep going; failures are listed at the end
        return member, f"error: {exc!r}", 0, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="e.g. Thursday-01-03-2018")
    ap.add_argument("--out", default="data/interim/pcap_flows")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the N smallest members (testing)")
    a = ap.parse_args()

    url = f"{BASE}/{a.day}/pcap.zip"
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", a.day)
    day = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    out_dir = Path(a.out) / day
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = [i for i in open_remote_zip(url).infolist() if not i.is_dir() and IP_RE.search(i.filename)]
    infos.sort(key=lambda i: i.file_size, reverse=not a.limit)
    if a.limit:
        infos = infos[: a.limit]
    jobs = []
    for i in infos:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(i.filename).name)
        jobs.append((i.filename, out_dir / f"{safe}.parquet", day))
    total_gb = sum(i.file_size for i in infos) / 1e9
    print(f"{len(jobs)} members, {total_gb:.1f} GB uncompressed -> {out_dir}", flush=True)

    t0, done_gb, errors = time.time(), 0.0, []
    size = {i.filename: i.file_size for i in infos}
    with mp.Pool(a.workers, initializer=_init, initargs=(url,)) as pool:
        for k, (member, status, n, dt) in enumerate(pool.imap_unordered(_work, jobs), 1):
            done_gb += size[member] / 1e9
            if status.startswith("error"):
                errors.append((member, status))
            print(f"[{k}/{len(jobs)}] {status:5s} {n:8d} flows {dt:6.1f}s  {member}  "
                  f"({done_gb:.1f}/{total_gb:.1f} GB, {time.time() - t0:.0f}s)", flush=True)
    print("errors:", errors if errors else "none", flush=True)


if __name__ == "__main__":
    main()
