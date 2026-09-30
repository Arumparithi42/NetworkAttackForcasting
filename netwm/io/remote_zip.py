"""Read single members of a huge remote ZIP (e.g. the CSE-CIC-IDS2018 per-day pcap.zip on S3)
through HTTP range requests, without downloading the whole archive."""
from __future__ import annotations

import http.client
import io
import time
import urllib.error
import urllib.request
import zipfile


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, retries: int = 5):
        self.url, self.pos, self.retries = url, 0, retries
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** (attempt + 1))
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(url: str, buffer_size: int = 8 << 20) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=buffer_size))
