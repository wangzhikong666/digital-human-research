#!/usr/bin/env python3
"""Download the exact README checkpoint, resuming verified-length segments."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "checkpoints/checkpoint.pt"
SEED = TARGET.with_suffix(".pt.part")
SIZE = 15458082007
SHA256 = "b563aec09dc8401d8270ed41c89e69e4c765f3806255607cae420af6d80efb1a"
URL = "https://huggingface.co/UmutKocasari/FaceAnything/resolve/main/checkpoint.pt"
CHUNK = 8 * 1024 * 1024
CACHE = TARGET.parent / ".download-chunks" / SHA256


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def main():
    if TARGET.exists():
        if TARGET.stat().st_size != SIZE or checksum(TARGET) != SHA256:
            raise RuntimeError("Existing checkpoint does not match the README release.")
        print("Official checkpoint already present; SHA256 verified.", flush=True)
        return

    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for offset in range(0, SIZE, CHUNK):
        length = min(CHUNK, SIZE - offset)
        path = CACHE / str(offset)
        if path.exists() and path.stat().st_size == length:
            continue
        if SEED.exists() and SEED.stat().st_size >= offset + length:
            with SEED.open("rb") as stream:
                stream.seek(offset)
                path.write_bytes(stream.read(length))
            continue
        jobs.append((offset, length, path))

    remaining = sum(length for _, length, _ in jobs)
    print(f"Official checkpoint: {(SIZE - remaining) / 1e9:.3f}/{SIZE / 1e9:.3f} GB saved.", flush=True)
    lock = threading.Lock()
    stop = threading.Event()
    completed = 0
    last_report = time.monotonic()

    def download(job):
        nonlocal completed, last_report
        offset, length, path = job
        for attempt in range(8):
            if stop.is_set():
                return
            try:
                end = offset + length - 1
                request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-{end}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{SIZE}":
                        raise RuntimeError("Server returned an unexpected byte range.")
                    data = response.read(length + 1)
                if len(data) != length:
                    raise RuntimeError("Incomplete segment.")
                partial = path.with_suffix(".partial")
                partial.write_bytes(data)
                os.replace(partial, path)
                with lock:
                    completed += length
                    if time.monotonic() - last_report >= 10:
                        print(f"Saved {(SIZE - remaining + completed) / 1e9:.3f}/{SIZE / 1e9:.3f} GB", flush=True)
                        last_report = time.monotonic()
                return
            except Exception:
                if attempt == 7:
                    raise
                stop.wait(min(attempt + 1, 5))

    executor = ThreadPoolExecutor(max_workers=32)
    try:
        futures = [executor.submit(download, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    print("Download complete; assembling and checking official SHA256...", flush=True)
    assembled = TARGET.with_suffix(".pt.assembling")
    digest = hashlib.sha256()
    with assembled.open("wb") as output:
        for offset in range(0, SIZE, CHUNK):
            data = (CACHE / str(offset)).read_bytes()
            output.write(data)
            digest.update(data)
    if assembled.stat().st_size != SIZE or digest.hexdigest() != SHA256:
        raise RuntimeError("Checkpoint checksum mismatch; segments preserved for inspection.")
    os.replace(assembled, TARGET)
    for path in CACHE.iterdir():
        path.unlink()
    CACHE.rmdir()
    CACHE.parent.rmdir()
    SEED.unlink(missing_ok=True)
    print(f"SHA256 verified: {SHA256}\nReady: {TARGET}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed segments preserved.", flush=True)
        raise SystemExit(130)
