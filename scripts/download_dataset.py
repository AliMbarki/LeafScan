"""Download the PlantVillage parquet shards from Hugging Face.

Robust against slow / flaky CDN connections: each file is fetched in parallel
byte-range chunks with curl; chunks on a stalled connection are aborted and retried.

    python scripts/download_dataset.py            # -> data/raw/hf/*.parquet
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8 * 1024 * 1024


def list_files() -> list[tuple[str, int]]:
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main/data", timeout=30) as r:
        return [(f["path"], f["size"]) for f in json.load(r) if f["path"].endswith(".parquet")]


def fetch_chunk(url: str, part: Path, start: int, end: int) -> None:
    want = end - start + 1
    for attempt in range(40):
        if part.exists() and part.stat().st_size == want:
            return
        cmd = [
            "curl", "-sL", "--fail", "-r", f"{start}-{end}", "-o", str(part),
            "--connect-timeout", "15", "--max-time", "240",
            "--speed-limit", "150000", "--speed-time", "6",  # abort connections slower than ~150 KB/s
            url,
        ]
        subprocess.run(cmd, check=False)
    raise RuntimeError(f"chunk {start}-{end} failed after retries")


def download(path: str, size: int, out_dir: Path, workers: int) -> Path:
    dest = out_dir / Path(path).name
    if dest.exists() and dest.stat().st_size == size:
        print(f"{dest.name}: already complete")
        return dest
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    parts_dir = out_dir / (dest.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fetch_chunk, url, parts_dir / f"{i:05d}", s, e): i for i, (s, e) in enumerate(ranges)}
        for f in as_completed(futs):
            f.result()
            done += 1
            print(f"\r{dest.name}: {done}/{len(ranges)} chunks", end="", flush=True)
    print()
    tmp = dest.with_suffix(".tmp")
    with tmp.open("wb") as out:
        for i in range(len(ranges)):
            with (parts_dir / f"{i:05d}").open("rb") as src:
                shutil.copyfileobj(src, out)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: size mismatch")
    tmp.replace(dest)
    shutil.rmtree(parts_dir)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "hf")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path, size in list_files():
        download(path, size, args.out, args.workers)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
