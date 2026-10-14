"""Extract the classes listed in labels.json from the PlantVillage parquet shards and
create a fixed, seeded, stratified train/val/test split.

    python scripts/prepare_dataset.py   # data/raw/hf/*.parquet -> data/processed/{train,val,test}/<class_key>/*.jpg

Images are written byte-for-byte (no re-encoding). Exact duplicates (same bytes) are
removed before splitting so the same picture can never land in both train and test.
The split depends only on the image hashes and the seed, not on file order.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.labels import get_labels  # noqa: E402

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def source_label_names(parquet_path: Path) -> list[str]:
    meta = pq.read_schema(parquet_path).metadata or {}
    hf = json.loads(meta.get(b"huggingface", b"{}"))
    names = hf.get("info", {}).get("features", {}).get("label", {}).get("names")
    if not names:
        raise SystemExit(f"{parquet_path.name}: no label names in parquet metadata")
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "hf")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    labels = get_labels()
    shards = sorted(args.raw.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet files in {args.raw} - run scripts/download_dataset.py first")
    names = source_label_names(shards[0])
    wanted = {}
    for info in labels:
        if info.source_label not in names:
            raise SystemExit(f"labels.json source_label {info.source_label!r} not in dataset labels")
        wanted[names.index(info.source_label)] = info.key
    print(f"selecting {len(wanted)} classes from {len(names)} in {len(shards)} shards")

    images: dict[str, tuple[str, bytes]] = {}
    conflicts: set[str] = set()
    raw_counts: Counter[str] = Counter()
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(columns=["image", "label"], batch_size=1024):
            cols = batch.to_pydict()
            for img, lab in zip(cols["image"], cols["label"]):
                key = wanted.get(lab)
                if key is None:
                    continue
                data = img["bytes"]
                raw_counts[key] += 1
                sha = hashlib.sha1(data).hexdigest()
                if sha in images and images[sha][0] != key:
                    conflicts.add(sha)
                images.setdefault(sha, (key, data))
        print(f"  read {shard.name}")
    for sha in conflicts:  # same image with two different labels: unusable
        images.pop(sha, None)

    shas = sorted(images)
    y = [images[s][0] for s in shas]
    trainval, test = train_test_split(shas, test_size=TEST_FRAC, stratify=y, random_state=args.seed)
    y_tv = [images[s][0] for s in trainval]
    train, val = train_test_split(trainval, test_size=VAL_FRAC / (1 - TEST_FRAC), stratify=y_tv, random_state=args.seed)
    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

    if args.out.exists():
        shutil.rmtree(args.out)
    rows = []
    for split, items in splits.items():
        for sha in items:
            key, data = images[sha]
            dest = args.out / split / key / f"{sha[:16]}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            rows.append((split, key, dest.relative_to(args.out).as_posix(), sha))
    with (args.out / "manifest.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("split", "class_key", "path", "sha1"), *rows])

    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for split, items in splits.items():
        c = Counter(images[s][0] for s in items)
        for key in labels.keys:
            counts[key][split] = c.get(key, 0)
    manifest_hash = hashlib.sha1("".join(f"{r[0]}{r[3]}" for r in rows).encode()).hexdigest()
    summary = {
        "source": labels.dataset,
        "seed": args.seed,
        "fractions": {"train": 1 - VAL_FRAC - TEST_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "raw_images": sum(raw_counts.values()),
        "unique_images": len(images),
        "duplicates_removed": sum(raw_counts.values()) - len(images) - len(conflicts),
        "conflicting_labels_removed": len(conflicts),
        "split_manifest_sha1": manifest_hash,
        "per_class": counts,
        "totals": {s: len(v) for s, v in splits.items()},
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "dataset_split.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["totals"]), f"duplicates removed: {summary['duplicates_removed']}")
    for key in labels.keys:
        print(f"  {key:24s} {counts[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
