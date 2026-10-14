"""Create a demo history database (separate from your real one) for showing the History screen.

    python scripts/seed_demo_history.py                 # -> data/demo/, uses the trained model
    python -m app --data-dir data/demo                  # open the app on the demo data

Each demo plant is a species from labels.json with a short "story" (e.g. healthy, then a
disease appears). Images are real held-out test images run through the full pipeline,
so the stored predictions/overlays are genuine model outputs.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.capture import load_image_file  # noqa: E402
from app.config import Settings  # noqa: E402
from app.contracts import ScanRecord  # noqa: E402
from app.labels import get_labels  # noqa: E402
from app.main import build_pipeline  # noqa: E402
from app.storage.db import SqliteScanStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "demo")
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    labels = get_labels()
    test_dir = ROOT / "data" / "processed" / "test"
    pools = {c.key: sorted((test_dir / c.key).glob("*.jpg")) for c in labels}
    if not any(pools.values()):
        raise SystemExit("no test images - run scripts/prepare_dataset.py first")

    # One demo plant per species: healthy scans first, then one of its diseases appears.
    stories = []
    for plant in labels.plants:
        classes = [c for c in labels if c.plant == plant]
        healthy = [c.key for c in classes if c.healthy]
        sick = [c.key for c in classes if not c.healthy]
        if healthy and sick:
            stories.append((f"{plant} bed", healthy[0], rng.choice(sick)))
    stories.append(("Plant 1", stories[0][1], stories[0][2]))

    if args.out.exists():
        shutil.rmtree(args.out)
    settings = Settings(data_dir=args.out, use_mocks=args.mock)
    pipeline, warning = build_pipeline(settings)
    if warning:
        print("warning:", warning)
    store = SqliteScanStore(settings.db_path, settings.scans_dir)
    now = datetime.now().astimezone().replace(second=0, microsecond=0)
    n = 0
    for name, healthy, sick in stories:
        plant = store.get_or_create_plant(name)
        onset = rng.randint(args.days // 3, 2 * args.days // 3)
        for day in range(args.days, -1, -1):
            if rng.random() > 0.45:
                continue
            key = healthy if (args.days - day) < onset else sick
            img = load_image_file(rng.choice(pools[key]))
            result = pipeline.analyze(img)
            ts = now - timedelta(days=day, hours=rng.randint(0, 9), minutes=rng.randint(0, 59))
            store.save_scan(ScanRecord.from_analysis(result, plant.id, ts))
            n += 1
    print(f"seeded {n} scans for {len(stories)} plants into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
