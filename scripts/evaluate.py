"""Evaluate the trained model.

    python scripts/evaluate.py            # held-out test split  -> reports/test_*
    python scripts/evaluate.py --real     # data/real_test/<class_key>/*.jpg -> reports/real_test_*

Real photos are run through the full app pipeline (leaf detection -> crop -> classify),
because that is what the user experiences. The whole-image classifier accuracy is
reported alongside for comparison. Dataset images are much cleaner than webcam photos,
so expect real-photo accuracy to be lower - that number is the honest one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.capture import load_image_file  # noqa: E402
from app.classify.data import list_folder  # noqa: E402
from app.classify.metrics import write_reports  # noqa: E402
from app.classify.predictor import MODEL_FILENAME, TorchClassifier  # noqa: E402
from app.contracts import DEFAULT_UNCERTAINTY_THRESHOLD  # noqa: E402
from app.labels import get_labels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "models" / MODEL_FILENAME)
    ap.add_argument("--real", action="store_true", help="evaluate data/real_test instead of the test split")
    ap.add_argument("--folder", type=Path, help="custom folder laid out as <class_key>/*.jpg")
    ap.add_argument("--threshold", type=float, default=DEFAULT_UNCERTAINTY_THRESHOLD)
    ap.add_argument("--device", default=None)
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--detector", action="store_true", help="leaf-detection rate + speed on the test split")
    args = ap.parse_args()

    labels = get_labels()
    if args.detector:
        return evaluate_detector(labels, args.folder or ROOT / "data" / "processed" / "test", args.reports)
    clf = TorchClassifier(args.model, args.threshold, labels, device=args.device)
    folder = args.folder or (ROOT / "data" / ("real_test" if args.real else "processed/test"))
    items = list_folder(folder, labels)
    if not items:
        print(f"No images found in {folder} (expected <class_key>/*.jpg, keys: {', '.join(labels.keys)})")
        return 1
    print(f"{len(items)} images from {folder}  |  model {clf.model_version}")

    if not (args.real or args.folder):
        probs = np.stack([clf.probabilities(load_image_file(p)) for p, _ in items])
        y = [t for _, t in items]
        m = write_reports("test", y, probs, labels, args.threshold, args.reports, f"Test split - {clf.model_version}",
                          extra={"model_version": clf.model_version})
        print(f"accuracy {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  uncertain {m['uncertain_rate']:.2%}")
        return 0

    from app.pipeline import AnalysisPipeline, crop_to_leaf
    from app.vision import build_detector, build_overlay

    detector = build_detector()
    whole, piped, y, no_leaf = [], [], [], []
    for path, target in items:
        img = load_image_file(path)
        leaf = detector.detect_leaf(img)
        whole.append(clf.probabilities(img))
        piped.append(clf.probabilities(crop_to_leaf(img, leaf)))
        y.append(target)
        if not leaf.found:
            no_leaf.append(path.relative_to(folder).as_posix())
    whole_m = write_reports("real_test_whole_image", y, np.stack(whole), labels, args.threshold, args.reports,
                            f"Real photos, whole image - {clf.model_version}")
    m = write_reports("real_test", y, np.stack(piped), labels, args.threshold, args.reports,
                      f"Real photos, full pipeline - {clf.model_version}",
                      extra={"model_version": clf.model_version, "no_leaf_detected": no_leaf,
                             "whole_image_accuracy": whole_m["accuracy"]})
    print(f"pipeline accuracy {m['accuracy']:.4f} (whole image {whole_m['accuracy']:.4f}), "
          f"uncertain {m['uncertain_rate']:.2%}, no leaf found in {len(no_leaf)}/{len(items)}")
    print(json.dumps({k: round(v['recall'], 3) for k, v in m['per_class'].items()}, indent=1))
    return 0


def evaluate_detector(labels, folder: Path, reports: Path) -> int:
    """Every test image contains exactly one leaf, so 'found' should be ~100%."""
    import time

    from app.vision import build_detector

    det = build_detector()
    items = list_folder(folder, labels)
    found: dict[str, list[int]] = {c.key: [0, 0] for c in labels}
    times, missed = [], []
    for path, target in items:
        img = load_image_file(path)
        t0 = time.perf_counter()
        r = det.detect_leaf(img)
        times.append((time.perf_counter() - t0) * 1000)
        key = labels.by_id(target).key
        found[key][1] += 1
        if r.found:
            found[key][0] += 1
        else:
            missed.append({"image": path.relative_to(folder).as_posix(), "reason": r.message})
    total_found = sum(f for f, _ in found.values())
    out = {
        "images": len(items),
        "found_rate": total_found / max(len(items), 1),
        "per_class_found_rate": {k: f / max(n, 1) for k, (f, n) in found.items()},
        "ms_mean": float(np.mean(times)),
        "ms_p95": float(np.percentile(times, 95)),
        "missed": missed,
        "note": "Dataset images (256x256, plain backgrounds). Webcam frames are larger: work size is capped at 448px.",
    }
    reports.mkdir(exist_ok=True)
    (reports / "leaf_detection.json").write_text(json.dumps(out, indent=2))
    print(f"leaf found in {total_found}/{len(items)} ({out['found_rate']:.2%}); {out['ms_mean']:.0f} ms mean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
