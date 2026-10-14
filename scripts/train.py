"""Fine-tune MobileNetV3-Small on data/processed and write models/ + reports/.

    python scripts/train.py                 # defaults: 20 epochs, batch 64, AdamW, cosine LR
    python scripts/train.py --epochs 2      # quick smoke run

Steps: train (augmented) -> pick best epoch by val macro-F1 -> fit softmax temperature
on val -> evaluate on the held-out test split -> save versioned weights + reports.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.classify.data import LeafFolderDataset, list_folder, load_eval_arrays  # noqa: E402
from app.classify.metrics import write_reports  # noqa: E402
from app.classify.model import build_model, save_checkpoint  # noqa: E402
from app.classify.predictor import MODEL_FILENAME  # noqa: E402
from app.classify.preprocess import IMG_SIZE, MEAN, RESIZE_TO, STD  # noqa: E402
from app.contracts import DEFAULT_UNCERTAINTY_THRESHOLD  # noqa: E402
from app.labels import get_labels  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int) -> None:
    import cv2

    torch.set_num_threads(1)  # avoid N workers x M threads oversubscribing the CPU
    cv2.setNumThreads(1)
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def normalize_uint8(x: torch.Tensor, device) -> torch.Tensor:
    """uint8 NCHW -> normalised float on device (same maths as app.classify.preprocess)."""
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (x.to(device, non_blocking=True).float().div_(255) - mean) / std


@torch.inference_mode()
def collect_logits(model, data: tuple[np.ndarray, np.ndarray], device, batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Logits for preloaded uint8 NHWC images; normalisation matches app.classify.preprocess."""
    model.eval()
    x_all, y = data
    out = []
    for i in range(0, len(x_all), batch):
        x = normalize_uint8(torch.from_numpy(x_all[i : i + batch]).permute(0, 3, 1, 2), device)
        out.append(model(x).float().cpu())
    return torch.cat(out).numpy(), y


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017): minimise val NLL over a single scalar T."""
    lg = torch.from_numpy(logits).double()
    yt = torch.from_numpy(y).long()
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.3, 10.0))


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macro_f1(y, pred, k) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=DEFAULT_UNCERTAINTY_THRESHOLD)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / MODEL_FILENAME)
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after N steps")
    args = ap.parse_args()

    seed_everything(args.seed)
    labels = get_labels()
    k = len(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    splits = {s: list_folder(args.data / s, labels) for s in ("train", "val", "test")}
    for s, items in splits.items():
        if not items:
            raise SystemExit(f"empty split {s!r} in {args.data} - run scripts/prepare_dataset.py")
        print(f"{s:5s}: {len(items)} images")
    train_ds = LeafFolderDataset(splits["train"], train=True)
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, drop_last=True, generator=g,
                          num_workers=args.workers,
                          persistent_workers=args.workers > 0, worker_init_fn=_worker_init)
    t0 = time.time()
    val_data = load_eval_arrays(splits["val"])
    test_data = load_eval_arrays(splits["test"])
    print(f"preloaded val/test in {time.time() - t0:.0f}s", flush=True)

    # Class-balanced loss weights (sqrt inverse frequency) for the small classes.
    counts = np.bincount(train_ds.targets, minlength=k).astype(np.float64)
    weights = np.sqrt(counts.sum() / (k * np.maximum(counts, 1)))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device),
                                    label_smoothing=0.05)

    model = build_model(k, pretrained=True).to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0}], lr=args.lr)
    steps = args.epochs * len(train_dl)
    warm = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    )

    history = []
    best = (-1.0, 0)
    best_state = None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, correct, loss_sum = time.time(), 0, 0, 0.0
        for x, y in train_dl:
            x, y = normalize_uint8(x, device), y.to(device, non_blocking=True)
            out = model(x)  # fp32: on this Turing GPU AMP was slower for MobileNetV3's depthwise convs
            loss = criterion(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
            if args.max_steps and total >= args.max_steps * args.batch:
                break
        vl, vy = collect_logits(model, val_data, device)
        vpred = vl.argmax(1)
        v_loss = float(nn.functional.cross_entropy(torch.from_numpy(vl), torch.from_numpy(vy).long()))
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_acc": correct / total,
            "val_loss": v_loss,
            "val_acc": float((vpred == vy).mean()),
            "val_macro_f1": macro_f1(vy, vpred, k),
            "lr": opt.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(f"epoch {epoch:2d}/{args.epochs}  loss {row['train_loss']:.4f}  train {row['train_acc']:.4f}  "
              f"val {row['val_acc']:.4f}  vF1 {row['val_macro_f1']:.4f}  ({row['seconds']:.0f}s)", flush=True)
        if row["val_macro_f1"] > best[0]:
            best = (row["val_macro_f1"], epoch)
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    vl, vy = collect_logits(model, val_data, device)
    temperature = fit_temperature(vl, vy)
    tl, ty = collect_logits(model, test_data, device)
    print(f"best epoch {best[1]} (val macro-F1 {best[0]:.4f}); temperature {temperature:.3f}")

    split_info = json.loads((args.reports / "dataset_split.json").read_text()) if (args.reports / "dataset_split.json").exists() else {}
    version = f"mnv3s-{k}c-{datetime.now():%Y%m%d-%H%M}"
    val_m = write_reports("val", vy, softmax(vl, temperature), labels, args.threshold, args.reports,
                          f"Validation - {version}")
    test_m = write_reports("test", ty, softmax(tl, temperature), labels, args.threshold, args.reports,
                           f"Test split - {version}", extra={"model_version": version})

    meta = {
        "version": version,
        "class_keys": labels.keys,
        "img_size": IMG_SIZE,
        "resize_to": RESIZE_TO,
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "temperature": temperature,
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": labels.dataset,
        "split_manifest_sha1": split_info.get("split_manifest_sha1"),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best[1],
        "hyperparams": {"batch": args.batch, "lr": args.lr, "wd": args.wd, "label_smoothing": 0.05},
        "val": {"accuracy": val_m["accuracy"], "macro_f1": val_m["macro_f1"]},
        "test": {"accuracy": test_m["accuracy"], "macro_f1": test_m["macro_f1"]},
        "torch": str(torch.__version__),
        "train_minutes": round((time.time() - t_start) / 60, 1),
    }
    save_checkpoint(args.out, model, meta)
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    with (args.reports / "training_history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    plot_history(history, args.reports / "training_curves.png")
    print(f"saved {args.out} ({version}); test acc {test_m['accuracy']:.4f}, macro-F1 {test_m['macro_f1']:.4f}")
    return 0


def plot_history(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=120)
    a.plot(e, [h["train_loss"] for h in history], label="train")
    a.plot(e, [h["val_loss"] for h in history], label="val")
    a.set_title("Loss")
    a.set_xlabel("epoch")
    a.legend()
    b.plot(e, [h["train_acc"] for h in history], label="train acc")
    b.plot(e, [h["val_acc"] for h in history], label="val acc")
    b.plot(e, [h["val_macro_f1"] for h in history], label="val macro-F1", linestyle="--")
    b.set_title("Accuracy")
    b.set_xlabel("epoch")
    b.legend()
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
