"""Metrics + report files (confusion matrix, per-class metrics). Used by the training/eval scripts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from app.labels import Labels  # noqa: E402


def compute_metrics(y_true, probs: np.ndarray, labels: Labels, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(1)
    conf = probs.max(1)
    ids = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    confident = conf >= threshold
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "uncertainty_threshold": threshold,
        "uncertain_rate": float(1 - confident.mean()) if len(conf) else 0.0,
        "accuracy_when_confident": float((y_pred[confident] == y_true[confident]).mean()) if confident.any() else None,
        "per_class": {
            labels.by_id(i).key: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in ids
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def plot_confusion(cm: np.ndarray, labels: Labels, path: Path, title: str) -> None:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = [c.display_name for c in labels]
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=130)
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.55 else "#1b2a22")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_reports(name: str, y_true, probs: np.ndarray, labels: Labels, threshold: float, out_dir: Path,
                  title: str, extra: dict | None = None) -> dict:
    m = compute_metrics(y_true, probs, labels, threshold)
    if extra:
        m.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(m, indent=2))
    ids = list(range(len(labels)))
    report = classification_report(
        y_true, probs.argmax(1), labels=ids, target_names=[c.key for c in labels], digits=4, zero_division=0
    )
    (out_dir / f"{name}_classification_report.txt").write_text(
        f"{title}\n\n{report}\nuncertain (<{threshold:.2f}) rate: {m['uncertain_rate']:.2%}\n"
    )
    plot_confusion(np.array(m["confusion_matrix"]), labels, out_dir / f"{name}_confusion_matrix.png",
                   f"{title}  (acc {m['accuracy']:.2%}, macro-F1 {m['macro_f1']:.3f}, n={m['n']})")
    return m
