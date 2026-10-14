"""ONNX Runtime implementation of the Classifier contract (pre-trained model, no PyTorch needed)."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from app.classify.preprocess import preprocess
from app.contracts import DEFAULT_UNCERTAINTY_THRESHOLD, ClassifierResult, RGBImage, validate_rgb
from app.labels import Labels, get_labels


class ModelLoadError(RuntimeError):
    pass


class OnnxClassifier:
    """Loads ``<name>.onnx`` plus its ``<name>.json`` (preprocessing + version)."""

    def __init__(
        self,
        model_path: Path | str,
        threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
        labels: Labels | None = None,
        plant_threshold: float | None = None,
    ) -> None:
        model_path = Path(model_path)
        self.labels = labels or get_labels()
        self.threshold = threshold
        self.plant_threshold = threshold if plant_threshold is None else plant_threshold
        meta_path = model_path.with_suffix(".json")
        if not model_path.exists() or not meta_path.exists():
            raise ModelLoadError(f"model file not found: {model_path}")
        try:
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelLoadError(f"cannot read {meta_path.name}: {exc}") from exc
        if self.meta.get("num_classes") != len(self.labels):
            raise ModelLoadError("model classes do not match labels.json")
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001 - corrupt file, missing runtime...
            raise ModelLoadError(f"could not load {model_path.name}: {exc}") from exc
        self.input_name = self.meta.get("input") or self.session.get_inputs()[0].name
        self.model_version: str = self.meta["version"]
        self._mean = np.asarray(self.meta["mean"], np.float32)
        self._std = np.asarray(self.meta["std"], np.float32)
        self._lock = threading.Lock()
        self.predict(np.zeros((224, 224, 3), np.uint8))  # warm-up so the first real scan is fast

    def probabilities_batch(self, images: list[RGBImage]) -> np.ndarray:
        for im in images:
            validate_rgb(im)
        x = np.stack([preprocess(im, self._mean, self._std, self.meta["resize_shortest"], self.meta["crop"])
                      for im in images])
        with self._lock:
            logits = self.session.run(None, {self.input_name: x})[0].astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def probabilities(self, image: RGBImage) -> np.ndarray:
        return self.probabilities_batch([image])[0]

    def _result(self, probs: np.ndarray) -> ClassifierResult:
        return ClassifierResult.from_probabilities(probs, self.labels, self.threshold, plant_threshold=self.plant_threshold)

    def predict(self, image: RGBImage) -> ClassifierResult:
        return self._result(self.probabilities(image))

    def predict_views(self, views: list[RGBImage]) -> ClassifierResult:
        """Average the probabilities of several views of the same leaf (test-time augmentation)."""
        return self._result(self.probabilities_batch(views).mean(axis=0))
