"""Torch-backed implementation of the Classifier contract."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch

from app.classify import MODEL_FILENAME  # noqa: F401  (re-exported)
from app.classify.model import ModelLoadError, load_checkpoint
from app.classify.preprocess import preprocess
from app.contracts import DEFAULT_UNCERTAINTY_THRESHOLD, ClassifierResult, RGBImage, validate_rgb
from app.labels import Labels, get_labels



class TorchClassifier:
    def __init__(
        self,
        model_path: Path | str,
        threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
        labels: Labels | None = None,
        device: str | None = None,
    ) -> None:
        self.labels = labels or get_labels()
        self.threshold = threshold
        self.device = torch.device(device or os.environ.get("LEAFSCAN_DEVICE", "cpu"))
        self.model, self.meta = load_checkpoint(Path(model_path), self.device)
        if list(self.meta["class_keys"]) != self.labels.keys:
            raise ModelLoadError(
                "model classes do not match labels.json - retrain or restore the matching labels.json"
            )
        self.model_version: str = self.meta["version"]
        self.temperature = float(self.meta.get("temperature", 1.0))
        self._lock = threading.Lock()
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.predict(np.zeros((224, 224, 3), np.uint8))  # warm-up so the first real scan is fast

    @torch.inference_mode()
    def probabilities(self, image: RGBImage) -> np.ndarray:
        validate_rgb(image)
        x = torch.from_numpy(preprocess(image)).unsqueeze(0).to(self.device)
        with self._lock:
            logits = self.model(x)[0]
        return torch.softmax(logits / self.temperature, dim=0).cpu().numpy()

    def predict(self, image: RGBImage) -> ClassifierResult:
        return ClassifierResult.from_probabilities(self.probabilities(image), self.labels, self.threshold)
