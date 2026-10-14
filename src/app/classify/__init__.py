"""Leaf disease classification (PyTorch MobileNetV3-Small).

``predict(rgb_image)`` is the module-level convenience API; the app builds its own
classifier instance on the analysis thread via ``load_classifier``.
"""
from __future__ import annotations

from pathlib import Path

from app.contracts import DEFAULT_UNCERTAINTY_THRESHOLD, Classifier, ClassifierResult, RGBImage

MODEL_FILENAME = "leafscan_mnv3s.pt"

_default: Classifier | None = None


def default_model_path() -> Path:
    from app.config import Settings

    return Settings().models_dir / MODEL_FILENAME


def load_classifier(model_path: Path | None = None, threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD) -> Classifier:
    """Load the trained model. Raises ModelLoadError (or ImportError if torch is missing)."""
    from app.classify.predictor import TorchClassifier

    return TorchClassifier(model_path or default_model_path(), threshold)


def predict(rgb_image: RGBImage) -> ClassifierResult:
    global _default
    if _default is None:
        _default = load_classifier()
    return _default.predict(rgb_image)
