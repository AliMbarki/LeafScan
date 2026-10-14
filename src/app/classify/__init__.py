"""Leaf disease classification with a pre-trained ONNX model (MobileNetV2, PlantVillage)."""
from __future__ import annotations

from pathlib import Path

from app.contracts import DEFAULT_UNCERTAINTY_THRESHOLD, Classifier, ClassifierResult, RGBImage

MODEL_FILENAME = "plant_disease.onnx"

_default: Classifier | None = None


def default_model_path() -> Path:
    from app.config import Settings

    return Settings().models_dir / MODEL_FILENAME


def load_classifier(
    model_path: Path | None = None,
    threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    plant_threshold: float | None = None,
) -> Classifier:
    """Load the model. Raises ModelLoadError if it is missing or invalid."""
    from app.classify.predictor import OnnxClassifier

    return OnnxClassifier(model_path or default_model_path(), threshold, plant_threshold=plant_threshold)


def predict(rgb_image: RGBImage) -> ClassifierResult:
    global _default
    if _default is None:
        _default = load_classifier()
    return _default.predict(rgb_image)
