"""Leaf disease classification. Phase 1: mock classifier."""
from __future__ import annotations

from app.contracts import Classifier, ClassifierResult, MockClassifier, RGBImage

_default: Classifier | None = None


def load_classifier(threshold: float | None = None) -> Classifier:
    return MockClassifier() if threshold is None else MockClassifier(threshold=threshold)


def predict(rgb_image: RGBImage) -> ClassifierResult:
    global _default
    if _default is None:
        _default = load_classifier()
    return _default.predict(rgb_image)
