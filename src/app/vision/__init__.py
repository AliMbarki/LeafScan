"""Leaf detection and overlays. Phase 1: mock detector."""
from __future__ import annotations

from app.contracts import LeafResult, MockLeafDetector, RGBImage

_detector = MockLeafDetector()


def detect_leaf(rgb_image: RGBImage) -> LeafResult:
    return _detector.detect_leaf(rgb_image)
