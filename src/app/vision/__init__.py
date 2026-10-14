"""Leaf detection and overlays. Phase 2: still mock (classical CV arrives in Phase 3)."""
from __future__ import annotations

from app.contracts import LeafDetector, LeafResult, MockLeafDetector, MockOverlayRenderer, OverlayRenderer, RGBImage


def build_detector() -> LeafDetector:
    return MockLeafDetector()


def build_overlay(highlight_discoloration: bool = True) -> OverlayRenderer:
    return MockOverlayRenderer()


_detector: LeafDetector | None = None


def detect_leaf(rgb_image: RGBImage) -> LeafResult:
    global _detector
    if _detector is None:
        _detector = build_detector()
    return _detector.detect_leaf(rgb_image)
