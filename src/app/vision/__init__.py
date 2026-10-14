"""Leaf detection (classical OpenCV) and result overlays."""
from __future__ import annotations

from app.contracts import LeafDetector, LeafResult, OverlayRenderer, RGBImage
from app.vision.leaf import ClassicalLeafDetector, LeafDetectorConfig
from app.vision.overlay import LeafOverlayRenderer


def build_detector() -> LeafDetector:
    return ClassicalLeafDetector()


def build_overlay(highlight_discoloration: bool = True) -> OverlayRenderer:
    return LeafOverlayRenderer(highlight_discoloration=highlight_discoloration)


_detector: LeafDetector | None = None


def detect_leaf(rgb_image: RGBImage) -> LeafResult:
    global _detector
    if _detector is None:
        _detector = build_detector()
    return _detector.detect_leaf(rgb_image)


__all__ = ["ClassicalLeafDetector", "LeafDetectorConfig", "LeafOverlayRenderer", "build_detector", "build_overlay",
           "detect_leaf"]
