"""capture -> leaf detection -> classification -> overlay (saving is a separate user action)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.contracts import (
    AnalysisResult,
    Classifier,
    GuidanceProvider,
    LeafDetector,
    LeafResult,
    OverlayRenderer,
    RGBImage,
    validate_rgb,
)


def crop_to_leaf(image: RGBImage, leaf: LeafResult, pad: float = 0.12) -> RGBImage:
    """Square crop around the leaf bbox so the classifier sees a framing similar to training data."""
    if not leaf.found or leaf.bbox is None:
        return image
    h, w = image.shape[:2]
    x, y, bw, bh = leaf.bbox
    side = int(max(bw, bh) * (1 + 2 * pad))
    cx, cy = x + bw // 2, y + bh // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(w, x0 + side), min(h, y0 + side)
    crop = image[y0:y1, x0:x1]
    return np.ascontiguousarray(crop) if crop.shape[0] >= 32 and crop.shape[1] >= 32 else image


@dataclass
class AnalysisPipeline:
    detector: LeafDetector
    classifier: Classifier
    overlay: OverlayRenderer
    guidance: GuidanceProvider
    require_leaf: bool = True

    @property
    def model_version(self) -> str:
        return self.classifier.model_version

    def analyze(self, image: RGBImage) -> AnalysisResult:
        t0 = time.perf_counter()
        validate_rgb(image)
        leaf = self.detector.detect_leaf(image)
        classification = None
        if leaf.found or not self.require_leaf:
            crop = crop_to_leaf(image, leaf)
            views = [crop, np.ascontiguousarray(crop[:, ::-1])]
            if leaf.method != "isolated":
                views.append(image)  # outline is less reliable on busy scenes: also look at the whole frame
            predict_views = getattr(self.classifier, "predict_views", None)
            classification = predict_views(views) if predict_views else self.classifier.predict(crop)
        overlay = self.overlay.render(image, leaf, classification)
        guidance = None
        if classification is not None and not classification.is_uncertain:
            guidance = self.guidance.get(classification.class_key)
        return AnalysisResult(
            image=image,
            leaf=leaf,
            classification=classification,
            overlay_image=overlay,
            guidance=guidance,
            model_version=self.model_version,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )


def build_mock_pipeline(threshold: float | None = None) -> AnalysisPipeline:
    from app.contracts import MockClassifier, MockGuidanceProvider, MockLeafDetector, MockOverlayRenderer

    clf = MockClassifier() if threshold is None else MockClassifier(threshold=threshold)
    return AnalysisPipeline(MockLeafDetector(), clf, MockOverlayRenderer(), MockGuidanceProvider())
