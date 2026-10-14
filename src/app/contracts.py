"""Shared data types and module interfaces, plus mock implementations.

Conventions
-----------
* Every image passed between modules is an RGB ``uint8`` numpy array of shape
  ``(H, W, 3)``. OpenCV's BGR is converted exactly once, in ``app.capture``.
* Masks are ``bool`` arrays of shape ``(H, W)`` matching the input image.
* Class identity is always a ``class_key`` from labels.json (or ``"uncertain"``);
  display names are resolved through ``app.labels``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Sequence, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

from app.labels import UNCERTAIN_KEY, UNCERTAIN_NAME, Labels, get_labels

RGBImage = NDArray[np.uint8]
Mask = NDArray[np.bool_]

DEFAULT_UNCERTAINTY_THRESHOLD = 0.60


def format_confidence(p: float) -> str:
    """Never display a model probability as a certainty: 0.9996 -> '>99%', not '100%'."""
    if p >= 0.995:
        return ">99%"
    if 0 < p < 0.005:
        return "<1%"
    return f"{p:.0%}"


class InvalidImageError(ValueError):
    """Raised when an image is not a usable RGB uint8 array."""


def validate_rgb(image: object) -> RGBImage:
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"expected numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise InvalidImageError(f"expected uint8 image, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(f"expected HxWx3 RGB image, got shape {image.shape}")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise InvalidImageError(f"image too small: {image.shape[1]}x{image.shape[0]}")
    return image


# --------------------------------------------------------------------------- classify


@dataclass(frozen=True)
class Prediction:
    class_id: int
    class_key: str
    name: str
    probability: float


@dataclass(frozen=True)
class ClassifierResult:
    """Output of ``classify.predict``.

    When ``is_uncertain`` is True, ``class_id`` is None and ``disease_name`` is
    "Not sure" -- the best guesses are still available in ``top_k``.
    """

    class_id: int | None
    disease_name: str
    confidence: float
    is_uncertain: bool
    class_key: str = UNCERTAIN_KEY
    top_k: tuple[Prediction, ...] = ()

    @classmethod
    def from_probabilities(
        cls,
        probs: Sequence[float] | NDArray[np.floating],
        labels: Labels,
        threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
        k: int = 3,
    ) -> "ClassifierResult":
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != (len(labels),):
            raise ValueError(f"expected {len(labels)} probabilities, got shape {p.shape}")
        order = np.argsort(-p)[:k]
        top = tuple(
            Prediction(int(i), labels.by_id(int(i)).key, labels.by_id(int(i)).display_name, float(p[i]))
            for i in order
        )
        best = top[0]
        if best.probability < threshold:
            return cls(None, UNCERTAIN_NAME, best.probability, True, UNCERTAIN_KEY, top)
        return cls(best.class_id, best.name, best.probability, False, best.class_key, top)


@runtime_checkable
class Classifier(Protocol):
    model_version: str
    threshold: float

    def predict(self, image: RGBImage) -> ClassifierResult: ...


# --------------------------------------------------------------------------- vision


@dataclass(frozen=True)
class LeafResult:
    mask: Mask
    annotated_image: RGBImage
    found: bool
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h in input coordinates
    area_fraction: float = 0.0
    message: str = ""
    contour: NDArray[np.int32] | None = None  # (N, 1, 2) outline in input coordinates
    discoloration: Mask | None = None  # optional: off-colour pixels inside the leaf
    discoloration_fraction: float = 0.0  # share of leaf area that is discoloured


@runtime_checkable
class LeafDetector(Protocol):
    def detect_leaf(self, image: RGBImage) -> LeafResult: ...


@runtime_checkable
class OverlayRenderer(Protocol):
    def render(self, image: RGBImage, leaf: LeafResult, result: ClassifierResult | None) -> RGBImage: ...


# --------------------------------------------------------------------------- guidance


@dataclass(frozen=True)
class Source:
    title: str
    url: str
    publisher: str = ""


@dataclass(frozen=True)
class Guidance:
    class_key: str
    title: str
    summary: str
    severity: str = "info"  # "none" | "low" | "moderate" | "high" | "info"
    symptoms: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    prevention: tuple[str, ...] = ()
    sources: tuple[Source, ...] = ()


@runtime_checkable
class GuidanceProvider(Protocol):
    def get(self, class_key: str) -> Guidance | None: ...


# --------------------------------------------------------------------------- analysis


@dataclass(frozen=True)
class AnalysisResult:
    image: RGBImage
    leaf: LeafResult
    classification: ClassifierResult | None  # None when no leaf was found
    overlay_image: RGBImage
    guidance: Guidance | None
    model_version: str
    elapsed_ms: float


# --------------------------------------------------------------------------- storage


@dataclass
class ScanRecord:
    image: RGBImage
    plant_id: int
    timestamp: datetime
    prediction: str  # class key from labels.json, or "uncertain"
    confidence: float
    model_version: str
    class_id: int | None = None
    is_uncertain: bool = False
    overlay_image: RGBImage | None = None

    @classmethod
    def from_analysis(cls, result: AnalysisResult, plant_id: int, timestamp: datetime | None = None) -> "ScanRecord":
        c = result.classification
        return cls(
            image=result.image,
            plant_id=plant_id,
            timestamp=timestamp or datetime.now().astimezone(),
            prediction=c.class_key if c else UNCERTAIN_KEY,
            confidence=c.confidence if c else 0.0,
            model_version=result.model_version,
            class_id=c.class_id if c else None,
            is_uncertain=c.is_uncertain if c else True,
            overlay_image=result.overlay_image,
        )


@dataclass(frozen=True)
class Plant:
    id: int
    name: str
    created_at: datetime
    scan_count: int = 0
    last_scan: datetime | None = None


@dataclass(frozen=True)
class StoredScan:
    id: int
    plant_id: int
    plant_name: str
    timestamp: datetime
    prediction: str
    class_id: int | None
    confidence: float
    is_uncertain: bool
    model_version: str
    image_path: str  # absolute
    overlay_path: str | None


@runtime_checkable
class ScanStore(Protocol):
    def save_scan(self, record: ScanRecord) -> int: ...
    def get_or_create_plant(self, name: str) -> Plant: ...
    def suggest_plant_name(self) -> str: ...
    def list_plants(self) -> list[Plant]: ...
    def list_scans(self, plant_id: int | None = None, limit: int | None = None) -> list[StoredScan]: ...
    def delete_scan(self, scan_id: int) -> None: ...
    def scans_per_day(self, plant_id: int | None = None) -> list[tuple[str, int]]: ...
    def prediction_counts(self, plant_id: int | None = None) -> dict[str, int]: ...


# =========================================================================== mocks


def _image_seed(image: RGBImage) -> int:
    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    return int.from_bytes(hashlib.sha1(small.tobytes()).digest()[:4], "little")


@dataclass
class MockClassifier:
    """Deterministic fake: the same image always yields the same prediction."""

    threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD
    labels: Labels = field(default_factory=get_labels)
    model_version: str = "mock-0"

    def predict(self, image: RGBImage) -> ClassifierResult:
        validate_rgb(image)
        rng = np.random.default_rng(_image_seed(image))
        logits = rng.normal(0, 1, len(self.labels))
        logits[rng.integers(len(self.labels))] += rng.uniform(0.5, 4.0)
        probs = np.exp(logits) / np.exp(logits).sum()
        return ClassifierResult.from_probabilities(probs, self.labels, self.threshold)


class MockLeafDetector:
    """Pretends the leaf is a centred ellipse covering ~half the frame."""

    def detect_leaf(self, image: RGBImage) -> LeafResult:
        validate_rgb(image)
        h, w = image.shape[:2]
        mask8 = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask8, (w // 2, h // 2), (int(w * 0.3), int(h * 0.38)), 0, 0, 360, 255, -1)
        contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = image.copy()
        cv2.drawContours(annotated, contours, -1, (62, 207, 142), max(2, w // 300), cv2.LINE_AA)
        mask = mask8 > 0
        return LeafResult(mask, annotated, True, cv2.boundingRect(contours[0]), float(mask.mean()), "mock leaf")


class MockOverlayRenderer:
    def render(self, image: RGBImage, leaf: LeafResult, result: ClassifierResult | None) -> RGBImage:
        out = leaf.annotated_image.copy()
        if not leaf.found or result is None:
            text = "No leaf detected"
        else:
            text = f"{result.disease_name} {format_confidence(result.confidence)}"
        x, y = (leaf.bbox[0], leaf.bbox[1]) if leaf.bbox else (10, 10)
        scale = max(0.5, image.shape[1] / 1400)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        y = max(y, th + base + 8)
        cv2.rectangle(out, (x, y - th - base - 8), (x + tw + 12, y), (20, 26, 23), -1)
        cv2.putText(out, text, (x + 6, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 2, cv2.LINE_AA)
        return out


class MockGuidanceProvider:
    def __init__(self, labels: Labels | None = None) -> None:
        self.labels = labels or get_labels()

    def get(self, class_key: str) -> Guidance | None:
        info = self.labels.by_key(class_key)
        if info is None:
            return None
        return Guidance(
            class_key=class_key,
            title=info.display_name,
            summary="Placeholder guidance (mock). Real, sourced content arrives in Phase 4.",
            severity="none" if info.healthy else "moderate",
            actions=("Re-scan in good light against a plain background.",),
            sources=(Source("Example source", "https://example.org"),),
        )
