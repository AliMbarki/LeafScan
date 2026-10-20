"""
LeafScan - Member 4 - Processing pipeline and decision logic
Main technology: Python / NumPy / Qt threads

Reading copy: the code below is collected from the real project files listed here.
Edit the originals (paths shown in each section), not this file. Files:
  - src/app/pipeline.py
  - src/app/contracts.py
  - src/app/labels.py
  - src/app/labels.json
  - src/app/ui/worker.py
  - src/app/config.py
"""


# ================================================================================================
# FILE: src/app/pipeline.py
# ================================================================================================

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

# ================================================================================================
# FILE: src/app/contracts.py
# ================================================================================================

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

from app.labels import UNCERTAIN_KEY, UNCERTAIN_NAME, UNKNOWN_GROUP, Labels, get_labels

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

    Two decisions are made separately:

    * **plant** - probabilities of all classes of a plant are summed. If the best plant is
      below the threshold, or is the unknown-plant group, ``plant`` is None.
    * **condition** - the best class of that plant. If it is below the threshold the
      condition is "Not sure" (``class_id`` None, ``is_uncertain`` True) but ``plant`` can
      still be known ("Tomato - not sure what's wrong").

    The best guesses are always available in ``top_k``. A label is never forced.
    """

    class_id: int | None
    disease_name: str
    confidence: float
    is_uncertain: bool
    class_key: str = UNCERTAIN_KEY
    top_k: tuple[Prediction, ...] = ()
    plant: str | None = None
    plant_confidence: float = 0.0
    unknown_plant: bool = False  # confidently "a plant LeafScan does not know"

    @classmethod
    def from_probabilities(
        cls,
        probs: Sequence[float] | NDArray[np.floating],
        labels: Labels,
        threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
        k: int = 3,
        plant_threshold: float | None = None,
    ) -> "ClassifierResult":
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != (len(labels),):
            raise ValueError(f"expected {len(labels)} probabilities, got shape {p.shape}")
        plant_threshold = threshold if plant_threshold is None else plant_threshold
        order = np.argsort(-p)[:k]
        top = tuple(
            Prediction(int(i), labels.by_id(int(i)).key, labels.by_id(int(i)).display_name, float(p[i]))
            for i in order
        )
        groups = labels.plant_groups
        plant_p = {name: float(p[ids].sum()) for name, ids in groups.items()}
        best_plant = max(plant_p, key=plant_p.get)
        pconf = plant_p[best_plant]
        if best_plant == UNKNOWN_GROUP and pconf >= plant_threshold:
            uid = groups[UNKNOWN_GROUP][0]
            return cls(None, labels.unknown_name, pconf, True, labels.by_id(uid).key, top, None, pconf, True)
        if best_plant == UNKNOWN_GROUP or pconf < plant_threshold:
            return cls(None, UNCERTAIN_NAME, float(p.max()), True, UNCERTAIN_KEY, top, None, pconf)
        ids = groups[best_plant]
        best = max(ids, key=lambda i: p[i])
        info = labels.by_id(best)
        if p[best] < threshold:
            return cls(None, UNCERTAIN_NAME, float(p[best]), True, UNCERTAIN_KEY, top, best_plant, pconf)
        return cls(best, info.display_name, float(p[best]), False, info.key, top, best_plant, pconf)


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
    method: str = "isolated"  # "isolated" (plain background) | "center" (busy background) | "area" (main green area)


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
    species: str | None = None  # predicted plant species (known even when the condition is "not sure")

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
            species=c.plant if c else None,
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
    species: str | None = None


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

    def predict_views(self, views: list[RGBImage]) -> ClassifierResult:
        return self.predict(views[0])

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

# ================================================================================================
# FILE: src/app/labels.py
# ================================================================================================

"""Loader for labels.json, the single source of truth for plants, class IDs and names.

No other module may hardcode class or plant names; everything goes through `Labels`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LABELS_PATH = Path(__file__).with_name("labels.json")

# Stored in place of a class key when the classifier is not confident enough.
UNCERTAIN_KEY = "uncertain"
UNCERTAIN_NAME = "Not sure"
UNKNOWN_GROUP = "__unknown__"  # internal plant-group key of the "some other plant" class


@dataclass(frozen=True)
class PlantInfo:
    name: str
    latin: str = ""


@dataclass(frozen=True)
class LabelInfo:
    id: int
    key: str
    plant: str
    condition: str
    healthy: bool
    condition_id: str = ""
    unknown: bool = False  # the "some other plant" class

    @property
    def display_name(self) -> str:
        """Human-readable name, ASCII-only so OpenCV can draw it."""
        return self.condition if self.unknown else f"{self.plant} - {self.condition}"


class Labels:
    def __init__(self, classes: list[LabelInfo], plants: list[PlantInfo] | None = None, dataset: str = "") -> None:
        ids = [c.id for c in classes]
        if ids != list(range(len(classes))):
            raise ValueError(f"labels.json class ids must be 0..N-1 in order, got {ids}")
        keys = [c.key for c in classes]
        if len(set(keys)) != len(keys) or UNCERTAIN_KEY in keys:
            raise ValueError("labels.json keys must be unique and must not be 'uncertain'")
        self.classes = tuple(classes)
        self.dataset = dataset
        self._by_key = {c.key: c for c in classes}
        known = sorted({c.plant for c in classes if not c.unknown})
        given = {p.name: p for p in plants or []}
        missing = [p for p in known if p not in given]
        self.plant_info = {p: given.get(p, PlantInfo(p)) for p in known}
        if plants and missing:
            raise ValueError(f"labels.json plants list is missing {missing}")
        # Plant groups: index lists used to sum class probabilities per plant.
        self._groups: dict[str, list[int]] = {}
        for c in classes:
            self._groups.setdefault(UNKNOWN_GROUP if c.unknown else c.plant, []).append(c.id)
        unknown = [c for c in classes if c.unknown]
        self.unknown_name = unknown[0].display_name if unknown else ""

    @classmethod
    def load(cls, path: Path | str = LABELS_PATH) -> "Labels":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = LabelInfo.__dataclass_fields__
        classes = [LabelInfo(**{k: v for k, v in c.items() if k in fields}) for c in raw["classes"]]
        plants = [PlantInfo(**p) for p in raw.get("plants", [])]
        return cls(classes, plants, raw.get("dataset", ""))

    def __len__(self) -> int:
        return len(self.classes)

    def __iter__(self):
        return iter(self.classes)

    def by_id(self, class_id: int) -> LabelInfo:
        return self.classes[class_id]

    def by_key(self, key: str) -> LabelInfo | None:
        return self._by_key.get(key)

    def display_name(self, key: str, plant: str | None = None) -> str:
        info = self.by_key(key)
        if info is not None:
            return info.display_name
        if key == UNCERTAIN_KEY:
            return f"{plant} - {UNCERTAIN_NAME}" if plant else UNCERTAIN_NAME
        return key

    @property
    def keys(self) -> list[str]:
        return [c.key for c in self.classes]

    @property
    def plants(self) -> list[str]:
        """Known plant names (sorted), excluding the unknown-plant group."""
        return list(self.plant_info)

    @property
    def plant_groups(self) -> dict[str, list[int]]:
        """Plant name (or UNKNOWN_GROUP for the unknown-plant class) -> class ids."""
        return self._groups

    def group_name(self, group: str) -> str:
        return self.unknown_name if group == UNKNOWN_GROUP else group

    def classes_of(self, plant: str) -> list[LabelInfo]:
        return [self.classes[i] for i in self._groups.get(plant, [])]


@lru_cache(maxsize=1)
def get_labels() -> Labels:
    return Labels.load()

# ================================================================================================
# FILE: src/app/labels.json
# ================================================================================================

# (JSON data file)
'''
{
 "schema": 2,
 "dataset": "PlantVillage (pre-trained model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification)",
 "note": "Single source of truth for plants and classes. Class ids are the model's output indices.",
 "plants": [
  {
   "name": "Apple",
   "latin": "Malus domestica"
  },
  {
   "name": "Blueberry",
   "latin": "Vaccinium spp."
  },
  {
   "name": "Cherry",
   "latin": "Prunus avium / P. cerasus"
  },
  {
   "name": "Citrus",
   "latin": "Citrus sinensis (orange)"
  },
  {
   "name": "Corn",
   "latin": "Zea mays"
  },
  {
   "name": "Grape",
   "latin": "Vitis vinifera"
  },
  {
   "name": "Peach",
   "latin": "Prunus persica"
  },
  {
   "name": "Pepper",
   "latin": "Capsicum annuum"
  },
  {
   "name": "Potato",
   "latin": "Solanum tuberosum"
  },
  {
   "name": "Raspberry",
   "latin": "Rubus idaeus"
  },
  {
   "name": "Soybean",
   "latin": "Glycine max"
  },
  {
   "name": "Squash",
   "latin": "Cucurbita spp."
  },
  {
   "name": "Strawberry",
   "latin": "Fragaria x ananassa"
  },
  {
   "name": "Tomato",
   "latin": "Solanum lycopersicum"
  }
 ],
 "classes": [
  {
   "id": 0,
   "key": "apple_scab",
   "plant": "Apple",
   "condition": "Apple scab",
   "condition_id": "apple_scab",
   "healthy": false
  },
  {
   "id": 1,
   "key": "apple_black_rot",
   "plant": "Apple",
   "condition": "Black rot",
   "condition_id": "black_rot",
   "healthy": false
  },
  {
   "id": 2,
   "key": "apple_cedar_apple_rust",
   "plant": "Apple",
   "condition": "Cedar apple rust",
   "condition_id": "rust",
   "healthy": false
  },
  {
   "id": 3,
   "key": "apple_healthy",
   "plant": "Apple",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 4,
   "key": "blueberry_healthy",
   "plant": "Blueberry",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 5,
   "key": "cherry_powdery_mildew",
   "plant": "Cherry",
   "condition": "Powdery mildew",
   "condition_id": "powdery_mildew",
   "healthy": false
  },
  {
   "id": 6,
   "key": "cherry_healthy",
   "plant": "Cherry",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 7,
   "key": "corn_gray_leaf_spot",
   "plant": "Corn",
   "condition": "Gray leaf spot",
   "condition_id": "gray_leaf_spot",
   "healthy": false
  },
  {
   "id": 8,
   "key": "corn_common_rust",
   "plant": "Corn",
   "condition": "Common rust",
   "condition_id": "rust",
   "healthy": false
  },
  {
   "id": 9,
   "key": "corn_northern_leaf_blight",
   "plant": "Corn",
   "condition": "Northern leaf blight",
   "condition_id": "leaf_blight",
   "healthy": false
  },
  {
   "id": 10,
   "key": "corn_healthy",
   "plant": "Corn",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 11,
   "key": "grape_black_rot",
   "plant": "Grape",
   "condition": "Black rot",
   "condition_id": "black_rot",
   "healthy": false
  },
  {
   "id": 12,
   "key": "grape_esca",
   "plant": "Grape",
   "condition": "Esca (black measles)",
   "condition_id": "esca",
   "healthy": false
  },
  {
   "id": 13,
   "key": "grape_leaf_blight",
   "plant": "Grape",
   "condition": "Leaf blight (Isariopsis)",
   "condition_id": "leaf_blight",
   "healthy": false
  },
  {
   "id": 14,
   "key": "grape_healthy",
   "plant": "Grape",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 15,
   "key": "citrus_greening",
   "plant": "Citrus",
   "condition": "Citrus greening (HLB)",
   "condition_id": "citrus_greening",
   "healthy": false
  },
  {
   "id": 16,
   "key": "peach_bacterial_spot",
   "plant": "Peach",
   "condition": "Bacterial spot",
   "condition_id": "bacterial_spot",
   "healthy": false
  },
  {
   "id": 17,
   "key": "peach_healthy",
   "plant": "Peach",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 18,
   "key": "pepper_bacterial_spot",
   "plant": "Pepper",
   "condition": "Bacterial spot",
   "condition_id": "bacterial_spot",
   "healthy": false
  },
  {
   "id": 19,
   "key": "pepper_healthy",
   "plant": "Pepper",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 20,
   "key": "potato_early_blight",
   "plant": "Potato",
   "condition": "Early blight",
   "condition_id": "early_blight",
   "healthy": false
  },
  {
   "id": 21,
   "key": "potato_late_blight",
   "plant": "Potato",
   "condition": "Late blight",
   "condition_id": "late_blight",
   "healthy": false
  },
  {
   "id": 22,
   "key": "potato_healthy",
   "plant": "Potato",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 23,
   "key": "raspberry_healthy",
   "plant": "Raspberry",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 24,
   "key": "soybean_healthy",
   "plant": "Soybean",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 25,
   "key": "squash_powdery_mildew",
   "plant": "Squash",
   "condition": "Powdery mildew",
   "condition_id": "powdery_mildew",
   "healthy": false
  },
  {
   "id": 26,
   "key": "strawberry_leaf_scorch",
   "plant": "Strawberry",
   "condition": "Leaf scorch",
   "condition_id": "leaf_scorch",
   "healthy": false
  },
  {
   "id": 27,
   "key": "strawberry_healthy",
   "plant": "Strawberry",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  },
  {
   "id": 28,
   "key": "tomato_bacterial_spot",
   "plant": "Tomato",
   "condition": "Bacterial spot",
   "condition_id": "bacterial_spot",
   "healthy": false
  },
  {
   "id": 29,
   "key": "tomato_early_blight",
   "plant": "Tomato",
   "condition": "Early blight",
   "condition_id": "early_blight",
   "healthy": false
  },
  {
   "id": 30,
   "key": "tomato_late_blight",
   "plant": "Tomato",
   "condition": "Late blight",
   "condition_id": "late_blight",
   "healthy": false
  },
  {
   "id": 31,
   "key": "tomato_leaf_mold",
   "plant": "Tomato",
   "condition": "Leaf mold",
   "condition_id": "leaf_mold",
   "healthy": false
  },
  {
   "id": 32,
   "key": "tomato_septoria_leaf_spot",
   "plant": "Tomato",
   "condition": "Septoria leaf spot",
   "condition_id": "septoria_leaf_spot",
   "healthy": false
  },
  {
   "id": 33,
   "key": "tomato_spider_mites",
   "plant": "Tomato",
   "condition": "Spider mites",
   "condition_id": "spider_mites",
   "healthy": false
  },
  {
   "id": 34,
   "key": "tomato_target_spot",
   "plant": "Tomato",
   "condition": "Target spot",
   "condition_id": "target_spot",
   "healthy": false
  },
  {
   "id": 35,
   "key": "tomato_yellow_leaf_curl_virus",
   "plant": "Tomato",
   "condition": "Yellow leaf curl virus",
   "condition_id": "leaf_curl_virus",
   "healthy": false
  },
  {
   "id": 36,
   "key": "tomato_mosaic_virus",
   "plant": "Tomato",
   "condition": "Mosaic virus",
   "condition_id": "mosaic_virus",
   "healthy": false
  },
  {
   "id": 37,
   "key": "tomato_healthy",
   "plant": "Tomato",
   "condition": "Healthy",
   "condition_id": "healthy",
   "healthy": true
  }
 ]
}
'''

# ================================================================================================
# FILE: src/app/ui/worker.py
# ================================================================================================

"""Background analysis: the pipeline is built and run on a dedicated QThread."""
from __future__ import annotations

import logging
import traceback
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot

from app.contracts import InvalidImageError, RGBImage
from app.pipeline import AnalysisPipeline

log = logging.getLogger(__name__)


class _AnalysisWorker(QObject):
    ready = Signal(str, str)  # model_version, warning ("" if none)
    finished = Signal(object)  # AnalysisResult
    failed = Signal(str)

    def __init__(self, factory: Callable[[], tuple[AnalysisPipeline, str]]) -> None:
        super().__init__()
        self._factory = factory
        self._pipeline: AnalysisPipeline | None = None

    @Slot()
    def initialize(self) -> None:
        try:
            self._pipeline, warning = self._factory()
            self.ready.emit(self._pipeline.model_version, warning)
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            log.exception("pipeline init failed")
            self.failed.emit(f"Could not load the analysis pipeline: {exc}")

    @Slot(bool)
    def set_highlight(self, on: bool) -> None:
        if self._pipeline is not None and hasattr(self._pipeline.overlay, "highlight_discoloration"):
            self._pipeline.overlay.highlight_discoloration = on

    @Slot(object)
    def analyze(self, image: RGBImage) -> None:
        if self._pipeline is None:
            self.failed.emit("The analysis pipeline is not ready yet.")
            return
        try:
            self.finished.emit(self._pipeline.analyze(image))
        except InvalidImageError as exc:
            self.failed.emit(f"That image can't be analysed: {exc}")
        except Exception as exc:  # noqa: BLE001
            log.error("analysis failed\n%s", traceback.format_exc())
            self.failed.emit(f"Analysis failed: {exc}")


class AnalysisController(QObject):
    """UI-thread facade. Inference never runs on the UI thread."""

    ready = Signal(str, str)
    finished = Signal(object)
    failed = Signal(str)
    busy_changed = Signal(bool)
    _request = Signal(object)
    _highlight = Signal(bool)

    def __init__(self, factory: Callable[[], tuple[AnalysisPipeline, str]], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._thread.setObjectName("analysis")
        self._worker = _AnalysisWorker(factory)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.initialize)
        self._request.connect(self._worker.analyze)
        self._highlight.connect(self._worker.set_highlight)
        self._worker.ready.connect(self._on_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._busy = False
        self.is_ready = False
        self.model_version = ""

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._thread.quit()
        self._thread.wait(5000)

    @property
    def busy(self) -> bool:
        return self._busy

    def analyze(self, image: RGBImage) -> bool:
        if self._busy:
            return False
        self._set_busy(True)
        self._request.emit(image)
        return True

    def set_highlight(self, on: bool) -> None:
        self._highlight.emit(on)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)

    def _on_ready(self, version: str, warning: str) -> None:
        self.is_ready = True
        self.model_version = version
        self.ready.emit(version, warning)

    def _on_finished(self, result) -> None:
        self._set_busy(False)
        self.finished.emit(result)

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self.failed.emit(message)

# ================================================================================================
# FILE: src/app/config.py
# ================================================================================================

"""Runtime configuration. Values can be overridden with LEAFSCAN_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.contracts import DEFAULT_UNCERTAINTY_THRESHOLD

APP_NAME = "LeafScan"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: _env_path("LEAFSCAN_DATA_DIR", PROJECT_ROOT / "data" / "app"))
    models_dir: Path = field(default_factory=lambda: _env_path("LEAFSCAN_MODELS_DIR", PROJECT_ROOT / "models"))
    uncertainty_threshold: float = field(
        default_factory=lambda: float(os.environ.get("LEAFSCAN_THRESHOLD", DEFAULT_UNCERTAINTY_THRESHOLD))
    )
    plant_threshold: float = field(default_factory=lambda: float(os.environ.get("LEAFSCAN_PLANT_THRESHOLD", 0.55)))
    camera_index: int = 0
    use_camera: bool = True
    use_mocks: bool = False

    @property
    def db_path(self) -> Path:
        return self.data_dir / "leafscan.db"

    @property
    def scans_dir(self) -> Path:
        return self.data_dir / "scans"
