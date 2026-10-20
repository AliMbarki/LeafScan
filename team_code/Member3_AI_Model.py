"""
LeafScan - Member 3 - AI model (MobileNetV2, ONNX Runtime)
Main technology: ONNX / ONNX Runtime / NumPy

Reading copy: the code below is collected from the real project files listed here.
Edit the originals (paths shown in each section), not this file. Files:
  - models/plant_disease.json
  - src/app/classify/__init__.py
  - src/app/classify/predictor.py
  - src/app/classify/preprocess.py
  - tests/test_hardening.py
  - tests/test_contracts.py
"""


# ================================================================================================
# FILE: models/plant_disease.json
# ================================================================================================

# (JSON data file)
'''
{
 "version": "mobilenetv2-plantvillage-38c",
 "file": "plant_disease.onnx",
 "input": "pixel_values",
 "num_classes": 38,
 "resize_shortest": 256,
 "crop": 224,
 "mean": [
  0.5,
  0.5,
  0.5
 ],
 "std": [
  0.5,
  0.5,
  0.5
 ],
 "source": "https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification (ONNX export: onnx-community)",
 "license": "other - see the model card",
 "reported_accuracy": "95.4% on the PlantVillage evaluation split (model card)"
}
'''

# ================================================================================================
# FILE: src/app/classify/__init__.py
# ================================================================================================

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

# ================================================================================================
# FILE: src/app/classify/predictor.py
# ================================================================================================

"""ONNX Runtime implementation of the Classifier contract (pre-trained model, no PyTorch needed)."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from app.classify.preprocess import preprocess
from app.contracts import DEFAULT_UNCERTAINTY_THRESHOLD, ClassifierResult, RGBImage, validate_rgb
from app.labels import Labels, get_labels


class ModelLoadError(RuntimeError):
    pass


class OnnxClassifier:
    """Loads ``<name>.onnx`` plus its ``<name>.json`` (preprocessing + version)."""

    def __init__(
        self,
        model_path: Path | str,
        threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
        labels: Labels | None = None,
        plant_threshold: float | None = None,
    ) -> None:
        model_path = Path(model_path)
        self.labels = labels or get_labels()
        self.threshold = threshold
        self.plant_threshold = threshold if plant_threshold is None else plant_threshold
        meta_path = model_path.with_suffix(".json")
        if not model_path.exists() or not meta_path.exists():
            raise ModelLoadError(f"model file not found: {model_path}")
        try:
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelLoadError(f"cannot read {meta_path.name}: {exc}") from exc
        if self.meta.get("num_classes") != len(self.labels):
            raise ModelLoadError("model classes do not match labels.json")
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001 - corrupt file, missing runtime...
            raise ModelLoadError(f"could not load {model_path.name}: {exc}") from exc
        self.input_name = self.meta.get("input") or self.session.get_inputs()[0].name
        self.model_version: str = self.meta["version"]
        self._mean = np.asarray(self.meta["mean"], np.float32)
        self._std = np.asarray(self.meta["std"], np.float32)
        self._lock = threading.Lock()
        self.predict(np.zeros((224, 224, 3), np.uint8))  # warm-up so the first real scan is fast

    def probabilities_batch(self, images: list[RGBImage]) -> np.ndarray:
        for im in images:
            validate_rgb(im)
        x = np.stack([preprocess(im, self._mean, self._std, self.meta["resize_shortest"], self.meta["crop"])
                      for im in images])
        with self._lock:
            logits = self.session.run(None, {self.input_name: x})[0].astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def probabilities(self, image: RGBImage) -> np.ndarray:
        return self.probabilities_batch([image])[0]

    def _result(self, probs: np.ndarray) -> ClassifierResult:
        return ClassifierResult.from_probabilities(probs, self.labels, self.threshold, plant_threshold=self.plant_threshold)

    def predict(self, image: RGBImage) -> ClassifierResult:
        return self._result(self.probabilities(image))

    def predict_views(self, views: list[RGBImage]) -> ClassifierResult:
        """Average the probabilities of several views of the same leaf (test-time augmentation)."""
        return self._result(self.probabilities_batch(views).mean(axis=0))

# ================================================================================================
# FILE: src/app/classify/preprocess.py
# ================================================================================================

"""Inference preprocessing: resize shorter side, centre crop, normalise to CHW float32."""
from __future__ import annotations

import cv2
import numpy as np

from app.contracts import RGBImage


def resize_center_crop(image: RGBImage, resize_to: int = 256, size: int = 224) -> RGBImage:
    h, w = image.shape[:2]
    s = resize_to / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    return resized[y0 : y0 + size, x0 : x0 + size]


def preprocess(image: RGBImage, mean: np.ndarray, std: np.ndarray, resize_to: int = 256, size: int = 224) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 3 x size x size, normalised with the model's mean/std."""
    x = resize_center_crop(image, resize_to, size).astype(np.float32) / 255.0
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1))

# ================================================================================================
# FILE: tests/test_hardening.py
# ================================================================================================

"""Failure modes: missing/corrupt/mismatched model, no leaf, missing camera."""
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.contracts import MockClassifier
from app.main import build_pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "plant_disease.onnx"


def test_missing_model_falls_back_to_mock_with_warning(tmp_path):
    pipeline, warning = build_pipeline(Settings(data_dir=tmp_path, models_dir=tmp_path / "nope"))
    assert isinstance(pipeline.classifier, MockClassifier)
    assert "MOCK" in warning


def test_corrupt_model_falls_back(tmp_path):
    (tmp_path / "plant_disease.onnx").write_bytes(b"definitely not a model")
    (tmp_path / "plant_disease.json").write_text((ROOT / "models" / "plant_disease.json").read_text())
    pipeline, warning = build_pipeline(Settings(data_dir=tmp_path, models_dir=tmp_path))
    assert isinstance(pipeline.classifier, MockClassifier) and warning


def test_model_labels_mismatch_is_rejected(tmp_path):
    import json
    import shutil

    from app.classify.predictor import ModelLoadError, OnnxClassifier

    shutil.copy(MODEL, tmp_path / "m.onnx")
    meta = json.loads(MODEL.with_suffix(".json").read_text())
    meta["num_classes"] = 3
    (tmp_path / "m.json").write_text(json.dumps(meta))
    with pytest.raises(ModelLoadError, match="labels.json"):
        OnnxClassifier(tmp_path / "m.onnx")


def test_model_contract(leaf_image):
    from app.classify.predictor import OnnxClassifier

    clf = OnnxClassifier(MODEL)
    p = clf.probabilities(leaf_image)
    assert p.shape == (38,) and abs(p.sum() - 1) < 1e-4
    r = clf.predict(leaf_image)
    assert 0 <= r.confidence <= 1 and len(r.top_k) == 3
    clf.threshold = clf.plant_threshold = 1.01  # nothing can be that confident -> always "not sure"
    r = clf.predict(leaf_image)
    assert r.is_uncertain and r.class_id is None


def test_pipeline_no_leaf_skips_classification(tmp_path):
    pipeline, _ = build_pipeline(Settings(data_dir=tmp_path, models_dir=tmp_path))
    blank = np.full((300, 400, 3), 128, np.uint8)
    r = pipeline.analyze(blank)
    assert not r.leaf.found and r.classification is None and r.guidance is None
    assert r.overlay_image.shape == blank.shape


def test_ui_shows_banner_without_model_and_handles_no_camera(qtbot, tmp_path):
    from app.storage.db import SqliteScanStore
    from app.ui.main_window import MainWindow

    settings = Settings(data_dir=tmp_path, models_dir=tmp_path / "none", use_camera=False)
    win = MainWindow(settings, SqliteScanStore(settings.db_path), lambda: build_pipeline(settings))
    qtbot.addWidget(win)
    qtbot.waitUntil(lambda: win.controller.is_ready, timeout=60000)
    assert win.scan_page.banner.isVisibleTo(win) and "MOCK" in win.scan_page.banner.text()
    assert not win.scan_page.capture_btn.isEnabled()  # no camera -> capture disabled, import still works
    assert win.scan_page.import_btn.isEnabled()
    win._on_camera_error("No camera found at index 0.")
    assert win.scan_page.retry_btn.isVisibleTo(win)
    win.close()

# ================================================================================================
# FILE: tests/test_contracts.py
# ================================================================================================

import numpy as np
import pytest

from app.contracts import (
    ClassifierResult,
    InvalidImageError,
    MockClassifier,
    MockLeafDetector,
    validate_rgb,
)
from app.labels import UNCERTAIN_KEY, Labels, get_labels


def test_labels_are_consistent():
    labels = get_labels()
    n = len(labels)
    assert [c.id for c in labels] == list(range(n))
    assert len(set(labels.keys)) == n
    assert len(labels.plants) >= 10
    assert sum(c.unknown for c in labels) <= 1  # at most one "some other plant" class
    for c in labels:
        assert c.display_name.isascii() and c.condition_id
    for plant in labels.plants:
        assert labels.classes_of(plant), plant
        assert labels.plant_info[plant].latin


def test_labels_reject_bad_ids(tmp_path):
    bad = tmp_path / "l.json"
    bad.write_text('{"classes":[{"id":1,"key":"a","plant":"P","condition":"c","healthy":true,"source_label":"x"}]}')
    with pytest.raises(ValueError):
        Labels.load(bad)


def _probs(labels, mass: dict[int, float]) -> np.ndarray:
    rest = (1 - sum(mass.values())) / (len(labels) - len(mass))
    p = np.full(len(labels), rest)
    for i, v in mass.items():
        p[i] = v
    return p


def test_from_probabilities_confident():
    labels = get_labels()
    r = ClassifierResult.from_probabilities(_probs(labels, {3: 0.95}), labels, threshold=0.6)
    assert not r.is_uncertain and r.class_id == 3
    assert r.class_key == labels.by_id(3).key
    assert r.plant == labels.by_id(3).plant and r.plant_confidence >= 0.95
    assert r.top_k[0].class_id == 3 and len(r.top_k) == 3


def test_plant_known_but_condition_not_sure():
    labels = get_labels()
    plant = next(p for p in labels.plants if len(labels.classes_of(p)) >= 2)
    a, b = (c.id for c in labels.classes_of(plant)[:2])
    r = ClassifierResult.from_probabilities(_probs(labels, {a: 0.45, b: 0.45}), labels, threshold=0.6)
    assert r.plant == plant and r.plant_confidence >= 0.9  # the plant is sure...
    assert r.is_uncertain and r.class_id is None  # ...the condition is not
    assert r.class_key == UNCERTAIN_KEY and not r.unknown_plant


def test_unknown_plant_is_reported_not_forced():
    labels = get_labels()
    other = next((c for c in labels if c.unknown), None)
    if other is None:
        pytest.skip("this model has no unknown-plant class")
    r = ClassifierResult.from_probabilities(_probs(labels, {other.id: 0.9}), labels, threshold=0.6)
    assert r.unknown_plant and r.plant is None and r.is_uncertain
    assert r.disease_name == labels.unknown_name == other.display_name


def test_plant_not_sure_when_spread_across_plants():
    labels = get_labels()
    a = labels.classes_of(labels.plants[0])[0].id
    b = labels.classes_of(labels.plants[1])[0].id
    r = ClassifierResult.from_probabilities(_probs(labels, {a: 0.4, b: 0.4}), labels, threshold=0.6)
    assert r.plant is None and r.is_uncertain and not r.unknown_plant


def test_from_probabilities_uncertain_never_forces_label():
    labels = get_labels()
    probs = np.full(len(labels), 1 / len(labels))
    r = ClassifierResult.from_probabilities(probs, labels, threshold=0.6)
    assert r.is_uncertain and r.class_id is None and r.class_key == UNCERTAIN_KEY
    assert r.disease_name == "Not sure"
    assert r.top_k  # best guesses are still available


@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.zeros((64, 64), np.uint8),
        np.zeros((64, 64, 3), np.float32),
        np.zeros((8, 8, 3), np.uint8),
        np.zeros((64, 64, 4), np.uint8),
    ],
)
def test_validate_rgb_rejects(bad):
    with pytest.raises(InvalidImageError):
        validate_rgb(bad)


def test_mock_classifier_deterministic(leaf_image):
    clf = MockClassifier()
    assert clf.predict(leaf_image) == clf.predict(leaf_image.copy())


def test_mock_detector_matches_input_geometry(leaf_image):
    r = MockLeafDetector().detect_leaf(leaf_image)
    assert r.found
    assert r.mask.shape == leaf_image.shape[:2] and r.mask.dtype == bool
    assert r.annotated_image.shape == leaf_image.shape


def test_format_confidence_never_claims_certainty():
    from app.contracts import format_confidence

    assert format_confidence(0.9996) == ">99%"
    assert format_confidence(0.987) == "99%"
    assert format_confidence(0.55) == "55%"
    assert format_confidence(0.001) == "<1%"
    assert format_confidence(0.0) == "0%"
