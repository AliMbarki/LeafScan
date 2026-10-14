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
