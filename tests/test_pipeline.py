import numpy as np
import pytest

from app.capture import load_image_file, save_rgb_png
from app.contracts import InvalidImageError, LeafResult
from app.pipeline import build_mock_pipeline, crop_to_leaf


def test_mock_pipeline_end_to_end(leaf_image):
    r = build_mock_pipeline().analyze(leaf_image)
    assert r.classification is not None
    assert r.overlay_image.shape == leaf_image.shape
    assert r.model_version == "mock-0"
    assert (r.guidance is None) == r.classification.is_uncertain


def test_pipeline_rejects_bad_image():
    with pytest.raises(InvalidImageError):
        build_mock_pipeline().analyze(np.zeros((10, 10), np.uint8))


def test_crop_to_leaf_bounds(leaf_image):
    mask = np.zeros(leaf_image.shape[:2], bool)
    leaf = LeafResult(mask, leaf_image, True, (400, 300, 70, 50))
    crop = crop_to_leaf(leaf_image, leaf)
    assert crop.ndim == 3 and crop.shape[0] >= 32 and crop.shape[1] >= 32


def test_load_image_file_roundtrip_rgb(tmp_path):
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0] = 255  # pure red in RGB
    path = tmp_path / "unicode été red.png"
    save_rgb_png(img, path)
    assert load_image_file(path)[0, 0].tolist() == [255, 0, 0]


def test_load_image_file_rejects_garbage(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image")
    with pytest.raises(InvalidImageError):
        load_image_file(p)
    with pytest.raises(InvalidImageError):
        load_image_file(tmp_path / "missing.png")
