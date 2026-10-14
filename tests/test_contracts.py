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
    assert len(labels) == 9
    assert [c.id for c in labels] == list(range(9))
    assert len(set(labels.keys)) == 9
    for c in labels:
        assert c.display_name.isascii()


def test_labels_reject_bad_ids(tmp_path):
    bad = tmp_path / "l.json"
    bad.write_text('{"classes":[{"id":1,"key":"a","plant":"P","condition":"c","healthy":true,"source_label":"x"}]}')
    with pytest.raises(ValueError):
        Labels.load(bad)


def test_from_probabilities_confident():
    labels = get_labels()
    probs = np.full(len(labels), 0.01)
    probs[3] = 1 - 0.01 * (len(labels) - 1)
    r = ClassifierResult.from_probabilities(probs, labels, threshold=0.6)
    assert not r.is_uncertain and r.class_id == 3
    assert r.class_key == labels.by_id(3).key
    assert r.top_k[0].class_id == 3 and len(r.top_k) == 3


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
