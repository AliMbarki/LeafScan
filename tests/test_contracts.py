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
