"""Leaf detector + overlay on synthetic scenes with known ground truth."""
import cv2
import numpy as np
import pytest

from app.contracts import ClassifierResult
from app.labels import get_labels
from app.vision import ClassicalLeafDetector, LeafOverlayRenderer
from app.vision.leaf import mask_iou


def scene(w=640, h=480, bg=(200, 198, 195), leaf=(55, 135, 45), shadow=False, spots=False, center=None, axes=None):
    img = np.full((h, w, 3), bg, np.uint8)
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(np.int16) + rng.integers(-4, 5, img.shape), 0, 255).astype(np.uint8)
    center = center or (w // 2, h // 2)
    axes = axes or (w // 4, h // 4)
    if shadow:  # soft, darker, same hue as background - must stay background
        cv2.ellipse(img, (center[0] + 30, center[1] + 25), axes, 25, 0, 360, (120, 119, 117), -1)
    gt = np.zeros((h, w), np.uint8)
    cv2.ellipse(gt, center, axes, 25, 0, 360, 255, -1)
    img[gt > 0] = leaf
    if spots:
        for dx, dy in ((-40, -10), (30, 20), (10, -35), (-15, 30)):
            cv2.circle(img, (center[0] + dx, center[1] + dy), 12, (115, 80, 35), -1)
    return img, gt > 0


@pytest.fixture(scope="module")
def det():
    return ClassicalLeafDetector()


def test_finds_leaf_on_plain_background(det):
    img, gt = scene()
    r = det.detect_leaf(img)
    assert r.found, r.message
    assert r.mask.shape == img.shape[:2] and r.mask.dtype == bool
    assert mask_iou(r.mask, gt) > 0.92
    x, y, w, h = r.bbox
    assert 0 <= x and 0 <= y and x + w <= img.shape[1] and y + h <= img.shape[0]


def test_shadow_is_excluded(det):
    img, gt = scene(shadow=True)
    r = det.detect_leaf(img)
    assert r.found and mask_iou(r.mask, gt) > 0.9


@pytest.mark.parametrize("bg", [(245, 245, 245), (30, 30, 32), (180, 160, 120)])
def test_various_plain_backgrounds(det, bg):
    img, gt = scene(bg=bg)
    r = det.detect_leaf(img)
    assert r.found and mask_iou(r.mask, gt) > 0.9


def test_coordinates_match_large_non_square_input(det):
    img, gt = scene(w=1920, h=1080, center=(1300, 400), axes=(300, 180))
    r = det.detect_leaf(img)
    assert r.found and r.mask.shape == (1080, 1920)
    assert mask_iou(r.mask, gt) > 0.9
    x, y, w, h = r.bbox
    assert abs((x + w / 2) - 1300) < 40 and abs((y + h / 2) - 400) < 40


def test_lesions_are_inside_mask_and_flagged(det):
    img, gt = scene(spots=True)
    r = det.detect_leaf(img)
    assert r.found and mask_iou(r.mask, gt) > 0.9  # brown spots are filled, not holes
    assert r.discoloration is not None and r.discoloration_fraction > 0.02
    healthy = det.detect_leaf(scene()[0])
    assert healthy.discoloration_fraction < 0.01


@pytest.mark.parametrize(
    "image",
    [
        np.full((480, 640, 3), 128, np.uint8),  # empty plain background
        np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8),  # noise
        scene(leaf=(40, 60, 200))[0],  # blue object: not a leaf
        scene(leaf=(214, 160, 130))[0],  # skin-toned object (a hand): not a leaf
        scene(axes=(20, 14))[0],  # far too small
    ],
    ids=["empty", "noise", "blue-object", "skin-object", "tiny"],
)
def test_no_leaf_cases(det, image):
    r = det.detect_leaf(image)
    assert not r.found
    assert r.message.startswith("No leaf detected")
    assert not r.mask.any() and r.mask.shape == image.shape[:2]


def test_mostly_brown_diseased_leaf_is_found(det):
    img, gt = scene(leaf=(105, 78, 40))  # dead, brown tissue...
    cv2.ellipse(img, (330, 260), (60, 30), 25, 0, 360, (70, 140, 50), -1)  # ...with a little green left
    r = det.detect_leaf(img)
    assert r.found, r.message
    assert mask_iou(r.mask, gt) > 0.85


def _clutter(img, rng, n=40, avoid=None):
    h, w = img.shape[:2]
    palette = [(200, 60, 60), (40, 70, 160), (230, 230, 230), (30, 30, 30), (190, 150, 90), (150, 60, 150)]
    for _ in range(n):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        bw, bh = int(rng.integers(20, 120)), int(rng.integers(20, 120))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), palette[int(rng.integers(len(palette)))], -1)


def test_leaf_on_cluttered_desk_is_found(det):
    rng = np.random.default_rng(3)
    img = np.full((480, 640, 3), (120, 110, 100), np.uint8)
    _clutter(img, rng)
    gt = np.zeros((480, 640), np.uint8)
    cv2.ellipse(gt, (320, 240), (150, 100), 20, 0, 360, 255, -1)
    img[gt > 0] = (60, 140, 50)
    r = det.detect_leaf(img)
    assert r.found, r.message
    assert r.method in {"isolated", "center", "area"}
    assert mask_iou(r.mask, gt > 0) > 0.75


def test_leaf_held_in_hand_is_found(det):
    img = np.full((480, 640, 3), (90, 95, 100), np.uint8)
    cv2.rectangle(img, (280, 330), (380, 480), (214, 160, 130), -1)  # arm/hand from the bottom edge
    gt = np.zeros((480, 640), np.uint8)
    cv2.ellipse(gt, (330, 210), (120, 140), 0, 0, 360, 255, -1)
    img[gt > 0] = (60, 140, 50)
    r = det.detect_leaf(img)
    assert r.found, r.message
    assert mask_iou(r.mask, gt > 0) > 0.7


def test_foliage_falls_back_to_main_green_area(det):
    rng = np.random.default_rng(5)
    img = np.full((480, 640, 3), (70, 55, 40), np.uint8)  # soil
    for _ in range(25):
        c = (int(rng.integers(0, 640)), int(rng.integers(0, 480)))
        cv2.ellipse(img, c, (int(rng.integers(40, 90)), int(rng.integers(20, 45))), int(rng.integers(0, 180)), 0, 360,
                    (int(rng.integers(40, 80)), int(rng.integers(110, 170)), int(rng.integers(30, 60))), -1)
    r = det.detect_leaf(img)
    assert r.found, r.message  # foliage must never be reported as "no leaf"
    assert r.area_fraction > 0.1


def test_leaf_filling_frame(det):
    img = np.full((480, 640, 3), (60, 140, 50), np.uint8)
    cv2.circle(img, (300, 240), 30, (120, 85, 40), -1)
    r = det.detect_leaf(img)
    assert r.found and r.area_fraction > 0.9


def test_overlay_shapes_and_no_leaf(det):
    labels = get_labels()
    img, _ = scene(spots=True)
    leaf = det.detect_leaf(img)
    probs = np.full(len(labels), 0.01)
    probs[2] = 0.92
    res = ClassifierResult.from_probabilities(probs, labels)
    ov = LeafOverlayRenderer()
    out = ov.render(img, leaf, res)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)
    blank = np.full((300, 400, 3), 128, np.uint8)
    nl = det.detect_leaf(blank)
    assert ov.render(blank, nl, None).shape == blank.shape
