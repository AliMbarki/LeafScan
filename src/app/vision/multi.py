"""Find several leaves in one frame (for video), using classical OpenCV - no training.

1. Plant-coloured pixels (green / yellow-green) in HSV, closed and hole-filled so lesions inside a
   leaf stay part of it.
2. Touching leaves form one blob, so big blobs are split with a distance transform + watershed.
3. Regions that are too small or not green enough are dropped; the largest ones are returned.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.contracts import RGBImage, validate_rgb
from app.vision.leaf import _pad_fill_holes, leaf_tissue_mask, plant_color_mask

WORK_SIZE = 480  # longest side used for detection (fast enough for video)


@dataclass(frozen=True)
class LeafRegion:
    bbox: tuple[int, int, int, int]  # x, y, w, h in input coordinates
    contour: np.ndarray  # (N, 1, 2) int32, input coordinates
    area_fraction: float  # share of the frame


def _split_touching(blob: np.ndarray) -> list[np.ndarray]:
    """Watershed on the distance transform: one marker per 'thick' part of the blob."""
    dist = cv2.distanceTransform(blob.astype(np.uint8), cv2.DIST_L2, 5)
    if dist.max() < 6:
        return [blob]
    peaks = (dist > 0.45 * dist.max()).astype(np.uint8)
    n, markers = cv2.connectedComponents(peaks)
    if n <= 2:  # one peak region -> one leaf
        return [blob]
    markers = markers + 1
    markers[~blob] = 0
    canvas = cv2.cvtColor((blob * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(canvas, markers)
    return [(markers == i) & blob for i in range(2, n + 1)]


def detect_leaves(image: RGBImage, max_leaves: int = 8, min_area: float = 0.012) -> list[LeafRegion]:
    validate_rgb(image)
    H, W = image.shape[:2]
    s = min(1.0, WORK_SIZE / max(H, W))
    small = cv2.resize(image, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else image
    small = cv2.GaussianBlur(small, (5, 5), 0)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    green = plant_color_mask(hsv)
    tissue = leaf_tissue_mask(hsv)

    mask = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0
    mask = _pad_fill_holes(mask)

    total = mask.shape[0] * mask.shape[1]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    regions: list[np.ndarray] = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area * total:
            continue
        blob = lab == i
        regions += _split_touching(blob) if stats[i, cv2.CC_STAT_AREA] > 0.08 * total else [blob]

    found: list[LeafRegion] = []
    for r in regions:
        area = r.sum()
        if area < min_area * total:
            continue
        if green[r].mean() < 0.35 or tissue[r].mean() < 0.6:  # mostly not plant-coloured
            continue
        r8 = cv2.resize(r.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours((r8 > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        found.append(LeafRegion((int(x), int(y), int(w), int(h)), c, float(area / total)))
    found.sort(key=lambda L: -L.area_fraction)
    return found[:max_leaves]
