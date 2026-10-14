"""Classical leaf segmentation for a single leaf on a plain background.

Pipeline (all on a downscaled working copy, mask scaled back to input size):
1. Colour cue: green/yellow pixels in HSV ("plant-coloured").
2. Background model: robust median Lab colour of the non-plant border pixels;
   foreground = pixels far from it (threshold adapts to background texture).
3. GrabCut refines the rough mask (sure-bg border, sure-fg eroded plant core).
4. Keep the largest component, fill holes (lesions), smooth.
5. Sanity checks -> found / "no leaf detected" with a human-readable reason.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.contracts import LeafResult, Mask, RGBImage, validate_rgb

OUTLINE_RGB = (62, 207, 142)


@dataclass
class LeafDetectorConfig:
    work_size: int = 448  # longest side of the working copy
    min_area: float = 0.04  # of the frame; smaller blobs are background clutter, not "the" leaf
    max_area: float = 0.985
    min_tissue_fraction: float = 0.60  # leaf-coloured (green, yellow, pale, brown) share inside the mask
    min_green_fraction: float = 0.03  # some true green/yellow must be present (rejects skin, wood)
    min_bg_distance: float = 8.0  # Lab chroma units
    max_edge_contact: float = 0.40  # share of the frame border covered by the mask (test split p99: 0.19)
    min_solidity: float = 0.50  # area / convex-hull area (test split p1: 0.68)
    grabcut_iters: int = 3
    border_frac: float = 0.03


def plant_color_mask(hsv: np.ndarray) -> np.ndarray:
    """Yellow-green to green, reasonably saturated, not too dark (OpenCV hue 0-180).

    Hue starts at 23 (~46 deg) so tan/wooden/beige surfaces (hue ~15-21) are not "plant".
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 23) & (h <= 95) & (s >= 45) & (v >= 35)


def leaf_tissue_mask(hsv: np.ndarray) -> np.ndarray:
    """Anything a (possibly very diseased) leaf is made of: green/yellow, pale green, or brown dead tissue."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    pale_green = (h >= 28) & (h <= 95) & (s >= 18) & (v >= 35)
    brown = (h <= 22) & (s >= 50) & (v >= 25) & (v <= 225)
    return plant_color_mask(hsv) | pale_green | brown


def healthy_green_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 30) & (h <= 90) & (s >= 50) & (v >= 40)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == best


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    h, w = m.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    inv = m.copy()
    cv2.floodFill(inv, flood, (0, 0), 255)  # (0,0) is background after padding below
    return mask | (inv == 0)


def _pad_fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    return _fill_holes(padded)[1:-1, 1:-1]


def _border(a: np.ndarray, b: int) -> np.ndarray:
    return np.concatenate([a[:b].reshape(-1, *a.shape[2:]), a[-b:].reshape(-1, *a.shape[2:]),
                           a[b:-b, :b].reshape(-1, *a.shape[2:]), a[b:-b, -b:].reshape(-1, *a.shape[2:])])


class ClassicalLeafDetector:
    def __init__(self, config: LeafDetectorConfig | None = None) -> None:
        self.cfg = config or LeafDetectorConfig()

    # ------------------------------------------------------------------ public

    def detect_leaf(self, image: RGBImage) -> LeafResult:
        validate_rgb(image)
        H, W = image.shape[:2]
        scale = min(1.0, self.cfg.work_size / max(H, W))
        small = cv2.resize(image, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
        small = cv2.GaussianBlur(small, (3, 3), 0)
        mask_small, reason = self._segment(small)
        if mask_small is None:
            return self._not_found(image, reason)

        mask = cv2.resize(mask_small.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_LINEAR) > 127
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._not_found(image, "No leaf detected.")
        contour = max(contours, key=cv2.contourArea)
        disc = self._discoloration(small, mask_small)
        disc_full = cv2.resize(disc.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST) > 127
        disc_full &= mask
        area = float(mask.mean())
        annotated = image.copy()
        cv2.drawContours(annotated, [contour], -1, OUTLINE_RGB, max(2, round(max(H, W) / 320)), cv2.LINE_AA)
        return LeafResult(
            mask=mask,
            annotated_image=annotated,
            found=True,
            bbox=tuple(int(v) for v in cv2.boundingRect(contour)),
            area_fraction=area,
            message=reason,
            contour=contour,
            discoloration=disc_full,
            discoloration_fraction=float(disc_full.sum() / max(mask.sum(), 1)),
        )

    # ----------------------------------------------------------------- private

    def _segment(self, rgb: np.ndarray) -> tuple[np.ndarray | None, str]:
        cfg = self.cfg
        h, w = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        plant = plant_color_mask(hsv)
        b = max(3, round(min(h, w) * cfg.border_frac))

        border_lab = _border(lab, b)
        border_plant = _border(plant, b)
        bg_pixels = border_lab[~border_plant]
        border_green = _border(healthy_green_mask(hsv), b).mean()
        if len(bg_pixels) < 0.2 * len(border_lab) and border_green < 0.5:
            bg_pixels = border_lab  # yellowish plain surface, not foliage: model the whole border
        if len(bg_pixels) < 0.2 * len(border_lab):
            # The border is mostly plant-coloured: leaf (or foliage) fills the frame.
            mask = _pad_fill_holes(_largest_component(cv2.morphologyEx(
                plant.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0))
            if mask.mean() < cfg.min_area:
                return None, "No leaf detected - the frame is full of something that isn't clearly a leaf."
            return mask, "Leaf fills the frame - step back a little for a cleaner outline."

        bg = np.median(bg_pixels, axis=0)
        # Chroma-only distance (a*, b*): cast shadows on a neutral background change lightness
        # but hardly any colour, so they stay background. Leaves (green/yellow/brown) are chromatic.
        dist = np.sqrt(((lab[..., 1:] - bg[1:]) ** 2).sum(-1))
        border_dist = np.sqrt(((bg_pixels[:, 1:] - bg[1:]) ** 2).sum(-1))
        thr = max(cfg.min_bg_distance, float(np.percentile(border_dist, 90)) * 1.5 + 3)
        light_diff = np.abs(lab[..., 0] - bg[0])
        shadow = (dist < thr) & (lab[..., 0] < bg[0])  # darker but same colour as the background
        rough = (dist > thr) | (plant & (dist > thr * 0.6)) | ((light_diff > 60) & ~shadow)
        rough = cv2.morphologyEx(rough.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        rough = _largest_component(rough)
        if rough.mean() < cfg.min_area * 0.5:
            return None, "No leaf detected. Place one leaf in the frame against a plain, contrasting background."

        mask = self._grabcut(rgb, rough, plant, (dist < thr * 0.5) | shadow, b)
        mask &= ~(shadow & ~plant)
        mask = _largest_component(mask)
        mask = _pad_fill_holes(mask)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        mask = _largest_component(mask)

        area = mask.mean()
        if area < cfg.min_area:
            return None, "No leaf detected - the object is too small. Move the leaf closer to the camera."
        if area > cfg.max_area:
            return None, "No leaf detected - could not separate the leaf from the background."
        # Badly diseased leaves can be mostly brown, so accept any leaf tissue colour, but still
        # require a little true green/yellow - that is what rejects hands, wood and cardboard.
        tissue_frac = float(leaf_tissue_mask(hsv)[mask].mean())
        green_frac = float(plant[mask].mean())
        if tissue_frac < cfg.min_tissue_fraction or green_frac < cfg.min_green_fraction:
            return None, "No leaf detected - the object in view doesn't look like a leaf."
        edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean()
        if edge > cfg.max_edge_contact:
            return None, ("No leaf detected - the region runs off the edges of the frame. "
                          "Use a plain background with the whole leaf in view.")
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        solidity = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        if solidity < cfg.min_solidity:
            return None, "No leaf detected - the shape is too irregular for a single leaf. Try a plainer background."
        return mask, ""

    def _grabcut(self, rgb, rough, plant, near_bg, b) -> np.ndarray:
        h, w = rough.shape
        gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        gc[cv2.dilate(rough.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0] = cv2.GC_PR_BGD
        gc[rough] = cv2.GC_PR_FGD
        core = cv2.erode((rough & plant).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        gc[core] = cv2.GC_FGD
        gc[near_bg & ~rough] = cv2.GC_BGD
        border = np.zeros((h, w), bool)
        border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
        gc[border & ~plant] = cv2.GC_BGD
        if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_BGD).any():
            return rough
        try:
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, self.cfg.grabcut_iters,
                        cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return rough
        out = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        # GrabCut occasionally collapses; fall back to the rough mask if it lost most of the leaf.
        return out if out.sum() > 0.4 * rough.sum() else rough

    @staticmethod
    def _discoloration(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inside the leaf (away from its edge): pixels that are not healthy green, ignoring glare."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        glare = (hsv[..., 2] > 230) & (hsv[..., 1] < 40)
        off = inner & ~healthy_green_mask(hsv) & ~glare
        off = cv2.morphologyEx(off.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(off, connectivity=8)
        min_blob = max(6, int(mask.sum() * 0.0008))
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob
        return keep[lab]

    @staticmethod
    def _not_found(image: RGBImage, reason: str) -> LeafResult:
        return LeafResult(
            mask=np.zeros(image.shape[:2], bool),
            annotated_image=image.copy(),
            found=False,
            message=reason or "No leaf detected.",
        )


def mask_iou(a: Mask, b: Mask) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
