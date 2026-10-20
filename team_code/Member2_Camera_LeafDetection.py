"""
LeafScan - Member 2 - Camera and leaf detection (computer vision)
Main technology: OpenCV

Reading copy: the code below is collected from the real project files listed here.
Edit the originals (paths shown in each section), not this file. Files:
  - src/app/capture.py
  - src/app/ui/camera_view.py
  - src/app/vision/leaf.py
  - src/app/vision/overlay.py
  - src/app/vision/__init__.py
"""


# ================================================================================================
# FILE: src/app/capture.py
# ================================================================================================

"""Image acquisition: webcam thread and file import.

This is the ONLY place where OpenCV's BGR images are converted to RGB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from app.contracts import InvalidImageError, RGBImage, validate_rgb

MAX_IMPORT_SIDE = 2000
IMAGE_FILE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"


def bgr_to_rgb(frame: np.ndarray) -> RGBImage:
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def load_image_file(path: str | Path) -> RGBImage:
    """Read an image file (unicode paths OK) as RGB uint8, downscaling huge images."""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise InvalidImageError(f"cannot read {path.name}: {exc}") from exc
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if frame is None:
        raise InvalidImageError(f"{path.name} is not a readable image")
    h, w = frame.shape[:2]
    if max(h, w) > MAX_IMPORT_SIDE:
        s = MAX_IMPORT_SIDE / max(h, w)
        frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return validate_rgb(bgr_to_rgb(frame))


def save_rgb_png(image: RGBImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"could not encode {path.name}")
    buf.tofile(str(path))


class CameraThread(QThread):
    """Grabs frames off the UI thread. Emits RGB frames; never blocks the UI."""

    frame_ready = Signal(object)  # RGBImage
    camera_opened = Signal(int, int)  # width, height
    camera_error = Signal(str)

    def __init__(self, index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.index = index

    def run(self) -> None:
        cap = self._open()
        if cap is None:
            self.camera_error.emit(f"No camera found at index {self.index}.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera_opened.emit(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        failures = 0
        try:
            while not self.isInterruptionRequested():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 50:
                        self.camera_error.emit("The camera stopped sending frames.")
                        return
                    self.msleep(40)
                    continue
                failures = 0
                self.frame_ready.emit(bgr_to_rgb(frame))
                self.msleep(15)
        finally:
            cap.release()

    def _open(self) -> cv2.VideoCapture | None:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
        for backend in backends:
            if self.isInterruptionRequested():
                return None
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened() and cap.read()[0]:
                return cap
            cap.release()
        return None

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(3000)

# ================================================================================================
# FILE: src/app/ui/camera_view.py
# ================================================================================================

"""Viewport that shows either the live camera feed or a frozen (captured/annotated) image."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.contracts import RGBImage
from app.ui.qt_utils import rgb_to_pixmap
from app.ui.theme import DARK, Theme


class CameraView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(360, 240)
        self._pixmap: QPixmap | None = None
        self._last_frame: RGBImage | None = None
        self._frozen = False
        self._message = "Starting camera..."
        self._badge = ""
        self._theme: Theme = DARK

    # ----------------------------------------------------------------- state

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def last_frame(self) -> RGBImage | None:
        return self._last_frame

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def show_frame(self, frame: RGBImage) -> None:
        """Live frame from the camera thread; ignored while a result is frozen on screen."""
        self._last_frame = frame
        if not self._frozen:
            self._pixmap = rgb_to_pixmap(frame)
            self._message = ""
            self.update()

    def freeze(self, image: RGBImage, badge: str = "") -> None:
        self._frozen = True
        self._badge = badge
        self._pixmap = rgb_to_pixmap(image)
        self._message = ""
        self.update()

    def unfreeze(self) -> None:
        self._frozen = False
        self._badge = ""
        self._pixmap = rgb_to_pixmap(self._last_frame) if self._last_frame is not None else None
        if self._pixmap is None and not self._message:
            self._message = "Waiting for camera..."
        self.update()

    def set_message(self, text: str) -> None:
        self._message = text
        if not self._frozen:
            self._pixmap = None
        self.update()

    # --------------------------------------------------------------- painting

    def paintEvent(self, _event) -> None:
        t = self._theme
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        radius = 22
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        p.fillPath(clip, QColor(t.brand))
        p.setClipPath(clip)

        has_image = self._pixmap is not None and not self._pixmap.isNull()
        if has_image:
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        if not self._frozen:
            self._draw_scan_frame(p, t)
        if self._message and not has_image:
            box = self.rect().adjusted(40, 0, -40, 0)
            f = QFont(self.font())
            f.setPointSizeF(30)
            p.setFont(f)
            p.setPen(QColor(t.lime))
            p.drawText(box.adjusted(0, 0, 0, -60), Qt.AlignCenter, "🌿")
            f.setPointSizeF(11)
            p.setFont(f)
            p.setPen(QColor(t.brand_text))
            p.drawText(box.adjusted(0, 70, 0, 0), Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextWordWrap, self._message)

        badge = self._badge or ("LIVE" if has_image and not self._frozen else "")
        if badge:
            f = QFont(self.font())
            f.setBold(True)
            f.setPointSizeF(8.5)
            p.setFont(f)
            w = p.fontMetrics().horizontalAdvance(badge) + 26
            r = QRectF(18, 18, w, 26)
            path = QPainterPath()
            path.addRoundedRect(r, 13, 13)
            p.fillPath(path, QColor(0, 0, 0, 140))
            dot = QColor("#ef6a5f") if badge == "LIVE" else QColor(t.lime)
            p.setBrush(dot)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(r.left() + 10, r.center().y() - 3.5, 7, 7))
            p.setPen(QColor("#ffffff"))
            p.drawText(r.adjusted(18, 0, 0, 0), Qt.AlignCenter, badge)

        p.setClipping(False)
        p.setPen(QColor(t.border))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect, radius, radius)

    def _draw_scan_frame(self, p: QPainter, t: Theme) -> None:
        """Viewfinder corner brackets: shows where to hold the leaf."""
        side = min(self.width(), self.height()) * 0.62
        cx, cy = self.width() / 2, self.height() / 2
        x0, y0, x1, y1 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
        arm = side * 0.14
        pen = QPen(QColor(t.lime), 4)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            path = QPainterPath()
            path.moveTo(x, y + dy * arm)
            path.quadTo(x, y, x + dx * arm, y)
            p.drawPath(path)

# ================================================================================================
# FILE: src/app/vision/leaf.py
# ================================================================================================

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
        """Cascade: isolated leaf on a plain background -> object in the centre -> main green area.

        Only when none of them finds coherent plant tissue is the result "no leaf", so a busy
        background degrades the outline quality instead of blocking the analysis.
        """
        validate_rgb(image)
        H, W = image.shape[:2]
        scale = min(1.0, self.cfg.work_size / max(H, W))
        small = cv2.resize(image, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
        small = cv2.GaussianBlur(small, (3, 3), 0)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

        method = "isolated"
        mask_small, reason = self._segment(small)
        if mask_small is None:
            first_reason = reason
            method = "center"
            mask_small, reason = self._center_object(small, hsv)
            if mask_small is None:
                method = "area"
                mask_small, reason = self._main_green_area(hsv)
            if mask_small is None:  # keep the plain-background step's reason when it is about size
                return self._not_found(image, first_reason if "too small" in first_reason else reason)

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
            method=method,
        )

    # ----------------------------------------------------------------- private

    def _plant_like(self, mask: np.ndarray, hsv: np.ndarray) -> bool:
        if not mask.any():
            return False
        tissue = float(leaf_tissue_mask(hsv)[mask].mean())
        green = float(plant_color_mask(hsv)[mask].mean())
        return tissue >= self.cfg.min_tissue_fraction and green >= self.cfg.min_green_fraction

    def _center_object(self, rgb: np.ndarray, hsv: np.ndarray) -> tuple[np.ndarray | None, str]:
        """GrabCut seeded from the middle of the frame: a leaf held up in front of anything."""
        h, w = hsv.shape[:2]
        tissue = leaf_tissue_mask(hsv)
        green = plant_color_mask(hsv)
        gc = np.full((h, w), cv2.GC_BGD, np.uint8)
        my, mx = int(h * 0.06), int(w * 0.06)
        gc[my : h - my, mx : w - mx] = cv2.GC_PR_BGD
        inner = np.zeros((h, w), bool)
        inner[my : h - my, mx : w - mx] = True
        gc[inner & tissue] = cv2.GC_PR_FGD
        cy0, cy1, cx0, cx1 = int(h * 0.3), int(h * 0.7), int(w * 0.3), int(w * 0.7)
        core = np.zeros((h, w), bool)
        core[cy0:cy1, cx0:cx1] = True
        core &= cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        if core.sum() < 0.01 * h * w:
            return None, "No leaf detected - hold the leaf in the middle of the frame."
        gc[core] = cv2.GC_FGD
        try:
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, self.cfg.grabcut_iters,
                        cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return None, "No leaf detected."
        mask = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        # keep the component that covers the centre seeds
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if n <= 1:
            return None, "No leaf detected."
        overlap = np.bincount(lab[core], minlength=n)
        overlap[0] = 0
        mask = _pad_fill_holes(lab == int(np.argmax(overlap)))
        if not (self.cfg.min_area <= mask.mean() <= 0.9) or not self._plant_like(mask, hsv):
            return None, "No leaf detected."
        return mask, "Busy background - outlined the leaf in the middle of the frame."

    def _main_green_area(self, hsv: np.ndarray) -> tuple[np.ndarray | None, str]:
        """Last resort: the largest coherent patch of plant tissue (foliage, a leaf filling the view)."""
        h, w = hsv.shape[:2]
        veg = leaf_tissue_mask(hsv) & (hsv[..., 1] >= 30)
        veg = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0
        mask = _pad_fill_holes(_largest_component(veg))
        if mask.mean() < self.cfg.min_area or not self._plant_like(mask, hsv):
            return None, "No leaf detected - no plant is visible. Hold one leaf in the middle of the frame."
        return mask, ("Several leaves or a busy background - analysed the main green area. "
                      "For best results, show one leaf against a plain background.")

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

# ================================================================================================
# FILE: src/app/vision/overlay.py
# ================================================================================================

"""Result overlay: dimmed background, leaf outline, optional discoloration tint and a label pill."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.contracts import ClassifierResult, LeafResult, RGBImage, format_confidence
from app.labels import Labels, get_labels

HEALTHY = (62, 207, 142)
DISEASE = (239, 106, 95)
UNSURE = (240, 180, 76)
NEUTRAL = (180, 190, 185)
TINT = (255, 64, 160)  # magenta stands out against green and brown


@lru_cache(maxsize=8)
def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    candidates = []
    if sys.platform == "win32":
        fonts = Path("C:/Windows/Fonts")
        candidates += [fonts / ("segoeuib.ttf" if bold else "segoeui.ttf"), fonts / "arialbd.ttf", fonts / "arial.ttf"]
    candidates += [Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
                   Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


class LeafOverlayRenderer:
    def __init__(self, labels: Labels | None = None, highlight_discoloration: bool = True, dim: float = 0.45) -> None:
        self.labels = labels or get_labels()
        self.highlight_discoloration = highlight_discoloration
        self.dim = dim

    def _status(self, result: ClassifierResult | None, leaf: LeafResult) -> tuple[str, tuple[int, int, int]]:
        if not leaf.found or result is None:
            return "No leaf detected", NEUTRAL
        if result.unknown_plant:
            return result.disease_name, UNSURE
        if result.plant is not None and result.is_uncertain:
            return f"{result.plant} - not sure", UNSURE
        if result.is_uncertain:
            return f"Not sure ({format_confidence(result.confidence)})", UNSURE
        info = self.labels.by_key(result.class_key)
        color = HEALTHY if info and info.healthy else DISEASE
        return f"{result.disease_name}  {format_confidence(result.confidence)}", color

    def render(self, image: RGBImage, leaf: LeafResult, result: ClassifierResult | None) -> RGBImage:
        h, w = image.shape[:2]
        text, color = self._status(result, leaf)
        out = image.astype(np.float32)
        if leaf.found:
            outside = ~leaf.mask
            out[outside] *= 1 - self.dim
            if self.highlight_discoloration and leaf.discoloration is not None and leaf.discoloration.any():
                d = leaf.discoloration
                out[d] = out[d] * 0.45 + np.array(TINT, np.float32) * 0.55
        out = np.clip(out, 0, 255).astype(np.uint8)
        thick = max(2, round(max(h, w) / 300))
        if leaf.found and leaf.contour is not None:
            cv2.drawContours(out, [leaf.contour], -1, color, thick + 1, cv2.LINE_AA)
        return self._draw_label(out, text, color, leaf)

    def _draw_label(self, img: RGBImage, text: str, color, leaf: LeafResult) -> RGBImage:
        h, w = img.shape[:2]
        size = max(14, round(max(h, w) / 34))
        font = _font(size)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil, "RGBA")
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        pad_x, pad_y = round(size * 0.6), round(size * 0.35)
        bw, bh = (r - l) + 2 * pad_x + size, (b - t) + 2 * pad_y
        if leaf.found and leaf.bbox:
            x, y, _, lh = leaf.bbox
            y0 = y - bh - thick_gap(h) if y - bh - thick_gap(h) >= 4 else min(h - bh - 4, y + lh + thick_gap(h))
            x0 = min(max(4, x), max(4, w - bw - 4))
        else:
            x0, y0 = (w - bw) // 2, (h - bh) // 2
        y0 = max(4, y0)
        draw.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=bh // 2, fill=(14, 19, 17, 215))
        dot = size * 0.42
        cy = y0 + bh / 2
        draw.ellipse((x0 + pad_x, cy - dot / 2, x0 + pad_x + dot, cy + dot / 2), fill=tuple(color) + (255,))
        draw.text((x0 + pad_x + size - l, y0 + pad_y - t), text, font=font, fill=(245, 248, 246, 255))
        return np.asarray(pil).copy()


def thick_gap(h: int) -> int:
    return max(6, h // 80)

# ================================================================================================
# FILE: src/app/vision/__init__.py
# ================================================================================================

"""Leaf detection (classical OpenCV) and result overlays."""
from __future__ import annotations

from app.contracts import LeafDetector, LeafResult, OverlayRenderer, RGBImage
from app.vision.leaf import ClassicalLeafDetector, LeafDetectorConfig
from app.vision.overlay import LeafOverlayRenderer


def build_detector() -> LeafDetector:
    return ClassicalLeafDetector()


def build_overlay(highlight_discoloration: bool = True) -> OverlayRenderer:
    return LeafOverlayRenderer(highlight_discoloration=highlight_discoloration)


_detector: LeafDetector | None = None


def detect_leaf(rgb_image: RGBImage) -> LeafResult:
    global _detector
    if _detector is None:
        _detector = build_detector()
    return _detector.detect_leaf(rgb_image)


__all__ = ["ClassicalLeafDetector", "LeafDetectorConfig", "LeafOverlayRenderer", "build_detector", "build_overlay",
           "detect_leaf"]
