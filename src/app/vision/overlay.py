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
