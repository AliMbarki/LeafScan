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
