"""Inference preprocessing, shared by the app predictor and the evaluation scripts.

Pure numpy/OpenCV so the exact same pixels reach the network at eval and in the app.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.contracts import RGBImage

IMG_SIZE = 224
RESIZE_TO = 256  # shorter side before the centre crop (crop ratio 0.875)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet
STD = np.array([0.229, 0.224, 0.225], np.float32)


def resize_center_crop(image: RGBImage, resize_to: int = RESIZE_TO, size: int = IMG_SIZE) -> RGBImage:
    h, w = image.shape[:2]
    s = resize_to / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    return resized[y0 : y0 + size, x0 : x0 + size]


def to_chw_normalized(image: RGBImage) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess(image: RGBImage) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 3x224x224, normalised."""
    return to_chw_normalized(resize_center_crop(image))
