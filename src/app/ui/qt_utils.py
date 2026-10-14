from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from app.contracts import RGBImage


def rgb_to_qimage(image: RGBImage) -> QImage:
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    return QImage(image.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def rgb_to_pixmap(image: RGBImage) -> QPixmap:
    return QPixmap.fromImage(rgb_to_qimage(image))


def label(text: str = "", name: str | None = None, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    return lbl


def card(name: str = "Card", margins: int = 16, spacing: int = 10) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName(name)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(margins, margins, margins, margins)
    lay.setSpacing(spacing)
    return frame, lay


def set_prop(widget: QWidget, name: str, value) -> None:
    """Set a dynamic property and re-polish so the stylesheet picks it up."""
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
