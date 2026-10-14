"""Viewport that shows either the live camera feed or a frozen (captured/annotated) image."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPixmap
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
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        clip = QPainterPath()
        clip.addRoundedRect(rect, 14, 14)
        p.fillPath(clip, QColor("#0b0f0d") if t.name == "dark" else QColor("#e4e9e6"))
        p.setClipPath(clip)

        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        if self._message and self._pixmap is None:
            p.setPen(QColor(t.muted))
            f = QFont(self.font())
            f.setPointSizeF(11)
            p.setFont(f)
            p.drawText(self.rect().adjusted(24, 24, -24, -24), Qt.AlignCenter | Qt.TextWordWrap, self._message)

        badge = self._badge or ("LIVE" if self._pixmap is not None and not self._frozen else "")
        if badge:
            f = QFont(self.font())
            f.setBold(True)
            f.setPointSizeF(8.5)
            p.setFont(f)
            w = p.fontMetrics().horizontalAdvance(badge) + 22
            r = QRectF(14, 14, w, 24)
            path = QPainterPath()
            path.addRoundedRect(r, 12, 12)
            p.fillPath(path, QColor(0, 0, 0, 150))
            dot = QColor("#ef6a5f") if badge == "LIVE" else QColor(t.accent)
            p.setBrush(dot)
            p.setPen(Qt.NoPen)
            p.drawEllipse(r.left() + 8, r.center().y() - 3, 6, 6)
            p.setPen(QColor("#ffffff"))
            p.drawText(r.adjusted(16, 0, 0, 0), Qt.AlignCenter, badge)

        p.setClipping(False)
        p.setPen(QColor(t.border))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect, 14, 14)
