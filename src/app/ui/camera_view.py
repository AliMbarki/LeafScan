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
