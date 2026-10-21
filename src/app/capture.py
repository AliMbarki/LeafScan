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


VIDEO_FILE_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm)"
MAX_VIDEO_SIDE = 1280  # bigger frames are shrunk: faster analysis, no visible loss on screen


class VideoSource:
    """A video file read frame by frame as RGB."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            self._cap.release()
            raise InvalidImageError(f"{self.path.name} is not a readable video")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 240 else 25.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = self.frame_count / self.fps if self.frame_count > 0 else 0.0

    def read(self) -> tuple[RGBImage | None, float]:
        """Next frame (RGB) and its time in seconds; (None, t) at the end."""
        ok, frame = self._cap.read()
        t = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not ok or frame is None:
            return None, t
        h, w = frame.shape[:2]
        if max(h, w) > MAX_VIDEO_SIDE:
            s = MAX_VIDEO_SIDE / max(h, w)
            frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        return bgr_to_rgb(frame), t

    def seek(self, seconds: float) -> None:
        self._cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)

    def close(self) -> None:
        self._cap.release()


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
