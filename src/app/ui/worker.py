"""Background analysis: the pipeline is built and run on a dedicated QThread."""
from __future__ import annotations

import logging
import traceback
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot

from app.contracts import InvalidImageError, RGBImage
from app.pipeline import AnalysisPipeline

log = logging.getLogger(__name__)


class _AnalysisWorker(QObject):
    ready = Signal(str, str)  # model_version, warning ("" if none)
    finished = Signal(object)  # AnalysisResult
    failed = Signal(str)

    def __init__(self, factory: Callable[[], tuple[AnalysisPipeline, str]]) -> None:
        super().__init__()
        self._factory = factory
        self._pipeline: AnalysisPipeline | None = None

    @Slot()
    def initialize(self) -> None:
        try:
            self._pipeline, warning = self._factory()
            self.ready.emit(self._pipeline.model_version, warning)
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            log.exception("pipeline init failed")
            self.failed.emit(f"Could not load the analysis pipeline: {exc}")

    @Slot(bool)
    def set_highlight(self, on: bool) -> None:
        if self._pipeline is not None and hasattr(self._pipeline.overlay, "highlight_discoloration"):
            self._pipeline.overlay.highlight_discoloration = on

    @Slot(object)
    def analyze(self, image: RGBImage) -> None:
        if self._pipeline is None:
            self.failed.emit("The analysis pipeline is not ready yet.")
            return
        try:
            self.finished.emit(self._pipeline.analyze(image))
        except InvalidImageError as exc:
            self.failed.emit(f"That image can't be analysed: {exc}")
        except Exception as exc:  # noqa: BLE001
            log.error("analysis failed\n%s", traceback.format_exc())
            self.failed.emit(f"Analysis failed: {exc}")


class AnalysisController(QObject):
    """UI-thread facade. Inference never runs on the UI thread."""

    ready = Signal(str, str)
    finished = Signal(object)
    failed = Signal(str)
    busy_changed = Signal(bool)
    _request = Signal(object)
    _highlight = Signal(bool)

    def __init__(self, factory: Callable[[], tuple[AnalysisPipeline, str]], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._thread.setObjectName("analysis")
        self._worker = _AnalysisWorker(factory)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.initialize)
        self._request.connect(self._worker.analyze)
        self._highlight.connect(self._worker.set_highlight)
        self._worker.ready.connect(self._on_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._busy = False
        self.is_ready = False
        self.model_version = ""

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._thread.quit()
        self._thread.wait(5000)

    @property
    def busy(self) -> bool:
        return self._busy

    def analyze(self, image: RGBImage) -> bool:
        if self._busy:
            return False
        self._set_busy(True)
        self._request.emit(image)
        return True

    def set_highlight(self, on: bool) -> None:
        self._highlight.emit(on)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)

    def _on_ready(self, version: str, warning: str) -> None:
        self.is_ready = True
        self.model_version = version
        self.ready.emit(version, warning)

    def _on_finished(self, result) -> None:
        self._set_busy(False)
        self.finished.emit(result)

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self.failed.emit(message)
