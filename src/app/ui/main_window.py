"""Main window: sidebar navigation, scan page, history page."""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.capture import IMAGE_FILE_FILTER, CameraThread, load_image_file
from app.config import APP_NAME, Settings
from app.contracts import AnalysisResult, InvalidImageError, RGBImage, ScanRecord
from app.pipeline import AnalysisPipeline
from app.storage.db import SqliteScanStore
from app.ui.camera_view import CameraView
from app.ui.history_view import HistoryView
from app.ui.qt_utils import label, set_prop
from app.ui.result_panel import ResultPanel
from app.ui.theme import THEMES, Theme, apply_theme
from app.ui.worker import AnalysisController

log = logging.getLogger(__name__)


class ScanPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Page")
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(12)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label("Scan a leaf", "H1"))
        titles.addWidget(label("Find out which plant it is and how healthy it looks", "Muted"))
        head.addLayout(titles)
        head.addStretch(1)
        self.hint = label("Space  scan     Ctrl+O  import     Esc  live", "Caption")
        head.addWidget(self.hint, 0, Qt.AlignBottom)
        left.addLayout(head)
        self.banner = label("", "Banner", wrap=True)
        self.banner.hide()
        left.addWidget(self.banner)
        self.camera = CameraView()
        left.addWidget(self.camera, 1)

        bar = QHBoxLayout()
        bar.setSpacing(10)
        self.live_btn = QPushButton("Back to live")
        self.import_btn = QPushButton("Import image")
        self.capture_btn = QPushButton("◉  Scan leaf")
        self.capture_btn.setObjectName("ScanButton")
        set_prop(self.capture_btn, "primary", True)
        self.capture_btn.setMinimumWidth(170)
        self.capture_btn.setCursor(Qt.PointingHandCursor)
        self.retry_btn = QPushButton("Retry camera")
        self.retry_btn.hide()
        bar.addWidget(self.live_btn)
        bar.addWidget(self.retry_btn)
        bar.addStretch(1)
        bar.addWidget(self.import_btn)
        bar.addWidget(self.capture_btn)
        left.addLayout(bar)
        root.addLayout(left, 1)

        self.result_panel = ResultPanel()
        root.addWidget(self.result_panel)


class MainWindow(QMainWindow):
    def __init__(
        self,
        settings: Settings,
        store: SqliteScanStore,
        pipeline_factory: Callable[[], tuple[AnalysisPipeline, str]],
    ) -> None:
        super().__init__()
        self.settings = settings
        self.store = store
        self.prefs = QSettings(APP_NAME, APP_NAME)
        self.setWindowTitle(APP_NAME)
        self.resize(1360, 840)
        self.setMinimumSize(1000, 660)
        self._camera: CameraThread | None = None
        self._camera_ok = False
        self._source_name = ""

        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.scan_page = ScanPage()
        self.history = HistoryView(store)
        from app.ui.video_page import VideoPage

        self.video_page = VideoPage(settings)
        self.stack.addWidget(self.scan_page)  # 0
        self.stack.addWidget(self.history)  # 1
        self.stack.addWidget(self.video_page)  # 2
        outer.addWidget(self.stack, 1)
        self.statusBar().showMessage("Loading model...")

        sp = self.scan_page
        sp.capture_btn.clicked.connect(self.capture)
        sp.import_btn.clicked.connect(lambda: self.import_image())
        sp.live_btn.clicked.connect(self.back_to_live)
        sp.retry_btn.clicked.connect(self.start_camera)
        sp.result_panel.save_requested.connect(self.save_current)
        sp.result_panel.highlight_toggled.connect(self.set_highlight)
        self.history.changed.connect(self._refresh_plants)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._space)
        QShortcut(QKeySequence.Open, self, activated=lambda: self.import_image())
        QShortcut(QKeySequence.Save, self, activated=lambda: sp.result_panel.save_btn.click())
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.back_to_live)

        self.controller = AnalysisController(pipeline_factory, self)
        self.controller.ready.connect(self._on_ready)
        self.controller.finished.connect(self._on_result)
        self.controller.failed.connect(self._on_failed)
        self.controller.busy_changed.connect(self._update_controls)

        self.set_theme(self.prefs.value("theme", "light"))
        self._renderer = None
        highlight = str(self.prefs.value("highlight_discoloration", "true")).lower() == "true"
        sp.result_panel.highlight.setChecked(highlight)
        self.controller.set_highlight(highlight)
        self._refresh_plants()
        self.history.refresh()
        self._update_controls()
        self.controller.start()
        if settings.use_camera:
            QTimer.singleShot(0, self.start_camera)
        else:
            sp.camera.set_message("Camera disabled. Use Import image to analyse a photo.")
            self.camera_status.setText("Camera: off")

    # --------------------------------------------------------------- layout

    def _build_sidebar(self) -> QFrame:
        side = QFrame()
        side.setObjectName("Sidebar")
        side.setFixedWidth(236)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(18, 24, 18, 18)
        lay.setSpacing(6)
        brand = QHBoxLayout()
        brand.setSpacing(10)
        brand.addWidget(label("🌿", "BrandMark"))
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(label(APP_NAME, "Brand"))
        names.addWidget(label("Plant health scanner", "BrandTag"))
        brand.addLayout(names)
        brand.addStretch(1)
        lay.addLayout(brand)
        lay.addSpacing(26)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        for i, text in ((0, "◎   Scan"), (2, "▶   Video"), (1, "▤   History")):
            b = QPushButton(text)
            b.setObjectName("NavButton")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.nav_group.addButton(b, i)
            lay.addWidget(b)
        self.nav_group.button(0).setChecked(True)
        self.nav_group.idClicked.connect(self.show_page)
        plants_btn = QPushButton("❀   Supported plants")
        plants_btn.setObjectName("NavButton")
        plants_btn.setCursor(Qt.PointingHandCursor)
        plants_btn.setToolTip("Which plants and conditions LeafScan knows")
        plants_btn.clicked.connect(self.show_supported_plants)
        lay.addWidget(plants_btn)
        lay.addStretch(1)
        tip = QFrame()
        tip.setObjectName("TipCard")
        tl = QVBoxLayout(tip)
        tl.setContentsMargins(14, 12, 14, 12)
        tl.setSpacing(4)
        tl.addWidget(label("Tip", "H3"))
        tl.addWidget(label("Hold one leaf in the middle of the frame, in even light. A plain background "
                           "gives the cleanest outline.", "Caption", wrap=True))
        lay.addWidget(tip)
        lay.addSpacing(10)

        self.camera_status = label("Camera: starting", "Caption", wrap=True)
        self.model_status = label("Model: loading", "Caption", wrap=True)
        lay.addWidget(self.camera_status)
        lay.addWidget(self.model_status)
        lay.addSpacing(8)
        self.theme_btn = QPushButton("Light mode")
        set_prop(self.theme_btn, "flat", True)
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.theme_btn.clicked.connect(self.toggle_theme)
        lay.addWidget(self.theme_btn, 0, Qt.AlignLeft)
        return side

    def show_supported_plants(self) -> None:
        from app.ui.plants_dialog import SupportedPlantsDialog

        SupportedPlantsDialog(parent=self).exec()

    def show_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.nav_group.button(index).setChecked(True)
        if index == 1:
            self.history.refresh()

    # ---------------------------------------------------------------- theme

    def set_theme(self, name: str) -> None:
        theme: Theme = THEMES.get(name, THEMES["dark"])
        self.theme = theme
        apply_theme(QApplication.instance(), theme)
        self.prefs.setValue("theme", theme.name)
        self.theme_btn.setText("Light mode" if theme.name == "dark" else "Dark mode")
        self.scan_page.camera.set_theme(theme)
        self.video_page.set_theme(theme)
        self.history.set_theme(theme)

    def toggle_theme(self) -> None:
        self.set_theme("light" if self.theme.name == "dark" else "dark")

    # --------------------------------------------------------------- camera

    def start_camera(self) -> None:
        if self._camera is not None and self._camera.isRunning():
            return
        self.scan_page.retry_btn.hide()
        self.camera_status.setText("Camera: starting")
        if not self.scan_page.camera.frozen:
            self.scan_page.camera.set_message("Starting camera...")
        self._camera = CameraThread(self.settings.camera_index, self)
        self._camera.frame_ready.connect(self.scan_page.camera.show_frame)
        self._camera.camera_opened.connect(self._on_camera_opened)
        self._camera.camera_error.connect(self._on_camera_error)
        self._camera.start()

    def _on_camera_opened(self, w: int, h: int) -> None:
        self._camera_ok = True
        self.camera_status.setText(f"Camera: {w}x{h}")
        self._update_controls()

    def _on_camera_error(self, message: str) -> None:
        self._camera_ok = False
        self.camera_status.setText("Camera: unavailable")
        self.scan_page.camera.set_message(
            f"{message}\nConnect a webcam (or start your phone-webcam app) and press Retry camera, "
            "or import an image."
        )
        self.scan_page.retry_btn.show()
        self.statusBar().showMessage(message, 8000)
        self._update_controls()

    # ------------------------------------------------------------- analysis

    def _space(self) -> None:
        if self.stack.currentIndex() == 0:
            if self.scan_page.camera.frozen:
                self.back_to_live()
            else:
                self.capture()

    def capture(self) -> None:
        frame = self.scan_page.camera.last_frame
        if frame is None:
            self.statusBar().showMessage("No camera frame available - import an image instead.", 5000)
            return
        self._source_name = "camera"
        self.analyze(frame.copy())

    def import_image(self, path: str | Path | None = None) -> None:
        if path is None:
            start = self.prefs.value("last_import_dir", str(Path.home()))
            path, _ = QFileDialog.getOpenFileName(self, "Import leaf image", start, IMAGE_FILE_FILTER)
            if not path:
                return
            self.prefs.setValue("last_import_dir", str(Path(path).parent))
        try:
            image = load_image_file(path)
        except InvalidImageError as exc:
            self.scan_page.result_panel.show_error(str(exc))
            self.statusBar().showMessage(str(exc), 6000)
            return
        self._source_name = Path(path).name
        self.analyze(image)

    def analyze(self, image: RGBImage) -> None:
        if not self.controller.is_ready:
            self.statusBar().showMessage("The model is still loading...", 4000)
            return
        if self.controller.analyze(image):
            self.scan_page.camera.freeze(image, "ANALYZING")
            self.scan_page.result_panel.show_busy()

    def _on_ready(self, version: str, warning: str) -> None:
        self.model_status.setText(f"Model: {version}")
        if warning:
            self.scan_page.banner.setText(warning)
            self.scan_page.banner.show()
        self.statusBar().showMessage(warning or "Ready.", 10000 if warning else 3000)
        self._update_controls()

    def _on_result(self, result: AnalysisResult) -> None:
        self.scan_page.camera.freeze(result.overlay_image, "RESULT")
        self.scan_page.result_panel.show_result(result)
        name = result.classification.disease_name if result.classification else "No leaf detected"
        self.statusBar().showMessage(f"{self._source_name}: {name}  ({result.elapsed_ms:.0f} ms)", 6000)

    def _on_failed(self, message: str) -> None:
        if not self.controller.is_ready:
            self.model_status.setText("Model: failed to load")
        self.scan_page.result_panel.show_error(message)
        self.statusBar().showMessage(message, 8000)

    def set_highlight(self, on: bool) -> None:
        """Toggle the discoloration tint; re-render the current overlay (drawing only, no inference)."""
        self.prefs.setValue("highlight_discoloration", "true" if on else "false")
        self.controller.set_highlight(on)
        panel = self.scan_page.result_panel
        result = panel.result
        if result is None or self.controller.busy:
            return
        if self._renderer is None:
            from app.vision import LeafOverlayRenderer

            self._renderer = LeafOverlayRenderer()
        self._renderer.highlight_discoloration = on
        overlay = self._renderer.render(result.image, result.leaf, result.classification)
        result = dataclasses.replace(result, overlay_image=overlay)
        saved_note = panel.save_note.text()
        can_save = panel.save_btn.isEnabled()
        panel.show_result(result)
        panel.save_btn.setEnabled(can_save)
        panel.save_note.setText(saved_note)
        self.scan_page.camera.freeze(overlay, "RESULT")

    def back_to_live(self) -> None:
        if self.controller.busy:
            return
        self.scan_page.camera.unfreeze()
        self.scan_page.result_panel.show_idle()

    def _update_controls(self, *_):
        busy = self.controller.busy
        ready = self.controller.is_ready
        sp = self.scan_page
        sp.capture_btn.setEnabled(ready and not busy and self._camera_ok)
        sp.import_btn.setEnabled(ready and not busy)
        sp.live_btn.setEnabled(not busy)
        sp.live_btn.setText("Back to live" if self._camera_ok else "Clear")

    # ----------------------------------------------------------------- save

    def save_current(self, plant_name: str) -> None:
        result = self.scan_page.result_panel.result
        if result is None:
            return
        try:
            plant = self.store.get_or_create_plant(plant_name)
            scan_id = self.store.save_scan(ScanRecord.from_analysis(result, plant.id))
        except Exception as exc:  # noqa: BLE001 - surface any disk/DB problem
            log.exception("save failed")
            self.statusBar().showMessage(f"Could not save scan: {exc}", 8000)
            return
        self.scan_page.result_panel.mark_saved(plant.name)
        self.statusBar().showMessage(f"Saved scan #{scan_id} to {plant.name}.", 5000)
        self._refresh_plants(keep=plant.name)
        self.history.refresh()

    def _refresh_plants(self, keep: str | None = None) -> None:
        names = [p.name for p in self.store.list_plants()]
        panel = self.scan_page.result_panel
        panel.set_plants(names, self.store.suggest_plant_name())
        if keep:
            panel.plant_combo.setCurrentText(keep)

    # ---------------------------------------------------------------- close

    def closeEvent(self, event) -> None:
        self.video_page.stop()
        if self._camera is not None:
            self._camera.stop()
        self.controller.shutdown()
        self.store.close()
        super().closeEvent(event)
