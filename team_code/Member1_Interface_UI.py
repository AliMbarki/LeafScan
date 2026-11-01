"""
LeafScan - Member 1 - Interface (UI) and project overview
Main technology: PySide6 / Qt

Reading copy: the code below is collected from the real project files listed here.
Edit the originals (paths shown in each section), not this file. Files:
  - src/app/main.py
  - src/app/ui/main_window.py
  - src/app/ui/theme.py
  - src/app/ui/result_panel.py
  - src/app/ui/plants_dialog.py
  - src/app/ui/qt_utils.py
"""


# ================================================================================================
# FILE: src/app/main.py
# ================================================================================================

"""Application entry point: ``python -m app`` (or the ``leafscan`` script)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import APP_NAME, Settings
from app.pipeline import AnalysisPipeline, build_mock_pipeline


log = logging.getLogger("app")


def build_pipeline(settings: Settings) -> tuple[AnalysisPipeline, str]:
    """Returns (pipeline, warning). Runs on the analysis thread, so loading never blocks the UI.

    If the trained model is missing or broken the app still starts, with the mock
    classifier and a visible warning, so history and the UI stay usable.
    """
    if settings.use_mocks:
        return build_mock_pipeline(settings.uncertainty_threshold), "Mock mode: predictions are fake."

    from app.contracts import MockClassifier
    from app.guidance import GuidanceRepository
    from app.vision import build_detector, build_overlay

    warning = ""
    try:
        from app.classify import MODEL_FILENAME, load_classifier

        classifier = load_classifier(settings.models_dir / MODEL_FILENAME, settings.uncertainty_threshold,
                                     settings.plant_threshold)
    except Exception as exc:  # noqa: BLE001 - ModelLoadError, missing onnxruntime, corrupt file...
        log.warning("classifier unavailable, using mock: %s", exc)
        classifier = MockClassifier(threshold=settings.uncertainty_threshold)
        warning = f"Model unavailable ({exc}). Using a MOCK classifier - predictions are fake."
    pipeline = AnalysisPipeline(build_detector(), classifier, build_overlay(), GuidanceRepository())
    return pipeline, warning


def setup_logging(data_dir: Path) -> None:
    """Console + rotating log file in the data folder; unexpected errors are logged, not lost."""
    from logging.handlers import RotatingFileHandler

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(data_dir / "leafscan.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as exc:
        log.warning("file logging disabled: %s", exc)

    def hook(exc_type, exc, tb):
        log.critical("unhandled exception", exc_info=(exc_type, exc, tb))
        from PySide6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is not None:
            QMessageBox.warning(None, APP_NAME, f"Something went wrong: {exc}\n\nDetails were written to the log file.")

    sys.excepthook = hook


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="leafscan", description=f"{APP_NAME} - plant disease scanner")
    p.add_argument("--mock", action="store_true", help="use mock classifier/detector")
    p.add_argument("--no-camera", action="store_true", help="do not open the webcam")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--data-dir", type=Path, help="where the history DB and images are stored")
    p.add_argument("--threshold", type=float, help="condition confidence below this is reported as 'not sure'")
    p.add_argument("--plant-threshold", type=float, help="plant confidence below this is reported as 'not sure'")
    p.add_argument("--theme", choices=["dark", "light"])
    p.add_argument("--image", type=Path, help="analyse this image at startup")
    p.add_argument("--page", choices=["scan", "history", "video"], default="scan")
    p.add_argument("--video", type=Path, help="open this video on the Video page at startup")
    p.add_argument("--screenshot", type=Path, help=argparse.SUPPRESS)  # dev: grab window and quit
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    from app.storage.db import SqliteScanStore
    from app.ui.main_window import MainWindow

    settings = Settings()
    if args.data_dir:
        settings.data_dir = args.data_dir
    if args.threshold is not None:
        settings.uncertainty_threshold = args.threshold
    if args.plant_threshold is not None:
        settings.plant_threshold = args.plant_threshold
    settings.camera_index = args.camera
    settings.use_camera = not args.no_camera
    settings.use_mocks = args.mock

    setup_logging(settings.data_dir)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    try:
        store = SqliteScanStore(settings.db_path, settings.scans_dir)
    except Exception as exc:  # noqa: BLE001 - unwritable folder, corrupt DB, ...
        log.exception("cannot open history database")
        QMessageBox.critical(
            None,
            APP_NAME,
            f"Cannot open the history database at\n{settings.db_path}\n\n{exc}\n\n"
            "Start with --data-dir pointing to a writable folder.",
        )
        return 1
    win = MainWindow(settings, store, lambda: build_pipeline(settings))
    if args.theme:
        win.set_theme(args.theme)
    win.show_page({"scan": 0, "history": 1, "video": 2}["video" if args.video else args.page])
    if args.video:
        win.video_page.open_video(args.video)
    win.show()

    if args.image:
        def _analyse_when_ready(*_):
            win.import_image(args.image)
        win.controller.ready.connect(_analyse_when_ready)
    if args.screenshot:  # dev aid: capture the window once the startup analysis (if any) is shown
        def _shot():
            if win.isVisible():
                win.grab().save(str(args.screenshot))
                win.close()
        if args.video:
            QTimer.singleShot(12000, _shot)
        elif args.image:
            win.controller.finished.connect(lambda *_: QTimer.singleShot(600, _shot))
            win.controller.failed.connect(lambda *_: QTimer.singleShot(600, _shot))
            QTimer.singleShot(60000, _shot)
        else:
            QTimer.singleShot(4000, _shot)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

# ================================================================================================
# FILE: src/app/ui/main_window.py
# ================================================================================================

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

# ================================================================================================
# FILE: src/app/ui/theme.py
# ================================================================================================

"""Design tokens and the Qt stylesheet generated from them.

Look: a plant-app style (deep forest-green brand panel, soft sage background, white rounded cards,
pill buttons, fresh lime accent), in a light and a dark variant.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    surface: str
    surface2: str
    border: str
    text: str
    muted: str
    accent: str
    accent_hover: str
    on_accent: str
    ok: str
    warn: str
    danger: str
    chart: tuple[str, ...]
    brand: str = "#173f30"  # sidebar / hero green
    brand_text: str = "#eaf3ec"
    brand_muted: str = "#9fbcab"
    lime: str = "#c5f36b"  # highlight accent (active nav, scan frame)
    on_lime: str = "#14311f"


LIGHT = Theme(
    name="light",
    bg="#eef3ee",
    surface="#ffffff",
    surface2="#f2f6f1",
    border="#dde7de",
    text="#15291f",
    muted="#6a7e72",
    accent="#1f5e45",
    accent_hover="#174a36",
    on_accent="#ffffff",
    ok="#2e8b57",
    warn="#c47f12",
    danger="#c8463a",
    chart=("#1f5e45", "#2f7fd1", "#c98a14", "#8a52c7", "#c8463a", "#1f9e93", "#c4508a", "#6f9a22", "#6b7785"),
)

DARK = Theme(
    name="dark",
    bg="#0e1813",
    surface="#15231b",
    surface2="#1c2e24",
    border="#26392f",
    text="#e8f1ea",
    muted="#8fa79a",
    accent="#6cc995",
    accent_hover="#86d6a8",
    on_accent="#0b1a12",
    ok="#6cc995",
    warn="#f0b44c",
    danger="#ef7a6f",
    chart=("#6cc995", "#5aa9f0", "#f0b44c", "#c78bf0", "#ef7a6f", "#4fd1c5", "#f08fc0", "#a3c95a", "#9aa5b1"),
    brand="#0b2a1f",
    brand_text="#e8f1ea",
    brand_muted="#86a596",
)

THEMES = {t.name: t for t in (DARK, LIGHT)}


def rgba(hex_color: str, alpha: float) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {int(alpha * 255)})"


def stylesheet(t: Theme) -> str:
    return f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif; font-size: 10pt; color: {t.text}; }}
QMainWindow, QWidget#Root {{ background: {t.bg}; }}
QWidget#Page {{ background: {t.bg}; }}
QToolTip {{ background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; padding: 6px; border-radius: 6px; }}

/* ---------- brand sidebar ---------- */
QFrame#Sidebar {{ background: {t.brand}; border: none; }}
QFrame#Sidebar QLabel {{ color: {t.brand_text}; background: transparent; }}
QFrame#Sidebar QLabel#Caption {{ color: {t.brand_muted}; }}
QLabel#Brand {{ font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 16pt; font-weight: 700; }}
QLabel#BrandMark {{
    background: {t.lime}; color: {t.on_lime}; border-radius: 12px;
    font-size: 13pt; font-weight: 800; padding: 2px 8px;
}}
QLabel#BrandTag {{ color: {t.brand_muted}; font-size: 8.5pt; }}
QPushButton#NavButton {{
    text-align: left; padding: 11px 16px; border: none; border-radius: 16px;
    background: transparent; color: {t.brand_muted}; font-weight: 600; font-size: 10.5pt;
}}
QPushButton#NavButton:hover {{ background: {rgba(t.brand_text, 0.08)}; color: {t.brand_text}; }}
QPushButton#NavButton:checked {{ background: {t.lime}; color: {t.on_lime}; }}
QFrame#Sidebar QPushButton[flat="true"] {{ color: {t.brand_muted}; }}
QFrame#Sidebar QPushButton[flat="true"]:hover {{ color: {t.brand_text}; }}
QFrame#TipCard {{ background: {rgba(t.brand_text, 0.07)}; border-radius: 16px; }}

/* ---------- cards & type ---------- */
QFrame#Card {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 20px; }}
QFrame#SubCard {{ background: {t.surface2}; border: none; border-radius: 16px; }}
QLabel#H1 {{ font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 20pt; font-weight: 700; }}
QLabel#H2 {{ font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 14pt; font-weight: 700; }}
QLabel#H3 {{ font-size: 10.5pt; font-weight: 700; }}
QLabel#Muted, QLabel#Caption {{ color: {t.muted}; }}
QLabel#Caption {{ font-size: 8.5pt; }}
QLabel#Latin {{ color: {t.muted}; font-style: italic; }}
QLabel#StatValue {{ font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 19pt; font-weight: 700; color: {t.accent}; }}
QLabel#Disclaimer {{
    color: {t.muted}; font-size: 8.5pt; padding: 10px 12px;
    background: {t.surface2}; border-radius: 14px;
}}
QLabel#Banner {{
    background: {rgba(t.warn, 0.14)}; color: {t.warn}; border: 1px solid {rgba(t.warn, 0.45)};
    border-radius: 14px; padding: 10px 14px; font-weight: 600;
}}
QLabel#Chip {{ border-radius: 9px; padding: 4px 12px; font-weight: 700; font-size: 8.5pt; }}
QLabel#Chip[tone="ok"]     {{ background: {rgba(t.ok, 0.16)}; color: {t.ok}; }}
QLabel#Chip[tone="warn"]   {{ background: {rgba(t.warn, 0.16)}; color: {t.warn}; }}
QLabel#Chip[tone="danger"] {{ background: {rgba(t.danger, 0.15)}; color: {t.danger}; }}
QLabel#Chip[tone="muted"]  {{ background: {t.surface2}; color: {t.muted}; }}

/* ---------- buttons & inputs ---------- */
QPushButton {{
    background: {t.surface}; border: 1px solid {t.border}; border-radius: 15px;
    padding: 9px 20px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {t.accent}; color: {t.accent}; }}
QPushButton:disabled {{ color: {t.muted}; border-color: {t.border}; background: {t.surface2}; }}
QPushButton[primary="true"] {{ background: {t.accent}; color: {t.on_accent}; border: none; }}
QPushButton[primary="true"]:hover {{ background: {t.accent_hover}; color: {t.on_accent}; }}
QPushButton[primary="true"]:disabled {{ background: {t.border}; color: {t.muted}; }}
QPushButton[flat="true"] {{ background: transparent; border: none; color: {t.muted}; padding: 6px 10px; }}
QPushButton[flat="true"]:hover {{ color: {t.text}; }}
QPushButton#ScanButton {{ font-size: 11pt; padding: 11px 28px; border-radius: 18px; }}

QComboBox, QLineEdit {{
    background: {t.surface2}; border: 1px solid {t.border}; border-radius: 12px; padding: 8px 12px;
    selection-background-color: {t.accent}; selection-color: {t.on_accent};
}}
QComboBox:focus, QLineEdit:focus {{ border-color: {t.accent}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {t.surface}; border: 1px solid {t.border}; selection-background-color: {t.surface2};
    selection-color: {t.text}; outline: none; padding: 4px;
}}
QCheckBox {{ spacing: 8px; }}

QProgressBar {{ background: {t.surface2}; border: none; border-radius: 5px; height: 10px; }}
QProgressBar::chunk {{ border-radius: 5px; background: {t.accent}; }}
QProgressBar[tone="ok"]::chunk {{ background: {t.accent}; }}
QProgressBar[tone="warn"]::chunk {{ background: {t.warn}; }}
QProgressBar[tone="danger"]::chunk {{ background: {t.danger}; }}

QSlider::groove:horizontal {{ height: 6px; background: {t.border}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {t.surface}; border: 3px solid {t.accent}; width: 12px; height: 12px;
    margin: -6px 0; border-radius: 9px;
}}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t.border}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}

QTableWidget {{
    background: {t.surface}; border: none; gridline-color: transparent; border-radius: 20px;
    selection-background-color: {t.surface2}; selection-color: {t.text}; alternate-background-color: {t.surface};
}}
QTableWidget::item {{ padding: 6px; border-bottom: 1px solid {t.border}; }}
QHeaderView::section {{
    background: {t.surface}; color: {t.muted}; border: none; border-bottom: 1px solid {t.border};
    padding: 10px 6px; font-weight: 600; font-size: 8.5pt;
}}
QStatusBar {{ background: {t.bg}; color: {t.muted}; border: none; }}
QDialog {{ background: {t.bg}; }}
"""


def apply_theme(app: QApplication, theme: Theme) -> None:
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(theme.bg))
    pal.setColor(QPalette.Base, QColor(theme.surface))
    pal.setColor(QPalette.Text, QColor(theme.text))
    pal.setColor(QPalette.WindowText, QColor(theme.text))
    pal.setColor(QPalette.Button, QColor(theme.surface2))
    pal.setColor(QPalette.ButtonText, QColor(theme.text))
    pal.setColor(QPalette.Highlight, QColor(theme.accent))
    pal.setColor(QPalette.HighlightedText, QColor(theme.on_accent))
    pal.setColor(QPalette.Link, QColor(theme.accent))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet(theme))

# ================================================================================================
# FILE: src/app/ui/result_panel.py
# ================================================================================================

"""Right-hand panel: prediction, confidence, guidance card, disclaimer and save controls."""
from __future__ import annotations

import html

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.contracts import AnalysisResult, Guidance, format_confidence
from app.labels import get_labels
from app.ui.qt_utils import card, label, set_prop

DISCLAIMER = (
    "Guidance, not a diagnosis. This tool can be wrong, especially on photos unlike its training data. "
    "Confirm with a local extension service or plant-health expert before treating plants."
)

SEVERITY_TONE = {"none": "ok", "low": "ok", "moderate": "warn", "high": "danger", "info": "muted"}


def _tone(p: float) -> str:
    return "ok" if p >= 0.85 else "warn"


def _bullets(items: tuple[str, ...]) -> str:
    return "".join(f"<li style='margin-bottom:3px'>{html.escape(i)}</li>" for i in items)


class GuidanceCard(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        frame, lay = card("SubCard", margins=14, spacing=8)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        head = QHBoxLayout()
        self.title = label("", "H3", wrap=True)
        self.severity = label("", "Chip")
        head.addWidget(self.title, 1)
        head.addWidget(self.severity, 0, Qt.AlignTop)
        lay.addLayout(head)
        self.body = label("", wrap=True)
        self.body.setTextFormat(Qt.RichText)
        self.body.setOpenExternalLinks(True)
        self.body.setTextInteractionFlags(Qt.TextBrowserInteraction)
        lay.addWidget(self.body)

    def set_guidance(self, g: Guidance) -> None:
        self.title.setText("What to do")
        self.severity.setText({"none": "No action", "info": "Info"}.get(g.severity, f"{g.severity.title()} risk"))
        set_prop(self.severity, "tone", SEVERITY_TONE.get(g.severity, "muted"))
        parts = [f"<p>{html.escape(g.summary)}</p>"]
        for heading, items in (("Look for", g.symptoms), ("Suggested steps", g.actions), ("Prevention", g.prevention)):
            if items:
                parts.append(f"<p style='margin-bottom:2px'><b>{heading}</b></p><ul style='margin-top:0'>{_bullets(items)}</ul>")
        if g.sources:
            links = "<br>".join(
                f"<a href='{html.escape(s.url)}'>{html.escape(s.title)}</a>"
                + (f" <span style='opacity:.7'>({html.escape(s.publisher)})</span>" if s.publisher else "")
                for s in g.sources
            )
            parts.append(f"<p style='margin-bottom:2px'><b>Sources</b></p><p style='margin-top:0'>{links}</p>")
        self.body.setText("".join(parts))


class Meter(QWidget):
    """Caption + value + thin bar, e.g. 'Plant  >99%'."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        head = QHBoxLayout()
        self.caption = label(caption, "Caption")
        head.addWidget(self.caption)
        head.addStretch(1)
        self.value = label("", "H3")
        head.addWidget(self.value)
        lay.addLayout(head)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        lay.addWidget(self.bar)

    def set(self, caption: str, p: float, tone: str) -> None:
        self.caption.setText(caption)
        self.value.setText(format_confidence(p))
        self.bar.setValue(round(p * 100))
        set_prop(self.bar, "tone", tone)


class ResultPanel(QWidget):
    save_requested = Signal(str)  # plant name
    highlight_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.labels = get_labels()
        self.setMinimumWidth(340)
        self.setMaximumWidth(420)
        self._result: AnalysisResult | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)
        col = QVBoxLayout(inner)
        col.setContentsMargins(0, 0, 4, 0)
        col.setSpacing(12)

        # --- result card
        res_card, lay = card()
        top = QHBoxLayout()
        self.chip = label("Ready", "Chip")
        set_prop(self.chip, "tone", "muted")
        top.addWidget(self.chip)
        top.addStretch(1)
        self.timing = label("", "Caption")
        top.addWidget(self.timing)
        lay.addLayout(top)
        self.title = label("Scan a leaf", "H1", wrap=True)
        lay.addWidget(self.title)
        self.subtitle = label("Hold one leaf in the middle of the frame and press Scan leaf (Space), or import a photo.", "Muted", wrap=True)
        lay.addWidget(self.subtitle)

        self.conf_row = QWidget()
        cr = QVBoxLayout(self.conf_row)
        cr.setContentsMargins(0, 6, 0, 0)
        cr.setSpacing(8)
        self.plant_meter = Meter("Plant")
        self.cond_meter = Meter("Condition")
        cr.addWidget(self.plant_meter)
        cr.addWidget(self.cond_meter)
        self.alternatives = label("", "Caption", wrap=True)
        cr.addWidget(self.alternatives)
        lay.addWidget(self.conf_row)
        self.conf_row.hide()
        self.disc_label = label("", "Caption", wrap=True)
        self.disc_label.hide()
        lay.addWidget(self.disc_label)
        self.scene_hint = label("", "Caption", wrap=True)
        self.scene_hint.hide()
        lay.addWidget(self.scene_hint)
        self.highlight = QCheckBox("Highlight discoloration")
        self.highlight.setToolTip("Tint off-colour areas inside the leaf (yellowing, spots, lesions)")
        self.highlight.toggled.connect(self.highlight_toggled)
        lay.addWidget(self.highlight)
        col.addWidget(res_card)

        # --- guidance
        self.guidance = GuidanceCard()
        self.guidance.hide()
        col.addWidget(self.guidance)

        # --- save
        save_card, slay = card(spacing=8)
        slay.addWidget(label("Save to history", "H3"))
        slay.addWidget(label("Pick a plant or type a new name.", "Caption"))
        row = QHBoxLayout()
        self.plant_combo = QComboBox()
        self.plant_combo.setEditable(True)
        self.plant_combo.setInsertPolicy(QComboBox.NoInsert)
        self.plant_combo.lineEdit().setPlaceholderText("Plant name")
        row.addWidget(self.plant_combo, 1)
        self.save_btn = QPushButton("Save scan")
        set_prop(self.save_btn, "primary", True)
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip("Save this scan (Ctrl+S)")
        self.save_btn.clicked.connect(self._emit_save)
        row.addWidget(self.save_btn)
        slay.addLayout(row)
        self.save_note = label("", "Caption", wrap=True)
        slay.addWidget(self.save_note)
        col.addWidget(save_card)

        self.disclaimer = label(DISCLAIMER, "Disclaimer", wrap=True)
        col.addWidget(self.disclaimer)
        col.addStretch(1)

    # ---------------------------------------------------------------- public

    @property
    def result(self) -> AnalysisResult | None:
        return self._result

    def set_plants(self, names: list[str], suggestion: str) -> None:
        current = self.plant_combo.currentText().strip()
        self.plant_combo.blockSignals(True)
        self.plant_combo.clear()
        self.plant_combo.addItems(names)
        if suggestion not in names:
            self.plant_combo.addItem(f"{suggestion}")
        self.plant_combo.setCurrentText(current or suggestion)
        self.plant_combo.blockSignals(False)
        self._suggestion = suggestion

    def show_idle(self) -> None:
        self.disc_label.hide()
        self.scene_hint.hide()
        self._result = None
        self._set_chip("Ready", "muted")
        self.title.setText("Scan a leaf")
        self.subtitle.setText("Hold one leaf in the middle of the frame and press Scan leaf (Space), or import a photo.")
        self.timing.setText("")
        self.conf_row.hide()
        self.guidance.hide()
        self.save_btn.setEnabled(False)
        self.save_note.setText("")

    def show_busy(self) -> None:
        self.disc_label.hide()
        self.scene_hint.hide()
        self._set_chip("Analyzing", "muted")
        self.title.setText("Analyzing...")
        self.subtitle.setText("Finding the leaf and classifying it.")
        self.conf_row.hide()
        self.guidance.hide()
        self.save_btn.setEnabled(False)
        self.save_note.setText("")

    def show_error(self, message: str) -> None:
        self.disc_label.hide()
        self.scene_hint.hide()
        self._result = None
        self._set_chip("Error", "danger")
        self.title.setText("Something went wrong")
        self.subtitle.setText(message)
        self.timing.setText("")
        self.conf_row.hide()
        self.guidance.hide()
        self.save_btn.setEnabled(False)

    def show_result(self, result: AnalysisResult) -> None:
        self._result = result
        self.timing.setText(f"{result.elapsed_ms:.0f} ms")
        self.save_btn.setEnabled(True)
        self.save_note.setText("")
        leaf = result.leaf
        if leaf.found and leaf.discoloration is not None:
            self.disc_label.setText(
                f"Off-colour area: {leaf.discoloration_fraction:.0%} of the leaf"
                + (" (tinted pink)" if self.highlight.isChecked() else "")
            )
            self.disc_label.show()
        else:
            self.disc_label.hide()
        c = result.classification
        hint = leaf.message if leaf.found and leaf.method != "isolated" else ""
        self.scene_hint.setText(hint)
        self.scene_hint.setVisible(bool(hint))
        if c is None:
            self._set_chip("No leaf", "warn")
            self.title.setText("No leaf detected")
            self.subtitle.setText(
                result.leaf.message or "Hold one leaf in the middle of the frame, ideally against a plain background."
            )
            self.conf_row.hide()
            self.guidance.hide()
            return

        self.conf_row.show()
        others = [p for p in c.top_k if p.class_id != c.class_id][:2] if not c.is_uncertain else list(c.top_k[:3])
        others = [p for p in others if p.probability >= 0.02]  # 0% alternatives are noise
        prefix = "Best guesses: " if c.is_uncertain else "Also possible: "
        self.alternatives.setText(
            prefix + ", ".join(f"{p.name} ({format_confidence(p.probability)})" for p in others) if others else ""
        )

        n_plants = len(self.labels.plants)
        if c.unknown_plant:
            self._set_chip(c.disease_name, "warn")
            self.title.setText(c.disease_name)
            self.subtitle.setText(f"This does not look like any of the {n_plants} plants LeafScan knows "
                                  "(see Supported plants), so no condition is given.")
            self.plant_meter.set("Not a known plant", c.plant_confidence, "warn")
            self.cond_meter.hide()
        elif c.plant is None:
            self._set_chip("Not sure", "warn")
            self.title.setText("Not sure")
            self.subtitle.setText("Could not tell which plant this is. Fill the frame with one leaf, in even light, "
                                  "and keep it sharp.")
            self.plant_meter.set("Plant", c.plant_confidence, "warn")
            self.cond_meter.hide()
        elif c.is_uncertain:
            self._set_chip("Not sure", "warn")
            self.title.setText(c.plant)
            self.subtitle.setText(f"{c.plant} leaf identified, but the condition is unclear. "
                                  "Try a closer, sharper photo of the affected area.")
            self.plant_meter.set(f"Plant: {c.plant}", c.plant_confidence, _tone(c.plant_confidence))
            self.cond_meter.set("Condition: not sure", c.confidence, "warn")
            self.cond_meter.show()
        else:
            info = self.labels.by_key(c.class_key)
            healthy = bool(info and info.healthy)
            condition = info.condition if info else c.disease_name
            self._set_chip("Looks healthy" if healthy else "Possible disease", "ok" if healthy else "danger")
            self.title.setText(condition)
            self.subtitle.setText(self._plant_line(c.plant))
            self.plant_meter.set(f"Plant: {c.plant}", c.plant_confidence, _tone(c.plant_confidence))
            self.cond_meter.set(f"Condition: {condition}", c.confidence, _tone(c.confidence))
            self.cond_meter.show()

        if result.guidance is not None:
            self.guidance.set_guidance(result.guidance)
            self.guidance.show()
        else:
            self.guidance.hide()

    def mark_saved(self, plant_name: str) -> None:
        self.save_btn.setEnabled(False)
        self.save_note.setText(f"Saved to {plant_name}.")

    def current_plant_name(self) -> str:
        return self.plant_combo.currentText().strip()

    # --------------------------------------------------------------- private

    def _emit_save(self) -> None:
        name = self.current_plant_name()
        if not name:
            self.save_note.setText("Enter a plant name first.")
            return
        self.save_requested.emit(name)

    def _plant_line(self, plant: str) -> str:
        info = self.labels.plant_info.get(plant)
        return f"{plant} leaf  ·  {info.latin}" if info and info.latin else f"{plant} leaf"

    def _set_chip(self, text: str, tone: str) -> None:
        self.chip.setText(text)
        set_prop(self.chip, "tone", tone)

# ================================================================================================
# FILE: src/app/ui/plants_dialog.py
# ================================================================================================

"""'Supported plants' dialog: every plant LeafScan can identify and the conditions it knows."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHeaderView, QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout

from app.labels import Labels, get_labels
from app.ui.qt_utils import label


class SupportedPlantsDialog(QDialog):
    def __init__(self, labels: Labels | None = None, parent=None) -> None:
        super().__init__(parent)
        self.labels = labels or get_labels()
        plants = self.labels.plants
        self.setWindowTitle("Supported plants")
        self.resize(760, 620)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 14)
        lay.addWidget(label(f"{len(plants)} plants", "H2"))
        lay.addWidget(label(
            "LeafScan identifies these plants from a leaf photo and checks for the listed conditions. "
            "Leaves of other plants are reported as 'Not sure' or matched to the closest known plant.",
            "Muted", wrap=True))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter, e.g. tomato or mildew")
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)

        self.table = QTableWidget(len(plants), 3)
        self.table.setHorizontalHeaderLabels(["Plant", "Scientific name", "Conditions it can detect"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setWordWrap(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        for row, plant in enumerate(plants):
            info = self.labels.plant_info[plant]
            conds = [c.condition for c in self.labels.classes_of(plant)]
            conds.sort(key=lambda c: (c != "Healthy", c))
            self.table.setItem(row, 0, QTableWidgetItem(plant))
            latin = QTableWidgetItem(info.latin)
            f = latin.font()
            f.setItalic(True)
            latin.setFont(f)
            self.table.setItem(row, 1, latin)
            self.table.setItem(row, 2, QTableWidgetItem(", ".join(conds) if conds != ["Healthy"]
                                                        else "Healthy (identification only)"))
        self.table.resizeRowsToContents()
        lay.addWidget(self.table, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _filter(self, text: str) -> None:
        t = text.strip().lower()
        for row in range(self.table.rowCount()):
            hay = " ".join(self.table.item(row, c).text().lower() for c in range(3))
            self.table.setRowHidden(row, bool(t) and t not in hay)

# ================================================================================================
# FILE: src/app/ui/qt_utils.py
# ================================================================================================

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
