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
    p.add_argument("--page", choices=["scan", "history"], default="scan")
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
    win.show_page(1 if args.page == "history" else 0)
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
        if args.image:
            win.controller.finished.connect(lambda *_: QTimer.singleShot(600, _shot))
            win.controller.failed.connect(lambda *_: QTimer.singleShot(600, _shot))
            QTimer.singleShot(60000, _shot)
        else:
            QTimer.singleShot(4000, _shot)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
