"""Application entry point: ``python -m app`` (or the ``leafscan`` script)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import APP_NAME, Settings
from app.pipeline import AnalysisPipeline, build_mock_pipeline


def build_pipeline(settings: Settings) -> tuple[AnalysisPipeline, str]:
    """Returns (pipeline, warning). Runs on the analysis thread."""
    return build_mock_pipeline(settings.uncertainty_threshold), "Running with mock model (Phase 1)."


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="leafscan", description=f"{APP_NAME} - plant disease scanner")
    p.add_argument("--mock", action="store_true", help="use mock classifier/detector")
    p.add_argument("--no-camera", action="store_true", help="do not open the webcam")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--data-dir", type=Path, help="where the history DB and images are stored")
    p.add_argument("--threshold", type=float, help="confidence below this is reported as 'not sure'")
    p.add_argument("--theme", choices=["dark", "light"])
    p.add_argument("--image", type=Path, help="analyse this image at startup")
    p.add_argument("--page", choices=["scan", "history"], default="scan")
    p.add_argument("--screenshot", type=Path, help=argparse.SUPPRESS)  # dev: grab window and quit
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from app.storage.db import SqliteScanStore
    from app.ui.main_window import MainWindow

    settings = Settings()
    if args.data_dir:
        settings.data_dir = args.data_dir
    if args.threshold is not None:
        settings.uncertainty_threshold = args.threshold
    settings.camera_index = args.camera
    settings.use_camera = not args.no_camera
    settings.use_mocks = args.mock

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    store = SqliteScanStore(settings.db_path, settings.scans_dir)
    win = MainWindow(settings, store, lambda: build_pipeline(settings))
    if args.theme:
        win.set_theme(args.theme)
    win.show_page(1 if args.page == "history" else 0)
    win.show()

    if args.image:
        def _analyse_when_ready(*_):
            win.import_image(args.image)
        win.controller.ready.connect(_analyse_when_ready)
    if args.screenshot:
        def _shot():
            win.grab().save(str(args.screenshot))
            win.close()
        QTimer.singleShot(4000, _shot)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
