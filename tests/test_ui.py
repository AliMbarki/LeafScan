"""Headless end-to-end: import -> analyse (worker thread) -> save -> history."""
import pytest

from app.capture import save_rgb_png
from app.config import Settings
from app.main import build_pipeline
from app.storage.db import SqliteScanStore
from app.ui.main_window import MainWindow


@pytest.fixture
def window(qtbot, tmp_path):
    settings = Settings(data_dir=tmp_path, use_camera=False, use_mocks=True, uncertainty_threshold=0.0)
    store = SqliteScanStore(settings.db_path, settings.scans_dir)
    win = MainWindow(settings, store, lambda: build_pipeline(settings))
    qtbot.addWidget(win)
    qtbot.waitUntil(lambda: win.controller.is_ready, timeout=60000)
    yield win
    win.close()


def test_import_analyse_save_history(window, qtbot, tmp_path, leaf_image):
    path = tmp_path / "leaf.png"
    save_rgb_png(leaf_image, path)
    with qtbot.waitSignal(window.controller.finished, timeout=30000) as sig:
        window.import_image(path)
    result = sig.args[0]
    panel = window.scan_page.result_panel
    assert panel.result is result and panel.save_btn.isEnabled()
    assert panel.current_plant_name() == "Plant 1"

    panel.save_btn.click()
    scans = window.store.list_scans()
    assert len(scans) == 1 and scans[0].plant_name == "Plant 1"
    assert not panel.save_btn.isEnabled()  # cannot double-save
    assert window.history.table.rowCount() == 1
    assert window.store.suggest_plant_name() == "Plant 2"


def test_bad_import_shows_error(window, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    window.import_image(bad)
    assert window.scan_page.result_panel.chip.text() == "Error"


def test_theme_toggle(window):
    before = window.theme.name
    window.toggle_theme()
    assert window.theme.name != before
    window.toggle_theme()
