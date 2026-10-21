"""Video page: open a recorded plant video; leaves are outlined and checked as it plays."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.capture import VIDEO_FILE_FILTER
from app.config import APP_NAME, Settings
from app.contracts import format_confidence
from app.ui.camera_view import CameraView
from app.ui.qt_utils import card, label, set_prop
from app.ui.result_panel import DISCLAIMER, GuidanceCard
from app.ui.theme import Theme
from app.video import Verdict, VideoThread, fmt_time

TONE = {"illness": "danger", "healthy": "ok", "unsure": "warn", "none": "muted"}


def make_classifier(settings: Settings):
    """The same pre-trained model as the Scan page, loaded in the video thread."""
    from app.classify import MODEL_FILENAME, load_classifier
    from app.contracts import MockClassifier

    try:
        return load_classifier(settings.models_dir / MODEL_FILENAME, settings.uncertainty_threshold,
                               settings.plant_threshold)
    except Exception:  # noqa: BLE001 - the Scan page already shows the model warning banner
        return MockClassifier(threshold=settings.uncertainty_threshold)


class VideoPage(QWidget):
    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Page")
        self.settings = settings
        self.prefs = QSettings(APP_NAME, APP_NAME)
        self._thread: VideoThread | None = None
        self._duration = 0.0
        self._paused = False
        self._last_frame = None
        self._guidance = None

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(12)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label("Video analysis", "H1"))
        titles.addWidget(label("Open a video of your plants: every leaf is found and checked while it plays",
                               "Muted"))
        left.addLayout(titles)
        self.view = CameraView()
        self.view.set_message("Open a video file (mp4, avi, mov...) to start.")
        left.addWidget(self.view, 1)

        seek_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self._seek)
        self.time_label = label("00:00 / 00:00", "Caption")
        seek_row.addWidget(self.slider, 1)
        seek_row.addWidget(self.time_label)
        left.addLayout(seek_row)

        bar = QHBoxLayout()
        bar.setSpacing(10)
        self.play_btn = QPushButton("Pause")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_pause)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.open_btn = QPushButton("▶  Open video")
        self.open_btn.setObjectName("ScanButton")
        set_prop(self.open_btn, "primary", True)
        self.open_btn.setCursor(Qt.PointingHandCursor)
        self.open_btn.clicked.connect(lambda: self.open_video())
        bar.addWidget(self.play_btn)
        bar.addWidget(self.stop_btn)
        bar.addStretch(1)
        bar.addWidget(self.open_btn)
        left.addLayout(bar)
        root.addLayout(left, 1)

        # ---- right panel: verdict, findings, guidance
        panel = QWidget()
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(420)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        pl.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)
        col = QVBoxLayout(inner)
        col.setContentsMargins(0, 0, 4, 0)
        col.setSpacing(12)

        vcard, vl = card()
        top = QHBoxLayout()
        self.chip = label("Waiting", "Chip")
        set_prop(self.chip, "tone", "muted")
        top.addWidget(self.chip)
        top.addStretch(1)
        self.count_label = label("", "Caption")
        top.addWidget(self.count_label)
        vl.addLayout(top)
        self.title = label("No video yet", "H1", wrap=True)
        vl.addWidget(self.title)
        self.detail = label("LeafScan outlines each leaf it finds and labels it with the plant and its "
                            "condition. The verdict updates while the video plays.", "Muted", wrap=True)
        vl.addWidget(self.detail)
        col.addWidget(vcard)

        fcard, fl = card(margins=14, spacing=6)
        fl.addWidget(label("Findings", "H3"))
        fl.addWidget(label("Conditions seen at least twice", "Caption"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Condition", "Seen at", "Max"])
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMinimumHeight(170)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        fl.addWidget(self.table)
        col.addWidget(fcard)

        self.guidance = GuidanceCard()
        self.guidance.hide()
        col.addWidget(self.guidance)
        col.addWidget(label(DISCLAIMER, "Disclaimer", wrap=True))
        col.addStretch(1)
        root.addWidget(panel)

    # ------------------------------------------------------------------ actions

    def set_theme(self, theme: Theme) -> None:
        self.view.set_theme(theme)

    def open_video(self, path: str | Path | None = None) -> None:
        if path is None:
            start = self.prefs.value("last_video_dir", str(Path.home()))
            path, _ = QFileDialog.getOpenFileName(self, "Open plant video", start, VIDEO_FILE_FILTER)
            if not path:
                return
            self.prefs.setValue("last_video_dir", str(Path(path).parent))
        self.stop()
        self._show_verdict(Verdict("none", "Analysing...", f"{Path(path).name}", []))
        set_prop(self.chip, "tone", "muted")
        self.chip.setText("Analysing")
        self.view.set_message("Loading video...")
        self._paused = False
        self.play_btn.setText("Pause")
        t = VideoThread(path, lambda: make_classifier(self.settings), self)
        t.opened.connect(self._on_opened)
        t.frame_ready.connect(self._on_frame)
        t.summary.connect(self._show_verdict)
        t.ended.connect(self._on_ended)
        t.failed.connect(self._on_failed)
        self._thread = t
        t.start()

    def toggle_pause(self) -> None:
        if self._thread is None:
            return
        self._paused = not self._paused
        self._thread.set_paused(self._paused)
        self.play_btn.setText("Play" if self._paused else "Pause")

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.stop()
            self._thread = None
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.slider.setEnabled(False)

    def _seek(self) -> None:
        if self._thread is not None:
            self._thread.seek(self.slider.value() / 10.0)

    # ------------------------------------------------------------------ signals

    def _on_opened(self, duration: float, fps: float) -> None:
        self._duration = duration
        self.slider.setRange(0, int(duration * 10))
        self.slider.setEnabled(duration > 0)
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def _on_frame(self, frame, t: float) -> None:
        self.view.freeze(frame, "PAUSED" if self._paused else "ANALYSING")
        if not self.slider.isSliderDown():
            self.slider.setValue(int(t * 10))
        self.time_label.setText(f"{fmt_time(t)} / {fmt_time(self._duration)}")

    def _on_ended(self, verdict: Verdict) -> None:
        self._show_verdict(verdict)
        self.count_label.setText(f"{verdict.leaves_analysed} leaf checks - finished")
        self.play_btn.setEnabled(False)
        self._thread = None

    def _on_failed(self, message: str) -> None:
        self.stop()
        set_prop(self.chip, "tone", "danger")
        self.chip.setText("Error")
        self.title.setText("Could not analyse the video")
        self.detail.setText(message)
        self.view.set_message(message)

    def _show_verdict(self, v: Verdict) -> None:
        self.chip.setText({"illness": "Illness detected", "healthy": "Looks healthy", "unsure": "Not sure",
                           "none": "Analysing"}[v.status])
        set_prop(self.chip, "tone", TONE[v.status])
        self.title.setText(v.title)
        self.detail.setText(v.detail)
        if v.leaves_analysed:
            self.count_label.setText(f"{v.leaves_analysed} leaf checks")
        self.table.setRowCount(len(v.findings))
        for row, f in enumerate(v.findings):
            name = QTableWidgetItem(f"{f.plant} - {f.condition}")
            if not f.healthy:
                f_ = name.font()
                f_.setBold(True)
                name.setFont(f_)
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, QTableWidgetItem(f"{fmt_time(f.first_t)}-{fmt_time(f.last_t)}"))
            self.table.setItem(row, 2, QTableWidgetItem(format_confidence(f.max_conf)))
        sick = [f for f in v.findings if not f.healthy]
        if sick:
            if self._guidance is None:
                from app.guidance import GuidanceRepository

                self._guidance = GuidanceRepository()
            g = self._guidance.get(sick[0].class_key)
            if g is not None:
                self.guidance.set_guidance(g)
                self.guidance.show()
        else:
            self.guidance.hide()
