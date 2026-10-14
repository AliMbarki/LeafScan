"""History screen: per-plant filter, summary tiles, charts and the scan list."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.contracts import StoredScan, format_confidence
from app.labels import get_labels
from app.storage.db import SqliteScanStore
from app.ui.charts import ConditionsPerPlantChart, ScansOverTimeChart
from app.ui.qt_utils import card, label, set_prop
from app.ui.theme import DARK, Theme

MAX_ROWS = 300
THUMB = 56


class StatTile(QFrame):
    def __init__(self, caption: str) -> None:
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        lay.addWidget(label(caption, "Caption"))
        self.value = label("-", "StatValue")
        lay.addWidget(self.value)
        self.sub = label("", "Caption", wrap=True)
        lay.addWidget(self.sub)

    def set(self, value: str, sub: str = "") -> None:
        self.value.setText(value)
        self.sub.setText(sub)


class ScanDetailDialog(QDialog):
    def __init__(self, scan: StoredScan, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        labels = get_labels()
        self.setWindowTitle(f"Scan #{scan.id} - {scan.plant_name}")
        self.resize(820, 640)
        lay = QVBoxLayout(self)
        img = QLabel()
        img.setAlignment(Qt.AlignCenter)
        pm = QPixmap(scan.overlay_path or scan.image_path)
        img.setPixmap(pm.scaled(780, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation) if not pm.isNull() else QPixmap())
        if pm.isNull():
            img.setText("Image file is missing.")
        lay.addWidget(img, 1)
        lay.addWidget(label(labels.display_name(scan.prediction), "H2"))
        lay.addWidget(
            label(
                f"{scan.plant_name}  |  {scan.timestamp:%Y-%m-%d %H:%M}  |  confidence {format_confidence(scan.confidence)}"
                f"  |  model {scan.model_version}",
                "Muted",
            )
        )
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class HistoryView(QWidget):
    changed = Signal()  # emitted after a scan is deleted

    def __init__(self, store: SqliteScanStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Page")
        self.store = store
        self.labels = get_labels()
        self.theme: Theme = DARK
        self._scans: list[StoredScan] = []
        self._thumbs: dict[str, QIcon] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(label("History", "H1"))
        head.addStretch(1)
        head.addWidget(label("Plant", "Muted"))
        self.plant_filter = QComboBox()
        self.plant_filter.setMinimumWidth(200)
        self.plant_filter.currentIndexChanged.connect(lambda _: self.refresh(keep_filter=True))
        head.addWidget(self.plant_filter)
        root.addLayout(head)

        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.t_scans = StatTile("Scans")
        self.t_plants = StatTile("Plants")
        self.t_top = StatTile("Most frequent result")
        self.t_top.setToolTip("Most common confident result ('Not sure' scans are excluded)")
        self.t_last = StatTile("Last scan")
        for t in (self.t_scans, self.t_plants, self.t_top, self.t_last):
            tiles.addWidget(t, 1)
        root.addLayout(tiles)

        self.charts_row = QHBoxLayout()
        self.charts_row.setSpacing(12)
        root.addLayout(self.charts_row, 4)
        self._build_charts()

        table_card, tl = card(margins=0, spacing=0)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["", "Date", "Plant", "Result", "Confidence", "Model"])
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setIconSize(QSize(THUMB, THUMB))
        self.table.verticalHeader().setDefaultSectionSize(THUMB + 10)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.doubleClicked.connect(lambda _: self.open_selected())
        self.table.itemSelectionChanged.connect(self._update_buttons)
        tl.addWidget(self.table)
        bar = QHBoxLayout()
        bar.setContentsMargins(12, 8, 12, 10)
        self.count_label = label("", "Caption")
        bar.addWidget(self.count_label)
        bar.addStretch(1)
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.open_selected)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_selected)
        bar.addWidget(self.open_btn)
        bar.addWidget(self.delete_btn)
        tl.addLayout(bar)
        root.addWidget(table_card, 5)
        self._update_buttons()

    # ---------------------------------------------------------------- charts

    def _build_charts(self) -> None:
        self.time_chart = ScansOverTimeChart()
        self.plant_chart = ConditionsPerPlantChart()
        for title, sub, chart in (
            ("Scans over time", "Last 30 days, by result", self.time_chart),
            ("Conditions per plant", "How often each condition was found", self.plant_chart),
        ):
            c, lay = card(margins=14, spacing=2)
            lay.addWidget(label(title, "H3"))
            lay.addWidget(label(sub, "Caption"))
            lay.addWidget(chart, 1)
            self.charts_row.addWidget(c, 1)

    def _update_charts(self, plant_id: int | None) -> None:
        self.time_chart.update_data(self.store.daily_prediction_counts(plant_id), self.labels, self.theme)
        by_plant = self.store.prediction_counts_by_plant()
        if plant_id is not None:
            name = next((p.name for p in self.store.list_plants() if p.id == plant_id), None)
            by_plant = {name: by_plant[name]} if name in by_plant else {}
        self.plant_chart.update_data(by_plant, self.labels, self.theme)

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self._update_charts(self.selected_plant_id())

    # ------------------------------------------------------------------ data

    def selected_plant_id(self) -> int | None:
        return self.plant_filter.currentData()

    def refresh(self, keep_filter: bool = False) -> None:
        if not keep_filter:
            current = self.selected_plant_id()
            self.plant_filter.blockSignals(True)
            self.plant_filter.clear()
            self.plant_filter.addItem("All plants", None)
            for p in self.store.list_plants():
                self.plant_filter.addItem(f"{p.name}  ({p.scan_count})", p.id)
            idx = self.plant_filter.findData(current)
            self.plant_filter.setCurrentIndex(max(idx, 0))
            self.plant_filter.blockSignals(False)

        plant_id = self.selected_plant_id()
        self._scans = self.store.list_scans(plant_id, limit=MAX_ROWS)
        self._fill_table()
        self._update_tiles(plant_id)
        self._update_charts(plant_id)

    def _fill_table(self) -> None:
        self.table.setRowCount(len(self._scans))
        for row, s in enumerate(self._scans):
            thumb = QTableWidgetItem()
            thumb.setIcon(self._thumb(s.overlay_path or s.image_path))
            self.table.setItem(row, 0, thumb)
            self.table.setItem(row, 1, QTableWidgetItem(f"{s.timestamp:%Y-%m-%d  %H:%M}"))
            self.table.setItem(row, 2, QTableWidgetItem(s.plant_name))
            self.table.setItem(row, 3, QTableWidgetItem(self.labels.display_name(s.prediction)))
            conf = QTableWidgetItem(format_confidence(s.confidence))
            conf.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 4, conf)
            self.table.setItem(row, 5, QTableWidgetItem(s.model_version))
        total = sum(n for _, n in self.store.scans_per_day(self.selected_plant_id()))
        shown = len(self._scans)
        self.count_label.setText(f"{total} scans" + (f" (showing latest {shown})" if total > shown else ""))
        self._update_buttons()

    def _update_tiles(self, plant_id: int | None) -> None:
        counts = self.store.prediction_counts(plant_id)
        total = sum(counts.values())
        self.t_scans.set(str(total), "for this plant" if plant_id else "across all plants")
        plants = self.store.list_plants()
        self.t_plants.set(str(len(plants)), "tracked")
        named = {k: n for k, n in counts.items() if self.labels.by_key(k) is not None}
        if named:
            key, n = max(named.items(), key=lambda kv: kv[1])
            self.t_top.set(f"{n}x", self.labels.display_name(key))
        else:
            self.t_top.set("-", "no confident results yet" if counts else "no scans yet")
        if self._scans:
            last = self._scans[0]
            self.t_last.set(f"{last.timestamp:%b %d}", f"{last.timestamp:%H:%M} - {last.plant_name}")
        else:
            self.t_last.set("-", "")

    def _thumb(self, path: str) -> QIcon:
        icon = self._thumbs.get(path)
        if icon is None:
            pm = QPixmap(path)
            icon = QIcon(pm.scaled(THUMB, THUMB, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)) if not pm.isNull() else QIcon()
            self._thumbs[path] = icon
        return icon

    # --------------------------------------------------------------- actions

    def _selected_scan(self) -> StoredScan | None:
        rows = self.table.selectionModel().selectedRows()
        return self._scans[rows[0].row()] if rows else None

    def _update_buttons(self) -> None:
        has = self._selected_scan() is not None
        self.open_btn.setEnabled(has)
        self.delete_btn.setEnabled(has)

    def open_selected(self) -> None:
        scan = self._selected_scan()
        if scan:
            ScanDetailDialog(scan, self).exec()

    def delete_selected(self) -> None:
        scan = self._selected_scan()
        if scan is None:
            return
        answer = QMessageBox.question(
            self, "Delete scan", f"Delete the scan from {scan.timestamp:%Y-%m-%d %H:%M} ({scan.plant_name})?"
        )
        if answer == QMessageBox.Yes:
            self.store.delete_scan(scan.id)
            self._thumbs.pop(scan.overlay_path or scan.image_path, None)
            self.refresh()
            self.changed.emit()
