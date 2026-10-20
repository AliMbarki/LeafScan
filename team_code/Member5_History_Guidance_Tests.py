"""
LeafScan - Member 5 - History, guidance and testing
Main technology: SQLite / Matplotlib / JSON / pytest

Reading copy: the code below is collected from the real project files listed here.
Edit the originals (paths shown in each section), not this file. Files:
  - src/app/storage/db.py
  - src/app/storage/__init__.py
  - src/app/ui/history_view.py
  - src/app/ui/charts.py
  - src/app/guidance/__init__.py
  - tests/test_storage.py
  - tests/test_ui.py
  - tests/test_guidance_history.py
"""


# ================================================================================================
# FILE: src/app/storage/db.py
# ================================================================================================

"""SQLite-backed scan history. Images live on disk; the DB stores relative paths."""
from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from app.capture import save_rgb_png
from app.contracts import Plant, ScanRecord, StoredScan

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS plants (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scans (
    id            INTEGER PRIMARY KEY,
    plant_id      INTEGER NOT NULL REFERENCES plants(id) ON DELETE CASCADE,
    timestamp     TEXT NOT NULL,            -- ISO-8601 local time with offset
    prediction    TEXT NOT NULL,            -- class key from labels.json, or 'uncertain'
    class_id      INTEGER,
    confidence    REAL NOT NULL,
    is_uncertain  INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    image_path    TEXT NOT NULL,            -- relative to the data dir
    overlay_path  TEXT,
    species       TEXT                      -- predicted plant species (v2)
);
CREATE INDEX IF NOT EXISTS idx_scans_plant_time ON scans(plant_id, timestamp);
"""

_PLANT_N = re.compile(r"^plant (\d+)$", re.IGNORECASE)


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SqliteScanStore:
    def __init__(self, db_path: Path | str, image_dir: Path | str | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.root = self.db_path.parent
        self.image_dir = Path(image_dir) if image_dir else self.root / "scans"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def _migrate(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(scans)")}
        if "species" not in cols:  # v1 -> v2
            self.conn.execute("ALTER TABLE scans ADD COLUMN species TEXT")

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ plants

    def get_or_create_plant(self, name: str) -> Plant:
        name = " ".join(name.split())
        if not name:
            raise ValueError("plant name cannot be empty")
        row = self.conn.execute("SELECT * FROM plants WHERE name = ?", (name,)).fetchone()
        if row is None:
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            cur = self.conn.execute("INSERT INTO plants(name, created_at) VALUES (?, ?)", (name, now))
            self.conn.commit()
            return Plant(cur.lastrowid, name, _dt(now))
        return Plant(row["id"], row["name"], _dt(row["created_at"]))

    def suggest_plant_name(self) -> str:
        taken = {
            int(m.group(1))
            for (name,) in self.conn.execute("SELECT name FROM plants")
            if (m := _PLANT_N.match(name))
        }
        n = 1
        while n in taken:
            n += 1
        return f"Plant {n}"

    def list_plants(self) -> list[Plant]:
        rows = self.conn.execute(
            """SELECT p.id, p.name, p.created_at, COUNT(s.id) AS n, MAX(s.timestamp) AS last
               FROM plants p LEFT JOIN scans s ON s.plant_id = p.id
               GROUP BY p.id ORDER BY p.name COLLATE NOCASE"""
        ).fetchall()
        return [Plant(r["id"], r["name"], _dt(r["created_at"]), r["n"], _dt(r["last"])) for r in rows]

    def rename_plant(self, plant_id: int, name: str) -> None:
        self.conn.execute("UPDATE plants SET name = ? WHERE id = ?", (" ".join(name.split()), plant_id))
        self.conn.commit()

    # ------------------------------------------------------------------- scans

    def save_scan(self, record: ScanRecord) -> int:
        if self.conn.execute("SELECT 1 FROM plants WHERE id = ?", (record.plant_id,)).fetchone() is None:
            raise ValueError(f"no plant with id {record.plant_id}")
        stem = f"{record.timestamp:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        month = self.image_dir / f"{record.timestamp:%Y-%m}"
        image_path = month / f"{stem}.png"
        save_rgb_png(record.image, image_path)
        overlay_rel = None
        if record.overlay_image is not None:
            overlay_path = month / f"{stem}_overlay.png"
            save_rgb_png(record.overlay_image, overlay_path)
            overlay_rel = overlay_path.relative_to(self.root).as_posix()
        cur = self.conn.execute(
            """INSERT INTO scans(plant_id, timestamp, prediction, class_id, confidence, is_uncertain,
                                 model_version, image_path, overlay_path, species)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.plant_id,
                record.timestamp.isoformat(timespec="seconds"),
                record.prediction,
                record.class_id,
                float(record.confidence),
                int(record.is_uncertain),
                record.model_version,
                image_path.relative_to(self.root).as_posix(),
                overlay_rel,
                record.species,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_scans(self, plant_id: int | None = None, limit: int | None = None) -> list[StoredScan]:
        sql = """SELECT s.*, p.name AS plant_name FROM scans s JOIN plants p ON p.id = s.plant_id"""
        args: list = []
        if plant_id is not None:
            sql += " WHERE s.plant_id = ?"
            args.append(plant_id)
        sql += " ORDER BY s.timestamp DESC, s.id DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [self._scan(r) for r in self.conn.execute(sql, args)]

    def delete_scan(self, scan_id: int) -> None:
        row = self.conn.execute("SELECT image_path, overlay_path FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if row is None:
            return
        self.conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        self.conn.commit()
        for rel in (row["image_path"], row["overlay_path"]):
            if rel:
                (self.root / rel).unlink(missing_ok=True)

    # ----------------------------------------------------------------- queries

    def scans_per_day(self, plant_id: int | None = None) -> list[tuple[str, int]]:
        where, args = ("WHERE plant_id = ?", (plant_id,)) if plant_id is not None else ("", ())
        rows = self.conn.execute(
            f"SELECT substr(timestamp, 1, 10) AS day, COUNT(*) FROM scans {where} GROUP BY day ORDER BY day", args
        )
        return [(day, n) for day, n in rows]

    def prediction_counts(self, plant_id: int | None = None) -> dict[str, int]:
        where, args = ("WHERE plant_id = ?", (plant_id,)) if plant_id is not None else ("", ())
        rows = self.conn.execute(
            f"SELECT prediction, COUNT(*) FROM scans {where} GROUP BY prediction ORDER BY COUNT(*) DESC", args
        )
        return {pred: n for pred, n in rows}

    def daily_prediction_counts(self, plant_id: int | None = None) -> list[tuple[str, str, int]]:
        """[(YYYY-MM-DD, prediction, count)] ordered by day."""
        where, args = ("WHERE plant_id = ?", (plant_id,)) if plant_id is not None else ("", ())
        rows = self.conn.execute(
            f"""SELECT substr(timestamp, 1, 10) AS day, prediction, COUNT(*) FROM scans {where}
                GROUP BY day, prediction ORDER BY day""",
            args,
        )
        return [(d, p, n) for d, p, n in rows]

    def prediction_counts_by_plant(self) -> dict[str, dict[str, int]]:
        rows = self.conn.execute(
            """SELECT p.name, s.prediction, COUNT(*) FROM scans s JOIN plants p ON p.id = s.plant_id
               GROUP BY p.id, s.prediction ORDER BY p.name COLLATE NOCASE"""
        )
        out: dict[str, dict[str, int]] = {}
        for plant, pred, n in rows:
            out.setdefault(plant, {})[pred] = n
        return out

    def _scan(self, r: sqlite3.Row) -> StoredScan:
        return StoredScan(
            id=r["id"],
            plant_id=r["plant_id"],
            plant_name=r["plant_name"],
            timestamp=_dt(r["timestamp"]),
            prediction=r["prediction"],
            class_id=r["class_id"],
            confidence=r["confidence"],
            is_uncertain=bool(r["is_uncertain"]),
            model_version=r["model_version"],
            image_path=str(self.root / r["image_path"]),
            overlay_path=str(self.root / r["overlay_path"]) if r["overlay_path"] else None,
            species=r["species"],
        )

# ================================================================================================
# FILE: src/app/storage/__init__.py
# ================================================================================================

"""Scan history storage (SQLite + PNG files)."""
from __future__ import annotations

from app.contracts import ScanRecord
from app.storage.db import SqliteScanStore

_default: SqliteScanStore | None = None


def default_store() -> SqliteScanStore:
    global _default
    if _default is None:
        from app.config import Settings

        s = Settings()
        _default = SqliteScanStore(s.db_path, s.scans_dir)
    return _default


def save_scan(record: ScanRecord) -> int:
    """Module-level convenience matching the documented interface."""
    return default_store().save_scan(record)


__all__ = ["SqliteScanStore", "save_scan", "default_store"]

# ================================================================================================
# FILE: src/app/ui/history_view.py
# ================================================================================================

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
        lay.addWidget(label(labels.display_name(scan.prediction, scan.species), "H2"))
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
        self.plant_chart.update_data(by_plant, self.labels, self.theme, self.store.prediction_counts(None))

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
            self.table.setItem(row, 3, QTableWidgetItem(self.labels.display_name(s.prediction, s.species)))
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

# ================================================================================================
# FILE: src/app/ui/charts.py
# ================================================================================================

"""History charts (matplotlib embedded in Qt), themed for dark/light.

Colour rules: status colours (good / warning / critical) encode scan state in the
time chart. The condition chart has dozens of possible conditions, more than a palette can
separate, so: Healthy is always green, "Not sure / unknown" always grey, the six most
frequent diseases in the WHOLE history get categorical colours, and the rest fold into
"Other conditions". Ranking uses the unfiltered history, so filtering by plant never
repaints a condition.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PySide6.QtGui import QCursor  # noqa: E402
from PySide6.QtWidgets import QSizePolicy, QToolTip  # noqa: E402

from app.labels import UNCERTAIN_KEY, UNCERTAIN_NAME, Labels  # noqa: E402
from app.ui.theme import Theme  # noqa: E402

# Validated reference palette (categorical slots 1-8), light / dark steps.
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
OTHER = "Other conditions"
HEALTHY = "Healthy"
DISEASE_SLOTS = (0, 1, 3, 4, 6, 7)  # categorical slots for diseases (slot 5, green, is reserved for Healthy)
DAYS = 30


def status_of(key: str, labels: Labels) -> str:
    info = labels.by_key(key)
    if info is None:
        return "warning"
    return "good" if info.healthy else "critical"


def condition_of(key: str, labels: Labels) -> str:
    info = labels.by_key(key)
    if info is None or info.unknown:
        return UNCERTAIN_NAME
    return HEALTHY if info.healthy else info.condition


def condition_order(labels: Labels, global_counts: dict[str, int] | None = None) -> list[str]:
    """Legend order: Healthy, the most frequent diseases (by whole-history count), Other, Not sure."""
    totals: dict[str, int] = {}
    for key, n in (global_counts or {}).items():
        cond = condition_of(key, labels)
        totals[cond] = totals.get(cond, 0) + n
    diseases = sorted({c.condition for c in labels if not c.healthy and not c.unknown},
                      key=lambda c: (-totals.get(c, 0), c))
    top = [d for d in diseases if totals.get(d, 0) > 0][: len(DISEASE_SLOTS)]
    return [HEALTHY, *top, OTHER, UNCERTAIN_NAME]


class _Chart(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(5, 2.6), dpi=100, layout="constrained")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(170)
        self.ax = self.fig.add_subplot(111)
        self._hits: list[tuple[Rectangle, str]] = []
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("figure_leave_event", lambda _e: QToolTip.hideText())

    def _style(self, t: Theme) -> None:
        self.fig.set_facecolor(t.surface)
        ax = self.ax
        ax.set_facecolor(t.surface)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t.border)
        ax.tick_params(colors=t.muted, labelsize=8, length=0)
        ax.grid(color=t.border, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    def _empty(self, t: Theme, text: str) -> None:
        self.ax.clear()
        self._style(t)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines["bottom"].set_visible(False)
        self.ax.text(0.5, 0.5, text, ha="center", va="center", color=t.muted, fontsize=9, transform=self.ax.transAxes)
        self._hits = []
        self.draw_idle()

    def _legend(self, t: Theme, handles, labels) -> None:
        leg = self.ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=min(len(labels), 3),
                             frameon=False, fontsize=8, handlelength=0.9, handleheight=0.9, borderaxespad=0.2,
                             columnspacing=1.0)
        for txt in leg.get_texts():
            txt.set_color(t.text)

    def _on_move(self, event) -> None:
        if event.inaxes is not self.ax:
            QToolTip.hideText()
            return
        for rect, tip in self._hits:
            if rect.contains(event)[0]:
                QToolTip.showText(QCursor.pos(), tip, self)
                return
        QToolTip.hideText()


class ScansOverTimeChart(_Chart):
    """Daily scan counts for the last 30 days, stacked by scan status."""

    def update_data(self, rows: list[tuple[str, str, int]], labels: Labels, t: Theme, today: date | None = None) -> None:
        if not rows:
            self._empty(t, "No scans yet")
            return
        last = max(date.fromisoformat(d) for d, _, _ in rows)
        end = max(today or date.today(), last)
        days = [end - timedelta(days=i) for i in range(DAYS - 1, -1, -1)]
        index = {d: i for i, d in enumerate(days)}
        series = {"good": [0] * DAYS, "critical": [0] * DAYS, "warning": [0] * DAYS}
        for d, key, n in rows:
            i = index.get(date.fromisoformat(d))
            if i is not None:
                series[status_of(key, labels)][i] += n
        names = {"good": "Healthy", "critical": "Possible disease", "warning": "Not sure"}

        ax = self.ax
        ax.clear()
        self._style(t)
        ax.grid(axis="x", visible=False)
        bottom = [0] * DAYS
        handles, hits = [], []
        for status in ("good", "critical", "warning"):
            vals = series[status]
            bars = ax.bar(range(DAYS), vals, bottom=bottom, width=0.72, color=STATUS[status],
                          edgecolor=t.surface, linewidth=1.5)
            handles.append(bars[0])
            for i, (rect, v) in enumerate(zip(bars, vals)):
                if v:
                    hits.append((rect, f"{days[i]:%a %b %d}\n{names[status]}: {v} scan{'s' * (v != 1)}"))
            bottom = [b + v for b, v in zip(bottom, vals)]
        self._hits = hits
        ticks = list(range(DAYS - 1, -1, -7))[::-1]
        ax.set_xticks(ticks, [f"{days[i]:%b %d}" for i in ticks])
        ax.set_xlim(-0.6, DAYS - 0.4)
        top = max(bottom) if any(bottom) else 1
        ax.set_ylim(0, top * 1.15 + 0.5)
        ax.yaxis.get_major_locator().set_params(integer=True)
        self._legend(t, handles, [names[s] for s in ("good", "critical", "warning")])
        self.draw_idle()


class ConditionsPerPlantChart(_Chart):
    """Per plant: how many scans showed each condition (stacked horizontal bars)."""

    def update_data(self, by_plant: dict[str, dict[str, int]], labels: Labels, t: Theme,
                    global_counts: dict[str, int] | None = None) -> None:
        if not by_plant:
            self._empty(t, "No scans yet")
            return
        if global_counts is None:
            global_counts = {}
            for preds in by_plant.values():
                for k, n in preds.items():
                    global_counts[k] = global_counts.get(k, 0) + n
        order = condition_order(labels, global_counts)
        palette = CATEGORICAL["dark" if t.name == "dark" else "light"]
        color = {HEALTHY: palette[5], OTHER: t.muted, UNCERTAIN_NAME: "#9aa5b1" if t.name == "dark" else "#b8c0c8"}
        for slot, cond in zip(DISEASE_SLOTS, order[1:-2]):
            color[cond] = palette[slot]
        counts: dict[str, dict[str, int]] = {}
        for plant, preds in by_plant.items():
            agg: dict[str, int] = defaultdict(int)
            for key, n in preds.items():
                cond = condition_of(key, labels)
                agg[cond if cond in color else OTHER] += n
            counts[plant] = agg
        plants = sorted(counts, key=lambda p: sum(counts[p].values()))[-8:]  # 8 busiest plants

        ax = self.ax
        ax.clear()
        self._style(t)
        ax.grid(axis="y", visible=False)
        left = [0] * len(plants)
        handles, names, hits = [], [], []
        for cond in order:
            vals = [counts[p].get(cond, 0) for p in plants]
            if not any(vals):
                continue
            bars = ax.barh(range(len(plants)), vals, left=left, height=0.6, color=color[cond],
                           edgecolor=t.surface, linewidth=1.5)
            handles.append(bars[0])
            names.append(cond)
            for rect, p, v in zip(bars, plants, vals):
                if v:
                    total = sum(counts[p].values())
                    hits.append((rect, f"{p}\n{cond}: {v} of {total} scan{'s' * (total != 1)}"))
            left = [a + b for a, b in zip(left, vals)]
        self._hits = hits
        ax.set_yticks(range(len(plants)), plants)
        ax.tick_params(axis="y", labelsize=9, labelcolor=t.text)
        for i, total in enumerate(left):  # selective direct label: the total at the bar end
            ax.text(total + max(left) * 0.015, i, str(total), va="center", fontsize=8, color=t.muted)
        ax.set_xlim(0, max(left) * 1.12 + 0.5)
        ax.xaxis.get_major_locator().set_params(integer=True)
        self._legend(t, handles, names)
        self.draw_idle()

# ================================================================================================
# FILE: src/app/guidance/__init__.py
# ================================================================================================

"""Local, sourced guidance per class (content.json). Titles come from labels.json.

content.json (schema 2):
* ``conditions`` - shared cards keyed by a class's ``condition_id`` (e.g. one powdery-mildew card
  for squash, cherry, grape and mango),
* ``classes`` - per-class overrides of any field,
* ``plant_care`` - extra plant-specific care sources shown with "Healthy" results.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.contracts import Guidance, Source
from app.labels import Labels, get_labels

CONTENT_PATH = Path(__file__).with_name("content.json")
SEVERITIES = {"none", "low", "moderate", "high", "info"}


class GuidanceError(ValueError):
    pass


def _sources(items: list[dict]) -> tuple[Source, ...]:
    return tuple(Source(s["title"], s["url"], s.get("publisher", "")) for s in items)


class GuidanceRepository:
    def __init__(self, labels: Labels | None = None, path: Path | str = CONTENT_PATH) -> None:
        self.labels = labels or get_labels()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        conditions, overrides = raw.get("conditions", {}), raw.get("classes", {})
        care = raw.get("plant_care", {})
        unknown = set(overrides) - set(self.labels.keys)
        missing = sorted({c.condition_id for c in self.labels if c.condition_id not in conditions})
        if missing or unknown:
            raise GuidanceError(f"guidance content out of sync with labels.json (no card for condition ids "
                                f"{missing}, overrides for unknown classes {sorted(unknown)})")
        self._items: dict[str, Guidance] = {}
        for info in self.labels:
            entry = {**conditions[info.condition_id], **overrides.get(info.key, {})}
            if entry.get("severity") not in SEVERITIES:
                raise GuidanceError(f"{info.key}: unknown severity {entry.get('severity')!r}")
            sources = list(entry.get("sources", []))
            if info.healthy:
                sources = care.get(info.plant, []) + [s for s in sources if s not in care.get(info.plant, [])]
            if not sources:
                raise GuidanceError(f"{info.key}: guidance has no sources")
            self._items[info.key] = Guidance(
                class_key=info.key,
                title=info.display_name,
                summary=entry["summary"],
                severity=entry["severity"],
                symptoms=tuple(entry.get("symptoms", ())),
                actions=tuple(entry.get("actions", ())),
                prevention=tuple(entry.get("prevention", ())),
                sources=_sources(sources),
            )

    def get(self, class_key: str) -> Guidance | None:
        return self._items.get(class_key)

    def __len__(self) -> int:
        return len(self._items)


__all__ = ["GuidanceRepository", "GuidanceError"]

# ================================================================================================
# FILE: tests/test_storage.py
# ================================================================================================

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.capture import load_image_file
from app.contracts import ScanRecord


def _record(plant_id, ts, pred="tomato_healthy", conf=0.9, img=None):
    img = img if img is not None else np.zeros((40, 40, 3), np.uint8)
    return ScanRecord(img, plant_id, ts, pred, conf, "test-1", class_id=None, overlay_image=img)


def test_plant_name_suggestions(store):
    assert store.suggest_plant_name() == "Plant 1"
    store.get_or_create_plant("Plant 1")
    store.get_or_create_plant("plant 3")
    assert store.suggest_plant_name() == "Plant 2"
    store.get_or_create_plant("Basil")
    assert store.suggest_plant_name() == "Plant 2"


def test_get_or_create_is_case_insensitive(store):
    a = store.get_or_create_plant("Tomato  bed")
    b = store.get_or_create_plant("tomato bed")
    assert a.id == b.id
    with pytest.raises(ValueError):
        store.get_or_create_plant("   ")


def test_save_list_delete_roundtrip(store):
    p = store.get_or_create_plant("Plant 1")
    img = np.full((50, 60, 3), (10, 200, 30), np.uint8)
    sid = store.save_scan(_record(p.id, datetime.now().astimezone(), img=img))
    scans = store.list_scans()
    assert [s.id for s in scans] == [sid]
    s = scans[0]
    assert s.plant_name == "Plant 1" and s.prediction == "tomato_healthy" and s.model_version == "test-1"
    assert np.array_equal(load_image_file(s.image_path), img)  # RGB preserved through PNG
    store.delete_scan(sid)
    assert store.list_scans() == []
    from pathlib import Path

    assert not Path(s.image_path).exists()


def test_unknown_plant_rejected(store):
    with pytest.raises(ValueError):
        store.save_scan(_record(999, datetime.now().astimezone()))


def test_history_queries(store):
    a = store.get_or_create_plant("A")
    b = store.get_or_create_plant("B")
    t0 = datetime(2026, 10, 1, 9, 0).astimezone()
    store.save_scan(_record(a.id, t0, "tomato_early_blight"))
    store.save_scan(_record(a.id, t0 + timedelta(hours=2), "tomato_early_blight"))
    store.save_scan(_record(a.id, t0 + timedelta(days=1), "uncertain", 0.3))
    store.save_scan(_record(b.id, t0 + timedelta(days=1), "pepper_healthy"))
    assert store.scans_per_day() == [("2026-10-01", 2), ("2026-10-02", 2)]
    assert store.scans_per_day(b.id) == [("2026-10-02", 1)]
    assert store.prediction_counts(a.id) == {"tomato_early_blight": 2, "uncertain": 1}
    assert store.prediction_counts_by_plant()["B"] == {"pepper_healthy": 1}
    plants = {p.name: p for p in store.list_plants()}
    assert plants["A"].scan_count == 3
    assert [s.prediction for s in store.list_scans(a.id, limit=1)] == ["uncertain"]


def test_v1_database_is_migrated(tmp_path):
    import sqlite3

    from app.storage.db import SqliteScanStore

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript(
        """CREATE TABLE plants (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE, created_at TEXT NOT NULL);
           CREATE TABLE scans (id INTEGER PRIMARY KEY, plant_id INTEGER NOT NULL, timestamp TEXT NOT NULL,
             prediction TEXT NOT NULL, class_id INTEGER, confidence REAL NOT NULL, is_uncertain INTEGER NOT NULL,
             model_version TEXT NOT NULL, image_path TEXT NOT NULL, overlay_path TEXT);
           INSERT INTO plants VALUES (1, 'Old', '2026-01-01T00:00:00+00:00');
           INSERT INTO scans VALUES (1, 1, '2026-01-02T00:00:00+00:00', 'uncertain', NULL, 0.3, 1, 'v1', 'a.png', NULL);"""
    )
    con.commit()
    con.close()
    s = SqliteScanStore(db)
    assert s.list_scans()[0].species is None
    p = s.get_or_create_plant("New")
    s.save_scan(_record(p.id, datetime.now().astimezone()))
    s.close()

# ================================================================================================
# FILE: tests/test_ui.py
# ================================================================================================

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

# ================================================================================================
# FILE: tests/test_guidance_history.py
# ================================================================================================

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from app.contracts import ScanRecord
from app.guidance import GuidanceError, GuidanceRepository
from app.labels import get_labels

ROOT = Path(__file__).resolve().parents[1]


def test_guidance_covers_every_class_with_sources():
    labels = get_labels()
    repo = GuidanceRepository()
    assert len(repo) == len(labels)
    for info in labels:
        g = repo.get(info.key)
        assert g is not None, info.key
        assert g.title == info.display_name  # title comes from labels.json
        assert g.summary and g.sources
        assert all(s.url.startswith("https://") and s.title for s in g.sources)
        assert (g.severity == "none") == info.healthy
        if not info.healthy and not info.unknown:
            assert g.symptoms and g.actions
    assert repo.get("uncertain") is None


def test_guidance_out_of_sync_is_rejected(tmp_path):
    raw = json.loads((ROOT / "src/app/guidance/content.json").read_text(encoding="utf-8"))
    raw["conditions"].pop("powdery_mildew")
    bad = tmp_path / "content.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(GuidanceError):
        GuidanceRepository(path=bad)


def test_no_class_names_hardcoded_in_code():
    """labels.json is the single source of truth: no class key/name may appear in code."""
    labels = get_labels()
    needles = set()
    raw = json.loads((ROOT / "src/app/labels.json").read_text(encoding="utf-8"))
    for c, r in zip(labels, raw["classes"]):
        needles |= {c.key, c.display_name}
        # dataset labels, except generic words like "Healthy" that legitimately appear in code
        needles |= {lab for labs in r.get("sources", {}).values() for lab in labs
                    if len(lab) > 6 and lab.lower() not in {"healthy", "fresh leaf"}}
        if not c.healthy and len(c.condition) > 6:
            needles.add(c.condition)
    offenders = []
    for path in [*ROOT.joinpath("src/app").rglob("*.py"), *ROOT.joinpath("scripts").glob("*.py")]:
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.name}: {n!r}" for n in needles if re.search(re.escape(n), text, re.IGNORECASE)]
    assert not offenders, offenders


def test_daily_prediction_counts(store):
    p = store.get_or_create_plant("A")
    t0 = datetime(2026, 10, 1, 9).astimezone()
    img = np.zeros((40, 40, 3), np.uint8)
    for dt, pred in ((0, "tomato_healthy"), (0, "uncertain"), (2, "tomato_healthy")):
        store.save_scan(ScanRecord(img, p.id, t0 + timedelta(days=dt), pred, 0.9, "t"))
    assert store.daily_prediction_counts() == [
        ("2026-10-01", "tomato_healthy", 1),
        ("2026-10-01", "uncertain", 1),
        ("2026-10-03", "tomato_healthy", 1),
    ]


def test_charts_render_and_fold(qtbot):
    from datetime import date

    from app.ui.charts import ConditionsPerPlantChart, ScansOverTimeChart, condition_order
    from app.ui.theme import DARK, LIGHT

    labels = get_labels()
    keys = labels.keys
    t = ScansOverTimeChart()
    qtbot.addWidget(t)
    t.update_data([("2026-10-01", keys[0], 2), ("2026-10-02", "uncertain", 1)], labels, DARK, today=date(2026, 10, 5))
    assert len(t._hits) == 2
    t.update_data([], labels, LIGHT)
    assert t._hits == []

    c = ConditionsPerPlantChart()
    qtbot.addWidget(c)
    c.update_data({"Plant 1": {keys[0]: 3, keys[1]: 1}, "Plant 2": {"uncertain": 2}}, labels, LIGHT)
    assert len(c._hits) == 3
    order = condition_order(labels)
    assert "Not sure" in order and len(order) == len(set(order))
