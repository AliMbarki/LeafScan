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
