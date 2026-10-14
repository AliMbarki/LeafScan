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
        self.subtitle = label("Hold one leaf against a plain background, then press Capture (Space) or import a photo.", "Muted", wrap=True)
        lay.addWidget(self.subtitle)

        self.conf_row = QWidget()
        cr = QVBoxLayout(self.conf_row)
        cr.setContentsMargins(0, 6, 0, 0)
        cr.setSpacing(4)
        crh = QHBoxLayout()
        crh.addWidget(label("Confidence", "Caption"))
        crh.addStretch(1)
        self.conf_value = label("", "H3")
        crh.addWidget(self.conf_value)
        cr.addLayout(crh)
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setFixedHeight(8)
        cr.addWidget(self.conf_bar)
        self.alternatives = label("", "Caption", wrap=True)
        cr.addWidget(self.alternatives)
        lay.addWidget(self.conf_row)
        self.conf_row.hide()
        self.disc_label = label("", "Caption", wrap=True)
        self.disc_label.hide()
        lay.addWidget(self.disc_label)
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
        self._result = None
        self._set_chip("Ready", "muted")
        self.title.setText("Scan a leaf")
        self.subtitle.setText("Hold one leaf against a plain background, then press Capture (Space) or import a photo.")
        self.timing.setText("")
        self.conf_row.hide()
        self.guidance.hide()
        self.save_btn.setEnabled(False)
        self.save_note.setText("")

    def show_busy(self) -> None:
        self.disc_label.hide()
        self._set_chip("Analyzing", "muted")
        self.title.setText("Analyzing...")
        self.subtitle.setText("Finding the leaf and classifying it.")
        self.conf_row.hide()
        self.guidance.hide()
        self.save_btn.setEnabled(False)
        self.save_note.setText("")

    def show_error(self, message: str) -> None:
        self.disc_label.hide()
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
        if c is None:
            self._set_chip("No leaf", "warn")
            self.title.setText("No leaf detected")
            self.subtitle.setText(
                result.leaf.message or "Place a single leaf in the frame against a plain, contrasting background."
            )
            self.conf_row.hide()
            self.guidance.hide()
            return

        self.conf_row.show()
        self.conf_value.setText(format_confidence(c.confidence))
        self.conf_bar.setValue(round(c.confidence * 100))
        others = [p for p in c.top_k if p.class_id != c.class_id][:2] if not c.is_uncertain else list(c.top_k[:3])
        others = [p for p in others if p.probability >= 0.02]  # 0% alternatives are noise
        prefix = "Best guesses: " if c.is_uncertain else "Also possible: "
        self.alternatives.setText(prefix + ", ".join(f"{p.name} ({format_confidence(p.probability)})" for p in others) if others else "")

        if c.is_uncertain:
            self._set_chip("Not sure", "warn")
            set_prop(self.conf_bar, "tone", "warn")
            self.title.setText("Not sure")
            self.subtitle.setText(
                "The model isn't confident enough to name a condition. Try a sharper photo of a single leaf in even light."
            )
        else:
            info = self.labels.by_key(c.class_key)
            healthy = bool(info and info.healthy)
            self._set_chip("Looks healthy" if healthy else "Possible disease", "ok" if healthy else "danger")
            set_prop(self.conf_bar, "tone", "ok" if c.confidence >= 0.85 else "warn")
            self.title.setText(info.condition if info else c.disease_name)
            self.subtitle.setText(f"{info.plant} leaf" if info else "")

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

    def _set_chip(self, text: str, tone: str) -> None:
        self.chip.setText(text)
        set_prop(self.chip, "tone", tone)
