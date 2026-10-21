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
