"""Dark/light design tokens and the Qt stylesheet generated from them."""
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


DARK = Theme(
    name="dark",
    bg="#0e1311",
    surface="#151b18",
    surface2="#1c2420",
    border="#27312c",
    text="#e6ece8",
    muted="#8d9b94",
    accent="#3ecf8e",
    accent_hover="#5ad9a0",
    on_accent="#06140d",
    ok="#3ecf8e",
    warn="#f0b44c",
    danger="#ef6a5f",
    chart=("#3ecf8e", "#5aa9f0", "#f0b44c", "#c78bf0", "#ef6a5f", "#4fd1c5", "#f08fc0", "#a3c95a", "#9aa5b1"),
)

LIGHT = Theme(
    name="light",
    bg="#f3f5f4",
    surface="#ffffff",
    surface2="#edf1ef",
    border="#dbe2de",
    text="#16201b",
    muted="#5b6a62",
    accent="#1c9a60",
    accent_hover="#17844f",
    on_accent="#ffffff",
    ok="#1c9a60",
    warn="#b7791f",
    danger="#c2412f",
    chart=("#1c9a60", "#2f7fd1", "#c98a14", "#8a52c7", "#c2412f", "#1f9e93", "#c4508a", "#6f9a22", "#6b7785"),
)

THEMES = {t.name: t for t in (DARK, LIGHT)}


def rgba(hex_color: str, alpha: float) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {int(alpha * 255)})"


def stylesheet(t: Theme) -> str:
    return f"""
* {{ font-family: "Segoe UI Variable", "Segoe UI", "Inter", sans-serif; font-size: 10pt; color: {t.text}; }}
QMainWindow, QWidget#Root {{ background: {t.bg}; }}
QWidget#Page {{ background: {t.bg}; }}
QToolTip {{ background: {t.surface2}; color: {t.text}; border: 1px solid {t.border}; padding: 4px; }}

QFrame#Sidebar {{ background: {t.surface}; border-right: 1px solid {t.border}; }}
QLabel#Brand {{ font-size: 15pt; font-weight: 700; }}
QLabel#BrandMark {{ color: {t.accent}; font-size: 18pt; font-weight: 800; }}
QPushButton#NavButton {{
    text-align: left; padding: 9px 14px; border: none; border-radius: 8px;
    background: transparent; color: {t.muted}; font-weight: 600;
}}
QPushButton#NavButton:hover {{ background: {t.surface2}; color: {t.text}; }}
QPushButton#NavButton:checked {{ background: {t.surface2}; color: {t.accent}; }}

QFrame#Card {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 14px; }}
QFrame#SubCard {{ background: {t.surface2}; border: none; border-radius: 10px; }}
QLabel#H1 {{ font-size: 18pt; font-weight: 700; }}
QLabel#H2 {{ font-size: 13pt; font-weight: 700; }}
QLabel#H3 {{ font-size: 10.5pt; font-weight: 700; }}
QLabel#Muted, QLabel#Caption {{ color: {t.muted}; }}
QLabel#Caption {{ font-size: 8.5pt; }}
QLabel#StatValue {{ font-size: 17pt; font-weight: 700; }}
QLabel#Disclaimer {{
    color: {t.muted}; font-size: 8.5pt; padding: 8px 10px;
    border: 1px dashed {t.border}; border-radius: 8px;
}}
QLabel#Banner {{
    background: {rgba(t.warn, 0.14)}; color: {t.warn}; border: 1px solid {rgba(t.warn, 0.45)};
    border-radius: 10px; padding: 8px 12px; font-weight: 600;
}}
QLabel#Chip {{ border-radius: 10px; padding: 3px 10px; font-weight: 700; font-size: 8.5pt; }}
QLabel#Chip[tone="ok"]     {{ background: {rgba(t.ok, 0.14)}; color: {t.ok}; }}
QLabel#Chip[tone="warn"]   {{ background: {rgba(t.warn, 0.14)}; color: {t.warn}; }}
QLabel#Chip[tone="danger"] {{ background: {rgba(t.danger, 0.14)}; color: {t.danger}; }}
QLabel#Chip[tone="muted"]  {{ background: {t.surface2}; color: {t.muted}; }}

QPushButton {{
    background: {t.surface2}; border: 1px solid {t.border}; border-radius: 9px;
    padding: 8px 16px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {t.accent}; }}
QPushButton:disabled {{ color: {t.muted}; border-color: {t.border}; background: {t.surface}; }}
QPushButton[primary="true"] {{ background: {t.accent}; color: {t.on_accent}; border: none; }}
QPushButton[primary="true"]:hover {{ background: {t.accent_hover}; }}
QPushButton[primary="true"]:disabled {{ background: {t.border}; color: {t.muted}; }}
QPushButton[flat="true"] {{ background: transparent; border: none; color: {t.muted}; padding: 6px 10px; }}
QPushButton[flat="true"]:hover {{ color: {t.text}; }}

QComboBox, QLineEdit {{
    background: {t.surface2}; border: 1px solid {t.border}; border-radius: 8px; padding: 6px 10px;
    selection-background-color: {t.accent}; selection-color: {t.on_accent};
}}
QComboBox:focus, QLineEdit:focus {{ border-color: {t.accent}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {t.surface}; border: 1px solid {t.border}; selection-background-color: {t.surface2};
    selection-color: {t.text}; outline: none;
}}

QProgressBar {{ background: {t.surface2}; border: none; border-radius: 4px; height: 8px; }}
QProgressBar::chunk {{ border-radius: 4px; background: {t.accent}; }}
QProgressBar[tone="warn"]::chunk {{ background: {t.warn}; }}
QProgressBar[tone="danger"]::chunk {{ background: {t.danger}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t.border}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}

QTableWidget {{
    background: {t.surface}; border: none; gridline-color: transparent;
    selection-background-color: {t.surface2}; selection-color: {t.text}; alternate-background-color: {t.surface};
}}
QTableWidget::item {{ padding: 4px; border-bottom: 1px solid {t.border}; }}
QHeaderView::section {{
    background: {t.surface}; color: {t.muted}; border: none; border-bottom: 1px solid {t.border};
    padding: 8px 6px; font-weight: 600; font-size: 8.5pt;
}}
QStatusBar {{ background: {t.surface}; color: {t.muted}; border-top: 1px solid {t.border}; }}
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
