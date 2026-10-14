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
