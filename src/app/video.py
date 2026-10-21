"""Video analysis: find the leaves in a recorded plant video and check each one for illness.

Per analysed frame (a few per second):  detect leaves -> crop each one -> classify (crop + mirror)
-> match to the leaves seen before (tracking) -> smooth the probabilities over time.
Every displayed frame gets the latest outlines and labels drawn on it. All findings are collected
into a summary: which conditions were seen, when, and an overall verdict.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from app.capture import VideoSource
from app.contracts import ClassifierResult, RGBImage, format_confidence
from app.labels import UNCERTAIN_KEY, Labels, get_labels
from app.vision.multi import LeafRegion, detect_leaves
from app.vision.overlay import DISEASE, HEALTHY, UNSURE, _font

log = logging.getLogger(__name__)

ANALYZE_EVERY_S = 0.33  # classify ~3 times per second of video
EMA = 0.5  # weight of the newest probabilities when smoothing a leaf's prediction
TRACK_TTL_S = 1.2  # forget a leaf not seen for this long
MIN_HITS = 2  # a condition must be seen in at least 2 analyses to count as a finding


# ----------------------------------------------------------------------------- data


@dataclass
class Track:
    id: int
    bbox: tuple[int, int, int, int]
    contour: np.ndarray
    probs: np.ndarray
    last_seen: float
    result: ClassifierResult | None = None


@dataclass
class Finding:
    class_key: str
    plant: str
    condition: str
    healthy: bool
    first_t: float
    last_t: float
    hits: int = 0
    max_conf: float = 0.0


@dataclass
class Verdict:
    status: str  # "illness" | "healthy" | "unsure" | "none"
    title: str
    detail: str
    findings: list[Finding] = field(default_factory=list)  # confirmed findings, diseases first
    leaves_analysed: int = 0


def fmt_time(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}"


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _square_crop(image: RGBImage, bbox: tuple[int, int, int, int], pad: float = 0.12) -> RGBImage:
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    side = int(max(bw, bh) * (1 + 2 * pad))
    cx, cy = x + bw // 2, y + bh // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    crop = image[y0 : min(h, y0 + side), x0 : min(w, x0 + side)]
    return np.ascontiguousarray(crop) if crop.shape[0] >= 32 and crop.shape[1] >= 32 else image


# ------------------------------------------------------------------------- analyzer


class VideoAnalyzer:
    """Pure logic (no Qt): testable with any classifier."""

    def __init__(self, classifier, labels: Labels | None = None) -> None:
        self.classifier = classifier
        self.labels = labels or get_labels()
        self.tracks: list[Track] = []
        self.findings: dict[str, Finding] = {}
        self.uncertain_hits = 0
        self.analyses = 0
        self._next_id = 1

    # -- probabilities for a batch of leaf crops (crop + mirror averaged)
    def _probabilities(self, crops: list[RGBImage]) -> np.ndarray:
        views = [v for c in crops for v in (c, np.ascontiguousarray(c[:, ::-1]))]
        batch = getattr(self.classifier, "probabilities_batch", None)
        if batch is not None:
            p = batch(views)
        else:  # classifiers without raw probabilities (e.g. the mock): rebuild them from top-k
            p = np.zeros((len(views), len(self.labels)))
            for i, v in enumerate(views):
                for pred in self.classifier.predict(v).top_k:
                    p[i, pred.class_id] = pred.probability
        return p.reshape(len(crops), 2, -1).mean(axis=1)

    def _result(self, probs: np.ndarray) -> ClassifierResult:
        thr = getattr(self.classifier, "threshold", 0.6)
        pthr = getattr(self.classifier, "plant_threshold", thr)
        return ClassifierResult.from_probabilities(probs, self.labels, thr, plant_threshold=pthr)

    def analyze(self, frame: RGBImage, t: float) -> list[Track]:
        regions: list[LeafRegion] = detect_leaves(frame)
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= TRACK_TTL_S]
        self.analyses += 1
        if not regions:
            return self.visible(t)
        probs = self._probabilities([_square_crop(frame, r.bbox) for r in regions])
        free = list(self.tracks)
        for region, p in zip(regions, probs):
            best = max(free, key=lambda tr: _iou(tr.bbox, region.bbox), default=None)
            if best is not None and _iou(best.bbox, region.bbox) > 0.3:
                free.remove(best)
                best.probs = EMA * p + (1 - EMA) * best.probs
                best.bbox, best.contour, best.last_seen = region.bbox, region.contour, t
                track = best
            else:
                track = Track(self._next_id, region.bbox, region.contour, p, t)
                self._next_id += 1
                self.tracks.append(track)
            track.result = self._result(track.probs)
            self._record(track.result, t)
        return self.visible(t)

    def visible(self, t: float) -> list[Track]:
        return [tr for tr in self.tracks if t - tr.last_seen <= 0.5 and tr.result is not None]

    def reset_tracks(self) -> None:
        self.tracks = []

    def _record(self, r: ClassifierResult, t: float) -> None:
        if r.is_uncertain or r.class_key == UNCERTAIN_KEY:
            self.uncertain_hits += 1
            return
        info = self.labels.by_key(r.class_key)
        if info is None:
            return
        f = self.findings.get(r.class_key)
        if f is None:
            f = self.findings[r.class_key] = Finding(r.class_key, info.plant, info.condition, info.healthy, t, t)
        f.hits += 1
        f.first_t, f.last_t = min(f.first_t, t), max(f.last_t, t)
        f.max_conf = max(f.max_conf, r.confidence)

    def verdict(self) -> Verdict:
        confirmed = [f for f in self.findings.values() if f.hits >= MIN_HITS]
        confirmed.sort(key=lambda f: (f.healthy, -f.hits))
        sick = [f for f in confirmed if not f.healthy]
        leaves = sum(f.hits for f in self.findings.values()) + self.uncertain_hits
        if sick:
            names = ", ".join(f"{f.condition} ({f.plant.lower()})" for f in sick[:3])
            return Verdict("illness", "Illness detected", f"Seen in the video: {names}.", confirmed, leaves)
        if confirmed:
            plants = ", ".join(sorted({f.plant for f in confirmed}))
            return Verdict("healthy", "No illness detected",
                           f"The leaves that were recognised look healthy ({plants}).", confirmed, leaves)
        if leaves:
            return Verdict("unsure", "Not sure",
                           "Leaves were found but none could be identified with confidence. Try a closer, "
                           "steadier video in good light.", confirmed, leaves)
        return Verdict("none", "No leaves found yet",
                       "No plant leaves have been detected in the video so far.", confirmed, 0)


# -------------------------------------------------------------------------- drawing


def _label(r: ClassifierResult, labels: Labels) -> tuple[str, tuple[int, int, int]]:
    if r.unknown_plant:
        return r.disease_name, UNSURE
    if r.plant is not None and r.is_uncertain:
        return f"{r.plant} - not sure", UNSURE
    if r.is_uncertain:
        return "Not sure", UNSURE
    info = labels.by_key(r.class_key)
    return f"{r.disease_name}  {format_confidence(r.confidence)}", HEALTHY if info and info.healthy else DISEASE


def draw_tracks(frame: RGBImage, tracks: list[Track], labels: Labels) -> RGBImage:
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(2, round(max(h, w) / 360))
    items = []
    for tr in tracks:
        text, color = _label(tr.result, labels)
        cv2.drawContours(out, [tr.contour], -1, color, thick, cv2.LINE_AA)
        items.append((tr, text, color))
    if not items:
        return out
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil, "RGBA")
    size = max(12, round(max(h, w) / 55))
    font = _font(size)
    for tr, text, color in items:
        l, t_, r, b = draw.textbbox((0, 0), text, font=font)
        bw, bh = (r - l) + size + 14, (b - t_) + 10
        x, y = tr.bbox[0], tr.bbox[1] - bh - 4
        if y < 2:
            y = tr.bbox[1] + 4
        x = min(max(2, x), max(2, w - bw - 2))
        draw.rounded_rectangle((x, y, x + bw, y + bh), radius=bh // 2, fill=(14, 19, 17, 210))
        d = size * 0.45
        cy = y + bh / 2
        draw.ellipse((x + 7, cy - d / 2, x + 7 + d, cy + d / 2), fill=tuple(color) + (255,))
        draw.text((x + 7 + size - l, y + 5 - t_), text, font=font, fill=(245, 248, 246, 255))
    return np.asarray(pil).copy()


# --------------------------------------------------------------------------- thread


class VideoThread(QThread):
    """Plays a video file off the UI thread, analysing it as it goes."""

    opened = Signal(float, float)  # duration (s), fps
    frame_ready = Signal(object, float)  # annotated RGB frame, time (s)
    summary = Signal(object)  # Verdict
    ended = Signal(object)  # final Verdict
    failed = Signal(str)

    def __init__(self, path: str | Path, classifier_factory: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self._factory = classifier_factory
        self._mutex = QMutex()
        self._paused = False
        self._seek_to: float | None = None

    # -- controls (called from the UI thread)
    def set_paused(self, paused: bool) -> None:
        with QMutexLocker(self._mutex):
            self._paused = paused

    def seek(self, seconds: float) -> None:
        with QMutexLocker(self._mutex):
            self._seek_to = seconds

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(5000)

    def run(self) -> None:
        try:
            source = VideoSource(self.path)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        try:
            analyzer = VideoAnalyzer(self._factory())
        except Exception as exc:  # noqa: BLE001
            source.close()
            self.failed.emit(f"Could not load the model: {exc}")
            return
        labels = analyzer.labels
        self.opened.emit(source.duration, source.fps)
        frame_dt = 1.0 / source.fps
        next_analysis = 0.0
        tracks: list[Track] = []
        last_summary = 0.0
        try:
            while not self.isInterruptionRequested():
                with QMutexLocker(self._mutex):
                    paused, seek_to, self._seek_to = self._paused, self._seek_to, None
                if seek_to is not None:
                    source.seek(seek_to)
                    analyzer.reset_tracks()
                    next_analysis = 0.0
                    tracks = []
                if paused and seek_to is None:
                    self.msleep(40)
                    continue
                started = time.perf_counter()
                frame, t = source.read()
                if frame is None:
                    break
                if t >= next_analysis:
                    tracks = analyzer.analyze(frame, t)
                    next_analysis = t + ANALYZE_EVERY_S
                else:
                    tracks = analyzer.visible(t)
                self.frame_ready.emit(draw_tracks(frame, tracks, labels), t)
                if t - last_summary >= 1.0 or t < last_summary:
                    self.summary.emit(analyzer.verdict())
                    last_summary = t
                spare = frame_dt - (time.perf_counter() - started)
                if spare > 0:  # play at the video's own speed when analysis is faster than real time
                    self.msleep(int(spare * 1000))
            self.ended.emit(analyzer.verdict())
        except Exception as exc:  # noqa: BLE001
            log.exception("video analysis failed")
            self.failed.emit(f"Video analysis failed: {exc}")
        finally:
            source.close()
