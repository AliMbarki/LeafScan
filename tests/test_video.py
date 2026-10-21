"""Video analysis: multi-leaf detection, tracking/verdict logic, video file reading, playback thread."""
import cv2
import numpy as np
import pytest

from app.capture import VideoSource
from app.contracts import InvalidImageError
from app.labels import get_labels
from app.video import VideoAnalyzer, VideoThread, draw_tracks
from app.vision.multi import detect_leaves

LEAF = (60, 140, 50)
SOIL = (90, 70, 50)


def scene(centres, w=640, h=480):
    img = np.full((h, w, 3), SOIL, np.uint8)
    for cx, cy in centres:
        cv2.ellipse(img, (cx, cy), (70, 45), 30, 0, 360, LEAF, -1)
    return img


class FakeClassifier:
    """Always the same class, with high confidence."""

    def __init__(self, key: str):
        self.labels = get_labels()
        self.idx = self.labels.by_key(key).id
        self.threshold = self.plant_threshold = 0.6

    def probabilities_batch(self, images):
        p = np.full((len(images), len(self.labels)), 0.05 / (len(self.labels) - 1))
        p[:, self.idx] = 0.95
        return p


def _keys():
    labels = get_labels()
    sick = next(c.key for c in labels if not c.healthy and not c.unknown)
    healthy = next(c.key for c in labels if c.healthy)
    return sick, healthy


def test_detect_several_leaves():
    leaves = detect_leaves(scene([(130, 120), (420, 140), (300, 360)]))
    assert len(leaves) == 3
    for L in leaves:
        x, y, w, h = L.bbox
        assert 0 <= x and 0 <= y and x + w <= 640 and y + h <= 480


def test_no_leaves_on_soil():
    assert detect_leaves(np.full((480, 640, 3), SOIL, np.uint8)) == []


def test_tracking_and_illness_verdict():
    sick, _ = _keys()
    an = VideoAnalyzer(FakeClassifier(sick))
    for i in range(4):  # leaves drift slowly: the same tracks must be kept
        tracks = an.analyze(scene([(130 + 5 * i, 120), (420 + 5 * i, 140)]), t=i * 0.33)
    assert len(tracks) == 2 and {t.id for t in tracks} == {1, 2}
    v = an.verdict()
    assert v.status == "illness" and v.findings[0].class_key == sick
    out = draw_tracks(scene([(150, 120), (440, 140)]), tracks, get_labels())
    assert out.shape == (480, 640, 3)


def test_healthy_and_empty_verdicts():
    _, healthy = _keys()
    an = VideoAnalyzer(FakeClassifier(healthy))
    for i in range(3):
        an.analyze(scene([(200, 200)]), t=i * 0.4)
    assert an.verdict().status == "healthy"
    empty = VideoAnalyzer(FakeClassifier(healthy))
    empty.analyze(np.full((480, 640, 3), SOIL, np.uint8), t=0)
    assert empty.verdict().status == "none"


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "plants.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    for i in range(30):
        out.write(cv2.cvtColor(scene([(150 + 3 * i, 150), (450, 300)]), cv2.COLOR_RGB2BGR))
    out.release()
    return path


def test_video_source_reads_rgb_frames(video_file, tmp_path):
    src = VideoSource(video_file)
    frame, t = src.read()
    assert frame.shape == (480, 640, 3) and src.fps == pytest.approx(10, abs=0.5)
    assert frame[240, 5, 0] > frame[240, 5, 2]  # soil is red > blue: RGB order, not BGR
    src.close()
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(InvalidImageError):
        VideoSource(bad)


def test_video_thread_end_to_end(qtbot, video_file):
    sick, _ = _keys()
    t = VideoThread(video_file, lambda: FakeClassifier(sick))
    frames = []
    t.frame_ready.connect(lambda f, ts: frames.append(f))
    with qtbot.waitSignal(t.ended, timeout=60000) as sig:
        t.start()
    t.wait(5000)
    assert frames and sig.args[0].status == "illness"
