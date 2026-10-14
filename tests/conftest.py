import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest


@pytest.fixture
def leaf_image() -> np.ndarray:
    """Synthetic RGB 'leaf': green ellipse with brown spots on a light background."""
    img = np.full((360, 480, 3), 235, np.uint8)
    cv2.ellipse(img, (240, 180), (150, 95), 20, 0, 360, (60, 140, 50), -1)
    for c in ((200, 160), (280, 200), (250, 140)):
        cv2.circle(img, c, 9, (110, 80, 40), -1)
    return img


@pytest.fixture
def store(tmp_path):
    from app.storage.db import SqliteScanStore

    s = SqliteScanStore(tmp_path / "test.db", tmp_path / "scans")
    yield s
    s.close()
