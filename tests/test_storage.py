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
