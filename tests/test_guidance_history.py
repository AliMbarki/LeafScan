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
        if not info.healthy:
            assert g.symptoms and g.actions
    assert repo.get("uncertain") is None


def test_guidance_out_of_sync_is_rejected(tmp_path):
    raw = json.loads((ROOT / "src/app/guidance/content.json").read_text(encoding="utf-8"))
    raw["classes"].pop(next(iter(raw["classes"])))
    bad = tmp_path / "content.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(GuidanceError):
        GuidanceRepository(path=bad)


def test_no_class_names_hardcoded_in_code():
    """labels.json is the single source of truth: no class key/name may appear in code."""
    labels = get_labels()
    needles = set()
    for c in labels:
        needles |= {c.key, c.source_label, c.display_name}
        if not c.healthy:
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
    assert order[-1] == "Not sure" and len(order) == len(set(order))
