"""Local, sourced guidance per class (content.json). Titles come from labels.json.

content.json (schema 2):
* ``conditions`` - shared cards keyed by a class's ``condition_id`` (e.g. one powdery-mildew card
  for squash, cherry, grape and mango),
* ``classes`` - per-class overrides of any field,
* ``plant_care`` - extra plant-specific care sources shown with "Healthy" results.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.contracts import Guidance, Source
from app.labels import Labels, get_labels

CONTENT_PATH = Path(__file__).with_name("content.json")
SEVERITIES = {"none", "low", "moderate", "high", "info"}


class GuidanceError(ValueError):
    pass


def _sources(items: list[dict]) -> tuple[Source, ...]:
    return tuple(Source(s["title"], s["url"], s.get("publisher", "")) for s in items)


class GuidanceRepository:
    def __init__(self, labels: Labels | None = None, path: Path | str = CONTENT_PATH) -> None:
        self.labels = labels or get_labels()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        conditions, overrides = raw.get("conditions", {}), raw.get("classes", {})
        care = raw.get("plant_care", {})
        unknown = set(overrides) - set(self.labels.keys)
        missing = sorted({c.condition_id for c in self.labels if c.condition_id not in conditions})
        if missing or unknown:
            raise GuidanceError(f"guidance content out of sync with labels.json (no card for condition ids "
                                f"{missing}, overrides for unknown classes {sorted(unknown)})")
        self._items: dict[str, Guidance] = {}
        for info in self.labels:
            entry = {**conditions[info.condition_id], **overrides.get(info.key, {})}
            if entry.get("severity") not in SEVERITIES:
                raise GuidanceError(f"{info.key}: unknown severity {entry.get('severity')!r}")
            sources = list(entry.get("sources", []))
            if info.healthy:
                sources = care.get(info.plant, []) + [s for s in sources if s not in care.get(info.plant, [])]
            if not sources:
                raise GuidanceError(f"{info.key}: guidance has no sources")
            self._items[info.key] = Guidance(
                class_key=info.key,
                title=info.display_name,
                summary=entry["summary"],
                severity=entry["severity"],
                symptoms=tuple(entry.get("symptoms", ())),
                actions=tuple(entry.get("actions", ())),
                prevention=tuple(entry.get("prevention", ())),
                sources=_sources(sources),
            )

    def get(self, class_key: str) -> Guidance | None:
        return self._items.get(class_key)

    def __len__(self) -> int:
        return len(self._items)


__all__ = ["GuidanceRepository", "GuidanceError"]
