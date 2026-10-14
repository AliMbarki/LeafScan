"""Local, sourced guidance per class (content.json). Titles come from labels.json."""
from __future__ import annotations

import json
from pathlib import Path

from app.contracts import Guidance, Source
from app.labels import Labels, get_labels

CONTENT_PATH = Path(__file__).with_name("content.json")
SEVERITIES = {"none", "low", "moderate", "high", "info"}


class GuidanceError(ValueError):
    pass


class GuidanceRepository:
    def __init__(self, labels: Labels | None = None, path: Path | str = CONTENT_PATH) -> None:
        self.labels = labels or get_labels()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))["classes"]
        missing = set(self.labels.keys) - set(raw)
        unknown = set(raw) - set(self.labels.keys)
        if missing or unknown:
            raise GuidanceError(f"guidance content out of sync with labels.json (missing {sorted(missing)}, "
                                f"unknown {sorted(unknown)})")
        self._items: dict[str, Guidance] = {}
        for key, entry in raw.items():
            if entry["severity"] not in SEVERITIES:
                raise GuidanceError(f"{key}: unknown severity {entry['severity']!r}")
            self._items[key] = Guidance(
                class_key=key,
                title=self.labels.display_name(key),
                summary=entry["summary"],
                severity=entry["severity"],
                symptoms=tuple(entry.get("symptoms", ())),
                actions=tuple(entry.get("actions", ())),
                prevention=tuple(entry.get("prevention", ())),
                sources=tuple(Source(s["title"], s["url"], s.get("publisher", "")) for s in entry["sources"]),
            )

    def get(self, class_key: str) -> Guidance | None:
        return self._items.get(class_key)

    def __len__(self) -> int:
        return len(self._items)


__all__ = ["GuidanceRepository", "GuidanceError"]
