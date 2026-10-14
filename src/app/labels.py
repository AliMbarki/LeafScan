"""Loader for labels.json, the single source of truth for class IDs and names.

No other module may hardcode class names; everything goes through `Labels`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LABELS_PATH = Path(__file__).with_name("labels.json")

# Stored in place of a class key when the classifier is not confident enough.
UNCERTAIN_KEY = "uncertain"
UNCERTAIN_NAME = "Not sure"


@dataclass(frozen=True)
class LabelInfo:
    id: int
    key: str
    plant: str
    condition: str
    healthy: bool
    source_label: str

    @property
    def display_name(self) -> str:
        """Human-readable name, ASCII-only so OpenCV can draw it."""
        return f"{self.plant} - {self.condition}"


class Labels:
    def __init__(self, classes: list[LabelInfo], dataset: str = "") -> None:
        ids = [c.id for c in classes]
        if ids != list(range(len(classes))):
            raise ValueError(f"labels.json class ids must be 0..N-1 in order, got {ids}")
        keys = [c.key for c in classes]
        if len(set(keys)) != len(keys) or UNCERTAIN_KEY in keys:
            raise ValueError("labels.json keys must be unique and must not be 'uncertain'")
        self.classes = tuple(classes)
        self.dataset = dataset
        self._by_key = {c.key: c for c in classes}

    @classmethod
    def load(cls, path: Path | str = LABELS_PATH) -> "Labels":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        classes = [LabelInfo(**{k: c[k] for k in LabelInfo.__dataclass_fields__}) for c in raw["classes"]]
        return cls(classes, raw.get("dataset", ""))

    def __len__(self) -> int:
        return len(self.classes)

    def __iter__(self):
        return iter(self.classes)

    def by_id(self, class_id: int) -> LabelInfo:
        return self.classes[class_id]

    def by_key(self, key: str) -> LabelInfo | None:
        return self._by_key.get(key)

    def display_name(self, key: str) -> str:
        info = self.by_key(key)
        if info is not None:
            return info.display_name
        return UNCERTAIN_NAME if key == UNCERTAIN_KEY else key

    @property
    def keys(self) -> list[str]:
        return [c.key for c in self.classes]

    @property
    def plants(self) -> list[str]:
        return sorted({c.plant for c in self.classes})


@lru_cache(maxsize=1)
def get_labels() -> Labels:
    return Labels.load()
