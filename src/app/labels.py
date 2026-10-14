"""Loader for labels.json, the single source of truth for plants, class IDs and names.

No other module may hardcode class or plant names; everything goes through `Labels`.
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
UNKNOWN_GROUP = "__unknown__"  # internal plant-group key of the "some other plant" class


@dataclass(frozen=True)
class PlantInfo:
    name: str
    latin: str = ""


@dataclass(frozen=True)
class LabelInfo:
    id: int
    key: str
    plant: str
    condition: str
    healthy: bool
    condition_id: str = ""
    unknown: bool = False  # the "some other plant" class

    @property
    def display_name(self) -> str:
        """Human-readable name, ASCII-only so OpenCV can draw it."""
        return self.condition if self.unknown else f"{self.plant} - {self.condition}"


class Labels:
    def __init__(self, classes: list[LabelInfo], plants: list[PlantInfo] | None = None, dataset: str = "") -> None:
        ids = [c.id for c in classes]
        if ids != list(range(len(classes))):
            raise ValueError(f"labels.json class ids must be 0..N-1 in order, got {ids}")
        keys = [c.key for c in classes]
        if len(set(keys)) != len(keys) or UNCERTAIN_KEY in keys:
            raise ValueError("labels.json keys must be unique and must not be 'uncertain'")
        self.classes = tuple(classes)
        self.dataset = dataset
        self._by_key = {c.key: c for c in classes}
        known = sorted({c.plant for c in classes if not c.unknown})
        given = {p.name: p for p in plants or []}
        missing = [p for p in known if p not in given]
        self.plant_info = {p: given.get(p, PlantInfo(p)) for p in known}
        if plants and missing:
            raise ValueError(f"labels.json plants list is missing {missing}")
        # Plant groups: index lists used to sum class probabilities per plant.
        self._groups: dict[str, list[int]] = {}
        for c in classes:
            self._groups.setdefault(UNKNOWN_GROUP if c.unknown else c.plant, []).append(c.id)
        unknown = [c for c in classes if c.unknown]
        self.unknown_name = unknown[0].display_name if unknown else ""

    @classmethod
    def load(cls, path: Path | str = LABELS_PATH) -> "Labels":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = LabelInfo.__dataclass_fields__
        classes = [LabelInfo(**{k: v for k, v in c.items() if k in fields}) for c in raw["classes"]]
        plants = [PlantInfo(**p) for p in raw.get("plants", [])]
        return cls(classes, plants, raw.get("dataset", ""))

    def __len__(self) -> int:
        return len(self.classes)

    def __iter__(self):
        return iter(self.classes)

    def by_id(self, class_id: int) -> LabelInfo:
        return self.classes[class_id]

    def by_key(self, key: str) -> LabelInfo | None:
        return self._by_key.get(key)

    def display_name(self, key: str, plant: str | None = None) -> str:
        info = self.by_key(key)
        if info is not None:
            return info.display_name
        if key == UNCERTAIN_KEY:
            return f"{plant} - {UNCERTAIN_NAME}" if plant else UNCERTAIN_NAME
        return key

    @property
    def keys(self) -> list[str]:
        return [c.key for c in self.classes]

    @property
    def plants(self) -> list[str]:
        """Known plant names (sorted), excluding the unknown-plant group."""
        return list(self.plant_info)

    @property
    def plant_groups(self) -> dict[str, list[int]]:
        """Plant name (or UNKNOWN_GROUP for the unknown-plant class) -> class ids."""
        return self._groups

    def group_name(self, group: str) -> str:
        return self.unknown_name if group == UNKNOWN_GROUP else group

    def classes_of(self, plant: str) -> list[LabelInfo]:
        return [self.classes[i] for i in self._groups.get(plant, [])]


@lru_cache(maxsize=1)
def get_labels() -> Labels:
    return Labels.load()
