"""SQLite-backed scan history. Images live on disk; the DB stores relative paths."""
from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from app.capture import save_rgb_png
from app.contracts import Plant, ScanRecord, StoredScan

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS plants (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scans (
    id            INTEGER PRIMARY KEY,
    plant_id      INTEGER NOT NULL REFERENCES plants(id) ON DELETE CASCADE,
    timestamp     TEXT NOT NULL,            -- ISO-8601 local time with offset
    prediction    TEXT NOT NULL,            -- class key from labels.json, or 'uncertain'
    class_id      INTEGER,
    confidence    REAL NOT NULL,
    is_uncertain  INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    image_path    TEXT NOT NULL,            -- relative to the data dir
    overlay_path  TEXT
);
CREATE INDEX IF NOT EXISTS idx_scans_plant_time ON scans(plant_id, timestamp);
"""

_PLANT_N = re.compile(r"^plant (\d+)$", re.IGNORECASE)


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SqliteScanStore:
    def __init__(self, db_path: Path | str, image_dir: Path | str | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.root = self.db_path.parent
        self.image_dir = Path(image_dir) if image_dir else self.root / "scans"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ plants

    def get_or_create_plant(self, name: str) -> Plant:
        name = " ".join(name.split())
        if not name:
            raise ValueError("plant name cannot be empty")
        row = self.conn.execute("SELECT * FROM plants WHERE name = ?", (name,)).fetchone()
        if row is None:
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            cur = self.conn.execute("INSERT INTO plants(name, created_at) VALUES (?, ?)", (name, now))
            self.conn.commit()
            return Plant(cur.lastrowid, name, _dt(now))
        return Plant(row["id"], row["name"], _dt(row["created_at"]))

    def suggest_plant_name(self) -> str:
        taken = {
            int(m.group(1))
            for (name,) in self.conn.execute("SELECT name FROM plants")
            if (m := _PLANT_N.match(name))
        }
        n = 1
        while n in taken:
            n += 1
        return f"Plant {n}"

    def list_plants(self) -> list[Plant]:
        rows = self.conn.execute(
            """SELECT p.id, p.name, p.created_at, COUNT(s.id) AS n, MAX(s.timestamp) AS last
               FROM plants p LEFT JOIN scans s ON s.plant_id = p.id
               GROUP BY p.id ORDER BY p.name COLLATE NOCASE"""
        ).fetchall()
        return [Plant(r["id"], r["name"], _dt(r["created_at"]), r["n"], _dt(r["last"])) for r in rows]

    def rename_plant(self, plant_id: int, name: str) -> None:
        self.conn.execute("UPDATE plants SET name = ? WHERE id = ?", (" ".join(name.split()), plant_id))
        self.conn.commit()

    # ------------------------------------------------------------------- scans

    def save_scan(self, record: ScanRecord) -> int:
        if self.conn.execute("SELECT 1 FROM plants WHERE id = ?", (record.plant_id,)).fetchone() is None:
            raise ValueError(f"unknown plant id {record.plant_id}")
        stem = f"{record.timestamp:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        month = self.image_dir / f"{record.timestamp:%Y-%m}"
        image_path = month / f"{stem}.png"
        save_rgb_png(record.image, image_path)
        overlay_rel = None
        if record.overlay_image is not None:
            overlay_path = month / f"{stem}_overlay.png"
            save_rgb_png(record.overlay_image, overlay_path)
            overlay_rel = overlay_path.relative_to(self.root).as_posix()
        cur = self.conn.execute(
            """INSERT INTO scans(plant_id, timestamp, prediction, class_id, confidence, is_uncertain,
                                 model_version, image_path, overlay_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.plant_id,
                record.timestamp.isoformat(timespec="seconds"),
                record.prediction,
                record.class_id,
                float(record.confidence),
                int(record.is_uncertain),
                record.model_version,
                image_path.relative_to(self.root).as_posix(),
                overlay_rel,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_scans(self, plant_id: int | None = None, limit: int | None = None) -> list[StoredScan]:
        sql = """SELECT s.*, p.name AS plant_name FROM scans s JOIN plants p ON p.id = s.plant_id"""
        args: list = []
        if plant_id is not None:
            sql += " WHERE s.plant_id = ?"
            args.append(plant_id)
        sql += " ORDER BY s.timestamp DESC, s.id DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [self._scan(r) for r in self.conn.execute(sql, args)]

    def delete_scan(self, scan_id: int) -> None:
        row = self.conn.execute("SELECT image_path, overlay_path FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if row is None:
            return
        self.conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        self.conn.commit()
        for rel in (row["image_path"], row["overlay_path"]):
            if rel:
                (self.root / rel).unlink(missing_ok=True)

    # ----------------------------------------------------------------- queries

    def scans_per_day(self, plant_id: int | None = None) -> list[tuple[str, int]]:
        where, args = ("WHERE plant_id = ?", (plant_id,)) if plant_id is not None else ("", ())
        rows = self.conn.execute(
            f"SELECT substr(timestamp, 1, 10) AS day, COUNT(*) FROM scans {where} GROUP BY day ORDER BY day", args
        )
        return [(day, n) for day, n in rows]

    def prediction_counts(self, plant_id: int | None = None) -> dict[str, int]:
        where, args = ("WHERE plant_id = ?", (plant_id,)) if plant_id is not None else ("", ())
        rows = self.conn.execute(
            f"SELECT prediction, COUNT(*) FROM scans {where} GROUP BY prediction ORDER BY COUNT(*) DESC", args
        )
        return {pred: n for pred, n in rows}

    def prediction_counts_by_plant(self) -> dict[str, dict[str, int]]:
        rows = self.conn.execute(
            """SELECT p.name, s.prediction, COUNT(*) FROM scans s JOIN plants p ON p.id = s.plant_id
               GROUP BY p.id, s.prediction ORDER BY p.name COLLATE NOCASE"""
        )
        out: dict[str, dict[str, int]] = {}
        for plant, pred, n in rows:
            out.setdefault(plant, {})[pred] = n
        return out

    def _scan(self, r: sqlite3.Row) -> StoredScan:
        return StoredScan(
            id=r["id"],
            plant_id=r["plant_id"],
            plant_name=r["plant_name"],
            timestamp=_dt(r["timestamp"]),
            prediction=r["prediction"],
            class_id=r["class_id"],
            confidence=r["confidence"],
            is_uncertain=bool(r["is_uncertain"]),
            model_version=r["model_version"],
            image_path=str(self.root / r["image_path"]),
            overlay_path=str(self.root / r["overlay_path"]) if r["overlay_path"] else None,
        )
