"""Scan history storage (SQLite + PNG files)."""
from __future__ import annotations

from app.contracts import ScanRecord
from app.storage.db import SqliteScanStore

_default: SqliteScanStore | None = None


def default_store() -> SqliteScanStore:
    global _default
    if _default is None:
        from app.config import Settings

        s = Settings()
        _default = SqliteScanStore(s.db_path, s.scans_dir)
    return _default


def save_scan(record: ScanRecord) -> int:
    """Module-level convenience matching the documented interface."""
    return default_store().save_scan(record)


__all__ = ["SqliteScanStore", "save_scan", "default_store"]
