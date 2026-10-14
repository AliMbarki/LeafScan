"""Runtime configuration. Values can be overridden with LEAFSCAN_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.contracts import DEFAULT_UNCERTAINTY_THRESHOLD

APP_NAME = "LeafScan"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: _env_path("LEAFSCAN_DATA_DIR", PROJECT_ROOT / "data" / "app"))
    models_dir: Path = field(default_factory=lambda: _env_path("LEAFSCAN_MODELS_DIR", PROJECT_ROOT / "models"))
    uncertainty_threshold: float = field(
        default_factory=lambda: float(os.environ.get("LEAFSCAN_THRESHOLD", DEFAULT_UNCERTAINTY_THRESHOLD))
    )
    plant_threshold: float = field(default_factory=lambda: float(os.environ.get("LEAFSCAN_PLANT_THRESHOLD", 0.55)))
    camera_index: int = 0
    use_camera: bool = True
    use_mocks: bool = False

    @property
    def db_path(self) -> Path:
        return self.data_dir / "leafscan.db"

    @property
    def scans_dir(self) -> Path:
        return self.data_dir / "scans"
