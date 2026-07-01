from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "settings.json"


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    config_path: Path

    @property
    def schema(self) -> str:
        return str(self.raw.get("schema", "tech_signal"))

    @property
    def output_root(self) -> Path:
        return Path(str(self.raw.get("output_root", "E:/technical_signals")))

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    @property
    def cache_dir(self) -> Path:
        return self.output_root / "cache"

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        return value if isinstance(value, dict) else {}


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else DEFAULT_CONFIG
    data = json.loads(config_path.read_text(encoding="utf-8"))
    settings = Settings(raw=data, config_path=config_path)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
