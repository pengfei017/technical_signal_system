from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_SECRET_FILES = (
    Path("D:/codex_research_private/secrets.json"),
    Path.home() / ".codex_research" / "secrets.json",
)


def _expand_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def secret_file_candidates() -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get("RESEARCH_SECRETS_FILE")
    if env_path:
        paths.append(_expand_path(env_path))
    paths.extend(DEFAULT_SECRET_FILES)
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _load_secret_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v not in (None, "")}


def _read_windows_user_env(name: str) -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value or "").strip()
    except Exception:
        return ""


def get_secret(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value not in (None, ""):
        return str(value)
    for path in secret_file_candidates():
        try:
            secrets = _load_secret_file(path)
        except Exception:
            continue
        value = secrets.get(name)
        if value not in (None, ""):
            return str(value)
    return _read_windows_user_env(name) or default
