from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ROOT, Settings


DEFAULT_FORMULA_SPEC = ROOT / "config" / "indicator_formulas.json"


def formula_spec_path(settings: Settings | None = None) -> Path:
    if settings is None:
        return DEFAULT_FORMULA_SPEC
    configured = settings.raw.get("formula_spec_path")
    if not configured:
        return DEFAULT_FORMULA_SPEC
    path = Path(str(configured))
    if path.is_absolute():
        return path
    return settings.config_path.parent / path


def load_formula_spec(settings: Settings | None = None) -> dict[str, Any]:
    path = formula_spec_path(settings)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def formula_section(spec: dict[str, Any], name: str) -> dict[str, Any]:
    value = spec.get(name, {})
    return value if isinstance(value, dict) else {}


def merged_signal_config(settings: Settings, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    formula = spec if spec is not None else load_formula_spec(settings)
    cfg = dict(settings.section("signals"))
    thresholds = formula_section(formula, "thresholds")
    cfg.update(thresholds)
    if "history_window_trading_days" in formula:
        cfg["stock_signal_history_trading_days"] = formula["history_window_trading_days"]

    indicators = formula_section(formula, "technical_indicators")
    ma = formula_section(indicators, "moving_averages")
    rsi = formula_section(indicators, "rsi")
    macd = formula_section(indicators, "macd")
    if ma.get("periods"):
        cfg["moving_averages"] = ma.get("periods")
    if rsi.get("period"):
        cfg["rsi_period"] = rsi.get("period")
    if macd.get("fast"):
        cfg["macd_fast"] = macd.get("fast")
    if macd.get("slow"):
        cfg["macd_slow"] = macd.get("slow")
    if macd.get("signal"):
        cfg["macd_signal"] = macd.get("signal")
    return cfg
