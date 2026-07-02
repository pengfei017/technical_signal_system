from __future__ import annotations

from typing import Any

import pandas as pd

from .formula_spec import formula_section, load_formula_spec


def _int_list(value: object, default: list[int]) -> list[int]:
    if not isinstance(value, list):
        return default
    out = []
    for item in value:
        try:
            period = int(item)
        except Exception:
            continue
        if period > 0 and period not in out:
            out.append(period)
    return out or default


def _int_value(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _str_value(value: object, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def add_indicators(df: pd.DataFrame, formula_spec: dict[str, Any] | None = None) -> pd.DataFrame:
    spec = formula_spec if formula_spec is not None else load_formula_spec()
    indicators = formula_section(spec, "technical_indicators")
    ma_cfg = formula_section(indicators, "moving_averages")
    volume_cfg = formula_section(indicators, "volume_averages")
    high_low_cfg = formula_section(indicators, "high_low")
    bias_cfg = formula_section(indicators, "bias")
    rsi_cfg = formula_section(indicators, "rsi")
    macd_cfg = formula_section(indicators, "macd")

    ma_periods = _int_list(ma_cfg.get("periods"), [5, 10, 20, 60])
    volume_periods = _int_list(volume_cfg.get("periods"), [5, 20])
    high_low_period = _int_value(high_low_cfg.get("period"), 20)
    bias_period = _int_value(bias_cfg.get("ma_period"), 5)
    rsi_period = _int_value(rsi_cfg.get("period"), 14)
    macd_fast = _int_value(macd_cfg.get("fast"), 12)
    macd_slow = _int_value(macd_cfg.get("slow"), 26)
    macd_signal = _int_value(macd_cfg.get("signal"), 9)
    macd_adjust = bool(macd_cfg.get("adjust", False))
    close_col = _str_value(ma_cfg.get("input") or rsi_cfg.get("input") or macd_cfg.get("input"), "adj_close")
    volume_col = _str_value(volume_cfg.get("input"), "vol")
    high_col = _str_value(high_low_cfg.get("high_input"), "adj_high")
    low_col = _str_value(high_low_cfg.get("low_input"), "adj_low")

    out = df.sort_values(["ts_code", "trade_date"]).copy()
    grouped = out.groupby("ts_code", group_keys=False)
    for period in ma_periods:
        out[f"ma{period}"] = grouped[close_col].transform(lambda s: s.rolling(period, min_periods=period).mean())
    for period in volume_periods:
        out[f"vol_ma{period}"] = grouped[volume_col].transform(lambda s: s.rolling(period, min_periods=period).mean())
        out[f"prev_vol_ma{period}"] = grouped[volume_col].transform(lambda s: s.rolling(period, min_periods=period).mean().shift(1))
        out[f"volume_ratio_{period}"] = out[volume_col] / out[f"prev_vol_ma{period}"].where(out[f"prev_vol_ma{period}"] > 0)
    out[f"high{high_low_period}"] = grouped[high_col].transform(lambda s: s.rolling(high_low_period, min_periods=high_low_period).max())
    out[f"low{high_low_period}"] = grouped[low_col].transform(lambda s: s.rolling(high_low_period, min_periods=high_low_period).min())
    if f"ma{bias_period}" in out.columns:
        bias_col = _str_value(bias_cfg.get("input"), close_col)
        out[f"bias{bias_period}"] = (out[bias_col] / out[f"ma{bias_period}"] - 1.0) * 100.0

    def _rsi(series: pd.Series, period: int = rsi_period) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100.0 - (100.0 / (1.0 + rs))

    rsi_col = _str_value(rsi_cfg.get("input"), close_col)
    out[f"rsi{rsi_period}"] = grouped[rsi_col].transform(_rsi)

    def _macd(series: pd.Series) -> pd.DataFrame:
        ema_fast = series.ewm(span=macd_fast, adjust=macd_adjust, min_periods=macd_fast).mean()
        ema_slow = series.ewm(span=macd_slow, adjust=macd_adjust, min_periods=macd_slow).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=macd_signal, adjust=macd_adjust, min_periods=macd_signal).mean()
        hist = macd - signal
        return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_hist": hist}, index=series.index)

    macd_col = _str_value(macd_cfg.get("input"), close_col)
    macd_frames = grouped[macd_col].apply(_macd).reset_index(level=0, drop=True)
    out[["macd", "macd_signal", "macd_hist"]] = macd_frames[["macd", "macd_signal", "macd_hist"]]
    return out
