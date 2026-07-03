from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from tech_signal.config import Settings
from tech_signal.db import connect, qname


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    group: str
    description: str
    higher_is_better: bool = True


FACTOR_DEFINITIONS: list[FactorDefinition] = [
    FactorDefinition("trend_rs_5", "trend", "5-day relative strength versus MA5"),
    FactorDefinition("trend_rs_10", "trend", "10-day relative strength versus MA10"),
    FactorDefinition("trend_rs_20", "trend", "20-day relative strength versus MA20"),
    FactorDefinition("trend_ma_bullish", "trend", "MA5 > MA10 > MA20 > MA60"),
    FactorDefinition("trend_new_high_20", "trend", "Adjusted close near 20-day high"),
    FactorDefinition("trend_breakout_20", "trend", "20-day breakout strength"),
    FactorDefinition("reversal_pullback_ma20", "reversal", "Pullback to MA20 without breaking it"),
    FactorDefinition("volume_amount_expansion", "volume", "Trading amount expansion versus 20-day volume"),
    FactorDefinition("volume_ratio_5", "volume", "Volume ratio versus previous 5-day average"),
    FactorDefinition("volume_ratio_20", "volume", "Volume ratio versus previous 20-day average"),
    FactorDefinition("volume_turnover_rate", "volume", "Turnover rate"),
    FactorDefinition("volume_turnover_change_5", "volume", "Turnover rate versus previous 5-day average"),
    FactorDefinition("volume_breakout", "volume", "20-day breakout with volume expansion"),
    FactorDefinition("volume_shrink_pullback", "volume", "Shrink-volume pullback near MA20"),
    FactorDefinition("moneyflow_net_rate", "moneyflow", "Net moneyflow amount divided by turnover amount"),
    FactorDefinition("moneyflow_net_yi", "moneyflow", "Net moneyflow amount in 100m CNY"),
    FactorDefinition("moneyflow_score", "moneyflow", "Current system moneyflow score"),
    FactorDefinition("moneyflow_consecutive_3", "moneyflow", "Recent 3-day net moneyflow persistence"),
    FactorDefinition("sentiment_limit_up", "sentiment", "Limit-up signal"),
    FactorDefinition("sentiment_limit_strength", "sentiment", "Limit-up times or board strength"),
    FactorDefinition("sentiment_score", "sentiment", "Current system limit/sentiment score"),
    FactorDefinition("lhb_net_buy_yi", "lhb", "LHB net buy in 100m CNY"),
    FactorDefinition("lhb_institution_net_buy_yi", "lhb", "Institution net buy in 100m CNY"),
    FactorDefinition("lhb_northbound_net_buy_yi", "lhb", "Northbound net buy in 100m CNY"),
    FactorDefinition("lhb_buy_amount_ratio", "lhb", "LHB net buy divided by trading amount"),
    FactorDefinition("risk_high_bias5", "risk", "High short-term BIAS risk", higher_is_better=False),
    FactorDefinition("risk_below_ma20", "risk", "Break below MA20 risk", higher_is_better=False),
    FactorDefinition("risk_high_turnover_stall", "risk", "High-turnover but stalled price risk", higher_is_better=False),
    FactorDefinition("risk_volume_long_black", "risk", "Heavy-volume negative candle risk", higher_is_better=False),
    FactorDefinition("risk_broken_board", "risk", "Broken-board risk", higher_is_better=False),
    FactorDefinition("reversal_low_rsi", "reversal", "Low RSI reversal pressure"),
    FactorDefinition("relative_strength_market", "relative_strength", "Daily return minus cross-sectional market return"),
]

FACTOR_BY_NAME = {item.name: item for item in FACTOR_DEFINITIONS}
DEFAULT_FACTOR_NAMES = [item.name for item in FACTOR_DEFINITIONS]
RISK_FACTOR_NAMES = {item.name for item in FACTOR_DEFINITIONS if not item.higher_is_better}


def normalize_date(value: str | date) -> str:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def factor_group(name: str) -> str:
    item = FACTOR_BY_NAME.get(name)
    return item.group if item else "unknown"


def selected_factor_names(names: Iterable[str] | None = None) -> list[str]:
    if not names:
        return DEFAULT_FACTOR_NAMES.copy()
    allowed = set(DEFAULT_FACTOR_NAMES)
    out = [name for name in names if name in allowed]
    return out or DEFAULT_FACTOR_NAMES.copy()


def _numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator.abs() > 1e-12)


def load_factor_base(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    lookback_calendar_days: int = 90,
    forward_calendar_days: int = 40,
) -> pd.DataFrame:
    start_value = normalize_date(start_date)
    end_value = normalize_date(end_date)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT s.trade_date, s.ts_code, s.name, s.industry, s.concepts,
                   s.close, s.pct_chg, s.amount_yi, s.turnover_rate,
                   s.ma5, s.ma10, s.ma20, s.ma60, s.bias5, s.rsi14,
                   s.macd, s.macd_signal, s.macd_hist,
                   s.vol_ma5, s.vol_ma20, s.prev_vol_ma5, s.prev_vol_ma20,
                   s.volume_ratio_5, s.volume_ratio_20, s.high20, s.low20,
                   s.volume_ratio, s.technical_score, s.price_volume_score,
                   s.moneyflow_score, s.limit_score, s.lhb_score,
                   s.total_signal_score, s.signal_level, s.trend_phase,
                   s.volume_state, s.limit_status, s.is_limit_up,
                   s.is_limit_down, s.is_broken_board, s.limit_times,
                   s.open_times, s.net_mf_amount, s.net_mf_amount_yi,
                   s.net_mf_rate, s.lhb_net_buy_yi,
                   s.institution_net_buy_yi, s.northbound_net_buy_yi,
                   b.open, b.high, b.low, b.pre_close, b.vol, b.amount,
                   b.adj_open, b.adj_high, b.adj_low, b.adj_close
            FROM {qname(settings, 'stock_signal_daily')} s
            JOIN {qname(settings, 'daily_bars')} b
              ON b.trade_date=s.trade_date AND b.ts_code=s.ts_code
            WHERE s.trade_date BETWEEN %s::date - (%s || ' days')::interval
                                  AND %s::date + (%s || ' days')::interval
              AND b.adj_close IS NOT NULL
            ORDER BY s.ts_code, s.trade_date
            """,
            (start_value, lookback_calendar_days, end_value, forward_calendar_days),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    numeric_columns = [
        "close",
        "pct_chg",
        "amount_yi",
        "turnover_rate",
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "bias5",
        "rsi14",
        "macd",
        "macd_signal",
        "macd_hist",
        "vol_ma5",
        "vol_ma20",
        "prev_vol_ma5",
        "prev_vol_ma20",
        "volume_ratio_5",
        "volume_ratio_20",
        "high20",
        "low20",
        "volume_ratio",
        "technical_score",
        "price_volume_score",
        "moneyflow_score",
        "limit_score",
        "lhb_score",
        "total_signal_score",
        "limit_times",
        "open_times",
        "net_mf_amount",
        "net_mf_amount_yi",
        "net_mf_rate",
        "lhb_net_buy_yi",
        "institution_net_buy_yi",
        "northbound_net_buy_yi",
        "open",
        "high",
        "low",
        "pre_close",
        "vol",
        "amount",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
    ]
    _numeric(df, numeric_columns)
    for column in ["is_limit_up", "is_limit_down", "is_broken_board"]:
        if column in df.columns:
            df[column] = df[column].fillna(False).astype(bool)
    return df


def compute_factor_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.sort_values(["ts_code", "trade_date"]).copy()
    grouped = out.groupby("ts_code", group_keys=False)

    out["trend_rs_5"] = _safe_div(out["adj_close"], out["ma5"]) - 1.0
    out["trend_rs_10"] = _safe_div(out["adj_close"], out["ma10"]) - 1.0
    out["trend_rs_20"] = _safe_div(out["adj_close"], out["ma20"]) - 1.0
    out["trend_ma_bullish"] = ((out["ma5"] > out["ma10"]) & (out["ma10"] > out["ma20"]) & (out["ma20"] > out["ma60"])).astype(float)
    out["trend_new_high_20"] = (out["adj_close"] >= out["high20"] * 0.995).astype(float)
    out["trend_breakout_20"] = (_safe_div(out["adj_close"], out["high20"]) - 0.995).clip(lower=-0.1, upper=0.1)
    out["reversal_pullback_ma20"] = (
        (out["adj_low"] <= out["ma20"] * 1.01)
        & (out["adj_close"] >= out["ma20"])
        & (out["volume_ratio_20"].fillna(1.0) <= 1.15)
    ).astype(float)

    out["volume_amount_expansion"] = out["volume_ratio_20"]
    out["volume_turnover_rate"] = out["turnover_rate"]
    turnover_ma5 = grouped["turnover_rate"].transform(lambda s: s.rolling(5, min_periods=3).mean().shift(1))
    out["volume_turnover_change_5"] = _safe_div(out["turnover_rate"], turnover_ma5)
    out["volume_breakout"] = out["trend_new_high_20"] * out["volume_ratio_20"]
    out["volume_shrink_pullback"] = out["reversal_pullback_ma20"] * (1.0 - out["volume_ratio_20"]).clip(lower=0, upper=1.0)

    out["moneyflow_net_rate"] = out["net_mf_rate"]
    out["moneyflow_net_yi"] = out["net_mf_amount_yi"]
    out["moneyflow_consecutive_3"] = grouped["net_mf_amount"].transform(
        lambda s: (s > 0).astype(float).rolling(3, min_periods=1).sum()
    )

    out["sentiment_limit_up"] = out["is_limit_up"].astype(float)
    out["sentiment_limit_strength"] = np.where(out["is_limit_up"], out["limit_times"].fillna(1.0), 0.0)
    out["sentiment_score"] = out["limit_score"]

    out["lhb_buy_amount_ratio"] = _safe_div(out["lhb_net_buy_yi"], out["amount_yi"])

    out["risk_high_bias5"] = (out["bias5"] - 8.0).clip(lower=0)
    out["risk_below_ma20"] = (out["adj_close"] < out["ma20"]).astype(float)
    out["risk_high_turnover_stall"] = (
        (out["turnover_rate"] >= 8.0)
        & (out["pct_chg"].abs() <= 1.0)
        & (out["volume_ratio_20"].fillna(0) >= 1.2)
    ).astype(float)
    out["risk_volume_long_black"] = ((out["pct_chg"] <= -3.0) & (out["volume_ratio_20"].fillna(0) >= 1.5)).astype(float)
    out["risk_broken_board"] = out["is_broken_board"].astype(float)

    out["reversal_low_rsi"] = (35.0 - out["rsi14"]).clip(lower=0)
    market_return = out.groupby("trade_date")["pct_chg"].transform("median")
    out["relative_strength_market"] = out["pct_chg"] - market_return

    for definition in FACTOR_DEFINITIONS:
        if definition.name in out.columns:
            out[definition.name] = pd.to_numeric(out[definition.name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def add_cross_sectional_ranks(df: pd.DataFrame, factor_names: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for name in factor_names:
        if name not in out.columns:
            continue
        rank_col = f"{name}__rank"
        pct_col = f"{name}__pct_rank"
        valid = out[name].notna()
        out[rank_col] = np.nan
        out[pct_col] = np.nan
        out.loc[valid, rank_col] = out.loc[valid].groupby("trade_date")[name].rank(method="average")
        counts = out.loc[valid].groupby("trade_date")[name].transform("count")
        out.loc[valid, pct_col] = out.loc[valid, rank_col] / counts.where(counts > 0)
    return out


def target_slice(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_value = pd.to_datetime(normalize_date(start_date)).date()
    end_value = pd.to_datetime(normalize_date(end_date)).date()
    return df[(df["trade_date"] >= start_value) & (df["trade_date"] <= end_value)].copy()

