from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from tech_signal.config import Settings
from tech_signal.db import connect, init_schema, qname, upsert_rows

from .backtest_engine import _add_execution_constraints, _load_trade_returns, _performance_from_daily_returns, _profit_loss_ratio
from .factor_definitions import normalize_date
from .factor_evaluator import _clean_float, _drawdown_stats, report_dir


TREND_MODEL_NAME = "trend_pure_v1"
TREND_TRACKING_HORIZONS = (3, 5, 10, 20)


@dataclass(frozen=True)
class TrendIndicator:
    name: str
    group: str
    direction: str
    description: str

    @property
    def higher_is_better(self) -> bool:
        return self.direction != "risk_high"


TREND_INDICATORS: tuple[TrendIndicator, ...] = (
    TrendIndicator("ret_5", "relative_strength", "positive", "近5日复权涨幅，相对强势"),
    TrendIndicator("ret_10", "relative_strength", "positive", "近10日复权涨幅，相对强势"),
    TrendIndicator("ret_20", "relative_strength", "positive", "近20日复权涨幅，相对强势"),
    TrendIndicator("ret_60", "relative_strength", "positive", "近60日复权涨幅，中期相对强势"),
    TrendIndicator("ma_bull_5_10_20", "trend", "positive", "MA5 > MA10 > MA20"),
    TrendIndicator("ma_bull_5_10_20_60", "trend", "positive", "MA5 > MA10 > MA20 > MA60"),
    TrendIndicator("price_above_ma20", "trend", "positive", "收盘站上MA20"),
    TrendIndicator("price_above_ma60", "trend", "positive", "收盘站上MA60"),
    TrendIndicator("ma20_slope_5", "trend", "positive", "MA20近5日斜率"),
    TrendIndicator("ma60_slope_10", "trend", "positive", "MA60近10日斜率"),
    TrendIndicator("new_high_20", "breakout", "positive", "接近20日复权收盘新高"),
    TrendIndicator("new_high_60", "breakout", "positive", "接近60日复权收盘新高"),
    TrendIndicator("new_high_120", "breakout", "positive", "接近120日复权收盘新高"),
    TrendIndicator("breakout_20", "breakout", "positive", "向上突破前20日高点幅度"),
    TrendIndicator("breakout_60", "breakout", "positive", "向上突破前60日高点幅度"),
    TrendIndicator("pullback_ma20", "pullback", "positive", "上升趋势中回踩MA20不破"),
    TrendIndicator("pullback_ma60", "pullback", "positive", "上升趋势中回踩MA60不破"),
    TrendIndicator("macd_above_signal", "momentum", "positive", "MACD在信号线上方"),
    TrendIndicator("macd_hist_rising", "momentum", "positive", "MACD柱较前一日改善"),
    TrendIndicator("rsi_mid_strong", "momentum", "positive", "RSI处于50-70强势但未过热区间"),
    TrendIndicator("volume_expansion_mild", "volume", "positive", "成交量温和放大"),
    TrendIndicator("amount_rank", "liquidity", "positive", "成交额横截面排名"),
    TrendIndicator("turnover_rank", "liquidity", "positive", "换手率横截面排名"),
    TrendIndicator("low_volatility_trend", "quality", "positive", "上升趋势中的较低波动质量"),
    TrendIndicator("bias5_overheat", "risk", "risk_high", "5日乖离过热，风险越高越扣分"),
    TrendIndicator("volume_expansion_extreme", "risk", "risk_high", "极端放量，风险越高越扣分"),
    TrendIndicator("long_upper_shadow", "risk", "risk_high", "长上影线，风险越高越扣分"),
    TrendIndicator("below_ma20", "risk", "risk_high", "跌破MA20，风险越高越扣分"),
    TrendIndicator("below_ma60", "risk", "risk_high", "跌破MA60，风险越高越扣分"),
    TrendIndicator("volatility_20", "risk", "risk_high", "20日波动率，风险越高越扣分"),
)


TREND_INDICATOR_MAP = {item.name: item for item in TREND_INDICATORS}

_FEATURE_CACHE_KEY: tuple[str, str] | None = None
_FEATURE_CACHE_FRAME: pd.DataFrame | None = None


TREND_COMBOS: dict[str, dict[str, float]] = {
    "ma_trend": {
        "ma_bull_5_10_20": 0.18,
        "ma_bull_5_10_20_60": 0.18,
        "price_above_ma20": 0.12,
        "price_above_ma60": 0.10,
        "ma20_slope_5": 0.18,
        "ma60_slope_10": 0.12,
        "ret_20": 0.12,
    },
    "breakout": {
        "new_high_20": 0.18,
        "new_high_60": 0.16,
        "breakout_20": 0.18,
        "breakout_60": 0.12,
        "volume_expansion_mild": 0.14,
        "amount_rank": 0.10,
        "ret_10": 0.12,
    },
    "pullback": {
        "pullback_ma20": 0.24,
        "pullback_ma60": 0.16,
        "price_above_ma20": 0.12,
        "ma20_slope_5": 0.16,
        "low_volatility_trend": 0.14,
        "macd_hist_rising": 0.10,
        "volume_expansion_extreme": 0.08,
    },
    "macd_momentum": {
        "macd_above_signal": 0.18,
        "macd_hist_rising": 0.18,
        "rsi_mid_strong": 0.14,
        "ret_5": 0.12,
        "ret_10": 0.12,
        "ma20_slope_5": 0.14,
        "volume_expansion_mild": 0.12,
    },
    "quality_trend": {
        "ma20_slope_5": 0.14,
        "ma60_slope_10": 0.12,
        "price_above_ma20": 0.10,
        "low_volatility_trend": 0.16,
        "amount_rank": 0.10,
        "turnover_rank": 0.08,
        "bias5_overheat": 0.10,
        "long_upper_shadow": 0.10,
        "below_ma20": 0.10,
        "volatility_20": 0.10,
    },
    "all_technical": {
        "ret_5": 0.06,
        "ret_10": 0.08,
        "ret_20": 0.08,
        "ma_bull_5_10_20": 0.08,
        "ma_bull_5_10_20_60": 0.06,
        "price_above_ma20": 0.06,
        "price_above_ma60": 0.05,
        "ma20_slope_5": 0.08,
        "new_high_20": 0.07,
        "new_high_60": 0.06,
        "breakout_20": 0.07,
        "pullback_ma20": 0.06,
        "macd_above_signal": 0.05,
        "macd_hist_rising": 0.05,
        "volume_expansion_mild": 0.06,
        "amount_rank": 0.04,
        "bias5_overheat": 0.05,
        "volume_expansion_extreme": 0.04,
        "long_upper_shadow": 0.04,
        "below_ma20": 0.04,
    },
}


def _open_date_floor(settings: Settings, start_date: str, lookback: int = 260) -> str:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT min(cal_date)::text AS d
            FROM (
                SELECT cal_date
                FROM {qname(settings, 'trade_calendar')}
                WHERE is_open=true AND cal_date <= %s
                ORDER BY cal_date DESC
                LIMIT %s
            ) x
            """,
            (normalize_date(start_date), int(lookback)),
        )
        row = cur.fetchone()
    return str(row["d"]) if row and row["d"] else normalize_date(start_date)


def _normalize_numeric(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _pct_rank(data: pd.DataFrame, column: str, *, higher_is_better: bool = True) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index)
    series = pd.to_numeric(data[column], errors="coerce")
    if higher_is_better:
        return series.groupby(data["trade_date"]).rank(pct=True, method="average")
    return 1.0 - series.groupby(data["trade_date"]).rank(pct=True, method="average") + (1.0 / data.groupby("trade_date")["ts_code"].transform("count"))


def load_trend_base(settings: Settings, start_date: str, end_date: str, *, lookback_days: int = 260) -> pd.DataFrame:
    start_value = normalize_date(start_date)
    end_value = normalize_date(end_date)
    floor_date = _open_date_floor(settings, start_value, lookback_days)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT b.trade_date,
                   b.ts_code,
                   max(s.name) AS name,
                   max(s.industry) AS industry,
                   max(s.concepts) AS concepts,
                   b.open,
                   b.high,
                   b.low,
                   b.close,
                   b.pct_chg,
                   b.vol,
                   b.amount,
                   b.adj_open,
                   b.adj_high,
                   b.adj_low,
                   b.adj_close,
                   max(d.turnover_rate) AS turnover_rate,
                   max(d.volume_ratio) AS daily_volume_ratio
            FROM {qname(settings, 'daily_bars')} b
            LEFT JOIN {qname(settings, 'daily_basic')} d
              ON d.trade_date=b.trade_date AND d.ts_code=b.ts_code
            LEFT JOIN {qname(settings, 'stock_signal_daily')} s
              ON s.trade_date=b.trade_date AND s.ts_code=b.ts_code
            WHERE b.trade_date BETWEEN %s AND %s
              AND b.adj_close IS NOT NULL
            GROUP BY b.trade_date, b.ts_code, b.open, b.high, b.low, b.close, b.pct_chg,
                     b.vol, b.amount, b.adj_open, b.adj_high, b.adj_low, b.adj_close
            ORDER BY b.ts_code, b.trade_date
            """,
            (floor_date, end_value),
        )
        rows = cur.fetchall()
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.date
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pct_chg",
        "vol",
        "amount",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "turnover_rate",
        "daily_volume_ratio",
    ]
    return _normalize_numeric(data, numeric_columns)


def _rolling_transform(grouped: pd.core.groupby.SeriesGroupBy, window: int, func: str, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or max(2, int(window * 0.7))
    if func == "mean":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    if func == "max":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).max())
    if func == "std":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std())
    raise ValueError(f"Unsupported rolling func: {func}")


def compute_trend_features(settings: Settings, start_date: str, end_date: str) -> pd.DataFrame:
    global _FEATURE_CACHE_KEY, _FEATURE_CACHE_FRAME
    cache_key = (normalize_date(start_date), normalize_date(end_date))
    if _FEATURE_CACHE_KEY == cache_key and _FEATURE_CACHE_FRAME is not None:
        return _FEATURE_CACHE_FRAME.copy()
    data = load_trend_base(settings, start_date, end_date)
    if data.empty:
        raise RuntimeError(f"No daily_bars rows for trend_pure_v1: {start_date} to {end_date}")
    data = data.sort_values(["ts_code", "trade_date"]).copy()
    grouped_close = data.groupby("ts_code", group_keys=False)["adj_close"]
    grouped_high = data.groupby("ts_code", group_keys=False)["adj_high"]
    grouped_vol = data.groupby("ts_code", group_keys=False)["vol"]
    grouped_pct = data.groupby("ts_code", group_keys=False)["pct_chg"]

    for window in (5, 10, 20, 60):
        data[f"ma{window}"] = _rolling_transform(grouped_close, window, "mean")
    data["ret_5"] = grouped_close.pct_change(5)
    data["ret_10"] = grouped_close.pct_change(10)
    data["ret_20"] = grouped_close.pct_change(20)
    data["ret_60"] = grouped_close.pct_change(60)
    data["ma20_slope_5"] = data["ma20"] / data.groupby("ts_code", group_keys=False)["ma20"].shift(5) - 1.0
    data["ma60_slope_10"] = data["ma60"] / data.groupby("ts_code", group_keys=False)["ma60"].shift(10) - 1.0

    data["high20"] = _rolling_transform(grouped_high, 20, "max")
    data["high60"] = _rolling_transform(grouped_high, 60, "max")
    data["high120"] = _rolling_transform(grouped_high, 120, "max")
    data["prev_high20"] = data.groupby("ts_code", group_keys=False)["high20"].shift(1)
    data["prev_high60"] = data.groupby("ts_code", group_keys=False)["high60"].shift(1)
    data["vol_ma5"] = _rolling_transform(grouped_vol, 5, "mean")
    data["vol_ma20"] = _rolling_transform(grouped_vol, 20, "mean")
    data["volatility_20"] = _rolling_transform(grouped_pct, 20, "std") / 100.0
    data["volume_ratio_20"] = data["vol"] / data["vol_ma20"]
    data["amount_yi"] = data["amount"] / 100000.0

    ema_fast = data.groupby("ts_code", group_keys=False)["adj_close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema_slow = data.groupby("ts_code", group_keys=False)["adj_close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    data["macd"] = ema_fast - ema_slow
    data["macd_signal"] = data.groupby("ts_code", group_keys=False)["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    data["macd_hist"] = data["macd"] - data["macd_signal"]
    data["macd_hist_prev"] = data.groupby("ts_code", group_keys=False)["macd_hist"].shift(1)

    delta = grouped_close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(data["ts_code"], group_keys=False).transform(lambda s: s.rolling(14, min_periods=10).mean())
    avg_loss = loss.groupby(data["ts_code"], group_keys=False).transform(lambda s: s.rolling(14, min_periods=10).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    data["ma_bull_5_10_20"] = ((data["ma5"] > data["ma10"]) & (data["ma10"] > data["ma20"])).astype(float)
    data["ma_bull_5_10_20_60"] = (
        (data["ma5"] > data["ma10"]) & (data["ma10"] > data["ma20"]) & (data["ma20"] > data["ma60"])
    ).astype(float)
    data["price_above_ma20"] = (data["adj_close"] > data["ma20"]).astype(float)
    data["price_above_ma60"] = (data["adj_close"] > data["ma60"]).astype(float)
    data["new_high_20"] = (data["adj_close"] >= data["high20"] * 0.995).astype(float)
    data["new_high_60"] = (data["adj_close"] >= data["high60"] * 0.995).astype(float)
    data["new_high_120"] = (data["adj_close"] >= data["high120"] * 0.995).astype(float)
    data["breakout_20"] = data["adj_close"] / data["prev_high20"] - 1.0
    data["breakout_60"] = data["adj_close"] / data["prev_high60"] - 1.0
    data.loc[data["breakout_20"] < 0, "breakout_20"] = 0.0
    data.loc[data["breakout_60"] < 0, "breakout_60"] = 0.0
    data["pullback_ma20"] = (
        (data["ma20_slope_5"] > 0)
        & (data["adj_low"] <= data["ma20"] * 1.015)
        & (data["adj_close"] >= data["ma20"])
        & (data["adj_close"] <= data["ma20"] * 1.08)
    ).astype(float)
    data["pullback_ma60"] = (
        (data["ma60_slope_10"] > 0)
        & (data["adj_low"] <= data["ma60"] * 1.02)
        & (data["adj_close"] >= data["ma60"])
        & (data["adj_close"] <= data["ma60"] * 1.10)
    ).astype(float)
    data["macd_above_signal"] = (data["macd"] > data["macd_signal"]).astype(float)
    data["macd_hist_rising"] = (data["macd_hist"] > data["macd_hist_prev"]).astype(float)
    data["rsi_mid_strong"] = ((data["rsi14"] >= 50.0) & (data["rsi14"] <= 70.0)).astype(float)
    data["volume_expansion_mild"] = ((data["volume_ratio_20"] >= 1.10) & (data["volume_ratio_20"] <= 2.50)).astype(float)
    data["volume_expansion_extreme"] = (data["volume_ratio_20"] > 3.0).astype(float)
    data["bias5_overheat"] = (data["adj_close"] / data["ma5"] - 1.0).clip(lower=0)
    data["upper_shadow_ratio"] = (data["adj_high"] - data[["adj_open", "adj_close"]].max(axis=1)) / data["adj_close"]
    data["long_upper_shadow"] = (data["upper_shadow_ratio"] > 0.04).astype(float)
    data["below_ma20"] = (data["adj_close"] < data["ma20"]).astype(float)
    data["below_ma60"] = (data["adj_close"] < data["ma60"]).astype(float)
    data["amount_rank"] = _pct_rank(data, "amount_yi", higher_is_better=True)
    data["turnover_rank"] = _pct_rank(data, "turnover_rate", higher_is_better=True)
    vol_score = _pct_rank(data, "volatility_20", higher_is_better=False)
    data["low_volatility_trend"] = vol_score.where((data["ma20_slope_5"] > 0) & (data["price_above_ma20"] > 0), 0.0)

    for indicator in TREND_INDICATORS:
        rank_col = f"{indicator.name}__pct_rank"
        data[rank_col] = _pct_rank(data, indicator.name, higher_is_better=indicator.higher_is_better)
        data[rank_col] = data[rank_col].clip(lower=0.0, upper=1.0)

    start_dt = pd.to_datetime(normalize_date(start_date)).date()
    end_dt = pd.to_datetime(normalize_date(end_date)).date()
    target = data[(data["trade_date"] >= start_dt) & (data["trade_date"] <= end_dt)].copy()
    target["is_st"] = target["name"].fillna("").astype(str).str.contains("ST", case=False, regex=False)
    target["eligible_base"] = (
        target["adj_close"].notna()
        & target["amount_yi"].fillna(0).ge(2.0)
        & (~target["is_st"].fillna(False))
        & target["ma60"].notna()
    )
    _FEATURE_CACHE_KEY = cache_key
    _FEATURE_CACHE_FRAME = target.copy()
    return target


def _split_labels(dates: list[Any]) -> dict[str, set[Any]]:
    unique = sorted(pd.to_datetime(pd.Series(dates)).dt.date.dropna().unique().tolist())
    if not unique:
        return {"full": set(), "train": set(), "validation": set(), "test": set()}
    train_end = int(len(unique) * 0.6)
    validation_end = int(len(unique) * 0.8)
    return {
        "full": set(unique),
        "train": set(unique[:train_end]),
        "validation": set(unique[train_end:validation_end]),
        "test": set(unique[validation_end:]),
    }


def _safe_corr(left: pd.Series, right: pd.Series, *, rank: bool = False) -> float | None:
    frame = pd.DataFrame({"x": left, "y": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 20 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    if rank:
        frame = frame.rank(method="average")
    return _clean_float(frame["x"].corr(frame["y"]))


def _daily_rank_ic(frame: pd.DataFrame, score_column: str, return_column: str) -> float | None:
    values = []
    for _, group in frame.groupby("trade_date"):
        corr = _safe_corr(group[score_column], group[return_column], rank=True)
        if corr is not None:
            values.append(corr)
    return _clean_float(float(np.mean(values))) if values else None


def _top_bottom_daily(frame: pd.DataFrame, score_column: str, return_column: str, quantile: float = 0.2) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date"):
        group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_column, return_column])
        if len(group) < 50:
            continue
        n = max(int(len(group) * quantile), 10)
        ordered = group.sort_values(score_column, ascending=False)
        top = ordered.head(n)[return_column]
        bottom = ordered.tail(n)[return_column]
        rows.append(
            {
                "trade_date": trade_date,
                "top_return": top.mean(),
                "bottom_return": bottom.mean(),
                "benchmark_return": group[return_column].mean(),
                "top_win": float((top > 0).mean()),
                "bottom_win": float((bottom > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _metric_rows_for_indicator(
    merged: pd.DataFrame,
    *,
    indicator: TrendIndicator,
    horizon_days: int,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    score_column = f"{indicator.name}__pct_rank"
    frame = merged[
        merged["eligible_base"].fillna(False)
        & merged[score_column].notna()
        & merged["future_return"].notna()
        & merged["entry_price"].notna()
        & merged["exit_price"].notna()
        & (~merged["one_word_limit_up"].fillna(False))
        & (~merged.get("exit_blocked", pd.Series(False, index=merged.index)).fillna(False))
    ].copy()
    splits = _split_labels(frame["trade_date"].tolist())
    rows: list[dict[str, Any]] = []
    for split_name, split_dates in splits.items():
        sub = frame[frame["trade_date"].isin(split_dates)].copy()
        if sub.empty:
            continue
        daily = _top_bottom_daily(sub, score_column, "future_return")
        if daily.empty:
            continue
        rank_ic = _daily_rank_ic(sub, score_column, "future_return")
        top_series = daily["top_return"] / float(max(horizon_days, 1))
        _, max_drawdown = _drawdown_stats(top_series)
        rows.append(
            {
                "indicator_name": indicator.name,
                "indicator_group": indicator.group,
                "direction": indicator.direction,
                "start_date": normalize_date(start_date),
                "end_date": normalize_date(end_date),
                "horizon_days": int(horizon_days),
                "sample_split": split_name,
                "sample_count": int(len(sub)),
                "rank_ic_mean": rank_ic,
                "top_quantile_return": _clean_float(daily["top_return"].mean()),
                "bottom_quantile_return": _clean_float(daily["bottom_return"].mean()),
                "long_short_return": _clean_float((daily["top_return"] - daily["bottom_return"]).mean()),
                "avg_excess_return": _clean_float((daily["top_return"] - daily["benchmark_return"]).mean()),
                "win_rate": _clean_float((daily["top_return"] > 0).mean()),
                "max_drawdown": max_drawdown,
                "profit_loss_ratio": _profit_loss_ratio(sub.nlargest(max(int(len(sub) * 0.2), 1), score_column)["future_return"]),
                "created_at": None,
            }
        )
    return rows


def evaluate_trend_indicators(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    horizons: tuple[int, ...] = (3, 5, 10, 20),
) -> dict[str, Any]:
    init_schema(settings)
    features = compute_trend_features(settings, start_date, end_date)
    if features.empty:
        raise RuntimeError(f"No trend feature rows for {start_date} to {end_date}")
    result_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        returns = _load_trade_returns(settings, start_date, end_date, horizon)
        if returns.empty:
            continue
        return_columns = [
            "trade_date",
            "ts_code",
            "entry_price",
            "entry_raw_open",
            "entry_raw_high",
            "entry_raw_low",
            "entry_pct_chg",
            "entry_amount_yi",
            "entry_limit_pct",
            "entry_one_word_limit_up",
            "scheduled_exit_date",
            "scheduled_exit_one_word_limit_down",
            "exit_price",
            "exit_blocked",
            "future_return",
        ]
        returns = returns[[column for column in return_columns if column in returns.columns]].copy()
        returns["one_word_limit_up"] = returns.get("entry_one_word_limit_up", pd.Series(False, index=returns.index)).fillna(False)
        returns["exit_blocked"] = returns.get("exit_blocked", pd.Series(False, index=returns.index)).fillna(False)
        merged = features.merge(returns, on=["trade_date", "ts_code"], how="left")
        for indicator in TREND_INDICATORS:
            result_rows.extend(
                _metric_rows_for_indicator(
                    merged,
                    indicator=indicator,
                    horizon_days=horizon,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
    columns = [
        "indicator_name",
        "indicator_group",
        "direction",
        "start_date",
        "end_date",
        "horizon_days",
        "sample_split",
        "sample_count",
        "rank_ic_mean",
        "top_quantile_return",
        "bottom_quantile_return",
        "long_short_return",
        "avg_excess_return",
        "win_rate",
        "max_drawdown",
        "profit_loss_ratio",
    ]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {qname(settings, 'trend_pure_indicator_performance')} WHERE start_date=%s AND end_date=%s",
                (normalize_date(start_date), normalize_date(end_date)),
            )
        inserted = upsert_rows(
            conn,
            table=qname(settings, "trend_pure_indicator_performance"),
            columns=columns,
            rows=result_rows,
            conflict_columns=["indicator_name", "start_date", "end_date", "horizon_days", "sample_split"],
        )
        conn.commit()
    report_file = write_trend_indicator_report(settings, start_date, end_date, result_rows)
    return {
        "model_name": TREND_MODEL_NAME,
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "feature_rows": int(len(features)),
        "indicator_rows": inserted,
        "report_file": str(report_file),
    }


def _combo_score(features: pd.DataFrame, combo_weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=features.index)
    total_abs = sum(abs(weight) for weight in combo_weights.values())
    if math.isclose(total_abs, 0.0):
        return score
    for name, weight in combo_weights.items():
        rank_col = f"{name}__pct_rank"
        if rank_col not in features.columns:
            continue
        oriented = pd.to_numeric(features[rank_col], errors="coerce").fillna(0.5) - 0.5
        if weight >= 0:
            score += oriented * abs(weight)
        else:
            score -= oriented * abs(weight)
    return 0.5 + score / total_abs


def score_trend_pure(settings: Settings, start_date: str, end_date: str) -> pd.DataFrame:
    features = compute_trend_features(settings, start_date, end_date)
    for combo_name, combo_weights in TREND_COMBOS.items():
        features[f"{combo_name}_score"] = _combo_score(features, combo_weights).clip(lower=0.0, upper=1.0)
    features["trend_pure_score"] = features["all_technical_score"]
    features["trend_pure_rank"] = features.groupby("trade_date")["trend_pure_score"].rank(method="first", ascending=False)
    features["trend_candidate"] = (
        features["eligible_base"].fillna(False)
        & features["amount_yi"].fillna(0).ge(2.0)
        & features["price_above_ma20"].fillna(0).gt(0)
        & (features["ma20_slope_5"].fillna(-1) > 0)
        & features["below_ma60"].fillna(1).lt(1)
    )
    return features


def _combo_reason(row: pd.Series) -> str:
    positives = []
    if row.get("ma_bull_5_10_20_60", 0) > 0:
        positives.append("均线多头")
    elif row.get("ma_bull_5_10_20", 0) > 0:
        positives.append("短均线多头")
    if row.get("new_high_20", 0) > 0 or row.get("breakout_20", 0) > 0:
        positives.append("20日突破/新高")
    if row.get("new_high_60", 0) > 0 or row.get("breakout_60", 0) > 0:
        positives.append("60日突破/新高")
    if row.get("pullback_ma20", 0) > 0:
        positives.append("回踩MA20不破")
    if row.get("macd_above_signal", 0) > 0:
        positives.append("MACD偏强")
    if row.get("volume_expansion_mild", 0) > 0:
        positives.append("温和放量")
    risks = []
    if row.get("bias5_overheat", 0) > 0.08:
        risks.append("乖离过热")
    if row.get("long_upper_shadow", 0) > 0:
        risks.append("长上影")
    if row.get("volume_expansion_extreme", 0) > 0:
        risks.append("极端放量")
    text = "、".join(positives[:4]) or "纯技术趋势综合评分靠前"
    if risks:
        text += "；风险：" + "、".join(risks[:3])
    return text


def _backtest_combo_once(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    combo_name: str,
    top_n: int,
    hold_days: int,
    min_amount_yi: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    capacity_pct: float = 0.005,
    portfolio_capital_yi: float = 0.1,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    score_col = f"{combo_name}_score"
    return_columns = [
        "trade_date",
        "ts_code",
        "entry_date",
        "entry_price",
        "entry_raw_open",
        "entry_raw_high",
        "entry_raw_low",
        "entry_pct_chg",
        "entry_amount_yi",
        "entry_limit_pct",
        "entry_one_word_limit_up",
        "scheduled_exit_date",
        "scheduled_exit_one_word_limit_down",
        "exit_date",
        "exit_price",
        "exit_raw_open",
        "exit_raw_high",
        "exit_raw_low",
        "exit_raw_close",
        "exit_pct_chg",
        "exit_limit_pct",
        "exit_delayed_days",
        "exit_was_delayed",
        "exit_blocked",
        "future_return",
    ]
    frame = features.merge(
        returns[[column for column in return_columns if column in returns.columns]],
        on=["trade_date", "ts_code"],
        how="inner",
    )
    frame = _add_execution_constraints(
        frame,
        top_n=top_n,
        capacity_pct=capacity_pct,
        portfolio_capital_yi=portfolio_capital_yi,
    )
    scored = frame[
        frame["trend_candidate"].fillna(False)
        & frame[score_col].notna()
        & frame["future_return"].notna()
        & frame["entry_price"].notna()
        & frame["exit_price"].notna()
        & (~frame["exit_blocked"].fillna(False))
        & (frame["amount_yi"].fillna(0) >= min_amount_yi)
    ].copy()
    if scored.empty:
        return [], scored
    before_buy_filter = (
        scored.sort_values(["trade_date", score_col, "ts_code"], ascending=[True, False, True])
        .groupby("trade_date", group_keys=False)
        .head(top_n)
        .copy()
    )
    selected = before_buy_filter[
        (~before_buy_filter["one_word_limit_up"].fillna(False))
        & before_buy_filter["capacity_pass"].fillna(False)
    ].copy()
    if selected.empty:
        return [], selected
    round_trip_cost = 2.0 * (transaction_cost_bps + slippage_bps) / 10000.0
    selected["net_return"] = selected["future_return"] - round_trip_cost
    selected["score"] = selected[score_col]
    selected["reason"] = selected.apply(_combo_reason, axis=1)

    daily = selected.groupby("trade_date").agg(cohort_return=("net_return", "mean"), trade_count=("ts_code", "count"))
    daily["daily_return"] = daily["cohort_return"] / float(max(hold_days, 1))
    benchmark_pool = scored[
        (~scored["one_word_limit_up"].fillna(False))
        & scored["capacity_pass"].fillna(False)
    ]
    benchmark = benchmark_pool.groupby("trade_date")["future_return"].mean()
    benchmark = (benchmark - round_trip_cost) / float(max(hold_days, 1))
    splits = _split_labels(list(daily.index))
    result_rows: list[dict[str, Any]] = []
    buyable_ratio = len(selected) / len(before_buy_filter) if len(before_buy_filter) else None
    previous: set[str] | None = None
    turnovers = []
    for _, group in selected.groupby("trade_date"):
        current = set(group["ts_code"].astype(str))
        if previous is not None and current:
            turnovers.append(1.0 - len(current & previous) / max(len(current), 1))
        previous = current
    avg_turnover = float(np.mean(turnovers)) if turnovers else 1.0
    for split_name, split_dates in splits.items():
        series = daily[daily.index.isin(split_dates)]["daily_return"]
        split_trades = selected[selected["trade_date"].isin(split_dates)]
        perf = _performance_from_daily_returns(series)
        bench_perf = _performance_from_daily_returns(benchmark[benchmark.index.isin(split_dates)])
        result_rows.append(
            {
                "combo_name": combo_name,
                "hold_days": int(hold_days),
                "top_n": int(top_n),
                "sample_split": split_name,
                "avg_return": _clean_float(split_trades["net_return"].mean()) if not split_trades.empty else None,
                "avg_excess_return": _clean_float(
                    split_trades["net_return"].mean() - benchmark_pool["future_return"].mean()
                )
                if not split_trades.empty and not benchmark_pool.empty
                else None,
                "total_return": perf["total_return"],
                "annual_return": perf["annual_return"],
                "max_drawdown": perf["max_drawdown"],
                "sharpe": perf["sharpe"],
                "win_rate": perf["win_rate"],
                "profit_loss_ratio": _profit_loss_ratio(split_trades["net_return"]) if not split_trades.empty else None,
                "buyable_ratio": _clean_float(buyable_ratio),
                "avg_turnover": _clean_float(avg_turnover),
                "trade_count": int(split_trades["ts_code"].count()),
                "benchmark_return": bench_perf["total_return"],
                "excess_return": _clean_float((perf["total_return"] or 0.0) - (bench_perf["total_return"] or 0.0)),
            }
        )
    return result_rows, selected


def run_trend_pure_backtest(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    top_ns: tuple[int, ...] = (10, 20, 30),
    hold_days_list: tuple[int, ...] = (3, 5, 10, 20),
    min_amount_yi: float = 2.0,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    capacity_pct: float = 0.005,
    portfolio_capital_yi: float = 0.1,
) -> dict[str, Any]:
    init_schema(settings)
    features = score_trend_pure(settings, start_date, end_date)
    result_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    for hold_days in hold_days_list:
        returns = _load_trade_returns(settings, start_date, end_date, hold_days)
        if returns.empty:
            continue
        for combo_name in TREND_COMBOS:
            for top_n in top_ns:
                rows, selected = _backtest_combo_once(
                    features,
                    returns,
                    combo_name=combo_name,
                    top_n=top_n,
                    hold_days=hold_days,
                    min_amount_yi=min_amount_yi,
                    transaction_cost_bps=transaction_cost_bps,
                    slippage_bps=slippage_bps,
                    capacity_pct=capacity_pct,
                    portfolio_capital_yi=portfolio_capital_yi,
                )
                result_rows.extend(
                    {
                        **row,
                        "start_date": normalize_date(start_date),
                        "end_date": normalize_date(end_date),
                    }
                    for row in rows
                )
                model_run_name = (
                    f"{TREND_MODEL_NAME}_{combo_name}_top{top_n}_hold{hold_days}_"
                    f"{normalize_date(start_date).replace('-', '')}_{normalize_date(end_date).replace('-', '')}"
                )
                if not selected.empty:
                    for row in selected.itertuples():
                        trade_rows.append(
                            {
                                "model_name": model_run_name,
                                "trade_date": row.trade_date,
                                "ts_code": row.ts_code,
                                "action": "BUY",
                                "price": _clean_float(row.entry_price),
                                "weight": _clean_float(1.0 / max(top_n, 1)),
                                "score": _clean_float(row.score),
                                "reason": row.reason,
                                "holding_days": int(hold_days),
                                "exit_date": row.exit_date,
                                "exit_price": _clean_float(row.exit_price),
                                "return_pct": _clean_float(row.net_return),
                            }
                        )

    if not result_rows:
        raise RuntimeError(f"No trend_pure_v1 combo backtest rows generated for {start_date} to {end_date}")

    perf_columns = [
        "combo_name",
        "start_date",
        "end_date",
        "hold_days",
        "top_n",
        "sample_split",
        "avg_return",
        "avg_excess_return",
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe",
        "win_rate",
        "profit_loss_ratio",
        "buyable_ratio",
        "avg_turnover",
        "trade_count",
        "benchmark_return",
        "excess_return",
    ]
    strategy_rows: list[dict[str, Any]] = []
    for row in result_rows:
        strategy_rows.append(
            {
                "model_name": (
                    f"{TREND_MODEL_NAME}_{row['combo_name']}_top{row['top_n']}_hold{row['hold_days']}_"
                    f"{normalize_date(start_date).replace('-', '')}_{normalize_date(end_date).replace('-', '')}"
                ),
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "rebalance_rule": "daily_pure_technical_trend",
                "hold_days": row["hold_days"],
                "top_n": row["top_n"],
                "sample_split": row["sample_split"],
                "total_return": row["total_return"],
                "annual_return": row["annual_return"],
                "max_drawdown": row["max_drawdown"],
                "sharpe": row["sharpe"],
                "win_rate": row["win_rate"],
                "profit_loss_ratio": row["profit_loss_ratio"],
                "buyable_ratio": row["buyable_ratio"],
                "avg_turnover": row["avg_turnover"],
                "avg_holding_days": row["hold_days"],
                "trade_count": row["trade_count"],
                "benchmark_return": row["benchmark_return"],
                "excess_return": row["excess_return"],
            }
        )
    strategy_columns = [
        "model_name",
        "start_date",
        "end_date",
        "rebalance_rule",
        "hold_days",
        "top_n",
        "sample_split",
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe",
        "win_rate",
        "profit_loss_ratio",
        "buyable_ratio",
        "avg_turnover",
        "avg_holding_days",
        "trade_count",
        "benchmark_return",
        "excess_return",
    ]
    trade_columns = [
        "model_name",
        "trade_date",
        "ts_code",
        "action",
        "price",
        "weight",
        "score",
        "reason",
        "holding_days",
        "exit_date",
        "exit_price",
        "return_pct",
    ]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {qname(settings, 'trend_pure_combo_performance')} WHERE start_date=%s AND end_date=%s",
                (normalize_date(start_date), normalize_date(end_date)),
            )
            cur.execute(
                f"""
                DELETE FROM {qname(settings, 'strategy_backtest_trades')}
                WHERE model_name LIKE %s AND trade_date BETWEEN %s AND %s
                """,
                (f"{TREND_MODEL_NAME}_%", normalize_date(start_date), normalize_date(end_date)),
            )
        combo_count = upsert_rows(
            conn,
            table=qname(settings, "trend_pure_combo_performance"),
            columns=perf_columns,
            rows=result_rows,
            conflict_columns=["combo_name", "start_date", "end_date", "hold_days", "top_n", "sample_split"],
        )
        strategy_count = upsert_rows(
            conn,
            table=qname(settings, "strategy_backtest_result"),
            columns=strategy_columns,
            rows=strategy_rows,
            conflict_columns=["model_name", "start_date", "end_date", "rebalance_rule", "hold_days", "top_n", "sample_split"],
        )
        trade_count = upsert_rows(
            conn,
            table=qname(settings, "strategy_backtest_trades"),
            columns=trade_columns,
            rows=trade_rows,
            conflict_columns=["model_name", "trade_date", "ts_code", "action"],
            batch_size=5000,
        )
        conn.commit()
    report_file = write_trend_combo_report(settings, start_date, end_date, result_rows)
    return {
        "model_name": TREND_MODEL_NAME,
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "combo_rows": combo_count,
        "strategy_rows": strategy_count,
        "trade_rows": trade_count,
        "report_file": str(report_file),
    }


def _fmt_pct(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return ""


def _fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def write_trend_indicator_report(settings: Settings, start_date: str, end_date: str, rows: list[dict[str, Any]]) -> str:
    compact = normalize_date(end_date).replace("-", "")
    path = report_dir(settings) / f"trend_pure_indicator_report_{compact}.md"
    data = pd.DataFrame(rows)
    lines = [
        f"# trend_pure_v1 指标检验 {normalize_date(start_date)} ~ {normalize_date(end_date)}",
        "",
        "范围：只使用复权价、成交量、成交额、换手率和由它们计算出的纯技术指标；不使用资金流、龙虎榜、涨停、主题热度或基本面判断。",
        "",
        "交易口径：T日收盘后形成信号，T+1复权开盘买入，持有 N 个交易日后复权收盘退出；按主板/创业板/科创板/北交所涨跌停幅度识别一字板，剔除一字涨停不可买入样本，计划退出日一字跌停则顺延到首个可卖日。",
        "",
    ]
    if data.empty:
        lines.append("- No indicator performance rows.")
    else:
        full_5 = data[(data["sample_split"] == "full") & (data["horizon_days"] == 5)].copy()
        full_5 = full_5.sort_values(["avg_excess_return", "long_short_return", "rank_ic_mean"], ascending=False)
        lines.extend(["## T+5 指标排序", ""])
        lines.append("| indicator | group | direction | RankIC | top20 | excess | long-short | win | mdd |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in full_5.head(30).to_dict("records"):
            lines.append(
                f"| {row['indicator_name']} | {row['indicator_group']} | {row['direction']} | "
                f"{_fmt_num(row.get('rank_ic_mean'))} | {_fmt_pct(row.get('top_quantile_return'))} | "
                f"{_fmt_pct(row.get('avg_excess_return'))} | {_fmt_pct(row.get('long_short_return'))} | "
                f"{_fmt_pct(row.get('win_rate'))} | {_fmt_pct(row.get('max_drawdown'))} |"
            )
        lines.extend(["", "## 分窗口摘要", ""])
        for horizon in sorted(data["horizon_days"].dropna().unique()):
            subset = data[(data["sample_split"] == "full") & (data["horizon_days"] == horizon)].copy()
            subset = subset.sort_values("avg_excess_return", ascending=False).head(8)
            names = ", ".join(
                f"{row.indicator_name}({_fmt_pct(row.avg_excess_return)})" for row in subset.itertuples()
            )
            lines.append(f"- T+{int(horizon)}：{names}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def write_trend_combo_report(settings: Settings, start_date: str, end_date: str, rows: list[dict[str, Any]]) -> str:
    compact = normalize_date(end_date).replace("-", "")
    path = report_dir(settings) / f"trend_pure_combo_report_{compact}.md"
    data = pd.DataFrame(rows)
    lines = [
        f"# trend_pure_v1 组合回测 {normalize_date(start_date)} ~ {normalize_date(end_date)}",
        "",
        "范围：纯技术趋势组合，不使用资金流、龙虎榜、涨停、主题或生产评分权重。",
        "",
        "交易口径：T日收盘出信号，T+1复权开盘买入，持有 N 日复权收盘退出；扣除双边交易成本和滑点，按板块涨跌停幅度排除一字涨停买不进样本，计划退出日一字跌停则顺延到首个可卖日，并加入单票成交额容量约束。",
        "",
    ]
    if data.empty:
        lines.append("- No combo backtest rows.")
    else:
        full = data[data["sample_split"] == "full"].copy()
        full = full.sort_values(["excess_return", "sharpe", "total_return"], ascending=False)
        lines.append("| combo | top | hold | total | excess | annual | mdd | sharpe | win | PL | buyable | trades |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in full.head(40).to_dict("records"):
            lines.append(
                f"| {row['combo_name']} | {row['top_n']} | {row['hold_days']} | {_fmt_pct(row.get('total_return'))} | "
                f"{_fmt_pct(row.get('excess_return'))} | {_fmt_pct(row.get('annual_return'))} | {_fmt_pct(row.get('max_drawdown'))} | "
                f"{_fmt_num(row.get('sharpe'), 2)} | {_fmt_pct(row.get('win_rate'))} | {_fmt_num(row.get('profit_loss_ratio'), 2)} | "
                f"{_fmt_pct(row.get('buyable_ratio'))} | {row.get('trade_count')} |"
            )
        lines.extend(["", "## 样本外拆分", ""])
        split = data[data["sample_split"].isin(["train", "validation", "test"])].copy()
        split = split.sort_values(["combo_name", "top_n", "hold_days", "sample_split"])
        lines.append("| combo | top | hold | split | total | excess | mdd | win | trades |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
        for row in split.head(80).to_dict("records"):
            lines.append(
                f"| {row['combo_name']} | {row['top_n']} | {row['hold_days']} | {row['sample_split']} | "
                f"{_fmt_pct(row.get('total_return'))} | {_fmt_pct(row.get('excess_return'))} | "
                f"{_fmt_pct(row.get('max_drawdown'))} | {_fmt_pct(row.get('win_rate'))} | {row.get('trade_count')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _period_ranges(start_date: str, end_date: str, period_years: int) -> list[tuple[str, str]]:
    start = pd.to_datetime(normalize_date(start_date)).date()
    end = pd.to_datetime(normalize_date(end_date)).date()
    years = max(int(period_years), 1)
    ranges: list[tuple[str, str]] = []
    current = start
    while current <= end:
        period_end_year = current.year + years - 1
        period_end = date(period_end_year, 12, 31)
        if period_end > end:
            period_end = end
        ranges.append((current.isoformat(), period_end.isoformat()))
        current = date(period_end.year + 1, 1, 1)
    return ranges


def _upsert_trend_indicator_rows(settings: Settings, rows: list[dict[str, Any]]) -> int:
    columns = [
        "indicator_name",
        "indicator_group",
        "direction",
        "start_date",
        "end_date",
        "horizon_days",
        "sample_split",
        "sample_count",
        "rank_ic_mean",
        "top_quantile_return",
        "bottom_quantile_return",
        "long_short_return",
        "avg_excess_return",
        "win_rate",
        "max_drawdown",
        "profit_loss_ratio",
    ]
    with connect() as conn:
        inserted = upsert_rows(
            conn,
            table=qname(settings, "trend_pure_indicator_performance"),
            columns=columns,
            rows=rows,
            conflict_columns=["indicator_name", "start_date", "end_date", "horizon_days", "sample_split"],
        )
        conn.commit()
    return inserted


def _upsert_trend_combo_rows(settings: Settings, rows: list[dict[str, Any]]) -> int:
    columns = [
        "combo_name",
        "start_date",
        "end_date",
        "hold_days",
        "top_n",
        "sample_split",
        "avg_return",
        "avg_excess_return",
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe",
        "win_rate",
        "profit_loss_ratio",
        "buyable_ratio",
        "avg_turnover",
        "trade_count",
        "benchmark_return",
        "excess_return",
    ]
    with connect() as conn:
        inserted = upsert_rows(
            conn,
            table=qname(settings, "trend_pure_combo_performance"),
            columns=columns,
            rows=rows,
            conflict_columns=["combo_name", "start_date", "end_date", "hold_days", "top_n", "sample_split"],
        )
        conn.commit()
    return inserted


def write_trend_period_report(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    period_years: int,
    indicator_rows: list[dict[str, Any]],
    combo_rows: list[dict[str, Any]],
) -> str:
    compact = normalize_date(end_date).replace("-", "")
    path = report_dir(settings) / f"trend_pure_period_study_{compact}.md"
    indicators = pd.DataFrame(indicator_rows)
    combos = pd.DataFrame(combo_rows)
    lines = [
        f"# trend_pure_v1 分段稳定性研究 {normalize_date(start_date)} ~ {normalize_date(end_date)}",
        "",
        f"分段口径：每 {int(period_years)} 年一段；主收益窗口为 T+5，组合口径为 Top20 Hold5。",
        "",
        "说明：该报告覆盖数据库已有纯技术历史，但避免一次性全历史大内存回测；每段独立计算指标和组合表现，更适合观察稳定性和阶段差异。",
        "",
    ]
    if indicators.empty:
        lines.append("- No indicator period rows.")
    else:
        full = indicators[indicators["sample_split"] == "full"].copy()
        full["avg_excess_return"] = pd.to_numeric(full["avg_excess_return"], errors="coerce")
        full["rank_ic_mean"] = pd.to_numeric(full["rank_ic_mean"], errors="coerce")
        stable = (
            full.groupby(["indicator_name", "indicator_group"], dropna=False)
            .agg(
                periods=("end_date", "count"),
                positive_periods=("avg_excess_return", lambda s: int((s > 0).sum())),
                avg_excess=("avg_excess_return", "mean"),
                median_excess=("avg_excess_return", "median"),
                min_excess=("avg_excess_return", "min"),
                max_excess=("avg_excess_return", "max"),
                avg_rank_ic=("rank_ic_mean", "mean"),
            )
            .reset_index()
        )
        stable["positive_ratio"] = stable["positive_periods"] / stable["periods"].replace(0, np.nan)
        stable = stable.sort_values(["positive_ratio", "avg_excess"], ascending=False)
        lines.extend(["## 指标跨阶段稳定性", ""])
        lines.append("| indicator | group | periods+ | avg_excess | median | worst | best | avg_rank_ic |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in stable.head(20).to_dict("records"):
            lines.append(
                f"| {row['indicator_name']} | {row['indicator_group']} | "
                f"{int(row['positive_periods'])}/{int(row['periods'])} | {_fmt_pct(row.get('avg_excess'))} | "
                f"{_fmt_pct(row.get('median_excess'))} | {_fmt_pct(row.get('min_excess'))} | "
                f"{_fmt_pct(row.get('max_excess'))} | {_fmt_num(row.get('avg_rank_ic'))} |"
            )
        lines.extend(["", "## 指标分段明细 Top10", ""])
        for period_key, group in full.groupby(["start_date", "end_date"]):
            period_start, period_end = period_key
            top = group.sort_values("avg_excess_return", ascending=False).head(10)
            names = ", ".join(f"{row.indicator_name}({_fmt_pct(row.avg_excess_return)})" for row in top.itertuples())
            lines.append(f"- {period_start} ~ {period_end}: {names}")
    if combos.empty:
        lines.extend(["", "## 组合跨阶段稳定性", "", "- No combo period rows."])
    else:
        data = combos[combos["sample_split"] == "full"].copy()
        for column in ["excess_return", "total_return", "max_drawdown", "sharpe", "win_rate"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        stable_combo = (
            data.groupby("combo_name", dropna=False)
            .agg(
                periods=("end_date", "count"),
                positive_periods=("excess_return", lambda s: int((s > 0).sum())),
                avg_excess=("excess_return", "mean"),
                median_excess=("excess_return", "median"),
                min_excess=("excess_return", "min"),
                max_excess=("excess_return", "max"),
                avg_total=("total_return", "mean"),
                worst_drawdown=("max_drawdown", "min"),
                avg_sharpe=("sharpe", "mean"),
                avg_win=("win_rate", "mean"),
            )
            .reset_index()
        )
        stable_combo["positive_ratio"] = stable_combo["positive_periods"] / stable_combo["periods"].replace(0, np.nan)
        stable_combo = stable_combo.sort_values(["positive_ratio", "avg_excess"], ascending=False)
        lines.extend(["", "## 组合跨阶段稳定性", ""])
        lines.append("| combo | periods+ | avg_excess | median | worst | best | avg_total | worst_mdd | avg_sharpe | avg_win |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in stable_combo.to_dict("records"):
            lines.append(
                f"| {row['combo_name']} | {int(row['positive_periods'])}/{int(row['periods'])} | "
                f"{_fmt_pct(row.get('avg_excess'))} | {_fmt_pct(row.get('median_excess'))} | "
                f"{_fmt_pct(row.get('min_excess'))} | {_fmt_pct(row.get('max_excess'))} | "
                f"{_fmt_pct(row.get('avg_total'))} | {_fmt_pct(row.get('worst_drawdown'))} | "
                f"{_fmt_num(row.get('avg_sharpe'), 2)} | {_fmt_pct(row.get('avg_win'))} |"
            )
        lines.extend(["", "## 组合分段明细", ""])
        lines.append("| period | combo | total | excess | mdd | sharpe | win | trades |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        data = data.sort_values(["start_date", "combo_name"])
        for row in data.to_dict("records"):
            lines.append(
                f"| {row['start_date']}~{row['end_date']} | {row['combo_name']} | "
                f"{_fmt_pct(row.get('total_return'))} | {_fmt_pct(row.get('excess_return'))} | "
                f"{_fmt_pct(row.get('max_drawdown'))} | {_fmt_num(row.get('sharpe'), 2)} | "
                f"{_fmt_pct(row.get('win_rate'))} | {row.get('trade_count')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def run_trend_pure_period_study(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    period_years: int = 3,
    horizon_days: int = 5,
    top_n: int = 20,
    hold_days: int = 5,
    min_amount_yi: float = 2.0,
    capacity_pct: float = 0.005,
    portfolio_capital_yi: float = 0.1,
) -> dict[str, Any]:
    init_schema(settings)
    periods = _period_ranges(start_date, end_date, period_years)
    all_indicator_rows: list[dict[str, Any]] = []
    all_combo_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for period_start, period_end in periods:
        features = score_trend_pure(settings, period_start, period_end)
        returns = _load_trade_returns(settings, period_start, period_end, horizon_days)
        if returns.empty or features.empty:
            metrics.append({"start_date": period_start, "end_date": period_end, "status": "empty"})
            continue
        return_columns = [
            "trade_date",
            "ts_code",
            "entry_date",
            "entry_price",
            "entry_raw_open",
            "entry_raw_high",
            "entry_raw_low",
            "entry_pct_chg",
            "entry_amount_yi",
            "entry_limit_pct",
            "entry_one_word_limit_up",
            "scheduled_exit_date",
            "scheduled_exit_one_word_limit_down",
            "exit_date",
            "exit_price",
            "exit_raw_open",
            "exit_raw_high",
            "exit_raw_low",
            "exit_raw_close",
            "exit_pct_chg",
            "exit_limit_pct",
            "exit_delayed_days",
            "exit_was_delayed",
            "exit_blocked",
            "future_return",
        ]
        returns = returns[[column for column in return_columns if column in returns.columns]].copy()
        returns["one_word_limit_up"] = returns.get("entry_one_word_limit_up", pd.Series(False, index=returns.index)).fillna(False)
        returns["exit_blocked"] = returns.get("exit_blocked", pd.Series(False, index=returns.index)).fillna(False)
        merged = features.merge(returns, on=["trade_date", "ts_code"], how="left")
        period_indicator_rows: list[dict[str, Any]] = []
        for indicator in TREND_INDICATORS:
            rows = _metric_rows_for_indicator(
                merged,
                indicator=indicator,
                horizon_days=horizon_days,
                start_date=period_start,
                end_date=period_end,
            )
            period_indicator_rows.extend([row for row in rows if row.get("sample_split") == "full"])
        period_combo_rows: list[dict[str, Any]] = []
        for combo_name in TREND_COMBOS:
            rows, _selected = _backtest_combo_once(
                features,
                returns,
                combo_name=combo_name,
                top_n=top_n,
                hold_days=hold_days,
                min_amount_yi=min_amount_yi,
                transaction_cost_bps=10.0,
                slippage_bps=5.0,
                capacity_pct=capacity_pct,
                portfolio_capital_yi=portfolio_capital_yi,
            )
            for row in rows:
                if row.get("sample_split") != "full":
                    continue
                period_combo_rows.append(
                    {
                        **row,
                        "start_date": normalize_date(period_start),
                        "end_date": normalize_date(period_end),
                    }
                )
        indicator_count = _upsert_trend_indicator_rows(settings, period_indicator_rows)
        combo_count = _upsert_trend_combo_rows(settings, period_combo_rows)
        all_indicator_rows.extend(period_indicator_rows)
        all_combo_rows.extend(period_combo_rows)
        metrics.append(
            {
                "start_date": period_start,
                "end_date": period_end,
                "feature_rows": int(len(features)),
                "indicator_rows": indicator_count,
                "combo_rows": combo_count,
                "status": "finished",
            }
        )
        global _FEATURE_CACHE_KEY, _FEATURE_CACHE_FRAME
        _FEATURE_CACHE_KEY = None
        _FEATURE_CACHE_FRAME = None
        del features, returns, merged, period_indicator_rows, period_combo_rows
        gc.collect()
    report_file = write_trend_period_report(
        settings,
        start_date,
        end_date,
        period_years=period_years,
        indicator_rows=all_indicator_rows,
        combo_rows=all_combo_rows,
    )
    return {
        "model_name": TREND_MODEL_NAME,
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "period_years": int(period_years),
        "horizon_days": int(horizon_days),
        "top_n": int(top_n),
        "hold_days": int(hold_days),
        "periods": metrics,
        "indicator_rows": len(all_indicator_rows),
        "combo_rows": len(all_combo_rows),
        "report_file": report_file,
    }


def build_trend_pure_shadow_rows(
    settings: Settings,
    *,
    trade_date: str,
    top_n: int,
    hold_days: int,
    production: pd.DataFrame,
    production_top_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = score_trend_pure(settings, trade_date, trade_date)
    scored = scored[scored["trend_candidate"].fillna(False)].copy()
    if scored.empty:
        raise RuntimeError(f"No trend_pure_v1 candidates for {trade_date}")
    scored = scored.sort_values(["trend_pure_score", "ts_code"], ascending=[False, True]).copy()
    scored["factor_rank"] = range(1, len(scored) + 1)
    model_top = scored.head(top_n)
    production_top_codes = set(production[production["production_rank"] <= production_top_n]["ts_code"].astype(str))
    model_top_codes = set(model_top["ts_code"].astype(str))
    union_codes = model_top_codes | production_top_codes
    candidates = scored[scored["ts_code"].astype(str).isin(union_codes)].merge(
        production[
            [
                "ts_code",
                "production_score",
                "production_rank",
                "name",
                "industry",
                "concepts",
            ]
        ],
        on="ts_code",
        how="outer",
        suffixes=("", "_prod"),
    )
    candidates = candidates[candidates["ts_code"].astype(str).isin(union_codes)].copy()
    for column in ["name", "industry", "concepts"]:
        prod_col = f"{column}_prod"
        if prod_col in candidates.columns:
            candidates[column] = candidates[column].fillna(candidates[prod_col])
            candidates = candidates.drop(columns=[prod_col])
    candidates["is_model_pick"] = candidates["ts_code"].astype(str).isin(model_top_codes)
    candidates["is_production_pick"] = candidates["ts_code"].astype(str).isin(production_top_codes)
    candidates["comparison_type"] = "factor_only"
    candidates.loc[candidates["is_model_pick"] & candidates["is_production_pick"], "comparison_type"] = "confluence"
    candidates.loc[(~candidates["is_model_pick"]) & candidates["is_production_pick"], "comparison_type"] = "production_only"

    rows: list[dict[str, Any]] = []
    for row in candidates.sort_values(["comparison_type", "factor_rank"], na_position="last").to_dict("records"):
        comparison_type = row.get("comparison_type")
        if comparison_type == "confluence":
            reason = f"生产排名{int(row['production_rank'])} + 纯技术趋势排名{int(row['factor_rank'])}；{_combo_reason(pd.Series(row))}"
        elif comparison_type == "factor_only":
            factor_rank = row.get("factor_rank")
            if pd.notna(factor_rank):
                reason = f"纯技术趋势排名{int(factor_rank)}，生产未进Top{production_top_n}；{_combo_reason(pd.Series(row))}"
            else:
                reason = "纯技术趋势候选，但排名缺失"
        else:
            factor_rank = row.get("factor_rank")
            if pd.notna(factor_rank):
                reason = f"生产Top但纯技术趋势排名{int(factor_rank)}未进Top{top_n}"
            else:
                reason = "生产Top但不满足纯技术趋势候选过滤"
        rows.append(
            {
                "trade_date": normalize_date(trade_date),
                "model_name": TREND_MODEL_NAME,
                "top_n": int(top_n),
                "hold_days": int(hold_days),
                "ts_code": row.get("ts_code"),
                "name": row.get("name"),
                "industry": row.get("industry"),
                "concepts": row.get("concepts"),
                "factor_rank": int(row["factor_rank"]) if pd.notna(row.get("factor_rank")) else None,
                "factor_score": _clean_float(row.get("trend_pure_score")),
                "production_rank": int(row["production_rank"]) if pd.notna(row.get("production_rank")) else None,
                "production_score": _clean_float(row.get("production_score")),
                "comparison_type": row.get("comparison_type"),
                "is_model_pick": bool(row.get("is_model_pick")),
                "is_production_pick": bool(row.get("is_production_pick")),
                "reason": reason,
            }
        )
    return rows, {
        "model_name": TREND_MODEL_NAME,
        "top_n": int(top_n),
        "hold_days": int(hold_days),
        "candidate_rows": len(rows),
        "model_pick_rows": len(model_top_codes),
        "production_top_rows": len(production_top_codes),
        "input_scope": "daily_bars/daily_basic pure technical only",
    }
