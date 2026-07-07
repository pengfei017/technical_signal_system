from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from tech_signal.config import Settings
from tech_signal.db import connect, init_schema, qname, upsert_rows

from .backtest_engine import _add_execution_constraints, _load_trade_returns, _performance_from_daily_returns, _profit_loss_ratio
from .factor_definitions import normalize_date
from .factor_evaluator import _clean_float, report_dir


SHORT_MODEL_NAME = "short_strength_v1"
SHORT_TRACKING_HORIZONS = (1, 3, 5)
DEFAULT_TOP_NS = (20, 30)
DEFAULT_HOLD_DAYS = (1, 3, 5)
_SHORT_SCORE_CACHE_KEY: tuple[str, str, str] | None = None
_SHORT_SCORE_CACHE_FRAME: pd.DataFrame | None = None


def _rank_pct(frame: pd.DataFrame, column: str, *, higher_is_better: bool = True) -> pd.Series:
    value = pd.to_numeric(frame[column], errors="coerce")
    rank = value.groupby(frame["trade_date"]).rank(method="average", ascending=higher_is_better)
    count = value.groupby(frame["trade_date"]).transform("count")
    return rank / count.where(count > 0)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator.abs() > 1e-12)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _split_concepts(value: Any, *, max_items: int = 8) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = [item.strip() for item in re.split(r"[|,，;；/、]", text) if item.strip()]
    return parts[:max_items]


def _load_theme_heat(settings: Settings, start_date: str, end_date: str) -> dict[tuple[str, str], float]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT trade_date::text AS trade_date, theme_name, max(heat_score) AS heat_score
            FROM {qname(settings, 'theme_signal_daily')}
            WHERE trade_date BETWEEN %s AND %s
            GROUP BY trade_date, theme_name
            """,
            (normalize_date(start_date), normalize_date(end_date)),
        )
        rows = cur.fetchall()
    out: dict[tuple[str, str], float] = {}
    for row in rows:
        try:
            out[(str(row["trade_date"]), str(row["theme_name"]))] = float(row["heat_score"] or 0.0) / 100.0
        except Exception:
            continue
    return out


def _theme_heat_for_row(row: pd.Series, heat_map: dict[tuple[str, str], float]) -> float:
    date_key = str(row.get("trade_date"))
    names = []
    industry = str(row.get("industry") or "").strip()
    if industry:
        names.append(industry)
    names.extend(_split_concepts(row.get("concepts"), max_items=8))
    values = [heat_map.get((date_key, name), 0.0) for name in names]
    return max(values) if values else 0.0


def load_short_strength_base(settings: Settings, start_date: str, end_date: str) -> pd.DataFrame:
    start_value = normalize_date(start_date)
    end_value = normalize_date(end_date)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            WITH first_dates AS (
                SELECT ts_code, min(trade_date) AS first_trade_date
                FROM {qname(settings, 'daily_bars')}
                GROUP BY ts_code
            )
            SELECT s.trade_date, s.ts_code, s.name, s.industry, s.concepts,
                   s.close, s.pct_chg, s.amount_yi, s.turnover_rate,
                   s.ma5, s.ma10, s.ma20, s.ma60, s.bias5, s.rsi14,
                   s.volume_ratio_5, s.volume_ratio_20, s.volume_ratio,
                   s.high20, s.low20, s.technical_score, s.price_volume_score,
                   s.moneyflow_score, s.limit_score, s.lhb_score, s.total_signal_score,
                   s.signal_level, s.trend_phase, s.volume_state, s.limit_status,
                   s.is_limit_up, s.is_limit_down, s.is_broken_board, s.limit_times,
                   s.open_times, s.net_mf_amount_yi, s.net_mf_rate,
                   s.lhb_net_buy_yi, s.institution_net_buy_yi, s.northbound_net_buy_yi,
                   b.open, b.high, b.low, b.vol, b.amount,
                   b.adj_open, b.adj_high, b.adj_low, b.adj_close,
                   f.first_trade_date
            FROM {qname(settings, 'stock_signal_daily')} s
            JOIN {qname(settings, 'daily_bars')} b
              ON b.trade_date=s.trade_date AND b.ts_code=s.ts_code
            LEFT JOIN first_dates f ON f.ts_code=s.ts_code
            WHERE s.trade_date BETWEEN %s::date - interval '90 days' AND %s
              AND b.adj_close IS NOT NULL
            ORDER BY s.ts_code, s.trade_date
            """,
            (start_value, end_value),
        )
        rows = cur.fetchall()
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    for column in ["trade_date", "first_trade_date"]:
        data[column] = pd.to_datetime(data[column]).dt.date
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
        "volume_ratio_5",
        "volume_ratio_20",
        "volume_ratio",
        "high20",
        "low20",
        "technical_score",
        "price_volume_score",
        "moneyflow_score",
        "limit_score",
        "lhb_score",
        "total_signal_score",
        "limit_times",
        "open_times",
        "net_mf_amount_yi",
        "net_mf_rate",
        "lhb_net_buy_yi",
        "institution_net_buy_yi",
        "northbound_net_buy_yi",
        "open",
        "high",
        "low",
        "vol",
        "amount",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["is_limit_up", "is_limit_down", "is_broken_board"]:
        data[column] = data[column].fillna(False).astype(bool)
    return data


def score_short_strength(settings: Settings, start_date: str, end_date: str) -> pd.DataFrame:
    global _SHORT_SCORE_CACHE_KEY, _SHORT_SCORE_CACHE_FRAME
    cache_key = (settings.schema, normalize_date(start_date), normalize_date(end_date))
    if _SHORT_SCORE_CACHE_KEY == cache_key and _SHORT_SCORE_CACHE_FRAME is not None:
        return _SHORT_SCORE_CACHE_FRAME.copy()
    start_value = pd.to_datetime(normalize_date(start_date)).date()
    end_value = pd.to_datetime(normalize_date(end_date)).date()
    data = load_short_strength_base(settings, start_date, end_date)
    if data.empty:
        return data
    data = data.sort_values(["ts_code", "trade_date"]).copy()
    grouped = data.groupby("ts_code", group_keys=False)
    data["ret_3"] = grouped["adj_close"].pct_change(3)
    data["ret_5"] = grouped["adj_close"].pct_change(5)
    data["ret_10"] = grouped["adj_close"].pct_change(10)
    data["amount_ma5"] = grouped["amount_yi"].transform(lambda s: s.rolling(5, min_periods=3).mean().shift(1))
    data["moneyflow_ma3"] = grouped["net_mf_amount_yi"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    data["trend_rs_5"] = _safe_div(data["adj_close"], data["ma5"]) - 1.0
    data["trend_rs_10"] = _safe_div(data["adj_close"], data["ma10"]) - 1.0
    data["trend_breakout_20"] = (_safe_div(data["adj_close"], data["high20"]) - 0.995).clip(lower=-0.1, upper=0.1)
    data["new_high_20"] = (data["adj_close"] >= data["high20"] * 0.995).astype(float)
    data["volume_breakout"] = data["new_high_20"] * data["volume_ratio_20"].fillna(0)
    data["amount_expansion"] = _safe_div(data["amount_yi"], data["amount_ma5"])
    data["moneyflow_strength"] = data["net_mf_amount_yi"].fillna(0) + data["moneyflow_ma3"].fillna(0) * 0.5

    data["industry_mean_pct"] = data.groupby(["trade_date", "industry"])["pct_chg"].transform("mean")
    data["industry_strong_count"] = data.groupby(["trade_date", "industry"])["pct_chg"].transform(lambda s: (pd.to_numeric(s, errors="coerce") >= 5).sum())
    data["industry_limit_count"] = data.groupby(["trade_date", "industry"])["is_limit_up"].transform("sum")
    data["industry_heat_raw"] = (
        data["industry_mean_pct"].fillna(0) / 10.0
        + data["industry_strong_count"].fillna(0).clip(upper=20) / 20.0
        + data["industry_limit_count"].fillna(0).clip(upper=10) / 10.0
    )

    target = data[(data["trade_date"] >= start_value) & (data["trade_date"] <= end_value)].copy()
    if target.empty:
        return target
    heat_map = _load_theme_heat(settings, start_date, end_date)
    if heat_map:
        target["theme_heat_score"] = target.apply(lambda row: _theme_heat_for_row(row, heat_map), axis=1)
    else:
        target["theme_heat_score"] = 0.0

    for column in [
        "pct_chg",
        "ret_3",
        "ret_5",
        "ret_10",
        "trend_rs_5",
        "trend_rs_10",
        "trend_breakout_20",
        "volume_breakout",
        "volume_ratio_5",
        "volume_ratio_20",
        "amount_yi",
        "turnover_rate",
        "moneyflow_strength",
        "industry_heat_raw",
    ]:
        target[f"{column}_rank"] = _rank_pct(target, column, higher_is_better=True).fillna(0.0)

    target["strength_condition"] = (
        (target["ret_5_rank"] >= 0.70)
        | (target["ret_10_rank"] >= 0.70)
        | (target["trend_rs_5_rank"] >= 0.70)
        | (target["trend_rs_10_rank"] >= 0.70)
        | (target["pct_chg_rank"] >= 0.70)
    )
    target["liquidity_condition"] = (
        (target["amount_yi"].fillna(0) >= 3.0)
        & (target["turnover_rate"].fillna(0) >= 1.0)
    ) | (target["amount_yi"].fillna(0) >= 8.0)
    target["breakout_condition"] = (
        (target["new_high_20"] > 0)
        | (target["trend_breakout_20"] > 0)
        | (target["volume_breakout"] >= 1.2)
    )
    target["volume_condition"] = (
        (target["volume_ratio_5"].fillna(0) >= 1.2)
        | (target["volume_ratio_20"].fillna(0) >= 1.2)
        | (target["amount_expansion"].fillna(0) >= 1.2)
    )
    target["moneyflow_condition"] = (
        (target["net_mf_amount_yi"].fillna(0) > 0)
        | (target["net_mf_rate"].fillna(0) > 0)
        | (target["moneyflow_score"].fillna(0) > 0)
    )
    target["heat_condition"] = (
        (target["industry_heat_raw_rank"] >= 0.70)
        | (target["theme_heat_score"].fillna(0) >= 0.70)
    )
    target["event_condition"] = (
        target["is_limit_up"].fillna(False)
        | (target["limit_times"].fillna(0) > 0)
        | (target["lhb_net_buy_yi"].fillna(0) > 0)
        | (target["institution_net_buy_yi"].fillna(0) > 0)
        | (target["northbound_net_buy_yi"].fillna(0) > 0)
        | (target["limit_score"].fillna(0) > 0)
        | (target["lhb_score"].fillna(0) > 0)
    )
    condition_cols = [
        "strength_condition",
        "liquidity_condition",
        "breakout_condition",
        "volume_condition",
        "moneyflow_condition",
        "heat_condition",
        "event_condition",
    ]
    target["attack_condition_count"] = target[condition_cols].sum(axis=1)

    target["st_or_delist"] = target["name"].astype(str).str.contains("ST|退", regex=True, na=False)
    target["listed_days"] = (
        pd.to_datetime(target["trade_date"]) - pd.to_datetime(target["first_trade_date"])
    ).dt.days
    high_low_range = (target["high"] - target["low"]).replace(0, np.nan)
    target["upper_shadow_ratio"] = (target["high"] - target[["open", "close"]].max(axis=1)) / high_low_range
    target["long_upper_shadow"] = (target["upper_shadow_ratio"].fillna(0) >= 0.45) & (target["pct_chg"].fillna(0) < 6)
    target["one_word_limit_signal"] = (
        target["is_limit_up"].fillna(False)
        & (target["open"] == target["high"])
        & (target["open"] == target["low"])
    )
    target["high_turnover_stall"] = (
        (target["turnover_rate"].fillna(0) >= 8.0)
        & (target["pct_chg"].fillna(0).abs() <= 1.0)
        & (target["volume_ratio_20"].fillna(0) >= 1.2)
    )
    target["consecutive_shrink"] = (
        (target["volume_ratio_5"].fillna(1.0) <= 0.75)
        & (target["volume_ratio_20"].fillna(1.0) <= 0.85)
    )
    target["below_ma20"] = target["adj_close"] < target["ma20"]
    target["low_liquidity"] = (target["amount_yi"].fillna(0) < 2.0) | (target["turnover_rate"].fillna(0) < 0.5)
    target["young_stock"] = target["listed_days"].fillna(9999) < 60

    target["is_short_candidate"] = (
        (~target["st_or_delist"])
        & (~target["young_stock"])
        & target["liquidity_condition"]
        & (
            (target["strength_condition"] & (target["attack_condition_count"] >= 3))
            | ((target["event_condition"] | target["breakout_condition"]) & (target["attack_condition_count"] >= 2))
        )
    )

    positive_score = (
        target["ret_5_rank"] * 0.16
        + target["ret_10_rank"] * 0.08
        + target["trend_rs_5_rank"] * 0.10
        + target["trend_rs_10_rank"] * 0.06
        + target["trend_breakout_20_rank"] * 0.10
        + target["new_high_20"].fillna(0) * 0.08
        + target["volume_breakout_rank"] * 0.09
        + target["volume_ratio_5_rank"] * 0.06
        + target["amount_yi_rank"] * 0.05
        + target["turnover_rate_rank"] * 0.05
        + target["moneyflow_strength_rank"] * 0.09
        + target["industry_heat_raw_rank"] * 0.04
        + target["theme_heat_score"].fillna(0).clip(0, 1) * 0.04
        + target["is_limit_up"].astype(float) * 0.04
        + (target["limit_times"].fillna(0).clip(0, 3) / 3.0) * 0.03
        + _rank_pct(target.assign(lhb_positive=target["lhb_net_buy_yi"].fillna(0).clip(lower=0)), "lhb_positive").fillna(0) * 0.03
    )
    risk_penalty = (
        target["high_turnover_stall"].astype(float) * 0.10
        + target["long_upper_shadow"].astype(float) * 0.08
        + target["is_broken_board"].astype(float) * 0.10
        + target["one_word_limit_signal"].astype(float) * 0.08
        + target["consecutive_shrink"].astype(float) * 0.05
        + target["below_ma20"].astype(float) * 0.08
        + target["low_liquidity"].astype(float) * 0.10
        + target["st_or_delist"].astype(float) * 0.50
        + target["young_stock"].astype(float) * 0.12
    )
    target["short_strength_positive_score"] = positive_score
    target["short_strength_risk_penalty"] = risk_penalty
    target["short_strength_score"] = positive_score - risk_penalty

    target["short_reason"] = target.apply(_short_reason, axis=1)
    target["risk_reason"] = target.apply(_risk_reason, axis=1)
    target = target.sort_values(["trade_date", "short_strength_score", "ts_code"], ascending=[True, False, True])
    target["short_strength_rank"] = target.groupby("trade_date")["short_strength_score"].rank(method="first", ascending=False)
    _SHORT_SCORE_CACHE_KEY = cache_key
    _SHORT_SCORE_CACHE_FRAME = target.copy()
    return target


def _short_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if bool(row.get("strength_condition")):
        reasons.append("短期相对强")
    if bool(row.get("breakout_condition")):
        reasons.append("突破/新高")
    if bool(row.get("volume_condition")):
        reasons.append("量能放大")
    if bool(row.get("moneyflow_condition")):
        reasons.append("资金改善")
    if bool(row.get("heat_condition")):
        reasons.append("行业/主题热")
    if bool(row.get("event_condition")):
        reasons.append("涨停/龙虎榜事件")
    return "、".join(reasons[:6])


def _risk_reason(row: pd.Series) -> str:
    risks: list[str] = []
    if bool(row.get("high_turnover_stall")):
        risks.append("高换手滞涨")
    if bool(row.get("long_upper_shadow")):
        risks.append("放量长上影")
    if bool(row.get("is_broken_board")):
        risks.append("炸板")
    if bool(row.get("one_word_limit_signal")):
        risks.append("一字板")
    if bool(row.get("consecutive_shrink")):
        risks.append("连续缩量")
    if bool(row.get("below_ma20")):
        risks.append("跌破MA20")
    if bool(row.get("low_liquidity")):
        risks.append("流动性不足")
    if bool(row.get("st_or_delist")):
        risks.append("ST/退市")
    if bool(row.get("young_stock")):
        risks.append("上市过短")
    return "、".join(risks[:6])


def _split_labels(dates: list[Any]) -> dict[str, set[Any]]:
    unique = sorted(dates)
    if not unique:
        return {"full": set(), "train": set(), "validation": set(), "test": set()}
    n = len(unique)
    train_end = int(n * 0.6)
    validation_end = int(n * 0.8)
    return {
        "full": set(unique),
        "train": set(unique[:train_end]),
        "validation": set(unique[train_end:validation_end]),
        "test": set(unique[validation_end:]),
    }


def _turnover(selected: pd.DataFrame) -> float | None:
    previous: set[str] | None = None
    values = []
    for _, group in selected.groupby("trade_date"):
        current = set(group["ts_code"].astype(str))
        if previous is not None and current:
            values.append(1.0 - len(current & previous) / max(len(current), 1))
        previous = current
    return _clean_float(float(np.mean(values))) if values else None


def _prepare_short_backtest_frame(
    settings: Settings,
    start_date: str,
    end_date: str,
    hold_days: int,
) -> pd.DataFrame:
    scored = score_short_strength(settings, start_date, end_date)
    if scored.empty:
        raise RuntimeError(f"No short_strength score rows for {start_date} to {end_date}")
    returns = _load_trade_returns(settings, start_date, end_date, hold_days)
    frame = scored.merge(
        returns[
            [
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
                "exit_delayed_days",
                "exit_was_delayed",
                "exit_blocked",
                "future_return",
            ]
        ],
        on=["trade_date", "ts_code"],
        how="left",
    )
    frame = _add_execution_constraints(frame, top_n=30, capacity_pct=0.0, portfolio_capital_yi=0.0)
    frame["one_word_limit_up_next"] = frame["one_word_limit_up"]
    frame["buyable_base"] = (
        frame["future_return"].notna()
        & frame["entry_price"].notna()
        & frame["exit_price"].notna()
        & (~frame["one_word_limit_up_next"].fillna(False))
        & (~frame["exit_blocked"].fillna(False))
        & (frame["amount_yi"].fillna(0) >= 2.0)
    )
    frame["buyable"] = frame["buyable_base"]
    return frame


def _metrics_for_split(
    *,
    split_name: str,
    split_dates: set[Any],
    selected: pd.DataFrame,
    selected_raw: pd.DataFrame,
    benchmark: pd.Series,
    hold_days: int,
    top_n: int,
    start_date: str,
    end_date: str,
    model_run_name: str,
) -> dict[str, Any]:
    split_selected = selected[selected["trade_date"].isin(split_dates)].copy()
    split_raw = selected_raw[selected_raw["trade_date"].isin(split_dates)].copy()
    if split_selected.empty:
        daily_return = pd.Series(dtype=float)
    else:
        daily_return = split_selected.groupby("trade_date")["net_return"].mean() / float(max(hold_days, 1))
    perf = _performance_from_daily_returns(daily_return)
    bench_series = benchmark[benchmark.index.isin(split_dates)]
    bench_perf = _performance_from_daily_returns(bench_series)
    raw_count = int(split_raw["ts_code"].count())
    buyable_count = int(split_selected["ts_code"].count())
    avg_return = _clean_float(split_selected["net_return"].mean()) if not split_selected.empty else None
    avg_benchmark = _clean_float((bench_series * float(max(hold_days, 1))).mean()) if not bench_series.empty else None
    avg_excess = _clean_float((avg_return or 0.0) - (avg_benchmark or 0.0)) if avg_return is not None and avg_benchmark is not None else None
    return {
        "model_name": model_run_name,
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "rebalance_rule": "daily_short_strength",
        "hold_days": hold_days,
        "top_n": top_n,
        "sample_split": split_name,
        "total_return": perf["total_return"],
        "annual_return": perf["annual_return"],
        "max_drawdown": perf["max_drawdown"],
        "sharpe": perf["sharpe"],
        "win_rate": _clean_float((split_selected["net_return"] > 0).mean()) if not split_selected.empty else None,
        "profit_loss_ratio": _profit_loss_ratio(split_selected["net_return"]) if not split_selected.empty else None,
        "buyable_ratio": _clean_float(buyable_count / raw_count) if raw_count else None,
        "avg_turnover": _turnover(split_selected),
        "avg_holding_days": _clean_float(hold_days),
        "trade_count": buyable_count,
        "benchmark_return": bench_perf["total_return"],
        "excess_return": _clean_float((perf["total_return"] or 0.0) - (bench_perf["total_return"] or 0.0)),
        "avg_return": avg_return,
        "avg_excess_return": avg_excess,
    }


def run_short_strength_backtest(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    top_ns: tuple[int, ...] = DEFAULT_TOP_NS,
    hold_days_list: tuple[int, ...] = DEFAULT_HOLD_DAYS,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    capacity_pct: float = 0.005,
    portfolio_capital_yi: float = 0.1,
) -> dict[str, Any]:
    init_schema(settings)
    result_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    report_snapshots: list[dict[str, Any]] = []
    for hold_days in hold_days_list:
        frame = _prepare_short_backtest_frame(settings, start_date, end_date, hold_days)
        round_trip_cost = 2.0 * (transaction_cost_bps + slippage_bps) / 10000.0
        frame["net_return"] = frame["future_return"] - round_trip_cost
        for top_n in top_ns:
            frame_top = _add_execution_constraints(
                frame,
                top_n=top_n,
                capacity_pct=capacity_pct,
                portfolio_capital_yi=portfolio_capital_yi,
            )
            frame_top["buyable"] = frame_top["buyable_base"] & frame_top["capacity_pass"].fillna(False)
            benchmark_pool = frame_top[frame_top["buyable"]].copy()
            benchmark = benchmark_pool.groupby("trade_date")["net_return"].mean() / float(max(hold_days, 1))
            selected_raw = (
                frame_top[frame_top["is_short_candidate"]]
                .sort_values(["trade_date", "short_strength_score"], ascending=[True, False])
                .groupby("trade_date", group_keys=False)
                .head(top_n)
                .copy()
            )
            selected = selected_raw[selected_raw["buyable"]].copy()
            if selected_raw.empty:
                continue
            start_key = normalize_date(start_date).replace("-", "")
            end_key = normalize_date(end_date).replace("-", "")
            model_run_name = f"{SHORT_MODEL_NAME}_top{top_n}_hold{hold_days}_{start_key}_{end_key}"
            split_labels = _split_labels(list(selected_raw["trade_date"].dropna().unique()))
            for split_name, split_dates in split_labels.items():
                row = _metrics_for_split(
                    split_name=split_name,
                    split_dates=split_dates,
                    selected=selected,
                    selected_raw=selected_raw,
                    benchmark=benchmark,
                    hold_days=hold_days,
                    top_n=top_n,
                    start_date=start_date,
                    end_date=end_date,
                    model_run_name=model_run_name,
                )
                result_rows.append(row)
                report_snapshots.append(row)
            weight = 1.0 / float(max(top_n, 1))
            for row in selected.itertuples():
                reason = row.short_reason
                if row.risk_reason:
                    reason = f"{reason}; 风险扣分：{row.risk_reason}"
                trade_rows.append(
                    {
                        "model_name": model_run_name,
                        "trade_date": row.trade_date,
                        "ts_code": row.ts_code,
                        "action": "BUY",
                        "price": _clean_float(row.entry_price),
                        "weight": weight,
                        "score": _clean_float(row.short_strength_score),
                        "reason": reason,
                        "holding_days": hold_days,
                        "exit_date": row.exit_date,
                        "exit_price": _clean_float(row.exit_price),
                        "return_pct": _clean_float(row.net_return),
                    }
                )
    if not result_rows:
        raise RuntimeError("No short_strength backtest rows generated")

    result_columns = [
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
                f"""
                DELETE FROM {qname(settings, 'strategy_backtest_result')}
                WHERE start_date=%s AND end_date=%s AND model_name LIKE %s
                """,
                (normalize_date(start_date), normalize_date(end_date), f"{SHORT_MODEL_NAME}_%"),
            )
            cur.execute(
                f"""
                DELETE FROM {qname(settings, 'strategy_backtest_trades')}
                WHERE trade_date BETWEEN %s AND %s AND model_name LIKE %s
                """,
                (normalize_date(start_date), normalize_date(end_date), f"{SHORT_MODEL_NAME}_%"),
            )
        result_count = upsert_rows(
            conn,
            table=qname(settings, "strategy_backtest_result"),
            columns=result_columns,
            rows=result_rows,
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

    report_file = write_short_strength_backtest_report(settings, start_date, end_date, report_snapshots)
    return {
        "model_name": SHORT_MODEL_NAME,
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "result_rows": result_count,
        "trade_rows": trade_count,
        "report_file": report_file,
    }


def _fmt_pct(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "NA"


def _fmt_num(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        return f"{float(value):.3f}"
    except Exception:
        return "NA"


def write_short_strength_backtest_report(
    settings: Settings,
    start_date: str,
    end_date: str,
    rows: list[dict[str, Any]],
) -> str:
    path = report_dir(settings) / f"short_strength_backtest_{normalize_date(end_date).replace('-', '')}.md"
    data = pd.DataFrame(rows)
    lines = [
        f"# Short Strength Backtest {normalize_date(start_date)} to {normalize_date(end_date)}",
        "",
        "研究影子模型输出，不修改 `stock_signal_daily` 生产评分权重。",
        "",
        "- 模型目标：短期强势股识别，主看 T+3/T+5 超额收益，T+1 辅助观察。",
        "- 候选池：先筛进攻属性，再排序；低波动/低风险本身不作为正向选股原因。",
        "- 风险项：高位滞涨、长上影、炸板、一字板、缩量、跌破 MA20、低流动性、ST/上市过短只扣分。",
        "- 交易口径：收盘信号，次日复权开盘买入，固定持有后复权收盘退出，扣交易成本和滑点。",
        "",
    ]
    if data.empty:
        lines.append("- No backtest rows.")
    else:
        display = data.sort_values(["hold_days", "top_n", "sample_split"])
        lines.append("| split | top | horizon | avg_return | avg_excess | win | max_dd | PL | buyable | trades |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in display.to_dict("records"):
            lines.append(
                f"| {row['sample_split']} | {row['top_n']} | T+{row['hold_days']} | "
                f"{_fmt_pct(row.get('avg_return'))} | {_fmt_pct(row.get('avg_excess_return'))} | "
                f"{_fmt_pct(row.get('win_rate'))} | {_fmt_pct(row.get('max_drawdown'))} | "
                f"{_fmt_num(row.get('profit_loss_ratio'))} | {_fmt_pct(row.get('buyable_ratio'))} | {row.get('trade_count')} |"
            )
        lines.extend(["", "## Walk-Forward View", ""])
        full_dates = sorted({item for item in data["start_date"].dropna().unique()})
        if full_dates:
            lines.append("- 当前版本为固定规则短线强势模型，未用测试段收益调权；walk-forward 以滚动时间切片观察稳定性。")
        for (top_n, hold_days), group in display[display["sample_split"].isin(["validation", "test"])].groupby(["top_n", "hold_days"]):
            val = group[group["sample_split"] == "validation"]
            test = group[group["sample_split"] == "test"]
            val_excess = val.iloc[0]["avg_excess_return"] if not val.empty else None
            test_excess = test.iloc[0]["avg_excess_return"] if not test.empty else None
            lines.append(f"- Top{top_n} T+{hold_days}: validation avg_excess={_fmt_pct(val_excess)}, test avg_excess={_fmt_pct(test_excess)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def build_short_strength_shadow_rows(
    settings: Settings,
    *,
    trade_date: str,
    top_n: int,
    hold_days: int,
    production: pd.DataFrame,
    production_top_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = score_short_strength(settings, trade_date, trade_date)
    if scored.empty:
        raise RuntimeError(f"No short_strength rows for {trade_date}")
    candidates = scored[scored["is_short_candidate"]].copy()
    candidates = candidates.sort_values(["short_strength_score", "ts_code"], ascending=[False, True]).copy()
    candidates["factor_rank"] = range(1, len(candidates) + 1)
    merged = candidates.merge(
        production[["ts_code", "production_score", "production_rank", "name", "industry", "concepts"]],
        on="ts_code",
        how="outer",
        suffixes=("", "_prod"),
    )
    for column in ["name", "industry", "concepts"]:
        prod_col = f"{column}_prod"
        if prod_col in merged.columns:
            merged[column] = merged[column].fillna(merged[prod_col])
            merged = merged.drop(columns=[prod_col])
    model_top = merged[merged["factor_rank"] <= top_n].copy()
    model_codes = set(model_top["ts_code"].astype(str))
    production_codes = set(production[production["production_rank"] <= production_top_n]["ts_code"].astype(str))
    union_codes = model_codes | production_codes
    out = merged[merged["ts_code"].astype(str).isin(union_codes)].copy()
    out["is_model_pick"] = out["ts_code"].astype(str).isin(model_codes)
    out["is_production_pick"] = out["ts_code"].astype(str).isin(production_codes)
    out["comparison_type"] = "factor_only"
    out.loc[out["is_model_pick"] & out["is_production_pick"], "comparison_type"] = "confluence"
    out.loc[(~out["is_model_pick"]) & out["is_production_pick"], "comparison_type"] = "production_only"

    rows: list[dict[str, Any]] = []
    for row in out.sort_values(["comparison_type", "factor_rank"], na_position="last").to_dict("records"):
        reason = _text(row.get("short_reason"))
        risk_reason = _text(row.get("risk_reason"))
        if risk_reason:
            reason = f"{reason}; 风险扣分：{risk_reason}"
        if row.get("comparison_type") == "confluence":
            reason = f"共振：生产排名{int(row['production_rank'])} + 短线排名{int(row['factor_rank'])}; {reason}"
        elif row.get("comparison_type") == "factor_only":
            prod_rank = row.get("production_rank")
            prod_text = f"生产排名{int(prod_rank)}" if pd.notna(prod_rank) else "生产无有效排名"
            reason = f"短线排名{int(row['factor_rank'])}，{prod_text}，未进生产Top{production_top_n}; {reason}"
        elif row.get("comparison_type") == "production_only":
            factor_rank = row.get("factor_rank")
            factor_text = f"短线排名{int(factor_rank)}" if pd.notna(factor_rank) else "短线无有效排名"
            reason = f"生产排名{int(row['production_rank'])}，{factor_text}，未进短线Top{top_n}; {reason}"
        rows.append(
            {
                "trade_date": normalize_date(trade_date),
                "model_name": SHORT_MODEL_NAME,
                "top_n": top_n,
                "hold_days": hold_days,
                "ts_code": row.get("ts_code"),
                "name": row.get("name"),
                "industry": row.get("industry"),
                "concepts": row.get("concepts"),
                "factor_rank": int(row["factor_rank"]) if pd.notna(row.get("factor_rank")) else None,
                "factor_score": _clean_float(row.get("short_strength_score")),
                "production_rank": int(row["production_rank"]) if pd.notna(row.get("production_rank")) else None,
                "production_score": _clean_float(row.get("production_score")),
                "comparison_type": row.get("comparison_type"),
                "is_model_pick": bool(row.get("is_model_pick")),
                "is_production_pick": bool(row.get("is_production_pick")),
                "reason": reason.strip("; "),
            }
        )
    return rows, {
        "model_name": SHORT_MODEL_NAME,
        "top_n": top_n,
        "hold_days": hold_days,
        "candidate_pool_rows": int(candidates["ts_code"].count()),
        "candidate_rows": len(rows),
        "model_pick_rows": len(model_codes),
        "production_top_rows": len(production_codes),
    }
