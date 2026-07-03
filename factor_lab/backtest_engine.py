from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tech_signal.config import Settings
from tech_signal.db import connect, init_schema, qname, upsert_rows

from .factor_definitions import (
    add_cross_sectional_ranks,
    compute_factor_values,
    load_factor_base,
    normalize_date,
    selected_factor_names,
    target_slice,
)
from .factor_evaluator import _clean_float, _drawdown_stats, _format_pct, report_dir
from .weight_optimizer import load_model_weights, optimize_weights


def _load_trade_returns(settings: Settings, start_date: str, end_date: str, hold_days: int) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT trade_date, ts_code, open, high, low, close, pct_chg,
                   adj_open, adj_high, adj_low, adj_close, amount
            FROM {qname(settings, 'daily_bars')}
            WHERE trade_date BETWEEN %s::date - interval '5 days'
                                  AND %s::date + (%s || ' days')::interval
              AND adj_close IS NOT NULL
            ORDER BY ts_code, trade_date
            """,
            (normalize_date(start_date), normalize_date(end_date), hold_days * 3 + 10),
        )
        rows = cur.fetchall()
    prices = pd.DataFrame(rows)
    if prices.empty:
        return prices
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date
    for column in ["open", "high", "low", "close", "pct_chg", "adj_open", "adj_close", "amount"]:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.sort_values(["ts_code", "trade_date"])
    grouped = prices.groupby("ts_code", group_keys=False)
    prices["entry_date"] = grouped["trade_date"].shift(-1)
    prices["entry_price"] = grouped["adj_open"].shift(-1)
    prices["entry_raw_open"] = grouped["open"].shift(-1)
    prices["entry_raw_high"] = grouped["high"].shift(-1)
    prices["entry_raw_low"] = grouped["low"].shift(-1)
    prices["entry_pct_chg"] = grouped["pct_chg"].shift(-1)
    prices["exit_date"] = grouped["trade_date"].shift(-hold_days)
    prices["exit_price"] = grouped["adj_close"].shift(-hold_days)
    prices["future_return"] = prices["exit_price"] / prices["entry_price"] - 1.0
    start_dt = pd.to_datetime(normalize_date(start_date)).date()
    end_dt = pd.to_datetime(normalize_date(end_date)).date()
    return prices[(prices["trade_date"] >= start_dt) & (prices["trade_date"] <= end_dt)].copy()


def _score_frame(
    settings: Settings,
    start_date: str,
    end_date: str,
    weights: dict[str, float],
) -> pd.DataFrame:
    factor_names = selected_factor_names(weights.keys())
    base = load_factor_base(settings, start_date, end_date, lookback_calendar_days=90, forward_calendar_days=0)
    factor_frame = target_slice(compute_factor_values(base), start_date, end_date)
    factor_frame = add_cross_sectional_ranks(factor_frame, factor_names)
    score = pd.Series(0.0, index=factor_frame.index)
    reason_parts: list[str] = []
    for name in factor_names:
        pct_col = f"{name}__pct_rank"
        if pct_col not in factor_frame.columns:
            continue
        weight = float(weights.get(name, 0.0))
        score = score + factor_frame[pct_col].fillna(0.5) * weight
        if abs(weight) > 0.03:
            reason_parts.append(name)
    factor_frame["factor_score"] = score
    factor_frame["score_reason"] = ", ".join(reason_parts[:8])
    return factor_frame


def _performance_from_daily_returns(returns: pd.Series, *, periods_per_year: float = 252.0) -> dict[str, float | None]:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    if series.empty:
        return {
            "total_return": None,
            "annual_return": None,
            "max_drawdown": None,
            "sharpe": None,
            "win_rate": None,
        }
    equity = (1.0 + series).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (periods_per_year / max(len(series), 1)) - 1.0)
    _, max_drawdown = _drawdown_stats(series)
    std = series.std()
    sharpe = None
    if std and not math.isclose(float(std), 0.0):
        sharpe = float(series.mean() / std * math.sqrt(periods_per_year))
    return {
        "total_return": _clean_float(total_return),
        "annual_return": _clean_float(annual_return),
        "max_drawdown": max_drawdown,
        "sharpe": _clean_float(sharpe),
        "win_rate": _clean_float((series > 0).mean()),
    }


def _split_labels(dates: list[Any]) -> dict[str, set[Any]]:
    if not dates:
        return {"full": set(), "train": set(), "validation": set(), "test": set()}
    unique = sorted(dates)
    n = len(unique)
    train_end = int(n * 0.6)
    validation_end = int(n * 0.8)
    return {
        "full": set(unique),
        "train": set(unique[:train_end]),
        "validation": set(unique[train_end:validation_end]),
        "test": set(unique[validation_end:]),
    }


def run_backtest(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    model_name: str = "decorrelated_v1",
    top_n: int = 20,
    hold_days: int = 5,
    min_amount_yi: float = 2.0,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> dict[str, Any]:
    init_schema(settings)
    try:
        weights = load_model_weights(settings, model_name, end_date)
    except Exception:
        optimize_weights(settings, start_date, end_date, as_of_date=end_date, horizon_days=hold_days)
        weights = load_model_weights(settings, model_name, end_date)
    if not weights:
        optimize_weights(settings, start_date, end_date, as_of_date=end_date, horizon_days=hold_days)
        weights = load_model_weights(settings, model_name, end_date)

    scored = _score_frame(settings, start_date, end_date, weights)
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
                "exit_date",
                "exit_price",
                "future_return",
            ]
        ],
        on=["trade_date", "ts_code"],
        how="inner",
    )
    if frame.empty:
        raise RuntimeError(f"No scored trade rows for backtest {start_date} to {end_date}")

    round_trip_cost = 2.0 * (transaction_cost_bps + slippage_bps) / 10000.0
    frame["one_word_limit_up"] = (
        (frame["entry_pct_chg"] >= 9.5)
        & (frame["entry_raw_open"] == frame["entry_raw_high"])
        & (frame["entry_raw_open"] == frame["entry_raw_low"])
    )
    eligible = frame[
        frame["future_return"].notna()
        & frame["entry_price"].notna()
        & frame["exit_price"].notna()
        & (~frame["one_word_limit_up"].fillna(False))
        & (frame["amount_yi"].fillna(0) >= min_amount_yi)
    ].copy()
    eligible["net_return"] = eligible["future_return"] - round_trip_cost
    selected = (
        eligible.sort_values(["trade_date", "factor_score"], ascending=[True, False])
        .groupby("trade_date", group_keys=False)
        .head(top_n)
        .copy()
    )
    if selected.empty:
        raise RuntimeError("No selected trades after filters")

    daily = selected.groupby("trade_date").agg(
        cohort_return=("net_return", "mean"),
        trade_count=("ts_code", "count"),
    )
    daily["daily_return"] = daily["cohort_return"] / float(max(hold_days, 1))
    benchmark = eligible.groupby("trade_date")["net_return"].mean() / float(max(hold_days, 1))
    previous: set[str] | None = None
    turnovers = []
    for _, group in selected.groupby("trade_date"):
        current = set(group["ts_code"].astype(str))
        if previous is not None and current:
            turnovers.append(1.0 - len(current & previous) / max(len(current), 1))
        previous = current
    avg_turnover = float(np.mean(turnovers)) if turnovers else 1.0

    start_key = normalize_date(start_date).replace("-", "")
    end_key = normalize_date(end_date).replace("-", "")
    model_run_name = f"{model_name}_top{top_n}_hold{hold_days}_{start_key}_{end_key}"
    splits = _split_labels(list(daily.index))
    result_rows: list[dict[str, Any]] = []
    benchmark_total = _performance_from_daily_returns(benchmark.reindex(daily.index).fillna(0))["total_return"]
    for split_name, split_dates in splits.items():
        series = daily[daily.index.isin(split_dates)]["daily_return"]
        perf = _performance_from_daily_returns(series)
        bench_series = benchmark[benchmark.index.isin(split_dates)]
        bench_perf = _performance_from_daily_returns(bench_series)
        result_rows.append(
            {
                "model_name": model_run_name,
                "start_date": normalize_date(start_date),
                "end_date": normalize_date(end_date),
                "rebalance_rule": "daily_cohort",
                "hold_days": hold_days,
                "top_n": top_n,
                "sample_split": split_name,
                "total_return": perf["total_return"],
                "annual_return": perf["annual_return"],
                "max_drawdown": perf["max_drawdown"],
                "sharpe": perf["sharpe"],
                "win_rate": perf["win_rate"],
                "avg_turnover": _clean_float(avg_turnover),
                "avg_holding_days": _clean_float(hold_days),
                "trade_count": int(selected[selected["trade_date"].isin(split_dates)]["ts_code"].count()),
                "benchmark_return": bench_perf["total_return"],
                "excess_return": _clean_float((perf["total_return"] or 0.0) - (bench_perf["total_return"] or 0.0)),
            }
        )

    trade_rows: list[dict[str, Any]] = []
    weight = 1.0 / float(max(top_n, 1))
    for row in selected.itertuples():
        trade_rows.append(
            {
                "model_name": model_run_name,
                "trade_date": row.trade_date,
                "ts_code": row.ts_code,
                "action": "BUY",
                "price": _clean_float(row.entry_price),
                "weight": weight,
                "score": _clean_float(row.factor_score),
                "reason": row.score_reason,
                "holding_days": hold_days,
                "exit_date": row.exit_date,
                "exit_price": _clean_float(row.exit_price),
                "return_pct": _clean_float(row.net_return),
            }
        )

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
                f"DELETE FROM {qname(settings, 'strategy_backtest_trades')} WHERE model_name=%s AND trade_date BETWEEN %s AND %s",
                (model_run_name, normalize_date(start_date), normalize_date(end_date)),
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

    path = write_backtest_report(settings, start_date, end_date, model_run_name, result_rows, selected)
    full = next(row for row in result_rows if row["sample_split"] == "full")
    return {
        "model_name": model_run_name,
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "top_n": top_n,
        "hold_days": hold_days,
        "result_rows": result_count,
        "trade_rows": trade_count,
        "total_return": full["total_return"],
        "annual_return": full["annual_return"],
        "max_drawdown": full["max_drawdown"],
        "sharpe": full["sharpe"],
        "benchmark_return": benchmark_total,
        "report_file": str(path),
    }


def write_backtest_report(
    settings: Settings,
    start_date: str,
    end_date: str,
    model_name: str,
    result_rows: list[dict[str, Any]],
    selected: pd.DataFrame,
) -> Path:
    path = report_dir(settings) / f"strategy_backtest_{normalize_date(end_date).replace('-', '')}.md"
    lines = [
        f"# Strategy Backtest {normalize_date(start_date)} to {normalize_date(end_date)}",
        "",
        f"Model: {model_name}",
        "",
        "Execution assumption: close signal, next-day open entry, fixed holding period, equal weight, low-liquidity and one-word limit-up filters, cost/slippage deducted.",
        "",
        "## Performance",
        "",
    ]
    for row in result_rows:
        lines.append(
            f"- {row['sample_split']}: total={_format_pct(row['total_return'])}, annual={_format_pct(row['annual_return'])}, "
            f"mdd={_format_pct(row['max_drawdown'])}, sharpe={row['sharpe'] if row['sharpe'] is not None else 'NA'}, "
            f"win={_format_pct(row['win_rate'])}, excess={_format_pct(row['excess_return'])}, trades={row['trade_count']}"
        )
    lines.extend(["", "## Yearly Cohort Return", ""])
    yearly = selected.copy()
    yearly["year"] = pd.to_datetime(yearly["trade_date"]).dt.year
    year_stats = yearly.groupby("year")["net_return"].mean().sort_index()
    for year, value in year_stats.items():
        lines.append(f"- {year}: avg trade return={_format_pct(value)}")
    lines.extend(
        [
            "",
            "## Reliability Notes",
            "",
            "- This is a first-pass daily cohort backtest, not a production execution simulator.",
            "- Overlapping holdings are approximated by dividing holding-period cohort return by hold_days.",
            "- It is suitable for comparing factor schemes, not for final capacity or execution conclusions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
