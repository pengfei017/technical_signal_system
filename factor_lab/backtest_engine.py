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
from .weight_optimizer import _split_dates, load_model_weights, optimize_weights


_SCORE_CACHE_KEY: tuple[Any, ...] | None = None
_SCORE_CACHE_FRAME: pd.DataFrame | None = None


def _board_limit_pct(ts_code: pd.Series, trade_date: pd.Series) -> pd.Series:
    """Return the daily up/down limit percentage by board."""
    code = ts_code.astype(str).str.upper()
    dates = pd.to_datetime(trade_date, errors="coerce")
    pct = pd.Series(10.0, index=ts_code.index, dtype="float64")

    pct[code.str.endswith(".BJ", na=False)] = 30.0
    pct[code.str.startswith(("688", "689"), na=False) & code.str.endswith(".SH", na=False)] = 20.0

    chinext = code.str.startswith(("300", "301"), na=False) & code.str.endswith(".SZ", na=False)
    pct[chinext & dates.ge(pd.Timestamp("2020-08-24"))] = 20.0
    pct[chinext & dates.lt(pd.Timestamp("2020-08-24"))] = 10.0
    return pct


def _one_word_limit_up(data: pd.DataFrame, *, prefix: str = "") -> pd.Series:
    open_col = f"{prefix}open"
    high_col = f"{prefix}high"
    low_col = f"{prefix}low"
    pct_col = f"{prefix}pct_chg"
    limit_col = f"{prefix}limit_pct"
    flat_board = np.isclose(data[open_col], data[high_col], equal_nan=False) & np.isclose(
        data[open_col], data[low_col], equal_nan=False
    )
    return pd.Series(flat_board, index=data.index) & (pd.to_numeric(data[pct_col], errors="coerce") >= data[limit_col] - 0.5)


def _one_word_limit_down(data: pd.DataFrame, *, prefix: str = "") -> pd.Series:
    open_col = f"{prefix}open"
    high_col = f"{prefix}high"
    low_col = f"{prefix}low"
    pct_col = f"{prefix}pct_chg"
    limit_col = f"{prefix}limit_pct"
    flat_board = np.isclose(data[open_col], data[high_col], equal_nan=False) & np.isclose(
        data[open_col], data[low_col], equal_nan=False
    )
    return pd.Series(flat_board, index=data.index) & (pd.to_numeric(data[pct_col], errors="coerce") <= -(data[limit_col] - 0.5))


def _apply_exit_delay_for_limit_down(prices: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    grouped = prices.groupby("ts_code", group_keys=False)
    prices["scheduled_exit_date"] = grouped["trade_date"].shift(-hold_days)
    prices["scheduled_exit_one_word_limit_down"] = grouped["raw_one_word_limit_down"].shift(-hold_days).fillna(False).astype(bool)
    prices["exit_date"] = prices["scheduled_exit_date"]
    prices["exit_price"] = grouped["adj_close"].shift(-hold_days)
    prices["exit_raw_open"] = grouped["open"].shift(-hold_days)
    prices["exit_raw_high"] = grouped["high"].shift(-hold_days)
    prices["exit_raw_low"] = grouped["low"].shift(-hold_days)
    prices["exit_raw_close"] = grouped["close"].shift(-hold_days)
    prices["exit_pct_chg"] = grouped["pct_chg"].shift(-hold_days)
    prices["exit_limit_pct"] = grouped["limit_pct"].shift(-hold_days)
    prices["exit_amount_yi"] = grouped["amount_yi"].shift(-hold_days)
    prices["exit_delayed_days"] = 0
    prices["exit_was_delayed"] = False
    prices["exit_blocked"] = False

    locked_rows = prices.index[prices["scheduled_exit_one_word_limit_down"]].to_numpy()
    if len(locked_rows) == 0:
        return prices

    group_indices = {code: group.index.to_numpy() for code, group in prices.groupby("ts_code", sort=False)}
    row_offsets = prices.groupby("ts_code").cumcount().to_numpy()
    raw_limit_down = prices["raw_one_word_limit_down"].to_numpy()

    for row_idx in locked_rows:
        code = prices.at[row_idx, "ts_code"]
        idx = group_indices.get(code)
        if idx is None:
            continue
        scheduled_offset = int(row_offsets[row_idx]) + hold_days
        final_offset = scheduled_offset
        while final_offset < len(idx) and bool(raw_limit_down[idx[final_offset]]):
            final_offset += 1
        if final_offset >= len(idx):
            prices.at[row_idx, "exit_blocked"] = True
            prices.at[row_idx, "exit_date"] = None
            prices.at[row_idx, "exit_price"] = np.nan
            prices.at[row_idx, "exit_raw_open"] = np.nan
            prices.at[row_idx, "exit_raw_high"] = np.nan
            prices.at[row_idx, "exit_raw_low"] = np.nan
            prices.at[row_idx, "exit_raw_close"] = np.nan
            prices.at[row_idx, "exit_pct_chg"] = np.nan
            prices.at[row_idx, "exit_limit_pct"] = np.nan
            prices.at[row_idx, "exit_amount_yi"] = np.nan
            continue

        final_idx = idx[final_offset]
        prices.at[row_idx, "exit_date"] = prices.at[final_idx, "trade_date"]
        prices.at[row_idx, "exit_price"] = prices.at[final_idx, "adj_close"]
        prices.at[row_idx, "exit_raw_open"] = prices.at[final_idx, "open"]
        prices.at[row_idx, "exit_raw_high"] = prices.at[final_idx, "high"]
        prices.at[row_idx, "exit_raw_low"] = prices.at[final_idx, "low"]
        prices.at[row_idx, "exit_raw_close"] = prices.at[final_idx, "close"]
        prices.at[row_idx, "exit_pct_chg"] = prices.at[final_idx, "pct_chg"]
        prices.at[row_idx, "exit_limit_pct"] = prices.at[final_idx, "limit_pct"]
        prices.at[row_idx, "exit_amount_yi"] = prices.at[final_idx, "amount_yi"]
        prices.at[row_idx, "exit_delayed_days"] = final_offset - scheduled_offset
        prices.at[row_idx, "exit_was_delayed"] = final_offset > scheduled_offset
    return prices


def _add_execution_constraints(
    frame: pd.DataFrame,
    *,
    top_n: int,
    capacity_pct: float = 0.005,
    portfolio_capital_yi: float = 0.1,
) -> pd.DataFrame:
    data = frame.copy()
    if "entry_one_word_limit_up" in data.columns:
        data["one_word_limit_up"] = data["entry_one_word_limit_up"].fillna(False).astype(bool)
    else:
        data["entry_limit_pct"] = _board_limit_pct(data["ts_code"], data["entry_date"])
        data = data.rename(
            columns={
                "entry_raw_open": "entry_open",
                "entry_raw_high": "entry_high",
                "entry_raw_low": "entry_low",
            }
        )
        data["one_word_limit_up"] = _one_word_limit_up(data, prefix="entry_")
        data = data.rename(
            columns={
                "entry_open": "entry_raw_open",
                "entry_high": "entry_raw_high",
                "entry_low": "entry_raw_low",
            }
        )

    data["exit_blocked"] = data.get("exit_blocked", False)
    data["exit_blocked"] = data["exit_blocked"].fillna(False).astype(bool)
    data["position_amount_yi"] = float(portfolio_capital_yi) / float(max(top_n, 1))
    if "entry_amount_yi" not in data.columns:
        data["entry_amount_yi"] = np.nan
    data["capacity_limit_yi"] = pd.to_numeric(data["entry_amount_yi"], errors="coerce") * float(capacity_pct)
    if capacity_pct > 0 and portfolio_capital_yi > 0:
        data["capacity_pass"] = data["capacity_limit_yi"].fillna(-1) >= data["position_amount_yi"]
    else:
        data["capacity_pass"] = True
    return data


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
    for column in ["open", "high", "low", "close", "pct_chg", "adj_open", "adj_high", "adj_low", "adj_close", "amount"]:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    prices["amount_yi"] = prices["amount"] / 100000.0
    prices["limit_pct"] = _board_limit_pct(prices["ts_code"], prices["trade_date"])
    prices["raw_one_word_limit_up"] = _one_word_limit_up(prices)
    prices["raw_one_word_limit_down"] = _one_word_limit_down(prices)
    grouped = prices.groupby("ts_code", group_keys=False)
    prices["entry_date"] = grouped["trade_date"].shift(-1)
    prices["entry_price"] = grouped["adj_open"].shift(-1)
    prices["entry_raw_open"] = grouped["open"].shift(-1)
    prices["entry_raw_high"] = grouped["high"].shift(-1)
    prices["entry_raw_low"] = grouped["low"].shift(-1)
    prices["entry_pct_chg"] = grouped["pct_chg"].shift(-1)
    prices["entry_amount_yi"] = grouped["amount_yi"].shift(-1)
    prices["entry_limit_pct"] = grouped["limit_pct"].shift(-1)
    prices["entry_one_word_limit_up"] = grouped["raw_one_word_limit_up"].shift(-1).fillna(False).astype(bool)
    prices = _apply_exit_delay_for_limit_down(prices, hold_days)
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
    global _SCORE_CACHE_FRAME, _SCORE_CACHE_KEY
    factor_names = selected_factor_names(weights.keys())
    weight_items = [(name, float(weights.get(name, 0.0))) for name in factor_names if name in weights]
    weight_items = [(name, weight) for name, weight in weight_items if not math.isclose(weight, 0.0)]
    if not weight_items:
        raise RuntimeError("No non-zero factor weights available for scoring")
    cache_key = (normalize_date(start_date), normalize_date(end_date), tuple(sorted(weight_items)))
    if _SCORE_CACHE_KEY == cache_key and _SCORE_CACHE_FRAME is not None:
        return _SCORE_CACHE_FRAME.copy()

    base_score = 0.5 * sum(weight for _, weight in weight_items)
    reason_parts = [name for name, weight in sorted(weight_items, key=lambda item: abs(item[1]), reverse=True) if abs(weight) > 0.03]
    reason = ", ".join(reason_parts[:8])
    values_sql = ", ".join(["(%s::text, %s::numeric)"] * len(weight_items))
    params: list[Any] = []
    for name, weight in weight_items:
        params.extend([name, weight])
    params.extend([base_score, normalize_date(start_date), normalize_date(end_date)])

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            WITH weights(factor_name, weight) AS (
                VALUES {values_sql}
            )
            SELECT f.trade_date,
                   f.ts_code,
                   max(s.name) AS name,
                   max(s.industry) AS industry,
                   max(s.concepts) AS concepts,
                   max(s.close) AS close,
                   max(s.pct_chg) AS pct_chg,
                   max(s.amount_yi) AS amount_yi,
                   %s::numeric + sum((coalesce(f.factor_pct_rank, 0.5) - 0.5) * w.weight) AS factor_score
            FROM {qname(settings, 'factor_daily')} f
            JOIN weights w ON w.factor_name=f.factor_name
            JOIN {qname(settings, 'stock_signal_daily')} s
              ON s.trade_date=f.trade_date AND s.ts_code=f.ts_code
            WHERE f.trade_date BETWEEN %s AND %s
              AND f.is_valid=true
            GROUP BY f.trade_date, f.ts_code
            """,
            tuple(params),
        )
        rows = cur.fetchall()
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No factor_daily score rows for {start_date} to {end_date}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    for column in ["close", "pct_chg", "amount_yi", "factor_score"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["score_reason"] = reason
    _SCORE_CACHE_KEY = cache_key
    _SCORE_CACHE_FRAME = frame
    return frame


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


def _profit_loss_ratio(returns: pd.Series) -> float | None:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    wins = series[series > 0]
    losses = series[series < 0]
    if wins.empty or losses.empty:
        return None
    return _clean_float(wins.mean() / abs(losses.mean()))


def _split_labels(dates: list[Any], *, fixed_label: str | None = None) -> dict[str, set[Any]]:
    if not dates:
        return {"full": set(), "train": set(), "validation": set(), "test": set()}
    unique = sorted(dates)
    if fixed_label:
        return {fixed_label: set(unique)}
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
    capacity_pct: float = 0.005,
    portfolio_capital_yi: float = 0.1,
    weight_as_of_date: str | None = None,
    weight_train_start: str | None = None,
    weight_train_end: str | None = None,
    auto_optimize: bool = True,
    sample_split_label: str | None = None,
    model_run_suffix: str | None = None,
) -> dict[str, Any]:
    init_schema(settings)
    weight_date = normalize_date(weight_as_of_date or end_date)
    if weight_train_start is None or weight_train_end is None:
        split = _split_dates(start_date, end_date)
        weight_train_start = str(split["train_start"])
        weight_train_end = str(split["train_end"])
    weights = load_model_weights(
        settings,
        model_name,
        weight_date,
        horizon_days=hold_days,
        train_start=weight_train_start,
        train_end=weight_train_end,
        fallback=False,
    )
    if not weights:
        if weight_as_of_date:
            raise RuntimeError(f"No model weights for {model_name} as of {weight_date}")
        if auto_optimize:
            result = optimize_weights(settings, start_date, end_date, as_of_date=weight_date, horizon_days=hold_days)
            weight_train_start = str(result.get("train_start") or weight_train_start)
            weight_train_end = str(result.get("train_end") or weight_train_end)
            weights = load_model_weights(
                settings,
                model_name,
                weight_date,
                horizon_days=hold_days,
                train_start=weight_train_start,
                train_end=weight_train_end,
                fallback=False,
            )
    if not weights:
        raise RuntimeError(f"No model weights for {model_name} as of {weight_date}")

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
        ],
        on=["trade_date", "ts_code"],
        how="inner",
    )
    if frame.empty:
        raise RuntimeError(f"No scored trade rows for backtest {start_date} to {end_date}")

    round_trip_cost = 2.0 * (transaction_cost_bps + slippage_bps) / 10000.0
    frame = _add_execution_constraints(
        frame,
        top_n=top_n,
        capacity_pct=capacity_pct,
        portfolio_capital_yi=portfolio_capital_yi,
    )
    eligible = frame[
        frame["future_return"].notna()
        & frame["entry_price"].notna()
        & frame["exit_price"].notna()
        & (~frame["one_word_limit_up"].fillna(False))
        & (~frame["exit_blocked"].fillna(False))
        & frame["capacity_pass"].fillna(False)
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
    pre_filter = frame[
        frame["future_return"].notna()
        & frame["entry_price"].notna()
        & frame["exit_price"].notna()
        & (frame["amount_yi"].fillna(0) >= min_amount_yi)
    ]
    buyable_ratio = len(eligible) / len(pre_filter) if len(pre_filter) else None

    start_key = normalize_date(start_date).replace("-", "")
    end_key = normalize_date(end_date).replace("-", "")
    suffix = f"_{model_run_suffix}" if model_run_suffix else ""
    model_run_name = f"{model_name}_top{top_n}_hold{hold_days}_{start_key}_{end_key}{suffix}"
    splits = _split_labels(list(daily.index), fixed_label=sample_split_label)
    result_rows: list[dict[str, Any]] = []
    benchmark_total = _performance_from_daily_returns(benchmark.reindex(daily.index).fillna(0))["total_return"]
    for split_name, split_dates in splits.items():
        series = daily[daily.index.isin(split_dates)]["daily_return"]
        split_trades = selected[selected["trade_date"].isin(split_dates)]
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
                "profit_loss_ratio": _profit_loss_ratio(split_trades["net_return"]) if not split_trades.empty else None,
                "buyable_ratio": _clean_float(buyable_ratio),
                "avg_turnover": _clean_float(avg_turnover),
                "avg_holding_days": _clean_float(hold_days),
                "trade_count": int(split_trades["ts_code"].count()),
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
    full = next((row for row in result_rows if row["sample_split"] == "full"), result_rows[0])
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


def _open_dates_between(settings: Settings, start_date: str, end_date: str) -> list[str]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT cal_date::text AS d
            FROM {qname(settings, 'trade_calendar')}
            WHERE is_open=true AND cal_date BETWEEN %s AND %s
            ORDER BY cal_date
            """,
            (normalize_date(start_date), normalize_date(end_date)),
        )
        rows = cur.fetchall()
    return [str(row["d"]) for row in rows]


def run_backtest_grid(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    model_name: str = "event_adjusted_v1",
    top_ns: tuple[int, ...] = (10, 20, 30),
    hold_days_list: tuple[int, ...] = (1, 3, 5, 10),
    min_amount_yi: float = 2.0,
    capacity_pct: float = 0.005,
    portfolio_capital_yi: float = 0.1,
) -> dict[str, Any]:
    init_schema(settings)
    results: list[dict[str, Any]] = []
    for hold_days in hold_days_list:
        optimize_weights(settings, start_date, end_date, as_of_date=end_date, horizon_days=hold_days)
        for top_n in top_ns:
            try:
                results.append(
                    run_backtest(
                        settings,
                        start_date,
                        end_date,
                        model_name=model_name,
                        top_n=top_n,
                        hold_days=hold_days,
                        min_amount_yi=min_amount_yi,
                        capacity_pct=capacity_pct,
                        portfolio_capital_yi=portfolio_capital_yi,
                        weight_as_of_date=end_date,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "model_name": f"{model_name}_top{top_n}_hold{hold_days}",
                        "top_n": top_n,
                        "hold_days": hold_days,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    path = write_backtest_overview_report(settings, start_date, end_date)
    return {
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "grid_runs": len(results),
        "grid_success": len([item for item in results if "error" not in item]),
        "grid_errors": [item for item in results if "error" in item][:10],
        "report_file": str(path),
    }


def run_walk_forward_backtest(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    model_name: str = "walk_forward_v1",
    top_n: int = 20,
    hold_days: int = 5,
    train_days: int = 126,
    test_days: int = 63,
    min_amount_yi: float = 2.0,
    capacity_pct: float = 0.005,
    portfolio_capital_yi: float = 0.1,
) -> dict[str, Any]:
    init_schema(settings)
    dates = _open_dates_between(settings, start_date, end_date)
    if len(dates) < train_days + max(test_days, hold_days):
        raise RuntimeError(f"Not enough open dates for walk-forward: {len(dates)}")

    period_results: list[dict[str, Any]] = []
    index = 0
    period_no = 1
    while index + train_days + hold_days < len(dates):
        train_start = dates[index]
        train_end = dates[index + train_days - 1]
        test_start_idx = index + train_days
        test_end_idx = min(index + train_days + test_days - 1, len(dates) - 1 - hold_days)
        if test_start_idx > test_end_idx:
            break
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]
        optimize_weights(
            settings,
            train_start,
            train_end,
            as_of_date=train_end,
            horizon_days=hold_days,
            train_only=True,
        )
        run = run_backtest(
            settings,
            test_start,
            test_end,
            model_name=model_name,
            top_n=top_n,
            hold_days=hold_days,
            min_amount_yi=min_amount_yi,
            capacity_pct=capacity_pct,
            portfolio_capital_yi=portfolio_capital_yi,
            weight_as_of_date=train_end,
            weight_train_start=train_start,
            weight_train_end=train_end,
            auto_optimize=False,
            sample_split_label=f"wf_{period_no:02d}",
            model_run_suffix=f"asof{train_end.replace('-', '')}",
        )
        run.update(
            {
                "period_no": period_no,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        period_results.append(run)
        index += test_days
        period_no += 1

    if not period_results:
        raise RuntimeError("No walk-forward periods produced")

    total_return = float(np.prod([1.0 + float(item.get("total_return") or 0.0) for item in period_results]) - 1.0)
    benchmark_return = float(np.prod([1.0 + float(item.get("benchmark_return") or 0.0) for item in period_results]) - 1.0)
    max_drawdown = min(float(item.get("max_drawdown") or 0.0) for item in period_results)
    trade_count = 0
    for item in period_results:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT coalesce(sum(trade_count), 0) AS n
                FROM {qname(settings, 'strategy_backtest_result')}
                WHERE model_name=%s
                """,
                (str(item["model_name"]),),
            )
            trade_count += int(cur.fetchone()["n"] or 0)

    span_days = max((pd.to_datetime(normalize_date(end_date)) - pd.to_datetime(normalize_date(start_date))).days, 1)
    annual_return = (1.0 + total_return) ** (365.0 / span_days) - 1.0
    aggregate_name = f"{model_name}_walk_forward_top{top_n}_hold{hold_days}_{normalize_date(start_date).replace('-', '')}_{normalize_date(end_date).replace('-', '')}"
    aggregate_row = {
        "model_name": aggregate_name,
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "rebalance_rule": f"walk_forward_train{train_days}_test{test_days}",
        "hold_days": hold_days,
        "top_n": top_n,
        "sample_split": "walk_forward_full",
        "total_return": _clean_float(total_return),
        "annual_return": _clean_float(annual_return),
        "max_drawdown": _clean_float(max_drawdown),
        "sharpe": None,
        "win_rate": _clean_float(np.mean([float(item.get("total_return") or 0.0) > 0 for item in period_results])),
        "profit_loss_ratio": None,
        "avg_turnover": None,
        "avg_holding_days": _clean_float(hold_days),
        "trade_count": trade_count,
        "benchmark_return": _clean_float(benchmark_return),
        "excess_return": _clean_float(total_return - benchmark_return),
    }
    with connect() as conn:
        upsert_rows(
            conn,
            table=qname(settings, "strategy_backtest_result"),
            columns=list(aggregate_row.keys()),
            rows=[aggregate_row],
            conflict_columns=["model_name", "start_date", "end_date", "rebalance_rule", "hold_days", "top_n", "sample_split"],
        )
        conn.commit()
    path = write_backtest_overview_report(settings, start_date, end_date)
    return {
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "periods": len(period_results),
        "total_return": _clean_float(total_return),
        "benchmark_return": _clean_float(benchmark_return),
        "excess_return": _clean_float(total_return - benchmark_return),
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
        "Execution assumption: close signal, next-day open entry, fixed holding period, equal weight, board-aware one-word limit-up entry filter, one-word limit-down exit delay, liquidity/capacity filters, cost/slippage deducted.",
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
            "- Capacity uses the configured portfolio size and entry-day amount participation cap; it is still a research approximation, not an order book simulator.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_backtest_overview_report(settings: Settings, start_date: str, end_date: str) -> Path:
    path = report_dir(settings) / f"strategy_backtest_{normalize_date(end_date).replace('-', '')}.md"
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT model_name, sample_split, rebalance_rule, top_n, hold_days,
                   total_return, annual_return, max_drawdown, sharpe, win_rate,
                   profit_loss_ratio, benchmark_return, excess_return, trade_count
            FROM {qname(settings, 'strategy_backtest_result')}
            WHERE start_date=%s AND end_date=%s
            ORDER BY sample_split, hold_days, top_n, model_name
            """,
            (normalize_date(start_date), normalize_date(end_date)),
        )
        rows = cur.fetchall()
    data = pd.DataFrame(rows)
    lines = [
        f"# Strategy Backtest {normalize_date(start_date)} to {normalize_date(end_date)}",
        "",
        "研究模块输出，不替换生产评分。",
        "收益口径：trade_date 收盘后生成信号，下一交易日复权开盘买入，固定持有期后按复权收盘退出。",
        "执行过滤：按主板/创业板/科创板/北交所涨跌停幅度识别一字板，剔除一字涨停买不进；计划退出日一字跌停则顺延到首个可卖日；同时过滤停牌/缺价、成交额过低和容量不足标的，扣除交易成本和滑点。",
        "",
        "## Parameter Grid",
        "",
    ]
    if data.empty:
        lines.append("- No backtest rows.")
    else:
        for column in [
            "total_return",
            "annual_return",
            "max_drawdown",
            "sharpe",
            "win_rate",
            "profit_loss_ratio",
            "benchmark_return",
            "excess_return",
        ]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        display = data[data["sample_split"].isin(["full", "walk_forward_full"])].copy()
        if display.empty:
            display = data.copy()
        display = display.sort_values(["excess_return", "total_return"], ascending=False)
        for row in display.head(60).to_dict("records"):
            sharpe = "NA" if pd.isna(row["sharpe"]) else f"{float(row['sharpe']):.2f}"
            plr = "NA" if pd.isna(row["profit_loss_ratio"]) else f"{float(row['profit_loss_ratio']):.2f}"
            lines.append(
                f"- {row['model_name']} / {row['sample_split']}: top={row['top_n']}, hold={row['hold_days']}, "
                f"total={_format_pct(row['total_return'])}, excess={_format_pct(row['excess_return'])}, "
                f"mdd={_format_pct(row['max_drawdown'])}, sharpe={sharpe}, win={_format_pct(row['win_rate'])}, "
                f"PL={plr}, trades={row['trade_count']}"
            )
        lines.extend(["", "## Sample Splits", ""])
        for split_name, group in data.groupby("sample_split"):
            best = group.sort_values("excess_return", ascending=False).head(5)
            detail = "; ".join(
                f"top{r.top_n}/hold{r.hold_days}: excess={_format_pct(r.excess_return)}"
                for r in best.itertuples()
            )
            lines.append(f"- {split_name}: {detail}")
    lines.extend(
        [
            "",
            "## Reliability Notes",
            "",
            "- 常规全窗口回测使用训练/验证/测试切分，权重只来自训练段。",
            "- walk-forward 回测每期只用过去训练窗口估权，再应用到未来窗口。",
            "- 容量约束使用配置的组合资金规模和入场日成交额参与上限，仍然是研究近似，不是订单簿级撮合模拟。",
            "- 当前仍是研究级 daily cohort 近似，不是生产交易执行系统。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
