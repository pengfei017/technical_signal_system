from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from tech_signal.config import Settings
from tech_signal.db import connect, init_schema, qname, upsert_rows

from .factor_definitions import (
    FACTOR_BY_NAME,
    FACTOR_DEFINITIONS,
    add_cross_sectional_ranks,
    compute_factor_values,
    factor_group,
    higher_is_better,
    load_factor_base,
    normalize_date,
    oriented_factor_value,
    selected_factor_names,
    target_slice,
)


HORIZONS = [1, 3, 5, 10]
REGIMES = ["all", "bull", "bear", "neutral", "high_turnover", "low_turnover"]


def report_dir(settings: Settings) -> Path:
    path = settings.output_root / "factor_lab" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _drawdown_stats(returns: pd.Series) -> tuple[float | None, float | None]:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    if series.empty:
        return None, None
    equity = (1.0 + series).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return _clean_float(drawdown.mean()), _clean_float(drawdown.min())


def _safe_corr(left: pd.Series, right: pd.Series, *, rank: bool = False) -> float | None:
    frame = pd.DataFrame({"x": left, "y": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 20 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    if rank:
        frame = frame.rank(method="average")
    value = frame["x"].corr(frame["y"])
    return _clean_float(value)


def _numeric_series(value: Any) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        series = value.stack()
    elif isinstance(value, pd.Series):
        series = value
    else:
        series = pd.Series(value)
    return pd.to_numeric(series, errors="coerce")


def build_factor_daily(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    factors: Iterable[str] | None = None,
) -> dict[str, Any]:
    init_schema(settings)
    factor_names = selected_factor_names(factors)
    base = load_factor_base(settings, start_date, end_date)
    if base.empty:
        raise RuntimeError(f"No base data for factor build: {start_date} to {end_date}")
    factor_frame = compute_factor_values(base)
    factor_frame = target_slice(factor_frame, start_date, end_date)
    factor_frame = add_cross_sectional_ranks(factor_frame, factor_names)
    if factor_frame.empty:
        raise RuntimeError(f"No target factor rows for {start_date} to {end_date}")

    start_value = normalize_date(start_date)
    end_value = normalize_date(end_date)
    columns = [
        "trade_date",
        "ts_code",
        "factor_name",
        "factor_group",
        "factor_value",
        "factor_rank",
        "factor_pct_rank",
        "is_valid",
    ]
    inserted = 0
    valid_counts: dict[str, int] = {}
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {qname(settings, 'factor_daily')} WHERE trade_date BETWEEN %s AND %s AND factor_name = ANY(%s)",
                (start_value, end_value, factor_names),
            )
            copy_sql = f"COPY {qname(settings, 'factor_daily')} ({', '.join(columns)}) FROM STDIN"
            with cur.copy(copy_sql) as copy:
                for name in factor_names:
                    if name not in factor_frame.columns:
                        continue
                    sub = factor_frame[["trade_date", "ts_code", name, f"{name}__rank", f"{name}__pct_rank"]].copy()
                    sub = sub.replace([np.inf, -np.inf], np.nan)
                    sub = sub[sub[name].notna()]
                    valid_counts[name] = int(len(sub))
                    group = factor_group(name)
                    for row in sub.itertuples(index=False):
                        copy.write_row(
                            (
                                row.trade_date,
                                row.ts_code,
                                name,
                                group,
                                _clean_float(getattr(row, name)),
                                _clean_float(getattr(row, f"{name}__rank")),
                                _clean_float(getattr(row, f"{name}__pct_rank")),
                                True,
                            )
                        )
                    inserted += int(len(sub))
        conn.commit()

    return {
        "start_date": start_value,
        "end_date": end_value,
        "factor_count": len(factor_names),
        "stock_day_rows": int(len(factor_frame)),
        "factor_daily_rows": inserted,
        "valid_count_min": min(valid_counts.values()) if valid_counts else 0,
        "valid_count_max": max(valid_counts.values()) if valid_counts else 0,
    }


def _load_forward_returns(settings: Settings, start_date: str, end_date: str, horizons: Iterable[int]) -> pd.DataFrame:
    start_value = normalize_date(start_date)
    end_value = normalize_date(end_date)
    max_horizon = max(horizons)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT trade_date, ts_code, adj_open, adj_close, pct_chg, amount
            FROM {qname(settings, 'daily_bars')}
            WHERE trade_date BETWEEN %s::date - interval '5 days'
                                  AND %s::date + (%s || ' days')::interval
              AND adj_close IS NOT NULL
            ORDER BY ts_code, trade_date
            """,
            (start_value, end_value, max_horizon * 3 + 10),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    for column in ["adj_open", "adj_close", "pct_chg", "amount"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values(["ts_code", "trade_date"])
    grouped = df.groupby("ts_code", group_keys=False)
    df["entry_price"] = grouped["adj_open"].shift(-1)
    for horizon in horizons:
        exit_price = grouped["adj_close"].shift(-horizon)
        df[f"fwd_return_{horizon}"] = exit_price / df["entry_price"] - 1.0
    start_dt = pd.to_datetime(start_value).date()
    end_dt = pd.to_datetime(end_value).date()
    return df[(df["trade_date"] >= start_dt) & (df["trade_date"] <= end_dt)].copy()


def _market_regime_dates(returns: pd.DataFrame) -> dict[str, set[Any]]:
    by_date = returns.groupby("trade_date").agg(
        market_pct=("pct_chg", "mean"),
        amount=("amount", "sum"),
    )
    if by_date.empty:
        return {name: set() for name in REGIMES}
    median_amount = by_date["amount"].median()
    regimes = {
        "all": set(by_date.index),
        "bull": set(by_date[by_date["market_pct"] > 0.3].index),
        "bear": set(by_date[by_date["market_pct"] < -0.3].index),
        "neutral": set(by_date[by_date["market_pct"].between(-0.3, 0.3)].index),
        "high_turnover": set(by_date[by_date["amount"] >= median_amount].index),
        "low_turnover": set(by_date[by_date["amount"] < median_amount].index),
    }
    return regimes


def _factor_rows(settings: Settings, factor_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT trade_date, ts_code, factor_value, factor_pct_rank
            FROM {qname(settings, 'factor_daily')}
            WHERE factor_name=%s
              AND trade_date BETWEEN %s AND %s
              AND is_valid=true
            """,
            (factor_name, normalize_date(start_date), normalize_date(end_date)),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["factor_value"] = pd.to_numeric(df["factor_value"], errors="coerce")
    df["factor_raw_value"] = df["factor_value"]
    df["factor_value"] = oriented_factor_value(factor_name, df["factor_value"])
    df["factor_pct_rank"] = pd.to_numeric(df["factor_pct_rank"], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def _evaluate_one(frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
    ret_col = f"fwd_return_{horizon}"
    data = frame[["trade_date", "factor_value", "factor_pct_rank", ret_col]].dropna()
    if data.empty:
        return {
            "sample_count": 0,
            "ic_mean": None,
            "rank_ic_mean": None,
            "ic_ir": None,
            "top_quantile_return": None,
            "bottom_quantile_return": None,
            "long_short_return": None,
            "quantile_returns": {},
            "win_rate": None,
            "avg_return": None,
            "avg_drawdown": None,
            "max_drawdown": None,
        }
    daily_ic = _numeric_series(
        data.groupby("trade_date").apply(lambda g: _safe_corr(g["factor_value"], g[ret_col]), include_groups=False)
    )
    daily_rank_ic = _numeric_series(
        data.groupby("trade_date").apply(
            lambda g: _safe_corr(g["factor_value"], g[ret_col], rank=True),
            include_groups=False,
        )
    )
    top = data[data["factor_pct_rank"] >= 0.8].groupby("trade_date")[ret_col].mean()
    bottom = data[data["factor_pct_rank"] <= 0.2].groupby("trade_date")[ret_col].mean()
    long_short = (top - bottom).dropna()
    quantile_returns: dict[str, float | None] = {}
    quantiles = pd.cut(
        data["factor_pct_rank"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.000001],
        labels=["q1", "q2", "q3", "q4", "q5"],
        include_lowest=True,
    )
    grouped_quantiles = data.assign(quantile=quantiles).groupby("quantile", observed=True)[ret_col].mean()
    for label in ["q1", "q2", "q3", "q4", "q5"]:
        quantile_returns[label] = _clean_float(grouped_quantiles.get(label))
    avg_drawdown, max_drawdown = _drawdown_stats(long_short)
    clean_ic = daily_ic.dropna()
    ic_std = _clean_float(clean_ic.std())
    ic_ir = None
    if ic_std is not None and not math.isclose(ic_std, 0.0):
        ic_ir = float(clean_ic.mean() / ic_std * math.sqrt(max(len(clean_ic), 1)))
    return {
        "sample_count": int(len(data)),
        "ic_mean": _clean_float(daily_ic.mean()),
        "rank_ic_mean": _clean_float(daily_rank_ic.mean()),
        "ic_ir": _clean_float(ic_ir),
        "top_quantile_return": _clean_float(top.mean()),
        "bottom_quantile_return": _clean_float(bottom.mean()),
        "long_short_return": _clean_float(long_short.mean()),
        "quantile_returns": quantile_returns,
        "win_rate": _clean_float((top > 0).mean()) if len(top) else None,
        "avg_return": _clean_float(top.mean()),
        "avg_drawdown": avg_drawdown,
        "max_drawdown": max_drawdown,
    }


def evaluate_factors(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    factors: Iterable[str] | None = None,
    horizons: Iterable[int] = HORIZONS,
) -> dict[str, Any]:
    init_schema(settings)
    factor_names = selected_factor_names(factors)
    horizon_list = [int(item) for item in horizons]
    returns = _load_forward_returns(settings, start_date, end_date, horizon_list)
    if returns.empty:
        raise RuntimeError(f"No return data for factor evaluation: {start_date} to {end_date}")
    regimes = _market_regime_dates(returns)
    rows: list[dict[str, Any]] = []
    decay_by_factor: dict[str, int | None] = {}

    for factor_name in factor_names:
        factor_data = _factor_rows(settings, factor_name, start_date, end_date)
        if factor_data.empty:
            continue
        frame = factor_data.merge(returns, on=["trade_date", "ts_code"], how="inner")
        all_horizon_stats: dict[int, dict[str, Any]] = {}
        for horizon in horizon_list:
            all_horizon_stats[horizon] = _evaluate_one(frame, horizon)
        base = abs(float(all_horizon_stats.get(1, {}).get("rank_ic_mean") or 0.0))
        decay = max(horizon_list) if base > 0 else None
        if base > 0:
            for horizon in sorted(horizon_list):
                if horizon == 1:
                    continue
                current = abs(float(all_horizon_stats.get(horizon, {}).get("rank_ic_mean") or 0.0))
                if current < base * 0.5:
                    decay = horizon
                    break
        decay_by_factor[factor_name] = decay
        for horizon in horizon_list:
            for regime, dates in regimes.items():
                subset = frame if regime == "all" else frame[frame["trade_date"].isin(dates)]
                stats = _evaluate_one(subset, horizon)
                rows.append(
                    {
                        "factor_name": factor_name,
                        "factor_group": factor_group(factor_name),
                        "start_date": normalize_date(start_date),
                        "end_date": normalize_date(end_date),
                        "horizon_days": horizon,
                        "sample_count": stats["sample_count"],
                        "ic_mean": stats["ic_mean"],
                        "rank_ic_mean": stats["rank_ic_mean"],
                        "ic_ir": stats["ic_ir"],
                        "top_quantile_return": stats["top_quantile_return"],
                        "bottom_quantile_return": stats["bottom_quantile_return"],
                        "long_short_return": stats["long_short_return"],
                        "quantile_returns": Jsonb(stats["quantile_returns"]),
                        "win_rate": stats["win_rate"],
                        "avg_return": stats["avg_return"],
                        "avg_drawdown": stats["avg_drawdown"],
                        "max_drawdown": stats["max_drawdown"],
                        "decay_days": decay_by_factor[factor_name],
                        "market_regime": regime,
                    }
                )

    columns = [
        "factor_name",
        "factor_group",
        "start_date",
        "end_date",
        "horizon_days",
        "sample_count",
        "ic_mean",
        "rank_ic_mean",
        "ic_ir",
        "top_quantile_return",
        "bottom_quantile_return",
        "long_short_return",
        "quantile_returns",
        "win_rate",
        "avg_return",
        "avg_drawdown",
        "max_drawdown",
        "decay_days",
        "market_regime",
    ]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "factor_performance"),
            columns=columns,
            rows=rows,
            conflict_columns=["factor_name", "start_date", "end_date", "horizon_days", "market_regime"],
        )
        conn.commit()
    write_factor_performance_report(settings, start_date, end_date, rows)
    return {
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "factor_count": len({row["factor_name"] for row in rows}),
        "performance_rows": count,
        "horizons": horizon_list,
        "regimes": REGIMES,
    }


def correlate_factors(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    factors: Iterable[str] | None = None,
) -> dict[str, Any]:
    init_schema(settings)
    factor_names = selected_factor_names(factors)
    base = load_factor_base(settings, start_date, end_date, lookback_calendar_days=90, forward_calendar_days=0)
    frame = target_slice(compute_factor_values(base), start_date, end_date)
    if frame.empty:
        raise RuntimeError(f"No factor base rows for correlation: {start_date} to {end_date}")
    available = [name for name in factor_names if name in frame.columns]
    daily_corrs: dict[tuple[str, str], list[float]] = {}
    for _, group in frame.groupby("trade_date"):
        if len(group) < 50:
            continue
        corr = group[available].corr(min_periods=50)
        for i, left in enumerate(available):
            for right in available[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    daily_corrs.setdefault((left, right), []).append(float(value))
    rows = []
    for (left, right), values in daily_corrs.items():
        rows.append(
            {
                "start_date": normalize_date(start_date),
                "end_date": normalize_date(end_date),
                "factor_a": left,
                "factor_b": right,
                "correlation": _clean_float(np.mean(values)),
                "sample_count": len(values),
            }
        )
    columns = ["start_date", "end_date", "factor_a", "factor_b", "correlation", "sample_count"]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "factor_correlation"),
            columns=columns,
            rows=rows,
            conflict_columns=["start_date", "end_date", "factor_a", "factor_b"],
        )
        conn.commit()
    write_factor_correlation_report(settings, start_date, end_date, rows)
    high_corr = [row for row in rows if row["correlation"] is not None and abs(float(row["correlation"])) >= 0.75]
    return {
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "factor_count": len(available),
        "correlation_rows": count,
        "high_correlation_pairs": len(high_corr),
    }


def _format_pct(value: Any) -> str:
    number = _clean_float(value)
    return "NA" if number is None else f"{number * 100:.2f}%"


def write_factor_performance_report(settings: Settings, start_date: str, end_date: str, rows: list[dict[str, Any]]) -> Path:
    path = report_dir(settings) / f"factor_performance_{normalize_date(end_date).replace('-', '')}.md"
    data = pd.DataFrame(rows)
    lines = [
        f"# Factor Performance {normalize_date(start_date)} to {normalize_date(end_date)}",
        "",
        "研究模块输出，不替换生产评分。",
        "收益口径：trade_date 收盘后生成信号，下一交易日复权开盘买入，持有 N 个交易日，按复权收盘退出。",
        "风险因子保留原始 factor_value，但评估和 pct_rank 使用正向口径：低风险为高分。",
        "",
    ]
    if data.empty:
        lines.append("No factor performance rows.")
    else:
        subset = data[(data["market_regime"] == "all") & (data["horizon_days"] == 5)].copy()
        subset["abs_rank_ic"] = subset["rank_ic_mean"].abs()
        best = subset.sort_values(["rank_ic_mean", "long_short_return"], ascending=False).head(12)
        weak = subset[subset["abs_rank_ic"].fillna(0) < 0.005].sort_values("sample_count", ascending=False).head(12)
        lines.extend(["## Most Effective Factors", ""])
        for row in best.to_dict("records"):
            lines.append(
                f"- {row['factor_name']} ({row['factor_group']}): RankIC={_format_pct(row['rank_ic_mean'])}, "
                f"LS={_format_pct(row['long_short_return'])}, win={_format_pct(row['win_rate'])}, decay={row.get('decay_days')}"
            )
        lines.extend(["", "## Likely Noise / Weak Factors", ""])
        for row in weak.to_dict("records"):
            lines.append(
                f"- {row['factor_name']} ({row['factor_group']}): RankIC={_format_pct(row['rank_ic_mean'])}, "
                f"LS={_format_pct(row['long_short_return'])}, samples={int(row['sample_count'])}"
            )
        lines.extend(["", "## Regime Notes", ""])
        for regime in [item for item in REGIMES if item != "all"]:
            regime_df = data[(data["market_regime"] == regime) & (data["horizon_days"] == 5)].copy()
            if regime_df.empty:
                continue
            top = regime_df.sort_values("rank_ic_mean", ascending=False).head(3)
            names = ", ".join(f"{r.factor_name}({_format_pct(r.rank_ic_mean)})" for r in top.itertuples())
            lines.append(f"- {regime}: {names}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_factor_correlation_report(settings: Settings, start_date: str, end_date: str, rows: list[dict[str, Any]]) -> Path:
    path = report_dir(settings) / f"factor_correlation_{normalize_date(end_date).replace('-', '')}.md"
    data = pd.DataFrame(rows)
    lines = [
        f"# Factor Correlation {normalize_date(start_date)} to {normalize_date(end_date)}",
        "",
        "每日横截面相关性的时间窗口平均值。绝对相关高于 0.75 的因子后续权重需要降权。",
        "",
    ]
    if data.empty:
        lines.append("No factor correlation rows.")
    else:
        data["abs_corr"] = data["correlation"].abs()
        high = data[data["abs_corr"] >= 0.75].sort_values("abs_corr", ascending=False)
        lines.extend(["## High-Correlation Pairs", ""])
        if high.empty:
            lines.append("- No pair above 0.75.")
        else:
            for row in high.head(50).to_dict("records"):
                lines.append(f"- {row['factor_a']} / {row['factor_b']}: corr={row['correlation']:.3f}, days={row['sample_count']}")
        lines.extend(["", "## Factor Definitions", ""])
        for item in FACTOR_DEFINITIONS:
            lines.append(f"- {item.name} ({item.group}): {item.description}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
