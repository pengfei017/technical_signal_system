#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tech_signal.config import load_settings
from tech_signal.db import connect, init_schema, qname

from factor_lab.backtest_engine import run_backtest, run_backtest_grid, run_walk_forward_backtest
from factor_lab.data_coverage import build_data_coverage
from factor_lab.event_study import run_event_study
from factor_lab.factor_definitions import FACTOR_DEFINITIONS, normalize_date
from factor_lab.factor_evaluator import build_factor_daily, correlate_factors, evaluate_factors, report_dir
from factor_lab.shadow_runner import run_shadow_pipeline
from factor_lab.short_strength import run_short_strength_backtest
from factor_lab.trend_pure import evaluate_trend_indicators, run_trend_pure_backtest, run_trend_pure_period_study
from factor_lab.weight_optimizer import optimize_weights


def _latest_daily_date(settings) -> str:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT max(trade_date)::text AS d FROM {qname(settings, 'daily_bars')}")
        row = cur.fetchone()
    if not row or not row["d"]:
        raise RuntimeError("No daily_bars date available")
    return str(row["d"])


def _latest_factor_range(settings, end_date: str | None = None) -> tuple[str, str]:
    params: list[Any] = []
    where = ""
    if end_date:
        where = "WHERE end_date=%s"
        params.append(normalize_date(end_date))
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT start_date::text AS start_date, end_date::text AS end_date, count(*) AS n
            FROM {qname(settings, 'factor_performance')}
            {where}
            GROUP BY start_date, end_date
            ORDER BY end_date DESC, n DESC
            LIMIT 1
            """,
            tuple(params),
        )
        row = cur.fetchone()
    if not row:
        raise RuntimeError("No factor_performance range available; run evaluate first")
    return str(row["start_date"]), str(row["end_date"])


def _default_start_date(settings, end_date: str, *, calendar_days: int = 365) -> str:
    target = pd.to_datetime(normalize_date(end_date)) - pd.Timedelta(days=calendar_days)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT min(trade_date)::text AS d
            FROM {qname(settings, 'stock_signal_daily')}
            WHERE trade_date >= %s
            """,
            (target.date().isoformat(),),
        )
        row = cur.fetchone()
    return str(row["d"]) if row and row["d"] else target.date().isoformat()


def _query_df(settings, sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def _factor_rows_available(settings, start_date: str, end_date: str, factor_names: list[str] | None = None) -> bool:
    params: list[Any] = [normalize_date(start_date), normalize_date(end_date)]
    factor_filter = ""
    if factor_names:
        factor_filter = "AND factor_name = ANY(%s)"
        params.append(factor_names)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT count(DISTINCT factor_name) AS factor_count, count(*) AS row_count
            FROM {qname(settings, 'factor_daily')}
            WHERE trade_date BETWEEN %s AND %s
            {factor_filter}
            """,
            tuple(params),
        )
        row = cur.fetchone()
    expected = len(factor_names or FACTOR_DEFINITIONS)
    return int(row["factor_count"] or 0) >= min(expected, 10) and int(row["row_count"] or 0) > 0


def _row_count(settings, sql: str, params: tuple[Any, ...]) -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return int(row["n"] or 0) if row else 0


def _performance_available(settings, start_date: str, end_date: str) -> bool:
    return _row_count(
        settings,
        f"""
        SELECT count(*) AS n
        FROM {qname(settings, 'factor_performance')}
        WHERE start_date=%s AND end_date=%s
        """,
        (normalize_date(start_date), normalize_date(end_date)),
    ) >= 100


def _correlation_available(settings, start_date: str, end_date: str) -> bool:
    return _row_count(
        settings,
        f"""
        SELECT count(*) AS n
        FROM {qname(settings, 'factor_correlation')}
        WHERE start_date=%s AND end_date=%s
        """,
        (normalize_date(start_date), normalize_date(end_date)),
    ) > 0


def _event_study_available(settings, start_date: str, end_date: str) -> bool:
    return _row_count(
        settings,
        f"""
        SELECT count(*) AS n
        FROM {qname(settings, 'factor_event_study')}
        WHERE start_date=%s AND end_date=%s
        """,
        (normalize_date(start_date), normalize_date(end_date)),
    ) > 0


def _weights_available(settings, as_of_date: str, horizon_days: int = 5) -> bool:
    return _row_count(
        settings,
        f"""
        SELECT count(DISTINCT model_name) AS n
        FROM {qname(settings, 'model_weight_history')}
        WHERE as_of_date=%s AND horizon_days=%s
        """,
        (normalize_date(as_of_date), int(horizon_days)),
    ) >= 5


def _grid_backtest_available(settings, start_date: str, end_date: str) -> bool:
    return _row_count(
        settings,
        f"""
        SELECT count(DISTINCT model_name) AS n
        FROM {qname(settings, 'strategy_backtest_result')}
        WHERE start_date=%s AND end_date=%s
          AND model_name LIKE 'event_adjusted_v1_top%%'
        """,
        (normalize_date(start_date), normalize_date(end_date)),
    ) >= 12


def _walk_forward_available(settings, start_date: str, end_date: str) -> bool:
    return _row_count(
        settings,
        f"""
        SELECT count(*) AS n
        FROM {qname(settings, 'strategy_backtest_result')}
        WHERE start_date=%s AND end_date=%s
          AND sample_split='walk_forward_full'
        """,
        (normalize_date(start_date), normalize_date(end_date)),
    ) > 0


def ensure_research_pipeline(
    settings,
    start_date: str,
    end_date: str,
    *,
    factor_names: list[str] | None = None,
    top_n: int = 20,
    hold_days: int = 5,
    min_amount_yi: float = 2.0,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics["coverage"] = build_data_coverage(settings, as_of_date=end_date)
    if not _factor_rows_available(settings, start_date, end_date, factor_names):
        metrics["build_factors"] = build_factor_daily(settings, start_date, end_date, factors=factor_names)
    else:
        metrics["build_factors"] = {"status": "reused"}
    if _performance_available(settings, start_date, end_date):
        metrics["evaluate"] = {"status": "reused"}
    else:
        metrics["evaluate"] = evaluate_factors(settings, start_date, end_date, factors=factor_names)
    if _correlation_available(settings, start_date, end_date):
        metrics["correlate"] = {"status": "reused"}
    else:
        metrics["correlate"] = correlate_factors(settings, start_date, end_date, factors=factor_names)
    if _event_study_available(settings, start_date, end_date):
        metrics["event_study"] = {"status": "reused"}
    else:
        metrics["event_study"] = run_event_study(settings, start_date, end_date)
    if _weights_available(settings, end_date, hold_days):
        metrics["weights"] = {"status": "reused"}
    else:
        metrics["weights"] = optimize_weights(settings, start_date, end_date, as_of_date=end_date, horizon_days=hold_days)
    if _grid_backtest_available(settings, start_date, end_date):
        metrics["backtest_grid"] = {"status": "reused"}
    else:
        metrics["backtest_grid"] = run_backtest_grid(
            settings,
            start_date,
            end_date,
            model_name="event_adjusted_v1",
            min_amount_yi=min_amount_yi,
        )
    if _walk_forward_available(settings, start_date, end_date):
        metrics["walk_forward"] = {"status": "reused"}
    else:
        try:
            metrics["walk_forward"] = run_walk_forward_backtest(
                settings,
                start_date,
                end_date,
                top_n=top_n,
                hold_days=hold_days,
                min_amount_yi=min_amount_yi,
            )
        except Exception as exc:
            metrics["walk_forward_error"] = f"{type(exc).__name__}: {exc}"
    return metrics


def write_summary_report(settings, report_date: str, start_date: str, end_date: str) -> Path:
    report_value = normalize_date(report_date)
    start_value = normalize_date(start_date)
    end_value = normalize_date(end_date)
    perf = _query_df(
        settings,
        f"""
        SELECT factor_name, factor_group, horizon_days, market_regime,
               rank_ic_mean, long_short_return, win_rate, max_drawdown, sample_count
        FROM {qname(settings, 'factor_performance')}
        WHERE start_date=%s AND end_date=%s AND horizon_days=5 AND market_regime='all'
        """,
        (start_value, end_value),
    )
    corr = _query_df(
        settings,
        f"""
        SELECT factor_a, factor_b, correlation, sample_count
        FROM {qname(settings, 'factor_correlation')}
        WHERE start_date=%s AND end_date=%s
        ORDER BY abs(correlation) DESC
        LIMIT 30
        """,
        (start_value, end_value),
    )
    backtest = _query_df(
        settings,
        f"""
        SELECT model_name, sample_split, total_return, annual_return, max_drawdown,
               sharpe, win_rate, profit_loss_ratio, benchmark_return, excess_return, trade_count
        FROM {qname(settings, 'strategy_backtest_result')}
        WHERE start_date=%s AND end_date=%s
        ORDER BY created_at DESC, model_name, sample_split
        """,
        (start_value, end_value),
    )
    weights = _query_df(
        settings,
        f"""
        SELECT model_name, factor_name, factor_group, weight, reason
        FROM {qname(settings, 'model_weight_history')}
        WHERE as_of_date=%s AND horizon_days=5
        """,
        (end_value,),
    )
    coverage = _query_df(
        settings,
        f"""
        SELECT table_name, min_date, max_date, row_count, stable_start,
               missing_or_low_dates, first_missing, last_missing
        FROM {qname(settings, 'factor_data_coverage')}
        WHERE as_of_date=%s
        ORDER BY table_name
        """,
        (end_value,),
    )
    events = _query_df(
        settings,
        f"""
        SELECT event_name, event_group, sample_count, avg_return, avg_excess_return, win_rate
        FROM {qname(settings, 'factor_event_study')}
        WHERE start_date=%s AND end_date=%s AND horizon_days=5 AND market_regime='all'
        ORDER BY avg_excess_return DESC NULLS LAST
        """,
        (start_value, end_value),
    )
    path = report_dir(settings) / f"factor_lab_summary_{report_value.replace('-', '')}.md"

    def fmt(value: Any) -> str:
        try:
            number = float(value)
        except Exception:
            return "NA"
        return f"{number * 100:.2f}%"

    lines = [
        f"# Factor Lab Summary {report_value}",
        "",
        f"Window: {start_value} to {end_value}",
        "",
        "研究模块输出，不修改生产 `stock_signal_daily` 评分权重。",
        "",
        "## 0. 数据覆盖和口径",
        "",
        "- 收益口径统一为：收盘后出信号，下一交易日复权开盘买入，持有 N 个交易日，按复权收盘退出。",
        "- 龙虎榜、资金流、涨跌停按盘后已知事件处理，只用于次日交易研究。",
        "- 风险因子统一为正向百分位：低风险 pct_rank 更高；建议权重只写研究表和报告。",
        "",
    ]
    if coverage.empty:
        lines.append("- No coverage rows.")
    else:
        coverage["missing_or_low_dates"] = pd.to_numeric(coverage["missing_or_low_dates"], errors="coerce").fillna(0)
        weak = coverage[coverage["missing_or_low_dates"] > 0]
        if weak.empty:
            lines.append("- 2020年以来核心表覆盖未发现低于阈值的交易日。")
        else:
            for row in weak.head(12).to_dict("records"):
                lines.append(
                    f"- {row['table_name']}: min={row['min_date']}, max={row['max_date']}, "
                    f"stable_start={row['stable_start']}, missing/low={int(row['missing_or_low_dates'])}, "
                    f"first={row['first_missing']}, last={row['last_missing']}"
                )

    lines.extend(["", "## 1. 哪些因子稳定有效", ""])
    if perf.empty:
        lines.append("- No performance rows.")
    else:
        for column in ["rank_ic_mean", "long_short_return", "win_rate", "max_drawdown"]:
            perf[column] = pd.to_numeric(perf[column], errors="coerce")
        best = perf.sort_values(["rank_ic_mean", "long_short_return"], ascending=False).head(12)
        for row in best.to_dict("records"):
            lines.append(
                f"- {row['factor_name']} ({row['factor_group']}): RankIC={fmt(row['rank_ic_mean'])}, "
                f"LS={fmt(row['long_short_return'])}, win={fmt(row['win_rate'])}"
            )

    lines.extend(["", "## 2. 哪些因子像噪音", ""])
    if not perf.empty:
        weak = perf[perf["rank_ic_mean"].abs().fillna(0) < 0.005].sort_values("sample_count", ascending=False).head(12)
        if weak.empty:
            lines.append("- 当前窗口没有明显接近零 RankIC 的因子。")
        else:
            for row in weak.to_dict("records"):
                lines.append(f"- {row['factor_name']} ({row['factor_group']}): RankIC={fmt(row['rank_ic_mean'])}, LS={fmt(row['long_short_return'])}")

    lines.extend(["", "## 3. 哪些因子高度重复", ""])
    if corr.empty:
        lines.append("- No correlation rows.")
    else:
        corr["correlation"] = pd.to_numeric(corr["correlation"], errors="coerce")
        high = corr[corr["correlation"].abs() >= 0.75]
        if high.empty:
            lines.append("- 没有绝对相关高于 0.75 的因子对。")
        else:
            for row in high.head(20).to_dict("records"):
                lines.append(f"- {row['factor_a']} / {row['factor_b']}: corr={float(row['correlation']):.3f}")

    lines.extend(["", "## 4. 哪些因子只在特定环境有效", ""])
    regime = _query_df(
        settings,
        f"""
        SELECT factor_name, factor_group, market_regime, rank_ic_mean, long_short_return
        FROM {qname(settings, 'factor_performance')}
        WHERE start_date=%s AND end_date=%s AND horizon_days=5 AND market_regime <> 'all'
        """,
        (start_value, end_value),
    )
    if regime.empty:
        lines.append("- No regime rows.")
    else:
        regime["rank_ic_mean"] = pd.to_numeric(regime["rank_ic_mean"], errors="coerce")
        for market_regime, group in regime.groupby("market_regime"):
            top = group.sort_values("rank_ic_mean", ascending=False).head(3)
            detail = ", ".join(f"{row.factor_name}({fmt(row.rank_ic_mean)})" for row in top.itertuples())
            lines.append(f"- {market_regime}: {detail}")

    lines.extend(["", "## 5. 事件型因子", ""])
    if events.empty:
        lines.append("- No event-study rows.")
    else:
        for row in events.head(15).to_dict("records"):
            lines.append(
                f"- {row['event_name']} ({row['event_group']}): samples={row['sample_count']}, "
                f"avg={fmt(row['avg_return'])}, excess={fmt(row['avg_excess_return'])}, win={fmt(row['win_rate'])}"
            )

    lines.extend(["", "## 6. 当前生产评分权重是否有明显问题", ""])
    lines.append("- 生产评分没有被本模块改动。")
    lines.append("- 如果趋势/量能高相关因子长期重复，人工评分可能存在重复加权；需要先用影子权重连续观察。")
    lines.append("- 样本较短的龙虎榜、资金流、涨跌停因子暂不应直接进入生产权重。")

    lines.extend(["", "## 7. 建议进入影子运行的权重方案", ""])
    if weights.empty:
        lines.append("- No weight suggestion rows.")
    else:
        weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
        for model_name, group in weights.groupby("model_name"):
            top = group.assign(abs_weight=group["weight"].abs()).sort_values("abs_weight", ascending=False).head(6)
            detail = ", ".join(f"{row.factor_name}:{row.weight:.3f}" for row in top.itertuples())
            lines.append(f"- {model_name}: {detail}")
        lines.append("- 当前默认影子主线已切到 trend_pure_v1；event_adjusted_v1、walk_forward_v1 和 short_strength_v1 仅保留为历史研究命令。")

    lines.extend(["", "## 8. 暂时不能进入生产的原因", ""])
    lines.append("- 这是研究校准输出，不是生产交易执行引擎。")
    lines.append("- 需要至少 10-20 个交易日影子运行，观察样本外、滚动窗口和不同市场环境的稳定性。")
    lines.append("- 若历史接口无法补齐更早资金流、龙虎榜或涨跌停，事件因子的长期可靠性要打折。")

    lines.extend(["", "## Backtest Snapshot", ""])
    if backtest.empty:
        lines.append("- No backtest result rows.")
    else:
        for row in backtest.head(20).to_dict("records"):
            plr = row.get("profit_loss_ratio")
            lines.append(
                f"- {row['model_name']} / {row['sample_split']}: total={fmt(row['total_return'])}, "
                f"excess={fmt(row['excess_return'])}, mdd={fmt(row['max_drawdown'])}, "
                f"PL={plr if plr is not None else 'NA'}, trades={row['trade_count']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only factor lab for technical_signal_system")
    parser.add_argument(
        "command",
        choices=[
            "init-schema",
            "coverage",
            "build-factors",
            "evaluate",
            "correlate",
            "event-study",
            "weights",
            "backtest",
            "backtest-grid",
            "walk-forward",
            "short-strength-backtest",
            "trend-pure-evaluate",
            "trend-pure-backtest",
            "trend-pure-full-run",
            "trend-pure-period-study",
            "shadow-run",
            "report",
            "full-run",
        ],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--factors", default=None, help="Comma-separated factor names")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--model-name", default="decorrelated_v1")
    parser.add_argument("--min-amount-yi", type=float, default=2.0)
    parser.add_argument("--capacity-pct", type=float, default=0.005, help="Max single-stock buy notional as a fraction of entry-day amount")
    parser.add_argument("--portfolio-capital-yi", type=float, default=0.1, help="Assumed portfolio capital in 100m CNY units")
    parser.add_argument("--train-days", type=int, default=126)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--period-years", type=int, default=3)
    parser.add_argument("--production-top-n", type=int, default=30)
    parser.add_argument("--tracking-lookback-days", type=int, default=30)
    parser.add_argument("--research-start-date", default="2020-01-02")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    factor_names = [item.strip() for item in args.factors.split(",") if item.strip()] if args.factors else None
    requested_end = args.end_date or args.date
    if str(requested_end or "").strip().lower() in {"latest", "最近交易日"}:
        requested_end = None
    end_date = None if args.command == "shadow-run" and not requested_end else normalize_date(requested_end or _latest_daily_date(settings))
    start_date = normalize_date(args.start_date) if args.start_date else None

    if args.command == "init-schema":
        init_schema(settings)
        result = {"schema_status": "initialized"}
    elif args.command == "shadow-run":
        result = run_shadow_pipeline(
            settings,
            trade_date=end_date,
            production_top_n=args.production_top_n,
            tracking_lookback_days=args.tracking_lookback_days,
            research_start_date=args.research_start_date,
        )
    else:
        if args.command in {"report", "full-run"} and not start_date:
            try:
                start_date, end_date = _latest_factor_range(settings, end_date if args.date or args.end_date else None)
            except Exception:
                start_date = _default_start_date(settings, end_date)
        elif args.command != "coverage" and not start_date:
            raise RuntimeError("--start-date is required for this factor-lab command")

        if args.command == "coverage":
            result = build_data_coverage(settings, as_of_date=end_date)
        elif args.command == "build-factors":
            result = build_factor_daily(settings, start_date, end_date, factors=factor_names)
        elif args.command == "evaluate":
            result = evaluate_factors(settings, start_date, end_date, factors=factor_names)
        elif args.command == "correlate":
            result = correlate_factors(settings, start_date, end_date, factors=factor_names)
        elif args.command == "event-study":
            result = run_event_study(settings, start_date, end_date)
        elif args.command == "weights":
            result = optimize_weights(settings, start_date, end_date, as_of_date=end_date, horizon_days=args.hold_days)
        elif args.command == "backtest":
            optimize_weights(settings, start_date, end_date, as_of_date=end_date, horizon_days=args.hold_days)
            result = run_backtest(
                settings,
                start_date,
                end_date,
                model_name=args.model_name,
                top_n=args.top_n,
                hold_days=args.hold_days,
                min_amount_yi=args.min_amount_yi,
                capacity_pct=args.capacity_pct,
                portfolio_capital_yi=args.portfolio_capital_yi,
            )
        elif args.command == "backtest-grid":
            result = run_backtest_grid(
                settings,
                start_date,
                end_date,
                model_name=args.model_name,
                min_amount_yi=args.min_amount_yi,
                capacity_pct=args.capacity_pct,
                portfolio_capital_yi=args.portfolio_capital_yi,
            )
        elif args.command == "walk-forward":
            result = run_walk_forward_backtest(
                settings,
                start_date,
                end_date,
                model_name="walk_forward_v1",
                top_n=args.top_n,
                hold_days=args.hold_days,
                train_days=args.train_days,
                test_days=args.test_days,
                min_amount_yi=args.min_amount_yi,
                capacity_pct=args.capacity_pct,
                portfolio_capital_yi=args.portfolio_capital_yi,
            )
        elif args.command == "short-strength-backtest":
            result = run_short_strength_backtest(
                settings,
                start_date,
                end_date,
                capacity_pct=args.capacity_pct,
                portfolio_capital_yi=args.portfolio_capital_yi,
            )
        elif args.command == "trend-pure-evaluate":
            result = evaluate_trend_indicators(settings, start_date, end_date)
        elif args.command == "trend-pure-backtest":
            top_ns = tuple(sorted({10, 20, 30, int(args.top_n)}))
            hold_days_list = tuple(sorted({3, 5, 10, 20, int(args.hold_days)}))
            result = run_trend_pure_backtest(
                settings,
                start_date,
                end_date,
                top_ns=top_ns,
                hold_days_list=hold_days_list,
                min_amount_yi=args.min_amount_yi,
                capacity_pct=args.capacity_pct,
                portfolio_capital_yi=args.portfolio_capital_yi,
            )
        elif args.command == "trend-pure-full-run":
            evaluate_result = evaluate_trend_indicators(settings, start_date, end_date)
            top_ns = tuple(sorted({10, 20, 30, int(args.top_n)}))
            hold_days_list = tuple(sorted({3, 5, 10, 20, int(args.hold_days)}))
            backtest_result = run_trend_pure_backtest(
                settings,
                start_date,
                end_date,
                top_ns=top_ns,
                hold_days_list=hold_days_list,
                min_amount_yi=args.min_amount_yi,
                capacity_pct=args.capacity_pct,
                portfolio_capital_yi=args.portfolio_capital_yi,
            )
            result = {"evaluate": evaluate_result, "backtest": backtest_result}
        elif args.command == "trend-pure-period-study":
            result = run_trend_pure_period_study(
                settings,
                start_date,
                end_date,
                period_years=args.period_years,
                horizon_days=args.hold_days,
                top_n=args.top_n,
                hold_days=args.hold_days,
                min_amount_yi=args.min_amount_yi,
                capacity_pct=args.capacity_pct,
                portfolio_capital_yi=args.portfolio_capital_yi,
            )
        elif args.command == "full-run":
            result = ensure_research_pipeline(
                settings,
                start_date,
                end_date,
                factor_names=factor_names,
                top_n=args.top_n,
                hold_days=args.hold_days,
                min_amount_yi=args.min_amount_yi,
            )
            path = write_summary_report(settings, args.date or end_date, start_date, end_date)
            result["summary_report_file"] = str(path)
        elif args.command == "report":
            pipeline = ensure_research_pipeline(
                settings,
                start_date,
                end_date,
                factor_names=factor_names,
                top_n=args.top_n,
                hold_days=args.hold_days,
                min_amount_yi=args.min_amount_yi,
            )
            path = write_summary_report(settings, args.date or end_date, start_date, end_date)
            result = {"report_file": str(path), "start_date": start_date, "end_date": end_date, "pipeline": pipeline}
        else:
            raise RuntimeError(f"Unsupported command: {args.command}")

    print("agent_name=factor_lab")
    print("agent_status=finished")
    print("metrics=" + json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
