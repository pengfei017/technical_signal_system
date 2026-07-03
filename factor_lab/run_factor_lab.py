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

from factor_lab.backtest_engine import run_backtest
from factor_lab.factor_definitions import FACTOR_DEFINITIONS, normalize_date
from factor_lab.factor_evaluator import (
    build_factor_daily,
    correlate_factors,
    evaluate_factors,
    report_dir,
)
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


def _query_df(settings, sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


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
               sharpe, win_rate, benchmark_return, excess_return, trade_count
        FROM {qname(settings, 'strategy_backtest_result')}
        WHERE start_date=%s AND end_date=%s
        ORDER BY created_at DESC, model_name, sample_split
        """,
        (start_value, end_value),
    )
    weights = _query_df(
        settings,
        f"""
        SELECT model_name, factor_name, factor_group, weight
        FROM {qname(settings, 'model_weight_history')}
        WHERE as_of_date=%s
        """,
        (end_value,),
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
        "This is research output only. It does not change production `stock_signal_daily` scoring.",
        "",
        "## 1. 哪些因子最有效",
        "",
    ]
    if perf.empty:
        lines.append("- No performance rows.")
    else:
        for column in ["rank_ic_mean", "long_short_return", "win_rate", "max_drawdown"]:
            perf[column] = pd.to_numeric(perf[column], errors="coerce")
        best = perf.sort_values(["rank_ic_mean", "long_short_return"], ascending=False).head(10)
        for row in best.to_dict("records"):
            lines.append(
                f"- {row['factor_name']} ({row['factor_group']}): RankIC={fmt(row['rank_ic_mean'])}, "
                f"LS={fmt(row['long_short_return'])}, win={fmt(row['win_rate'])}"
            )
    lines.extend(["", "## 2. 哪些因子是噪音", ""])
    if not perf.empty:
        weak = perf[perf["rank_ic_mean"].abs().fillna(0) < 0.005].sort_values("sample_count", ascending=False).head(10)
        if weak.empty:
            lines.append("- No clear near-zero RankIC factor in this window.")
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
            lines.append("- No pair above abs(corr)=0.75.")
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
    lines.extend(["", "## 5. 当前生产评分权重是否明显不合理", ""])
    lines.append("- 第一版只能判断方向：若趋势/量能高度重复，现有人工打分可能存在重复加权。")
    lines.append("- 还不能直接证明生产权重错误，因为当前样本仍短，且 2025 年以前资金流、龙虎榜、涨跌停辅助因子不完整。")
    lines.extend(["", "## 6. 是否建议调整权重", ""])
    if weights.empty:
        lines.append("- No weight suggestion rows.")
    else:
        weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
        for model_name, group in weights.groupby("model_name"):
            top = group.assign(abs_weight=group["weight"].abs()).sort_values("abs_weight", ascending=False).head(6)
            detail = ", ".join(f"{row.factor_name}:{row.weight:.3f}" for row in top.itertuples())
            lines.append(f"- {model_name}: {detail}")
    lines.extend(["", "## 7. 暂时不建议进入生产的原因", ""])
    lines.append("- 这是首次离线评估，尚未经过连续影子运行。")
    lines.append("- 回测是 daily cohort 近似，不是完整交易执行引擎。")
    lines.append("- 需要看滚动窗口、样本外、不同市场环境下是否稳定后，才能讨论替换生产评分。")
    lines.extend(["", "## Backtest Snapshot", ""])
    if backtest.empty:
        lines.append("- No backtest result rows.")
    else:
        for row in backtest.head(12).to_dict("records"):
            lines.append(
                f"- {row['model_name']} / {row['sample_split']}: total={fmt(row['total_return'])}, "
                f"excess={fmt(row['excess_return'])}, mdd={fmt(row['max_drawdown'])}, trades={row['trade_count']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only factor lab for technical_signal_system")
    parser.add_argument("command", choices=["init-schema", "build-factors", "evaluate", "correlate", "weights", "backtest", "report"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--factors", default=None, help="Comma-separated factor names")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--model-name", default="decorrelated_v1")
    parser.add_argument("--min-amount-yi", type=float, default=2.0)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    factor_names = [item.strip() for item in args.factors.split(",") if item.strip()] if args.factors else None
    end_date = normalize_date(args.end_date or args.date or _latest_daily_date(settings))
    start_date = normalize_date(args.start_date) if args.start_date else None

    if args.command == "init-schema":
        init_schema(settings)
        result = {"schema_status": "initialized"}
    else:
        if args.command == "report" and not start_date:
            start_date, end_date = _latest_factor_range(settings, end_date if args.date or args.end_date else None)
        elif not start_date:
            raise RuntimeError("--start-date is required for this factor-lab command")

        if args.command == "build-factors":
            result = build_factor_daily(settings, start_date, end_date, factors=factor_names)
        elif args.command == "evaluate":
            result = evaluate_factors(settings, start_date, end_date, factors=factor_names)
        elif args.command == "correlate":
            result = correlate_factors(settings, start_date, end_date, factors=factor_names)
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
            )
        elif args.command == "report":
            optimize_weights(settings, start_date, end_date, as_of_date=end_date, horizon_days=args.hold_days)
            path = write_summary_report(settings, args.date or end_date, start_date, end_date)
            result = {"report_file": str(path), "start_date": start_date, "end_date": end_date}
        else:
            raise RuntimeError(f"Unsupported command: {args.command}")

    print("agent_name=factor_lab")
    print("agent_status=finished")
    print("metrics=" + json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

