from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tech_signal.config import Settings
from tech_signal.db import connect, init_schema, qname, upsert_rows

from .factor_definitions import normalize_date
from .factor_evaluator import report_dir


COVERAGE_THRESHOLDS = {
    "daily_bars": 1000,
    "daily_basic": 1000,
    "stock_signal_daily": 1000,
    "theme_signal_daily": 1,
    "moneyflow_daily": 1000,
    "moneyflow_stock": 1000,
    "moneyflow_market": 1,
    "moneyflow_industry": 1,
    "moneyflow_concept": 1,
    "limit_events": 1,
    "limit_market_stats": 1,
    "lhb_stocks": 1,
    "lhb_seats": 1,
    "index_daily": 8,
    "global_index_daily": 1,
    "dragon_leader_daily": 1,
    "factor_daily": 1,
    "factor_performance": 1,
    "factor_correlation": 1,
    "model_weight_history": 1,
    "strategy_backtest_result": 1,
    "strategy_backtest_trades": 1,
}

DATE_COLUMNS = {
    "factor_performance": "end_date",
    "factor_correlation": "end_date",
    "model_weight_history": "as_of_date",
    "strategy_backtest_result": "end_date",
    "global_index_daily": "trade_date",
}


def _date_col(table: str) -> str:
    return DATE_COLUMNS.get(table, "trade_date")


def build_data_coverage(
    settings: Settings,
    *,
    as_of_date: str | None = None,
    baseline_start: str = "2020-01-01",
) -> dict[str, Any]:
    init_schema(settings)
    as_of = normalize_date(as_of_date or _latest_daily_date(settings))
    baseline = normalize_date(baseline_start)
    rows: list[dict[str, Any]] = []
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT max(trade_date)::text AS d FROM {qname(settings, 'daily_bars')}")
        latest_daily = cur.fetchone()["d"]
        for table, threshold in COVERAGE_THRESHOLDS.items():
            col = _date_col(table)
            cur.execute(
                f"""
                SELECT min({col}) AS min_date, max({col}) AS max_date,
                       count(*) AS row_count, count(DISTINCT {col}) AS date_count
                FROM {qname(settings, table)}
                """
            )
            summary = cur.fetchone()
            cur.execute(
                f"""
                WITH per_day AS (
                    SELECT {col} AS trade_date, count(*) AS n
                    FROM {qname(settings, table)}
                    GROUP BY {col}
                ), ok AS (
                    SELECT trade_date, n FROM per_day WHERE n >= %s
                )
                SELECT min(trade_date) AS stable_start, max(trade_date) AS stable_end, count(*) AS ok_dates
                FROM ok
                """,
                (threshold,),
            )
            stable = cur.fetchone()
            cur.execute(
                f"""
                WITH open_dates AS (
                    SELECT cal_date
                    FROM {qname(settings, 'trade_calendar')}
                    WHERE is_open=true AND cal_date BETWEEN %s AND %s
                ), per_day AS (
                    SELECT {col} AS trade_date, count(*) AS n
                    FROM {qname(settings, table)}
                    GROUP BY {col}
                )
                SELECT count(*) AS open_dates,
                       count(*) FILTER (WHERE coalesce(p.n, 0) >= %s) AS ok_dates,
                       count(*) FILTER (WHERE coalesce(p.n, 0) < %s) AS missing_or_low_dates,
                       min(o.cal_date) FILTER (WHERE coalesce(p.n, 0) < %s) AS first_missing,
                       max(o.cal_date) FILTER (WHERE coalesce(p.n, 0) < %s) AS last_missing
                FROM open_dates o
                LEFT JOIN per_day p ON p.trade_date=o.cal_date
                """,
                (baseline, latest_daily, threshold, threshold, threshold, threshold),
            )
            missing = cur.fetchone()
            notes = ""
            if table in {"moneyflow_daily", "moneyflow_stock", "moneyflow_market", "moneyflow_industry", "moneyflow_concept", "limit_events", "limit_market_stats", "lhb_stocks", "lhb_seats", "theme_signal_daily", "dragon_leader_daily", "index_daily"}:
                notes = "research-sensitive auxiliary layer; early missing dates reduce event/factor reliability"
            rows.append(
                {
                    "as_of_date": as_of,
                    "table_name": table,
                    "min_date": summary["min_date"],
                    "max_date": summary["max_date"],
                    "row_count": int(summary["row_count"] or 0),
                    "date_count": int(summary["date_count"] or 0),
                    "stable_start": stable["stable_start"],
                    "stable_end": stable["stable_end"],
                    "open_dates": int(missing["open_dates"] or 0),
                    "ok_dates": int(missing["ok_dates"] or 0),
                    "missing_or_low_dates": int(missing["missing_or_low_dates"] or 0),
                    "first_missing": missing["first_missing"],
                    "last_missing": missing["last_missing"],
                    "threshold_rows": threshold,
                    "notes": notes,
                }
            )
    columns = [
        "as_of_date",
        "table_name",
        "min_date",
        "max_date",
        "row_count",
        "date_count",
        "stable_start",
        "stable_end",
        "open_dates",
        "ok_dates",
        "missing_or_low_dates",
        "first_missing",
        "last_missing",
        "threshold_rows",
        "notes",
    ]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "factor_data_coverage"),
            columns=columns,
            rows=rows,
            conflict_columns=["as_of_date", "table_name"],
        )
        conn.commit()
    path = write_coverage_report(settings, as_of, rows)
    return {"as_of_date": as_of, "coverage_rows": count, "report_file": str(path)}


def _latest_daily_date(settings: Settings) -> str:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT max(trade_date)::text AS d FROM {qname(settings, 'daily_bars')}")
        row = cur.fetchone()
    return str(row["d"]) if row and row["d"] else ""


def write_coverage_report(settings: Settings, as_of_date: str, rows: list[dict[str, Any]]) -> Path:
    path = report_dir(settings) / f"factor_data_coverage_{normalize_date(as_of_date).replace('-', '')}.md"
    data = pd.DataFrame(rows)
    lines = [
        f"# Factor Data Coverage {normalize_date(as_of_date)}",
        "",
        "覆盖统计以交易日历开市日为基准。threshold_rows 是判断某表当天是否可用于研究的最低行数。",
        "",
        "| table | min | max | rows | dates | stable_start | 2020+ ok/missing | first_missing | last_missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['table_name']} | {row['min_date'] or ''} | {row['max_date'] or ''} | {row['row_count']} | "
            f"{row['date_count']} | {row['stable_start'] or ''} | {row['ok_dates']}/{row['missing_or_low_dates']} | "
            f"{row['first_missing'] or ''} | {row['last_missing'] or ''} |"
        )
    weak = data[data["missing_or_low_dates"].fillna(0).astype(int) > 0]
    lines.extend(["", "## Gaps", ""])
    if weak.empty:
        lines.append("- No 2020+ missing/low dates under configured thresholds.")
    else:
        for row in weak.to_dict("records"):
            lines.append(
                f"- {row['table_name']}: {row['missing_or_low_dates']} missing/low open dates since 2020; "
                f"first={row['first_missing']}, last={row['last_missing']}. {row['notes']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
