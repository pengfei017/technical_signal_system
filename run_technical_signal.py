#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from tech_signal.analyzer import compute_signals
from tech_signal.config import load_settings
from tech_signal.db import connect, init_schema, qname
from tech_signal.market_layers import (
    backfill_market_structure_layers,
    refresh_dragon_leader_daily,
    refresh_global_index_daily,
    refresh_index_daily,
    refresh_market_structure_layers,
)
from tech_signal.report import generate_report
from tech_signal.trading_signals import update_trading_auxiliary
from tech_signal.tushare_fetcher import TushareFetcher
from tech_signal.universe import load_focus_universe, sync_signal_universe


def setup_logging(log_file) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def start_run(settings, task: str) -> str:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {qname(settings, 'signal_runs')} (run_id, status, task) VALUES (%s, %s, %s)",
            (run_id, "running", task),
        )
        conn.commit()
    return run_id


def finish_run(settings, run_id: str, status: str, message: str = "", metrics: dict | None = None) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {qname(settings, 'signal_runs')}
            SET finished_at=now(), status=%s, message=%s, metrics=%s::jsonb
            WHERE run_id=%s
            """,
            (status, message, json.dumps(metrics or {}, ensure_ascii=False), run_id),
        )
        conn.commit()


def acquire_lock(lock_path: Path, *, stale_hours: int = 12) -> bool:
    if lock_path.exists():
        age_seconds = datetime.now().timestamp() - lock_path.stat().st_mtime
        if age_seconds > stale_hours * 3600:
            lock_path.unlink(missing_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\nstarted_at={datetime.now().isoformat()}\n")
    return True


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def update_calendar(settings) -> dict[str, Any]:
    fetcher = TushareFetcher(settings)
    data_cfg = settings.section("data")
    calendar_start = str(data_cfg.get("trade_calendar_start_date", "2010-01-01"))
    calendar_future_years = int(data_cfg.get("trade_calendar_future_years", 2))
    calendar_end = f"{datetime.now().year + calendar_future_years}1231"
    return fetcher.sync_trade_calendar_range(calendar_start, calendar_end)


def _recent_open_dates_from_db(settings, *, lookback: int, end_date: str | None = None) -> list[str]:
    if end_date:
        text = str(end_date).replace("-", "")[:8]
        end_value = f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) == 8 else datetime.now().strftime("%Y-%m-%d")
    else:
        end_value = datetime.now().strftime("%Y-%m-%d")

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT cal_date
            FROM {qname(settings, 'trade_calendar')}
            WHERE is_open=true AND cal_date <= %s
            ORDER BY cal_date DESC
            LIMIT %s
            """,
            (end_value, lookback),
        )
        rows = [row["cal_date"].strftime("%Y%m%d") for row in cur.fetchall()]
    return list(reversed(rows))


def _open_trade_dates(settings, fetcher: TushareFetcher, *, lookback: int, end_date: str | None = None) -> list[str]:
    dates = _recent_open_dates_from_db(settings, lookback=lookback, end_date=end_date)
    if len(dates) >= min(lookback, 3):
        return dates
    return fetcher.open_trade_dates(lookback)


def _latest_daily_bar_trade_date(settings) -> str:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT max(trade_date) AS d FROM {qname(settings, 'daily_bars')}")
        row = cur.fetchone()
        return row["d"].strftime("%Y%m%d") if row and row["d"] else ""


def _expected_latest_open_date(settings) -> str:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT max(cal_date) AS d
            FROM {qname(settings, 'trade_calendar')}
            WHERE is_open=true AND cal_date <= CURRENT_DATE
            """
        )
        row = cur.fetchone()
        return row["d"].strftime("%Y%m%d") if row and row["d"] else ""


def _dash_date(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).replace("-", "")[:8]
    if len(text) == 8:
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return value


def _compact_date(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).replace("-", "")[:8]
    return text if len(text) == 8 else None


def _open_trade_dates_between(settings, *, start_date: str, end_date: str) -> list[str]:
    start_value = _dash_date(start_date)
    end_value = _dash_date(end_date)
    if not start_value or not end_value:
        raise RuntimeError(f"Invalid backfill date range: {start_date} to {end_date}")

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT cal_date
            FROM {qname(settings, 'trade_calendar')}
            WHERE is_open=true
              AND cal_date BETWEEN %s AND %s
            ORDER BY cal_date
            """,
            (start_value, end_value),
        )
        return [row["cal_date"].strftime("%Y%m%d") for row in cur.fetchall()]


def _count_rows_for_date(cur, settings, table: str, trade_date: str, extra_where: str = "") -> int:
    cur.execute(
        f"SELECT count(*) AS n FROM {qname(settings, table)} WHERE trade_date=%s {extra_where}",
        (trade_date,),
    )
    row = cur.fetchone()
    return int(row["n"] or 0)


def ensure_market_data_current(settings) -> dict[str, Any]:
    expected = _expected_latest_open_date(settings)
    latest = _latest_daily_bar_trade_date(settings)
    if not expected or (latest and latest >= expected):
        return {
            "market_data_checked": True,
            "market_data_refreshed": False,
            "expected_trade_date": _dash_date(expected),
            "latest_daily_bar_trade_date": _dash_date(latest),
        }

    logging.warning("daily market data is stale, refreshing before evening pipeline: latest=%s expected=%s", latest, expected)
    refresh_metrics = update_market_data(settings)
    refreshed_latest = _latest_daily_bar_trade_date(settings)
    return {
        "market_data_checked": True,
        "market_data_refreshed": True,
        "expected_trade_date": _dash_date(expected),
        "previous_daily_bar_trade_date": _dash_date(latest),
        "latest_daily_bar_trade_date": _dash_date(refreshed_latest),
        **{f"market_refresh_{key}": value for key, value in refresh_metrics.items()},
    }


def validate_data_ready(
    settings,
    *,
    trade_date: str | None = None,
    require_trading: bool = True,
    require_moneyflow: bool = True,
    require_current_trade_date: bool = False,
) -> dict[str, Any]:
    validation_cfg = settings.section("validation")
    data_cfg = settings.section("data")
    date_value = _dash_date(trade_date)
    if not date_value:
        date_value = _dash_date(_latest_daily_bar_trade_date(settings))
    if not date_value:
        raise RuntimeError("Data validation failed: no daily_bars trade_date available")

    expected = _dash_date(_expected_latest_open_date(settings))
    counts: dict[str, int] = {}
    with connect() as conn, conn.cursor() as cur:
        counts["daily_bars"] = _count_rows_for_date(cur, settings, "daily_bars", date_value)
        counts["daily_bars_with_adj"] = _count_rows_for_date(
            cur,
            settings,
            "daily_bars",
            date_value,
            "AND adj_close IS NOT NULL",
        )
        counts["daily_basic"] = _count_rows_for_date(cur, settings, "daily_basic", date_value)
        counts["moneyflow_daily"] = _count_rows_for_date(cur, settings, "moneyflow_daily", date_value)
        counts["moneyflow_stock"] = _count_rows_for_date(cur, settings, "moneyflow_stock", date_value)
        counts["limit_events"] = _count_rows_for_date(cur, settings, "limit_events", date_value)
        counts["limit_market_stats"] = _count_rows_for_date(cur, settings, "limit_market_stats", date_value)
        counts["lhb_stocks"] = _count_rows_for_date(cur, settings, "lhb_stocks", date_value)
        counts["moneyflow_market"] = _count_rows_for_date(cur, settings, "moneyflow_market", date_value)
        counts["moneyflow_industry"] = _count_rows_for_date(cur, settings, "moneyflow_industry", date_value)
        counts["moneyflow_concept"] = _count_rows_for_date(cur, settings, "moneyflow_concept", date_value)

    failures: list[str] = []

    def require_count(key: str, min_count: int) -> None:
        actual = int(counts.get(key, 0))
        if actual < min_count:
            failures.append(f"{key} rows {actual} < {min_count}")

    if require_current_trade_date and expected and date_value != expected:
        failures.append(f"daily_bars latest trade_date {date_value} != expected open date {expected}")

    min_daily = int(validation_cfg.get("min_daily_bar_rows", 1000))
    require_count("daily_bars", min_daily)
    require_count("daily_bars_with_adj", min_daily)
    require_count("daily_basic", int(validation_cfg.get("min_daily_basic_rows", 1000)))

    if require_trading:
        moneyflow_scope = str(data_cfg.get("moneyflow_scope", "all_market"))
        if require_moneyflow and moneyflow_scope == "all_market":
            min_moneyflow = int(validation_cfg.get("min_moneyflow_rows", 1000))
            require_count("moneyflow_daily", min_moneyflow)
            require_count("moneyflow_stock", int(validation_cfg.get("min_moneyflow_stock_rows", min_moneyflow)))

        if bool(data_cfg.get("limit_lhb_enabled", True)) and bool(validation_cfg.get("require_limit_lhb", True)):
            require_count("limit_events", int(validation_cfg.get("min_limit_events", 1)))
            require_count("limit_market_stats", int(validation_cfg.get("min_limit_market_stats_rows", 1)))
            require_count("lhb_stocks", int(validation_cfg.get("min_lhb_stocks", 1)))

        if bool(data_cfg.get("market_moneyflow_enabled", True)) and bool(validation_cfg.get("require_market_moneyflow", True)):
            require_count("moneyflow_market", int(validation_cfg.get("min_moneyflow_market_rows", 1)))
            require_count("moneyflow_industry", int(validation_cfg.get("min_moneyflow_industry_rows", 1)))
            require_count("moneyflow_concept", int(validation_cfg.get("min_moneyflow_concept_rows", 1)))

    metrics: dict[str, Any] = {
        "trade_date": date_value,
        "expected_trade_date": expected,
        "validation_status": "passed" if not failures else "failed",
        **{f"validation_{key}_rows": value for key, value in counts.items()},
    }
    if failures:
        metrics["validation_failures"] = failures
        raise RuntimeError(f"Data validation failed for {date_value}: " + "; ".join(failures))
    return metrics


def update_market_data(settings, *, days: int | None = None) -> dict[str, Any]:
    fetcher = TushareFetcher(settings)
    data_cfg = settings.section("data")
    history_days = int(data_cfg.get("all_a_lookback_trading_days", 90))
    fetch_recent_days = int(data_cfg.get("daily_fetch_recent_trading_days", data_cfg.get("refresh_recent_trading_days", 5)))
    existing_bar_dates = fetcher.existing_daily_bar_date_count()
    bootstrap_backfill = days is not None or existing_bar_dates < history_days
    fetch_days = int(days or history_days) if bootstrap_backfill else fetch_recent_days
    # Add one open-day buffer so a current intraday date with no settled daily data
    # does not crowd out the latest completed trading day.
    dates = _open_trade_dates(settings, fetcher, lookback=fetch_days + 1)
    if not dates:
        raise RuntimeError("No open trade dates found")

    market_metrics = fetcher.fetch_market_for_dates(dates)
    effective_trade_date = market_metrics.get("latest_trade_date") or dates[-1]
    index_metrics = refresh_index_daily(settings, trade_date=str(effective_trade_date))
    members = load_focus_universe(settings)
    universe_count = sync_signal_universe(settings, members)
    return {
        "trade_date": effective_trade_date,
        "target_lookback_trading_days": history_days,
        "fetch_window_trading_days": fetch_days,
        "existing_daily_bar_dates": existing_bar_dates,
        "bootstrap_backfill": bootstrap_backfill,
        "universe_rows": universe_count,
        **market_metrics,
        **index_metrics,
    }


def backfill_daily_history(
    settings,
    *,
    start_date: str,
    end_date: str,
    sleep_seconds: float | None = None,
) -> dict[str, Any]:
    fetcher = TushareFetcher(settings)
    start_compact = _compact_date(start_date)
    end_compact = _compact_date(end_date)
    if not start_compact or not end_compact:
        raise RuntimeError(f"Invalid backfill date range: {start_date} to {end_date}")
    if start_compact > end_compact:
        raise RuntimeError(f"Invalid backfill date range: {start_date} > {end_date}")

    data_cfg = settings.section("data")
    if sleep_seconds is None:
        sleep_seconds = float(data_cfg.get("historical_daily_backfill_sleep_seconds", 1.2))
    fetcher.sleep_seconds = float(sleep_seconds)

    dates = _open_trade_dates_between(settings, start_date=start_compact, end_date=end_compact)
    if not dates:
        fetcher.sync_trade_calendar_range(start_compact, end_compact)
        dates = _open_trade_dates_between(settings, start_date=start_compact, end_date=end_compact)
    if not dates:
        raise RuntimeError(f"No open trade dates found for {start_compact}-{end_compact}")

    metrics = fetcher.fetch_market_for_dates(dates, refresh_adjusted_scope="range")
    return {
        "backfill_start_date": _dash_date(start_compact),
        "backfill_end_date": _dash_date(end_compact),
        "backfill_open_dates": len(dates),
        "backfill_sleep_seconds": fetcher.sleep_seconds,
        "trade_date": _dash_date(metrics.get("latest_trade_date") or dates[-1]),
        **metrics,
    }


def update_trading_data(settings, *, skip_moneyflow: bool = False) -> dict[str, Any]:
    fetcher = TushareFetcher(settings)
    data_cfg = settings.section("data")
    latest_trade_date = _latest_daily_bar_trade_date(settings)
    if not latest_trade_date:
        raise RuntimeError("No daily_bars trade_date available; run update-market-data first")

    moneyflow_count = 0
    moneyflow_scope = str(data_cfg.get("moneyflow_scope", "all_market"))
    if not skip_moneyflow and moneyflow_scope == "all_market":
        mf_days = int(data_cfg.get("stock_moneyflow_fetch_recent_trading_days", data_cfg.get("focus_moneyflow_fetch_recent_trading_days", 5)))
        moneyflow_dates = _open_trade_dates(settings, fetcher, lookback=mf_days, end_date=latest_trade_date)
        moneyflow_count = fetcher.fetch_market_moneyflow(moneyflow_dates)
    elif not skip_moneyflow and moneyflow_scope == "focus":
        members = load_focus_universe(settings)
        mf_days = int(data_cfg.get("focus_moneyflow_fetch_recent_trading_days", data_cfg.get("focus_moneyflow_lookback_trading_days", 5)))
        moneyflow_dates = _open_trade_dates(settings, fetcher, lookback=mf_days, end_date=latest_trade_date)
        moneyflow_count = fetcher.fetch_focus_moneyflow([m.ts_code for m in members], moneyflow_dates)

    auxiliary_metrics = update_trading_auxiliary(settings, fetcher, latest_trade_date)
    return {
        "trade_date": latest_trade_date,
        "moneyflow_scope": moneyflow_scope,
        "moneyflow_rows": moneyflow_count,
        **auxiliary_metrics,
    }


def update_data(settings, *, days: int | None = None, skip_moneyflow: bool = False) -> dict[str, Any]:
    market_metrics = update_market_data(settings, days=days)
    trading_metrics = update_trading_data(settings, skip_moneyflow=skip_moneyflow)
    return {**market_metrics, **trading_metrics}


def process_signals(
    settings,
    *,
    trade_date: str | None = None,
    require_trading: bool = True,
    require_moneyflow: bool = True,
    require_current_trade_date: bool = False,
) -> dict[str, Any]:
    validation_metrics = validate_data_ready(
        settings,
        trade_date=trade_date,
        require_trading=require_trading,
        require_moneyflow=require_moneyflow,
        require_current_trade_date=require_current_trade_date,
    )
    members = load_focus_universe(settings)
    sync_signal_universe(settings, members)
    signal_metrics = compute_signals(settings, trade_date=_dash_date(trade_date))
    leader_metrics = refresh_dragon_leader_daily(settings, trade_date=str(signal_metrics["trade_date"]))
    report_path = generate_report(settings, trade_date=str(signal_metrics["trade_date"]))
    return {**validation_metrics, **signal_metrics, **leader_metrics, "report_file": str(report_path)}


def evening_pipeline(settings, *, skip_moneyflow: bool = False) -> dict[str, Any]:
    market_metrics = ensure_market_data_current(settings)
    trading_metrics = update_trading_data(settings, skip_moneyflow=skip_moneyflow)
    process_metrics = process_signals(
        settings,
        trade_date=str(trading_metrics["trade_date"]),
        require_moneyflow=not skip_moneyflow,
        require_current_trade_date=True,
    )
    return {**market_metrics, **trading_metrics, **process_metrics}


def run_all(settings, args) -> dict[str, Any]:
    init_schema(settings)
    calendar_metrics = update_calendar(settings)
    market_metrics = update_market_data(settings, days=args.days)
    trading_metrics = update_trading_data(settings, skip_moneyflow=args.skip_moneyflow)
    process_metrics = process_signals(
        settings,
        require_moneyflow=not args.skip_moneyflow,
        require_current_trade_date=True,
    )
    return {
        **calendar_metrics,
        **market_metrics,
        **trading_metrics,
        **process_metrics,
    }


RETRYABLE_COMMANDS = {
    "update-calendar",
    "update-market-data",
    "update-trading-data",
    "update-data",
    "update-indexes",
    "update-global-indexes",
    "backfill-daily",
    "backfill-market-layers",
    "refresh-dragon-leaders",
    "evening-pipeline",
    "run",
}


def execute_command(settings, args) -> dict[str, Any]:
    if args.command == "update-calendar":
        return update_calendar(settings)
    if args.command == "update-market-data":
        return update_market_data(settings, days=args.days)
    if args.command == "update-trading-data":
        return update_trading_data(settings, skip_moneyflow=args.skip_moneyflow)
    if args.command == "update-data":
        return update_data(settings, days=args.days, skip_moneyflow=args.skip_moneyflow)
    if args.command == "update-indexes":
        return refresh_market_structure_layers(settings, trade_date=args.trade_date)
    if args.command == "update-global-indexes":
        return refresh_global_index_daily(settings, trade_date=args.trade_date)
    if args.command == "backfill-daily":
        if args.year:
            start_date = f"{args.year}0101"
            end_date = f"{args.year}1231"
        else:
            start_date = args.start_date
            end_date = args.end_date
        if not start_date or not end_date:
            raise RuntimeError("backfill-daily requires --year or both --start-date and --end-date")
        return backfill_daily_history(
            settings,
            start_date=start_date,
            end_date=end_date,
            sleep_seconds=args.sleep_seconds,
        )
    if args.command == "backfill-market-layers":
        if args.year:
            start_date = args.start_date or f"{args.year}0101"
            if args.end_date:
                end_date = args.end_date
            elif args.year >= datetime.now().year:
                end_date = _expected_latest_open_date(settings) or _latest_daily_bar_trade_date(settings)
            else:
                end_date = f"{args.year}1231"
        else:
            start_date = args.start_date or f"{datetime.now().year}0101"
            end_date = args.end_date or _expected_latest_open_date(settings) or _latest_daily_bar_trade_date(settings)
        if not start_date or not end_date:
            raise RuntimeError("backfill-market-layers requires a valid date range")
        return backfill_market_structure_layers(
            settings,
            start_date=start_date,
            end_date=end_date,
            include_dragon_leaders=not args.skip_dragon_leaders,
        )
    if args.command == "validate-data":
        return validate_data_ready(
            settings,
            trade_date=args.trade_date,
            require_current_trade_date=args.trade_date is None,
        )
    if args.command == "analyze":
        validation_metrics = validate_data_ready(
            settings,
            trade_date=args.trade_date,
            require_current_trade_date=args.trade_date is None,
        )
        members = load_focus_universe(settings)
        sync_signal_universe(settings, members)
        signal_metrics = compute_signals(settings, trade_date=args.trade_date)
        return {**validation_metrics, **signal_metrics}
    if args.command == "report":
        path = generate_report(settings, trade_date=args.trade_date)
        return {"report_file": str(path)}
    if args.command == "refresh-dragon-leaders":
        return refresh_dragon_leader_daily(settings, trade_date=args.trade_date)
    if args.command == "process":
        return process_signals(
            settings,
            trade_date=args.trade_date,
            require_current_trade_date=args.trade_date is None,
        )
    if args.command == "evening-pipeline":
        return evening_pipeline(settings, skip_moneyflow=args.skip_moneyflow)
    return run_all(settings, args)


def execute_with_retries(settings, args) -> dict[str, Any]:
    retry_cfg = settings.section("automation")
    retry_count = int(retry_cfg.get("task_retry_count", 0)) if args.command in RETRYABLE_COMMANDS else 0
    sleep_seconds = float(retry_cfg.get("task_retry_sleep_seconds", 0))
    attempts = retry_count + 1
    for attempt in range(1, attempts + 1):
        try:
            metrics = execute_command(settings, args)
            if attempts > 1:
                metrics["task_attempt"] = attempt
                metrics["task_max_attempts"] = attempts
            return metrics
        except Exception:
            if attempt >= attempts:
                raise
            logging.exception("technical signal command failed, retrying: command=%s attempt=%s/%s", args.command, attempt, attempts)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    raise RuntimeError("unreachable retry state")


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent A-share technical signal system")
    parser.add_argument(
        "command",
        nargs="?",
        default="process",
        choices=[
            "init-db",
            "update-calendar",
            "update-market-data",
            "update-trading-data",
            "update-data",
            "update-indexes",
            "update-global-indexes",
            "backfill-daily",
            "backfill-market-layers",
            "validate-data",
            "analyze",
            "report",
            "refresh-dragon-leaders",
            "process",
            "evening-pipeline",
            "run",
        ],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--days", type=int, default=None, help="Backfill trading days for all-A daily data")
    parser.add_argument("--year", type=int, default=None, help="Year for historical daily backfill, for example 2010")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD for historical daily backfill")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD for historical daily backfill")
    parser.add_argument("--sleep-seconds", type=float, default=None, help="Override per-request sleep for historical backfill")
    parser.add_argument("--trade-date", default=None, help="YYYY-MM-DD, default latest")
    parser.add_argument("--date", default=None, help="Alias of --trade-date; use 最近交易日 or latest for default latest")
    parser.add_argument("--skip-moneyflow", action="store_true")
    parser.add_argument("--skip-dragon-leaders", action="store_true")
    args = parser.parse_args()
    if args.date and not args.trade_date:
        args.trade_date = args.date
    if str(args.trade_date or "").strip().lower() in {"latest", "最近交易日"}:
        args.trade_date = "__latest_available__"
    if str(args.trade_date or "").strip().lower() in {""}:
        args.trade_date = None

    settings = load_settings(args.config)
    setup_logging(settings.logs_dir / "technical_signal.log")
    run_id = ""
    lock_path = settings.output_root / "technical_signal.lock"
    locked = False
    try:
        if args.command == "init-db":
            init_schema(settings)
            print("agent_name=technical_signal")
            print("agent_status=finished")
            print("message=database initialized")
            return 0

        locked = acquire_lock(lock_path)
        if not locked:
            raise RuntimeError(f"Another technical_signal task is running: {lock_path}")
        init_schema(settings)
        if args.trade_date == "__latest_available__":
            args.trade_date = _dash_date(_latest_daily_bar_trade_date(settings))
        run_id = start_run(settings, args.command)
        metrics = execute_with_retries(settings, args)
        finish_run(settings, run_id, "finished", metrics=metrics)
        print("agent_name=technical_signal")
        print("agent_status=finished")
        print("agent_exit_code=0")
        if "trade_date" in metrics:
            print(f"trade_date={metrics['trade_date']}")
        if "report_file" in metrics:
            print(f"report_file={metrics['report_file']}")
        print(f"metrics={json.dumps(metrics, ensure_ascii=False)}")
        return 0
    except Exception as exc:
        if run_id:
            try:
                finish_run(settings, run_id, "failed", message=str(exc))
            except Exception:
                pass
        logging.exception("technical signal task failed")
        print("agent_name=technical_signal")
        print("agent_status=failed")
        print("agent_exit_code=1")
        print(f"error={exc}")
        return 1
    finally:
        if locked:
            release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
