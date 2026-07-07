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
from tech_signal.trading_signals import (
    fetch_lhb,
    fetch_limit_events,
    fetch_moneyflow_layers,
    refresh_stock_signal_daily,
    refresh_stock_signal_daily_range,
    refresh_theme_signal_daily,
    sync_moneyflow_stock_from_daily,
    update_trading_auxiliary,
)
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


def update_trading_data(settings, *, trade_date: str | None = None, skip_moneyflow: bool = False) -> dict[str, Any]:
    fetcher = TushareFetcher(settings)
    data_cfg = settings.section("data")
    target_trade_date = _compact_date(trade_date) if trade_date else _latest_daily_bar_trade_date(settings)
    if not target_trade_date:
        raise RuntimeError("No daily_bars trade_date available; run update-market-data first")

    moneyflow_count = 0
    moneyflow_scope = str(data_cfg.get("moneyflow_scope", "all_market"))
    if not skip_moneyflow and moneyflow_scope == "all_market":
        mf_days = int(data_cfg.get("stock_moneyflow_fetch_recent_trading_days", data_cfg.get("focus_moneyflow_fetch_recent_trading_days", 5)))
        moneyflow_dates = _open_trade_dates(settings, fetcher, lookback=mf_days, end_date=target_trade_date)
        moneyflow_count = fetcher.fetch_market_moneyflow(moneyflow_dates)
    elif not skip_moneyflow and moneyflow_scope == "focus":
        members = load_focus_universe(settings)
        mf_days = int(data_cfg.get("focus_moneyflow_fetch_recent_trading_days", data_cfg.get("focus_moneyflow_lookback_trading_days", 5)))
        moneyflow_dates = _open_trade_dates(settings, fetcher, lookback=mf_days, end_date=target_trade_date)
        moneyflow_count = fetcher.fetch_focus_moneyflow([m.ts_code for m in members], moneyflow_dates)

    auxiliary_metrics = update_trading_auxiliary(settings, fetcher, target_trade_date)
    return {
        "trade_date": _dash_date(target_trade_date),
        "moneyflow_scope": moneyflow_scope,
        "moneyflow_rows": moneyflow_count,
        **auxiliary_metrics,
    }


def _trading_counts_for_date(settings, trade_date: str) -> dict[str, int]:
    date_value = _dash_date(trade_date)
    tables = [
        "daily_bars",
        "moneyflow_daily",
        "moneyflow_stock",
        "limit_events",
        "limit_market_stats",
        "lhb_stocks",
        "lhb_seats",
        "moneyflow_market",
        "moneyflow_industry",
        "moneyflow_concept",
    ]
    with connect() as conn, conn.cursor() as cur:
        counts = {}
        for table in tables:
            counts[table] = _count_rows_for_date(cur, settings, table, str(date_value))
    return counts


def _trading_data_complete(counts: dict[str, int], validation_cfg: dict[str, Any]) -> bool:
    min_moneyflow = int(validation_cfg.get("min_moneyflow_rows", 1000))
    return (
        counts.get("moneyflow_daily", 0) >= min_moneyflow
        and counts.get("moneyflow_stock", 0) >= int(validation_cfg.get("min_moneyflow_stock_rows", min_moneyflow))
        and counts.get("limit_events", 0) >= int(validation_cfg.get("min_limit_events", 1))
        and counts.get("limit_market_stats", 0) >= int(validation_cfg.get("min_limit_market_stats_rows", 1))
        and counts.get("lhb_stocks", 0) >= int(validation_cfg.get("min_lhb_stocks", 1))
        and counts.get("moneyflow_market", 0) >= int(validation_cfg.get("min_moneyflow_market_rows", 1))
        and counts.get("moneyflow_industry", 0) >= int(validation_cfg.get("min_moneyflow_industry_rows", 1))
        and counts.get("moneyflow_concept", 0) >= int(validation_cfg.get("min_moneyflow_concept_rows", 1))
    )


def backfill_trading_data(
    settings,
    *,
    start_date: str,
    end_date: str,
    force: bool = False,
    sleep_seconds: float | None = None,
) -> dict[str, Any]:
    fetcher = TushareFetcher(settings)
    if sleep_seconds is not None:
        fetcher.sleep_seconds = float(sleep_seconds)
    validation_cfg = settings.section("validation")
    start_compact = _compact_date(start_date)
    end_compact = _compact_date(end_date)
    if not start_compact or not end_compact:
        raise RuntimeError(f"Invalid backfill date range: {start_date} to {end_date}")
    dates = _open_trade_dates_between(settings, start_date=start_compact, end_date=end_compact)
    if not dates:
        raise RuntimeError(f"No open trade dates found for {start_compact}-{end_compact}")

    min_moneyflow = int(validation_cfg.get("min_moneyflow_rows", 1000))
    min_moneyflow_stock = int(validation_cfg.get("min_moneyflow_stock_rows", min_moneyflow))
    warnings: list[str] = []
    totals: dict[str, Any] = {
        "backfill_trading_start_date": _dash_date(start_compact),
        "backfill_trading_end_date": _dash_date(end_compact),
        "backfill_trading_target_dates": len(dates),
        "backfill_trading_force": force,
        "backfill_trading_sleep_seconds": fetcher.sleep_seconds,
        "backfill_trading_skipped_complete_dates": 0,
        "backfill_trading_completed_dates": 0,
        "moneyflow_rows": 0,
        "moneyflow_stock_rows": 0,
        "limit_events": 0,
        "lhb_rows": 0,
        "lhb_seats": 0,
        "moneyflow_market_rows": 0,
        "moneyflow_industry_rows": 0,
        "moneyflow_concept_rows": 0,
    }

    for index, trade_date in enumerate(dates, start=1):
        counts = _trading_counts_for_date(settings, trade_date)
        if not force and _trading_data_complete(counts, validation_cfg):
            totals["backfill_trading_skipped_complete_dates"] += 1
            totals["backfill_trading_completed_dates"] += 1
            logging.info("trading backfill %s/%s %s skipped complete", index, len(dates), _dash_date(trade_date))
            continue

        logging.info("trading backfill %s/%s %s start counts=%s", index, len(dates), _dash_date(trade_date), counts)
        if counts.get("daily_bars", 0) < int(validation_cfg.get("min_daily_bar_rows", 1000)):
            warnings.append(f"{_dash_date(trade_date)} skipped: daily_bars rows {counts.get('daily_bars', 0)}")
            continue

        if force or counts.get("moneyflow_daily", 0) < min_moneyflow:
            try:
                rows = fetcher.fetch_market_moneyflow([trade_date])
                totals["moneyflow_rows"] += rows
            except Exception as exc:
                warnings.append(f"{_dash_date(trade_date)} moneyflow_daily: {exc}")

        counts = _trading_counts_for_date(settings, trade_date)
        if force or counts.get("moneyflow_stock", 0) < min_moneyflow_stock:
            try:
                rows = sync_moneyflow_stock_from_daily(settings, trade_date)
                totals["moneyflow_stock_rows"] += rows
            except Exception as exc:
                warnings.append(f"{_dash_date(trade_date)} moneyflow_stock: {exc}")

        counts = _trading_counts_for_date(settings, trade_date)
        if force or counts.get("limit_events", 0) < int(validation_cfg.get("min_limit_events", 1)) or counts.get("limit_market_stats", 0) < int(validation_cfg.get("min_limit_market_stats_rows", 1)):
            try:
                metrics = fetch_limit_events(settings, fetcher, trade_date)
                totals["limit_events"] += int(metrics.get("limit_events", 0) or 0)
                warnings.extend(str(x) for x in metrics.get("limit_warnings", []) if x)
            except Exception as exc:
                warnings.append(f"{_dash_date(trade_date)} limit_events: {exc}")

        counts = _trading_counts_for_date(settings, trade_date)
        if force or counts.get("lhb_stocks", 0) < int(validation_cfg.get("min_lhb_stocks", 1)):
            try:
                metrics = fetch_lhb(settings, fetcher, trade_date)
                totals["lhb_rows"] += int(metrics.get("lhb_rows", 0) or 0)
                totals["lhb_seats"] += int(metrics.get("lhb_seats", 0) or 0)
                warnings.extend(str(x) for x in metrics.get("lhb_warnings", []) if x)
            except Exception as exc:
                warnings.append(f"{_dash_date(trade_date)} lhb: {exc}")

        counts = _trading_counts_for_date(settings, trade_date)
        if (
            force
            or counts.get("moneyflow_market", 0) < int(validation_cfg.get("min_moneyflow_market_rows", 1))
            or counts.get("moneyflow_industry", 0) < int(validation_cfg.get("min_moneyflow_industry_rows", 1))
            or counts.get("moneyflow_concept", 0) < int(validation_cfg.get("min_moneyflow_concept_rows", 1))
        ):
            try:
                metrics = fetch_moneyflow_layers(settings, fetcher, trade_date)
                totals["moneyflow_market_rows"] += int(metrics.get("moneyflow_market_rows", 0) or 0)
                totals["moneyflow_industry_rows"] += int(metrics.get("moneyflow_industry_rows", 0) or 0)
                totals["moneyflow_concept_rows"] += int(metrics.get("moneyflow_concept_rows", 0) or 0)
                warnings.extend(str(x) for x in metrics.get("moneyflow_layer_warnings", []) if x)
            except Exception as exc:
                warnings.append(f"{_dash_date(trade_date)} moneyflow_layers: {exc}")

        final_counts = _trading_counts_for_date(settings, trade_date)
        if _trading_data_complete(final_counts, validation_cfg):
            totals["backfill_trading_completed_dates"] += 1
        logging.info("trading backfill %s/%s %s done counts=%s", index, len(dates), _dash_date(trade_date), final_counts)

    totals["backfill_trading_warning_count"] = len(warnings)
    totals["backfill_trading_warning_samples"] = warnings[:20]
    return totals


def _signal_layer_counts_for_date(settings, trade_date: str) -> dict[str, int]:
    date_value = _dash_date(trade_date)
    tables = ["technical_signals", "stock_signal_daily", "theme_signal_daily"]
    with connect() as conn, conn.cursor() as cur:
        return {table: _count_rows_for_date(cur, settings, table, str(date_value)) for table in tables}


def _signal_layers_complete(counts: dict[str, int], validation_cfg: dict[str, Any]) -> bool:
    min_technical = int(validation_cfg.get("min_technical_signal_rows", 1))
    return (
        counts.get("technical_signals", 0) >= min_technical
        and _final_signal_layers_complete(counts, validation_cfg)
    )


def _final_signal_layers_complete(counts: dict[str, int], validation_cfg: dict[str, Any]) -> bool:
    min_daily = int(validation_cfg.get("min_daily_bar_rows", 1000))
    return (
        counts.get("stock_signal_daily", 0) >= min_daily
        and counts.get("theme_signal_daily", 0) >= int(validation_cfg.get("min_theme_signal_rows", 1))
    )


def backfill_signal_layers(
    settings,
    *,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> dict[str, Any]:
    start_compact = _compact_date(start_date)
    end_compact = _compact_date(end_date)
    if not start_compact or not end_compact:
        raise RuntimeError(f"Invalid signal-layer backfill date range: {start_date} to {end_date}")
    dates = _open_trade_dates_between(settings, start_date=start_compact, end_date=end_compact)
    if not dates:
        raise RuntimeError(f"No open trade dates found for {start_compact}-{end_compact}")

    validation_cfg = settings.section("validation")
    members = load_focus_universe(settings)
    universe_count = sync_signal_universe(settings, members)
    warnings: list[str] = []
    totals: dict[str, Any] = {
        "signal_layer_backfill_start_date": _dash_date(start_compact),
        "signal_layer_backfill_end_date": _dash_date(end_compact),
        "signal_layer_target_dates": len(dates),
        "signal_layer_force": force,
        "signal_layer_universe_rows": universe_count,
        "signal_layer_skipped_complete_dates": 0,
        "signal_layer_refreshed_dates": 0,
        "technical_signals_rows": 0,
        "latest_signals_rows": 0,
        "stock_signal_daily_rows": 0,
        "theme_signal_daily_rows": 0,
    }

    for index, trade_date in enumerate(dates, start=1):
        counts = _signal_layer_counts_for_date(settings, trade_date)
        if not force and _signal_layers_complete(counts, validation_cfg):
            totals["signal_layer_skipped_complete_dates"] += 1
            logging.info("signal-layer backfill %s/%s %s skipped complete", index, len(dates), _dash_date(trade_date))
            continue

        try:
            logging.info("signal-layer backfill %s/%s %s start counts=%s", index, len(dates), _dash_date(trade_date), counts)
            metrics = compute_signals(
                settings,
                trade_date=str(_dash_date(trade_date)),
                refresh_final_layers=False,
            )
            stock_metrics: dict[str, Any] = {}
            theme_metrics: dict[str, Any] = {}
            if force or counts.get("stock_signal_daily", 0) < int(validation_cfg.get("min_daily_bar_rows", 1000)):
                stock_metrics = refresh_stock_signal_daily(settings, trade_date)
            if (
                force
                or stock_metrics
                or counts.get("theme_signal_daily", 0) < int(validation_cfg.get("min_theme_signal_rows", 1))
            ):
                theme_metrics = refresh_theme_signal_daily(settings, trade_date)
            metrics = {**metrics, **stock_metrics, **theme_metrics}
            totals["signal_layer_refreshed_dates"] += 1
            totals["technical_signals_rows"] += int(metrics.get("signals", 0) or 0)
            totals["latest_signals_rows"] = int(metrics.get("latest", 0) or totals["latest_signals_rows"])
            totals["stock_signal_daily_rows"] += int(metrics.get("stock_signal_daily_rows", 0) or 0)
            totals["theme_signal_daily_rows"] += int(metrics.get("theme_signal_daily_rows", 0) or 0)
            logging.info("signal-layer backfill %s/%s %s done metrics=%s", index, len(dates), _dash_date(trade_date), metrics)
        except Exception as exc:
            warnings.append(f"{_dash_date(trade_date)}: {type(exc).__name__}: {exc}")
            logging.exception("signal-layer backfill failed for %s", _dash_date(trade_date))

    totals["signal_layer_warning_count"] = len(warnings)
    totals["signal_layer_warning_samples"] = warnings[:20]
    return totals


def _stock_signal_counts_for_date(settings, trade_date: str) -> dict[str, int]:
    date_value = _dash_date(trade_date)
    with connect() as conn, conn.cursor() as cur:
        return {
            "daily_bars": _count_rows_for_date(cur, settings, "daily_bars", str(date_value), "AND adj_close IS NOT NULL"),
            "stock_signal_daily": _count_rows_for_date(cur, settings, "stock_signal_daily", str(date_value)),
        }


def backfill_stock_signals(
    settings,
    *,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> dict[str, Any]:
    start_compact = _compact_date(start_date)
    end_compact = _compact_date(end_date)
    if not start_compact or not end_compact:
        raise RuntimeError(f"Invalid stock-signal backfill date range: {start_date} to {end_date}")
    dates = _open_trade_dates_between(settings, start_date=start_compact, end_date=end_compact)
    if not dates:
        raise RuntimeError(f"No open trade dates found for {start_compact}-{end_compact}")

    warnings: list[str] = []
    totals: dict[str, Any] = {
        "stock_signal_backfill_start_date": _dash_date(start_compact),
        "stock_signal_backfill_end_date": _dash_date(end_compact),
        "stock_signal_target_dates": len(dates),
        "stock_signal_force": force,
        "stock_signal_skipped_complete_dates": 0,
        "stock_signal_skipped_no_bars": 0,
        "stock_signal_refreshed_dates": 0,
        "stock_signal_expected_rows": 0,
        "stock_signal_daily_rows": 0,
    }

    chunks: list[tuple[str, str]] = []
    chunk_start = dates[0]
    previous = dates[0]
    chunk_year = dates[0][:4]
    for trade_date in dates[1:]:
        if trade_date[:4] != chunk_year:
            chunks.append((chunk_start, previous))
            chunk_start = trade_date
            chunk_year = trade_date[:4]
        previous = trade_date
    chunks.append((chunk_start, previous))

    for index, (chunk_start_date, chunk_end_date) in enumerate(chunks, start=1):
        try:
            logging.info("stock-signal range backfill %s/%s %s to %s start", index, len(chunks), _dash_date(chunk_start_date), _dash_date(chunk_end_date))
            metrics = refresh_stock_signal_daily_range(
                settings,
                str(_dash_date(chunk_start_date)),
                str(_dash_date(chunk_end_date)),
                force=force,
            )
            totals["stock_signal_skipped_complete_dates"] += int(metrics.get("stock_signal_skipped_complete_dates", 0) or 0)
            totals["stock_signal_skipped_no_bars"] += int(metrics.get("stock_signal_skipped_no_bars", 0) or 0)
            totals["stock_signal_refreshed_dates"] += int(metrics.get("stock_signal_refreshed_dates", 0) or 0)
            totals["stock_signal_expected_rows"] += int(metrics.get("stock_signal_expected_rows", 0) or 0)
            totals["stock_signal_daily_rows"] += int(metrics.get("stock_signal_daily_rows", 0) or 0)
            logging.info("stock-signal range backfill %s/%s %s to %s done metrics=%s", index, len(chunks), _dash_date(chunk_start_date), _dash_date(chunk_end_date), metrics)
        except Exception as exc:
            warnings.append(f"{_dash_date(chunk_start_date)} to {_dash_date(chunk_end_date)}: {type(exc).__name__}: {exc}")
            logging.exception("stock-signal range backfill failed for %s to %s", _dash_date(chunk_start_date), _dash_date(chunk_end_date))

    totals["stock_signal_warning_count"] = len(warnings)
    totals["stock_signal_warning_samples"] = warnings[:20]
    return totals


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


def run_factor_shadow(settings, *, trade_date: str) -> dict[str, Any]:
    shadow_cfg = settings.section("factor_lab_shadow")
    if shadow_cfg.get("enabled", True) is False:
        return {"factor_shadow_enabled": False}
    from factor_lab.shadow_runner import run_shadow_pipeline

    return run_shadow_pipeline(
        settings,
        trade_date=trade_date,
        production_top_n=int(shadow_cfg.get("production_top_n", 30)),
        tracking_lookback_days=int(shadow_cfg.get("tracking_lookback_days", 30)),
        research_start_date=str(shadow_cfg.get("research_start_date", "2020-01-02")),
    )


def evening_pipeline(settings, *, skip_moneyflow: bool = False) -> dict[str, Any]:
    market_metrics = ensure_market_data_current(settings)
    trading_metrics = update_trading_data(settings, skip_moneyflow=skip_moneyflow)
    process_metrics = process_signals(
        settings,
        trade_date=str(trading_metrics["trade_date"]),
        require_moneyflow=not skip_moneyflow,
        require_current_trade_date=True,
    )
    shadow_metrics = run_factor_shadow(settings, trade_date=str(process_metrics["trade_date"]))
    return {**market_metrics, **trading_metrics, **process_metrics, **shadow_metrics}


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
    shadow_metrics = run_factor_shadow(settings, trade_date=str(process_metrics["trade_date"]))
    return {
        **calendar_metrics,
        **market_metrics,
        **trading_metrics,
        **process_metrics,
        **shadow_metrics,
    }


RETRYABLE_COMMANDS = {
    "update-calendar",
    "update-market-data",
    "update-trading-data",
    "update-data",
    "update-indexes",
    "update-global-indexes",
    "backfill-daily",
    "backfill-trading-data",
    "backfill-market-layers",
    "backfill-signal-layers",
    "backfill-stock-signals",
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
        return update_trading_data(settings, trade_date=args.trade_date, skip_moneyflow=args.skip_moneyflow)
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
    if args.command == "backfill-trading-data":
        if args.year:
            start_date = args.start_date or f"{args.year}0101"
            if args.end_date:
                end_date = args.end_date
            elif args.year >= datetime.now().year:
                end_date = _expected_latest_open_date(settings) or _latest_daily_bar_trade_date(settings)
            else:
                end_date = f"{args.year}1231"
        else:
            start_date = args.start_date
            end_date = args.end_date or _expected_latest_open_date(settings) or _latest_daily_bar_trade_date(settings)
        if not start_date or not end_date:
            raise RuntimeError("backfill-trading-data requires --year or both --start-date and --end-date")
        return backfill_trading_data(
            settings,
            start_date=start_date,
            end_date=end_date,
            force=args.force,
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
            force_dragon_signals=args.force_signals,
        )
    if args.command == "backfill-signal-layers":
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
            raise RuntimeError("backfill-signal-layers requires a valid date range")
        return backfill_signal_layers(
            settings,
            start_date=start_date,
            end_date=end_date,
            force=args.force,
        )
    if args.command == "backfill-stock-signals":
        if args.year:
            start_date = args.start_date or f"{args.year}0101"
            end_date = args.end_date or f"{args.year}1231"
        else:
            start_date = args.start_date
            end_date = args.end_date
        if not start_date or not end_date:
            raise RuntimeError("backfill-stock-signals requires --year or both --start-date and --end-date")
        return backfill_stock_signals(
            settings,
            start_date=start_date,
            end_date=end_date,
            force=args.force,
        )
    if args.command in {"validate-data", "validate"}:
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
    if len(sys.argv) > 1 and sys.argv[1] == "factor-lab":
        from factor_lab.run_factor_lab import main as factor_lab_main

        return factor_lab_main(sys.argv[2:])

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
            "backfill-trading-data",
            "backfill-market-layers",
            "backfill-signal-layers",
            "backfill-stock-signals",
            "validate",
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
    parser.add_argument("--force", action="store_true", help="Refresh backfill data even when the date already looks complete")
    parser.add_argument("--force-signals", action="store_true", help="Regenerate stock/theme signal layers during market-layer backfill")
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
