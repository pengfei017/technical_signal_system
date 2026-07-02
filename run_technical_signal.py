#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from tech_signal.analyzer import compute_signals
from tech_signal.config import load_settings
from tech_signal.db import connect, init_schema, qname
from tech_signal.report import generate_report
from tech_signal.trading_signals import refresh_final_signal_layers, update_trading_auxiliary
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


def update_data(settings, *, days: int | None = None, skip_moneyflow: bool = False) -> dict:
    fetcher = TushareFetcher(settings)
    data_cfg = settings.section("data")
    calendar_start = str(data_cfg.get("trade_calendar_start_date", "2010-01-01"))
    calendar_future_years = int(data_cfg.get("trade_calendar_future_years", 2))
    calendar_end = f"{datetime.now().year + calendar_future_years}1231"
    history_days = int(data_cfg.get("all_a_lookback_trading_days", 90))
    fetch_recent_days = int(data_cfg.get("daily_fetch_recent_trading_days", data_cfg.get("refresh_recent_trading_days", 5)))
    existing_bar_dates = fetcher.existing_daily_bar_date_count()
    bootstrap_backfill = days is not None or existing_bar_dates < history_days
    fetch_days = int(days or history_days) if bootstrap_backfill else fetch_recent_days
    # Add one open-day buffer so a current intraday date with no settled daily data
    # does not crowd out the latest completed trading day.
    dates = fetcher.open_trade_dates(fetch_days + 1)
    if not dates:
        raise RuntimeError("No open trade dates found")
    calendar_metrics = fetcher.sync_trade_calendar_range(calendar_start, calendar_end)
    market_metrics = fetcher.fetch_market_for_dates(dates)
    effective_trade_date = market_metrics.get("latest_trade_date") or dates[-1]
    members = load_focus_universe(settings)
    universe_count = sync_signal_universe(settings, members)
    moneyflow_count = 0
    moneyflow_scope = str(data_cfg.get("moneyflow_scope", "all_market"))
    if not skip_moneyflow and moneyflow_scope == "all_market":
        mf_days = int(
            data_cfg.get(
                "stock_moneyflow_fetch_recent_trading_days",
                data_cfg.get("focus_moneyflow_fetch_recent_trading_days", fetch_recent_days),
            )
        )
        moneyflow_dates = dates[-mf_days:]
        moneyflow_count = fetcher.fetch_market_moneyflow(moneyflow_dates)
    elif not skip_moneyflow and moneyflow_scope == "focus":
        mf_days = int(
            data_cfg.get(
                "focus_moneyflow_fetch_recent_trading_days",
                data_cfg.get("focus_moneyflow_lookback_trading_days", fetch_recent_days),
            )
        )
        moneyflow_dates = dates[-mf_days:]
        moneyflow_count = fetcher.fetch_focus_moneyflow([m.ts_code for m in members], moneyflow_dates)
    auxiliary_metrics = update_trading_auxiliary(settings, fetcher, str(effective_trade_date))
    return {
        "trade_date": effective_trade_date,
        "target_lookback_trading_days": history_days,
        "fetch_window_trading_days": fetch_days,
        "existing_daily_bar_dates": existing_bar_dates,
        "bootstrap_backfill": bootstrap_backfill,
        "moneyflow_scope": moneyflow_scope,
        **calendar_metrics,
        "universe_rows": universe_count,
        "moneyflow_rows": moneyflow_count,
        **auxiliary_metrics,
        **market_metrics,
    }


def run_all(settings, args) -> dict:
    init_schema(settings)
    data_metrics = update_data(settings, days=args.days, skip_moneyflow=args.skip_moneyflow)
    signal_metrics = compute_signals(settings, trade_date=None)
    final_signal_metrics = refresh_final_signal_layers(settings, trade_date=str(signal_metrics["trade_date"]))
    report_path = generate_report(settings, trade_date=str(signal_metrics["trade_date"]))
    return {
        **data_metrics,
        **signal_metrics,
        **final_signal_metrics,
        "report_file": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent A-share technical signal system")
    parser.add_argument("command", choices=["init-db", "update-data", "analyze", "report", "run"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--days", type=int, default=None, help="Backfill trading days for all-A daily data")
    parser.add_argument("--trade-date", default=None, help="YYYY-MM-DD, default latest")
    parser.add_argument("--skip-moneyflow", action="store_true")
    args = parser.parse_args()

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
        run_id = start_run(settings, args.command)
        if args.command == "update-data":
            metrics = update_data(settings, days=args.days, skip_moneyflow=args.skip_moneyflow)
        elif args.command == "analyze":
            members = load_focus_universe(settings)
            sync_signal_universe(settings, members)
            metrics = compute_signals(settings, trade_date=args.trade_date)
            metrics.update(refresh_final_signal_layers(settings, trade_date=str(metrics["trade_date"])))
        elif args.command == "report":
            path = generate_report(settings, trade_date=args.trade_date)
            metrics = {"report_file": str(path)}
        else:
            metrics = run_all(settings, args)
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
