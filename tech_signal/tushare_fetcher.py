from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd
import tushare as ts

from .config import Settings
from .db import connect, qname, upsert_rows
from .secrets import get_secret


LOGGER = logging.getLogger(__name__)


def _date_text(value: object) -> str:
    return str(value or "").replace("-", "")[:8]


def _to_date(value: object) -> str | None:
    text = _date_text(value)
    if len(text) != 8:
        return None
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _num(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


class TushareFetcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        token = get_secret("TUSHARE_TOKEN", os.environ.get("TUSHARE_TOKEN", ""))
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is missing")
        ts.set_token(token)
        self.pro = ts.pro_api(token)
        data_cfg = settings.section("data")
        self.sleep_seconds = float(data_cfg.get("request_sleep_seconds", 0.25))
        self.retry_count = int(data_cfg.get("retry_count", 3))
        self.retry_sleep_seconds = float(data_cfg.get("retry_sleep_seconds", 2.0))

    def _existing_bar_count(self, trade_date: str) -> int:
        date_value = _to_date(trade_date)
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) AS n FROM {qname(self.settings, 'daily_bars')} WHERE trade_date=%s",
                (date_value,),
            )
            row = cur.fetchone()
            return int(row["n"] if row else 0)

    def existing_daily_bar_date_count(self) -> int:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(DISTINCT trade_date) AS n FROM {qname(self.settings, 'daily_bars')}")
            row = cur.fetchone()
            return int(row["n"] if row else 0)

    def _call(self, label: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_count + 1):
            try:
                df = fn()
                time.sleep(self.sleep_seconds)
                return df if df is not None else pd.DataFrame()
            except Exception as exc:
                last_error = exc
                LOGGER.warning("%s failed attempt %s/%s: %s", label, attempt, self.retry_count, exc)
                time.sleep(self.retry_sleep_seconds * attempt)
        raise RuntimeError(f"{label} failed after {self.retry_count} attempts: {last_error}")

    def open_trade_dates(self, lookback: int) -> list[str]:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=max(lookback * 3, 180))).strftime("%Y%m%d")
        df = self._call(
            "trade_cal",
            lambda: self.pro.trade_cal(exchange="SSE", start_date=start, end_date=end),
        )
        if df.empty:
            return []
        rows = df[df["is_open"].astype(str) == "1"].sort_values("cal_date")
        dates = [str(x) for x in rows["cal_date"].tolist()]
        return dates[-lookback:]

    def sync_trade_calendar(self, dates: list[str]) -> int:
        if not dates:
            return 0
        start, end = dates[0], dates[-1]
        df = self._call(
            "trade_cal_sync",
            lambda: self.pro.trade_cal(exchange="SSE", start_date=start, end_date=end),
        )
        rows = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "cal_date": _to_date(row.get("cal_date")),
                    "is_open": str(row.get("is_open")) == "1",
                    "pretrade_date": _to_date(row.get("pretrade_date")),
                }
            )
        with connect() as conn:
            count = upsert_rows(
                conn,
                table=qname(self.settings, "trade_calendar"),
                columns=["cal_date", "is_open", "pretrade_date"],
                rows=rows,
                conflict_columns=["cal_date"],
            )
            conn.commit()
        return count

    def fetch_market_for_dates(self, dates: list[str]) -> dict[str, int | str]:
        metrics: dict[str, int | str] = {
            "daily_bars": 0,
            "daily_basic": 0,
            "dates": 0,
            "empty_dates": 0,
            "latest_trade_date": "",
            "adjusted_price_rows": 0,
        }
        refresh_recent = int(self.settings.section("data").get("refresh_recent_trading_days", 3))
        refresh_set = set(dates[-refresh_recent:]) if refresh_recent > 0 else set()
        for trade_date in dates:
            existing = self._existing_bar_count(trade_date)
            if existing > 1000 and trade_date not in refresh_set:
                metrics["dates"] += 1
                metrics["latest_trade_date"] = trade_date
                LOGGER.info("skip existing all-A daily data for %s rows=%s", trade_date, existing)
                continue
            LOGGER.info("fetch all-A daily data for %s", trade_date)
            daily = self._call("daily", lambda d=trade_date: self.pro.daily(trade_date=d))
            adj = self._call("adj_factor", lambda d=trade_date: self.pro.adj_factor(trade_date=d))
            basic = self._call("daily_basic", lambda d=trade_date: self.pro.daily_basic(trade_date=d))
            bars_count = self._write_daily_bars(daily, adj)
            basic_count = self._write_daily_basic(basic)
            metrics["daily_bars"] += bars_count
            metrics["daily_basic"] += basic_count
            if bars_count:
                metrics["dates"] += 1
                metrics["latest_trade_date"] = trade_date
            else:
                metrics["empty_dates"] += 1
            LOGGER.info("date %s saved bars=%s basic=%s", trade_date, bars_count, basic_count)
        metrics["adjusted_price_rows"] = self.refresh_adjusted_prices()
        return metrics

    def _write_daily_bars(self, daily: pd.DataFrame, adj: pd.DataFrame) -> int:
        if daily.empty:
            return 0
        if not adj.empty:
            daily = daily.merge(adj[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
        else:
            daily["adj_factor"] = None
        rows = []
        for _, row in daily.iterrows():
            rows.append(
                {
                    "ts_code": row.get("ts_code"),
                    "trade_date": _to_date(row.get("trade_date")),
                    "open": _num(row.get("open")),
                    "high": _num(row.get("high")),
                    "low": _num(row.get("low")),
                    "close": _num(row.get("close")),
                    "pre_close": _num(row.get("pre_close")),
                    "change": _num(row.get("change")),
                    "pct_chg": _num(row.get("pct_chg")),
                    "vol": _num(row.get("vol")),
                    "amount": _num(row.get("amount")),
                    "adj_factor": _num(row.get("adj_factor")),
                }
            )
        columns = [
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
            "adj_factor",
        ]
        with connect() as conn:
            count = upsert_rows(
                conn,
                table=qname(self.settings, "daily_bars"),
                columns=columns,
                rows=rows,
                conflict_columns=["ts_code", "trade_date"],
            )
            conn.commit()
        return count

    def _write_daily_basic(self, basic: pd.DataFrame) -> int:
        if basic.empty:
            return 0
        columns = [
            "ts_code",
            "trade_date",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "dv_ratio",
            "dv_ttm",
            "total_share",
            "float_share",
            "free_share",
            "total_mv",
            "circ_mv",
        ]
        rows = []
        for _, row in basic.iterrows():
            out = {"ts_code": row.get("ts_code"), "trade_date": _to_date(row.get("trade_date"))}
            for col in columns[2:]:
                out[col] = _num(row.get(col))
            rows.append(out)
        with connect() as conn:
            count = upsert_rows(
                conn,
                table=qname(self.settings, "daily_basic"),
                columns=columns,
                rows=rows,
                conflict_columns=["ts_code", "trade_date"],
            )
            conn.commit()
        return count

    def refresh_adjusted_prices(self, dates: list[str] | None = None) -> int:
        start = _to_date(dates[0]) if dates else None
        end = _to_date(dates[-1]) if dates else None
        range_sql = ""
        params: tuple[str | None, ...] = ()
        if start and end:
            range_sql = "AND b.trade_date BETWEEN %s AND %s"
            params = (start, end)
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                WITH latest_factor AS (
                    SELECT DISTINCT ON (ts_code) ts_code, adj_factor AS latest_adj_factor
                    FROM {qname(self.settings, 'daily_bars')}
                    WHERE adj_factor IS NOT NULL
                    ORDER BY ts_code, trade_date DESC
                )
                UPDATE {qname(self.settings, 'daily_bars')} b
                SET adj_open = CASE WHEN l.latest_adj_factor IS NULL OR l.latest_adj_factor = 0 THEN b.open ELSE b.open * b.adj_factor / l.latest_adj_factor END,
                    adj_high = CASE WHEN l.latest_adj_factor IS NULL OR l.latest_adj_factor = 0 THEN b.high ELSE b.high * b.adj_factor / l.latest_adj_factor END,
                    adj_low = CASE WHEN l.latest_adj_factor IS NULL OR l.latest_adj_factor = 0 THEN b.low ELSE b.low * b.adj_factor / l.latest_adj_factor END,
                    adj_close = CASE WHEN l.latest_adj_factor IS NULL OR l.latest_adj_factor = 0 THEN b.close ELSE b.close * b.adj_factor / l.latest_adj_factor END
                FROM latest_factor l
                WHERE b.ts_code = l.ts_code
                  {range_sql}
                """,
                params,
            )
            count = cur.rowcount
            conn.commit()
            return int(count or 0)

    def fetch_focus_moneyflow(self, ts_codes: list[str], dates: list[str]) -> int:
        if not ts_codes or not dates:
            return 0
        start, end = dates[0], dates[-1]
        total = 0
        for ts_code in sorted(set(ts_codes)):
            df = self._call(
                f"moneyflow:{ts_code}",
                lambda code=ts_code: self.pro.moneyflow(ts_code=code, start_date=start, end_date=end),
            )
            if df.empty:
                continue
            total += self._write_moneyflow(df)
        return total

    def _write_moneyflow(self, df: pd.DataFrame) -> int:
        rows = []
        for _, row in df.iterrows():
            buy = sum(_num(row.get(col)) or 0.0 for col in ["buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount"])
            sell = sum(_num(row.get(col)) or 0.0 for col in ["sell_sm_amount", "sell_md_amount", "sell_lg_amount", "sell_elg_amount"])
            rows.append(
                {
                    "ts_code": row.get("ts_code"),
                    "trade_date": _to_date(row.get("trade_date")),
                    "buy_sm_amount": _num(row.get("buy_sm_amount")),
                    "sell_sm_amount": _num(row.get("sell_sm_amount")),
                    "buy_md_amount": _num(row.get("buy_md_amount")),
                    "sell_md_amount": _num(row.get("sell_md_amount")),
                    "buy_lg_amount": _num(row.get("buy_lg_amount")),
                    "sell_lg_amount": _num(row.get("sell_lg_amount")),
                    "buy_elg_amount": _num(row.get("buy_elg_amount")),
                    "sell_elg_amount": _num(row.get("sell_elg_amount")),
                    "net_mf_amount": buy - sell,
                }
            )
        columns = [
            "ts_code",
            "trade_date",
            "buy_sm_amount",
            "sell_sm_amount",
            "buy_md_amount",
            "sell_md_amount",
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
            "net_mf_amount",
        ]
        with connect() as conn:
            count = upsert_rows(
                conn,
                table=qname(self.settings, "moneyflow_daily"),
                columns=columns,
                rows=rows,
                conflict_columns=["ts_code", "trade_date"],
            )
            conn.commit()
        return count
