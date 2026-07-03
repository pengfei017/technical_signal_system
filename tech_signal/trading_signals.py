from __future__ import annotations

import json
import logging
import math
from typing import Any

import pandas as pd

from .config import Settings
from .db import connect, qname, upsert_rows
from .formula_spec import formula_section, load_formula_spec, merged_signal_config
from .indicators import add_indicators
from .tushare_fetcher import TushareFetcher


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


def _clean(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return _clean(value.item())
        except Exception:
            return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value: object) -> str:
    return json.dumps(_clean(value), ensure_ascii=False, default=str, allow_nan=False)


def _yuan_to_yi(value: object) -> float | None:
    n = _num(value)
    return None if n is None else round(n / 100000000.0, 6)


def _wan_to_yi(value: object) -> float | None:
    n = _num(value)
    return None if n is None else round(n / 10000.0, 6)


def _thousand_yuan_to_yi(value: object) -> float | None:
    n = _num(value)
    return None if n is None else round(n / 100000.0, 6)


def _auto_amount_to_yi(value: object) -> float | None:
    n = _num(value)
    if n is None:
        return None
    # Tushare daily amount is usually thousand yuan, while LHB/limit amount
    # fields are often yuan. Use magnitude to keep both common cases sane.
    if abs(n) >= 100000000:
        return round(n / 100000000.0, 6)
    return round(n / 100000.0, 6)


def _bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _classify_seat(exalter: str) -> str:
    name = str(exalter or "")
    if "机构专用" in name:
        return "institution"
    if "沪股通" in name or "深股通" in name or "陆股通" in name:
        return "northbound"
    return "broker_seat"


def _safe_call(fetcher: TushareFetcher, label: str, fn) -> tuple[pd.DataFrame, str]:
    try:
        return fetcher._call(label, fn), ""
    except Exception as exc:
        LOGGER.warning("%s failed: %s", label, exc)
        return pd.DataFrame(), f"{label} failed: {type(exc).__name__}: {exc}"


def _history_start_date(settings: Settings, trade_date: str, trading_days: int) -> str | None:
    date_value = _to_date(trade_date)
    if not date_value or trading_days <= 0:
        return None
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT cal_date
            FROM {qname(settings, 'trade_calendar')}
            WHERE is_open=true AND cal_date <= %s
            ORDER BY cal_date DESC
            OFFSET %s
            LIMIT 1
            """,
            (date_value, max(0, trading_days - 1)),
        )
        row = cur.fetchone()
        return str(row["cal_date"]) if row and row["cal_date"] else None


def _stock_meta() -> dict[str, dict[str, Any]]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.stock_master') AS table_name")
        if not cur.fetchone()["table_name"]:
            return {}
        cur.execute("SELECT ts_code, name, industry, concepts FROM public.stock_master")
        return {str(row["ts_code"]): dict(row) for row in cur.fetchall()}


def sync_moneyflow_stock_from_daily(settings: Settings, trade_date: str | None = None) -> int:
    date_value = _to_date(trade_date) if trade_date else None
    where_sql = "WHERE mf.trade_date=%s" if date_value else ""
    params: tuple[Any, ...] = (date_value,) if date_value else ()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {qname(settings, 'moneyflow_stock')} (
                ts_code, trade_date, name, close, pct_chg, amount_yi, turnover_rate,
                buy_sm_amount, sell_sm_amount, buy_md_amount, sell_md_amount,
                buy_lg_amount, sell_lg_amount, buy_elg_amount, sell_elg_amount,
                net_mf_amount, net_mf_amount_yi, net_mf_rate, source, updated_at
            )
            SELECT
                mf.ts_code,
                mf.trade_date,
                COALESCE(sm.name, mf.ts_code),
                b.close,
                b.pct_chg,
                b.amount / 100000.0,
                db.turnover_rate,
                mf.buy_sm_amount,
                mf.sell_sm_amount,
                mf.buy_md_amount,
                mf.sell_md_amount,
                mf.buy_lg_amount,
                mf.sell_lg_amount,
                mf.buy_elg_amount,
                mf.sell_elg_amount,
                mf.net_mf_amount,
                mf.net_mf_amount / 10000.0,
                CASE WHEN b.amount IS NOT NULL AND b.amount <> 0 THEN mf.net_mf_amount * 1000.0 / b.amount ELSE NULL END,
                'tushare.moneyflow',
                now()
            FROM {qname(settings, 'moneyflow_daily')} mf
            LEFT JOIN {qname(settings, 'daily_bars')} b
              ON b.ts_code=mf.ts_code AND b.trade_date=mf.trade_date
            LEFT JOIN {qname(settings, 'daily_basic')} db
              ON db.ts_code=mf.ts_code AND db.trade_date=mf.trade_date
            LEFT JOIN public.stock_master sm
              ON sm.ts_code=mf.ts_code
            {where_sql}
            ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                name=EXCLUDED.name,
                close=EXCLUDED.close,
                pct_chg=EXCLUDED.pct_chg,
                amount_yi=EXCLUDED.amount_yi,
                turnover_rate=EXCLUDED.turnover_rate,
                buy_sm_amount=EXCLUDED.buy_sm_amount,
                sell_sm_amount=EXCLUDED.sell_sm_amount,
                buy_md_amount=EXCLUDED.buy_md_amount,
                sell_md_amount=EXCLUDED.sell_md_amount,
                buy_lg_amount=EXCLUDED.buy_lg_amount,
                sell_lg_amount=EXCLUDED.sell_lg_amount,
                buy_elg_amount=EXCLUDED.buy_elg_amount,
                sell_elg_amount=EXCLUDED.sell_elg_amount,
                net_mf_amount=EXCLUDED.net_mf_amount,
                net_mf_amount_yi=EXCLUDED.net_mf_amount_yi,
                net_mf_rate=EXCLUDED.net_mf_rate,
                source=EXCLUDED.source,
                updated_at=now()
            """,
            params,
        )
        count = int(cur.rowcount or 0)
        conn.commit()
        return count


def fetch_lhb(settings: Settings, fetcher: TushareFetcher, trade_date: str) -> dict[str, Any]:
    tushare_date = _date_text(trade_date)
    date_value = _to_date(tushare_date)
    if not date_value:
        return {"lhb_rows": 0, "lhb_seats": 0, "lhb_warnings": ["invalid_trade_date"]}

    top_df, top_error = _safe_call(fetcher, "top_list", lambda: fetcher.pro.top_list(trade_date=tushare_date))
    inst_df, inst_error = _safe_call(fetcher, "top_inst", lambda: fetcher.pro.top_inst(trade_date=tushare_date))
    warnings = [x for x in [top_error, inst_error] if x]

    top_records = top_df.to_dict("records") if not top_df.empty else []
    inst_records = inst_df.to_dict("records") if not inst_df.empty else []

    by_code: dict[str, dict[str, Any]] = {}
    for raw in top_records:
        code = str(raw.get("ts_code") or "").strip()
        if not code:
            continue
        item = by_code.setdefault(
            code,
            {
                "trade_date": date_value,
                "ts_code": code,
                "name": str(raw.get("name") or "").strip(),
                "close": _num(raw.get("close")),
                "pct_change": _num(raw.get("pct_change")),
                "turnover_rate": _num(raw.get("turnover_rate")),
                "amount_yi": 0.0,
                "lhb_amount_yi": 0.0,
                "lhb_net_buy_yi": 0.0,
                "net_rate": 0.0,
                "amount_rate": 0.0,
                "institution_net_buy_yi": 0.0,
                "northbound_net_buy_yi": 0.0,
                "broker_seat_net_buy_yi": 0.0,
                "top_count": 0,
                "primary_reason": "",
                "reasons": [],
                "top_seats": [],
                "_primary_abs_net": -1.0,
            },
        )
        item["top_count"] += 1
        reason = str(raw.get("reason") or "").strip()
        if reason and reason not in item["reasons"]:
            item["reasons"].append(reason)
        net_yi = _yuan_to_yi(raw.get("net_amount")) or 0.0
        if abs(net_yi) > float(item.get("_primary_abs_net") or -1):
            item["_primary_abs_net"] = abs(net_yi)
            item["primary_reason"] = reason
            item["amount_yi"] = _yuan_to_yi(raw.get("amount")) or 0.0
            item["lhb_amount_yi"] = _yuan_to_yi(raw.get("l_amount")) or 0.0
            item["lhb_net_buy_yi"] = net_yi
            item["net_rate"] = _num(raw.get("net_rate")) or 0.0
            item["amount_rate"] = _num(raw.get("amount_rate")) or 0.0

    seat_totals: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in inst_records:
        code = str(raw.get("ts_code") or "").strip()
        if not code:
            continue
        item = by_code.setdefault(
            code,
            {
                "trade_date": date_value,
                "ts_code": code,
                "name": str(raw.get("name") or "").strip(),
                "close": None,
                "pct_change": None,
                "turnover_rate": None,
                "amount_yi": 0.0,
                "lhb_amount_yi": 0.0,
                "lhb_net_buy_yi": 0.0,
                "net_rate": 0.0,
                "amount_rate": 0.0,
                "institution_net_buy_yi": 0.0,
                "northbound_net_buy_yi": 0.0,
                "broker_seat_net_buy_yi": 0.0,
                "top_count": 0,
                "primary_reason": "",
                "reasons": [],
                "top_seats": [],
                "_primary_abs_net": -1.0,
            },
        )
        reason = str(raw.get("reason") or "").strip()
        if reason and reason not in item["reasons"]:
            item["reasons"].append(reason)
        exalter = str(raw.get("exalter") or "").strip()
        if not exalter:
            continue
        seat_type = _classify_seat(exalter)
        buy_yi = _yuan_to_yi(raw.get("buy")) or 0.0
        sell_yi = _yuan_to_yi(raw.get("sell")) or 0.0
        net_yi = _yuan_to_yi(raw.get("net_buy")) or (buy_yi - sell_yi)
        item[f"{seat_type}_net_buy_yi"] += net_yi
        key = (code, exalter, reason)
        seat = seat_totals.setdefault(
            key,
            {
                "trade_date": date_value,
                "ts_code": code,
                "name": item.get("name"),
                "exalter": exalter,
                "seat_type": seat_type,
                "buy_yi": 0.0,
                "sell_yi": 0.0,
                "net_buy_yi": 0.0,
                "reason": reason,
            },
        )
        seat["buy_yi"] += buy_yi
        seat["sell_yi"] += sell_yi
        seat["net_buy_yi"] += net_yi

    for seat in seat_totals.values():
        code = str(seat["ts_code"])
        if code in by_code:
            by_code[code]["top_seats"].append(
                {
                    "exalter": seat["exalter"],
                    "seat_type": seat["seat_type"],
                    "net_buy_yi": round(float(seat["net_buy_yi"] or 0), 4),
                }
            )

    stock_rows = []
    for item in by_code.values():
        item.pop("_primary_abs_net", None)
        item["top_seats"] = sorted(
            item["top_seats"],
            key=lambda x: abs(float(x.get("net_buy_yi") or 0)),
            reverse=True,
        )[:8]
        item["reasons"] = _json(item.get("reasons") or [])
        item["top_seats"] = _json(item.get("top_seats") or [])
        stock_rows.append(item)

    stock_cols = [
        "trade_date",
        "ts_code",
        "name",
        "close",
        "pct_change",
        "turnover_rate",
        "amount_yi",
        "lhb_amount_yi",
        "lhb_net_buy_yi",
        "net_rate",
        "amount_rate",
        "institution_net_buy_yi",
        "northbound_net_buy_yi",
        "broker_seat_net_buy_yi",
        "top_count",
        "primary_reason",
        "reasons",
        "top_seats",
    ]
    seat_cols = [
        "trade_date",
        "ts_code",
        "name",
        "exalter",
        "seat_type",
        "buy_yi",
        "sell_yi",
        "net_buy_yi",
        "reason",
    ]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {qname(settings, 'lhb_stocks')} WHERE trade_date=%s", (date_value,))
            cur.execute(f"DELETE FROM {qname(settings, 'lhb_seats')} WHERE trade_date=%s", (date_value,))
        stock_count = upsert_rows(
            conn,
            table=qname(settings, "lhb_stocks"),
            columns=stock_cols,
            rows=stock_rows,
            conflict_columns=["trade_date", "ts_code"],
        )
        seat_count = upsert_rows(
            conn,
            table=qname(settings, "lhb_seats"),
            columns=seat_cols,
            rows=list(seat_totals.values()),
            conflict_columns=["trade_date", "ts_code", "exalter", "reason"],
        )
        conn.commit()
    return {
        "lhb_rows": stock_count,
        "lhb_seats": seat_count,
        "lhb_top_list_rows": len(top_records),
        "lhb_top_inst_rows": len(inst_records),
        "lhb_warnings": warnings,
    }


def fetch_limit_events(settings: Settings, fetcher: TushareFetcher, trade_date: str) -> dict[str, Any]:
    tushare_date = _date_text(trade_date)
    date_value = _to_date(tushare_date)
    if not date_value:
        return {"limit_events": 0, "limit_warnings": ["invalid_trade_date"]}

    frames: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for limit_type in ["U", "D", "Z"]:
        df, error = _safe_call(
            fetcher,
            f"limit_list_d:{limit_type}",
            lambda lt=limit_type: fetcher.pro.limit_list_d(trade_date=tushare_date, limit_type=lt),
        )
        frames[limit_type] = df
        if error:
            warnings.append(error)

    rows: list[dict[str, Any]] = []
    for limit_type, df in frames.items():
        if df.empty:
            continue
        for _, raw in df.iterrows():
            row = raw.to_dict()
            rows.append(
                {
                    "trade_date": date_value,
                    "ts_code": row.get("ts_code"),
                    "name": row.get("name"),
                    "industry": row.get("industry"),
                    "limit_type": limit_type,
                    "close": _num(row.get("close")),
                    "pct_chg": _num(row.get("pct_chg")),
                    "amount_yi": _auto_amount_to_yi(row.get("amount")),
                    "turnover_rate": _num(row.get("turnover_rate")),
                    "fd_amount_yi": _auto_amount_to_yi(row.get("fd_amount")),
                    "first_limit_time": str(row.get("first_time") or row.get("first_limit_time") or ""),
                    "last_limit_time": str(row.get("last_time") or row.get("last_limit_time") or ""),
                    "open_times": _num(row.get("open_times")),
                    "limit_times": _num(row.get("limit_times")),
                    "source": "tushare.limit_list_d",
                    "raw": _json(row),
                }
            )

    stats = _build_limit_stats(settings, date_value, frames, warnings)
    columns = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "limit_type",
        "close",
        "pct_chg",
        "amount_yi",
        "turnover_rate",
        "fd_amount_yi",
        "first_limit_time",
        "last_limit_time",
        "open_times",
        "limit_times",
        "source",
        "raw",
    ]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {qname(settings, 'limit_events')} WHERE trade_date=%s", (date_value,))
        count = upsert_rows(
            conn,
            table=qname(settings, "limit_events"),
            columns=columns,
            rows=rows,
            conflict_columns=["trade_date", "ts_code", "limit_type"],
        )
        upsert_rows(
            conn,
            table=qname(settings, "limit_market_stats"),
            columns=[
                "trade_date",
                "limit_up_count",
                "limit_down_count",
                "broken_count",
                "broken_rate",
                "max_board",
                "limit_up_industry_distribution",
                "previous_limit_positive",
                "previous_limit_negative",
                "source",
                "warnings",
            ],
            rows=[stats],
            conflict_columns=["trade_date"],
        )
        conn.commit()
    return {
        "limit_events": count,
        "limit_up_count": stats["limit_up_count"],
        "limit_down_count": stats["limit_down_count"],
        "broken_count": stats["broken_count"],
        "limit_warnings": warnings,
    }


def _build_limit_stats(
    settings: Settings,
    trade_date: str,
    frames: dict[str, pd.DataFrame],
    warnings: list[str],
) -> dict[str, Any]:
    up = frames.get("U", pd.DataFrame())
    down = frames.get("D", pd.DataFrame())
    broken = frames.get("Z", pd.DataFrame())
    limit_up_count = int(len(up))
    broken_count = int(len(broken))
    denom = limit_up_count + broken_count
    max_board = 0
    if not up.empty and "limit_times" in up.columns:
        max_board = int(pd.to_numeric(up["limit_times"], errors="coerce").fillna(0).max())
    distribution: list[dict[str, Any]] = []
    if not up.empty and "industry" in up.columns:
        distribution = [
            {"industry": str(k), "count": int(v)}
            for k, v in up["industry"].dropna().astype(str).value_counts().head(20).items()
        ]
    previous_positive, previous_negative = _previous_limit_feedback(settings, trade_date)
    if limit_up_count == 0 and int(len(down)) == 0 and broken_count == 0:
        warnings.append("limit_list_d returned all zero rows")
    return {
        "trade_date": trade_date,
        "limit_up_count": limit_up_count,
        "limit_down_count": int(len(down)),
        "broken_count": broken_count,
        "broken_rate": round(broken_count / denom * 100.0, 2) if denom else None,
        "max_board": max_board,
        "limit_up_industry_distribution": _json(distribution),
        "previous_limit_positive": _json(previous_positive),
        "previous_limit_negative": _json(previous_negative),
        "source": "tushare.limit_list_d",
        "warnings": _json(warnings),
    }


def _previous_limit_feedback(settings: Settings, trade_date: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT max(cal_date) AS d
            FROM {qname(settings, 'trade_calendar')}
            WHERE is_open=true AND cal_date < %s
            """,
            (trade_date,),
        )
        row = cur.fetchone()
        prev_date = row["d"] if row and row["d"] else None
        if not prev_date:
            return [], []
        cur.execute(
            f"""
            SELECT e.ts_code, e.name, e.industry, e.limit_times, b.pct_chg
            FROM {qname(settings, 'limit_events')} e
            LEFT JOIN {qname(settings, 'daily_bars')} b
              ON b.ts_code=e.ts_code AND b.trade_date=%s
            WHERE e.trade_date=%s AND e.limit_type='U'
              AND b.pct_chg IS NOT NULL
            """,
            (trade_date, prev_date),
        )
        rows = [dict(r) for r in cur.fetchall()]
    rows.sort(key=lambda x: float(x.get("pct_chg") or 0), reverse=True)
    positive = rows[:5]
    negative = list(reversed(rows[-5:])) if len(rows) > 5 else []
    return positive, negative


def fetch_moneyflow_layers(settings: Settings, fetcher: TushareFetcher, trade_date: str) -> dict[str, Any]:
    tushare_date = _date_text(trade_date)
    date_value = _to_date(tushare_date)
    if not date_value:
        return {"moneyflow_layer_warnings": ["invalid_trade_date"]}

    metrics: dict[str, Any] = {}
    warnings: list[str] = []
    market_df, error = _safe_call(
        fetcher,
        "moneyflow_mkt_dc",
        lambda: fetcher.pro.moneyflow_mkt_dc(start_date=tushare_date, end_date=tushare_date),
    )
    if error:
        warnings.append(error)
    if not market_df.empty:
        row = market_df.iloc[0].to_dict()
        with connect() as conn:
            upsert_rows(
                conn,
                table=qname(settings, "moneyflow_market"),
                columns=[
                    "trade_date",
                    "pct_change_sh",
                    "pct_change_sz",
                    "net_amount_yi",
                    "net_amount_rate",
                    "buy_elg_amount_yi",
                    "buy_lg_amount_yi",
                    "buy_md_amount_yi",
                    "buy_sm_amount_yi",
                    "source",
                    "raw",
                ],
                rows=[
                    {
                        "trade_date": _to_date(row.get("trade_date")) or date_value,
                        "pct_change_sh": _num(row.get("pct_change_sh")),
                        "pct_change_sz": _num(row.get("pct_change_sz")),
                        "net_amount_yi": _yuan_to_yi(row.get("net_amount")),
                        "net_amount_rate": _num(row.get("net_amount_rate")),
                        "buy_elg_amount_yi": _yuan_to_yi(row.get("buy_elg_amount")),
                        "buy_lg_amount_yi": _yuan_to_yi(row.get("buy_lg_amount")),
                        "buy_md_amount_yi": _yuan_to_yi(row.get("buy_md_amount")),
                        "buy_sm_amount_yi": _yuan_to_yi(row.get("buy_sm_amount")),
                        "source": "tushare.moneyflow_mkt_dc",
                        "raw": _json(row),
                    }
                ],
                conflict_columns=["trade_date"],
            )
            conn.commit()
        metrics["moneyflow_market_rows"] = 1
    else:
        metrics["moneyflow_market_rows"] = 0

    industry_count = 0
    for api_name, source, amount_unit in [
        ("moneyflow_ind_ths", "tushare.moneyflow_ind_ths", "yi"),
        ("moneyflow_ind_dc", "tushare.moneyflow_ind_dc", "yuan"),
    ]:
        df, error = _safe_call(fetcher, api_name, lambda name=api_name: getattr(fetcher.pro, name)(trade_date=tushare_date))
        if error:
            warnings.append(error)
        rows = []
        if not df.empty:
            for _, raw_row in df.iterrows():
                row = raw_row.to_dict()
                name = str(row.get("industry") or row.get("name") or row.get("content") or "").strip()
                if not name:
                    continue
                to_yi = _yuan_to_yi if amount_unit == "yuan" else _num
                rows.append(
                    {
                        "trade_date": _to_date(row.get("trade_date")) or date_value,
                        "theme_name": name,
                        "source": source,
                        "pct_chg": _num(row.get("pct_change") or row.get("pct_chg")),
                        "net_amount_yi": to_yi(row.get("net_amount")),
                        "net_buy_amount_yi": to_yi(row.get("net_buy_amount")),
                        "net_sell_amount_yi": to_yi(row.get("net_sell_amount")),
                        "lead_stock": row.get("lead_stock"),
                        "rank": _num(row.get("rank")),
                        "raw": _json(row),
                    }
                )
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {qname(settings, 'moneyflow_industry')} WHERE trade_date=%s AND source=%s",
                    (date_value, source),
                )
            industry_count += upsert_rows(
                conn,
                table=qname(settings, "moneyflow_industry"),
                columns=[
                    "trade_date",
                    "theme_name",
                    "source",
                    "pct_chg",
                    "net_amount_yi",
                    "net_buy_amount_yi",
                    "net_sell_amount_yi",
                    "lead_stock",
                    "rank",
                    "raw",
                ],
                rows=rows,
                conflict_columns=["trade_date", "source", "theme_name"],
            )
            conn.commit()
    metrics["moneyflow_industry_rows"] = industry_count

    concept_df, error = _safe_call(fetcher, "moneyflow_cnt_ths", lambda: fetcher.pro.moneyflow_cnt_ths(trade_date=tushare_date))
    if error:
        warnings.append(error)
    concept_rows = []
    if not concept_df.empty:
        for _, raw_row in concept_df.iterrows():
            row = raw_row.to_dict()
            name = str(row.get("name") or row.get("concept") or "").strip()
            if not name:
                continue
            concept_rows.append(
                {
                    "trade_date": _to_date(row.get("trade_date")) or date_value,
                    "theme_name": name,
                    "source": "tushare.moneyflow_cnt_ths",
                    "pct_chg": _num(row.get("pct_change") or row.get("pct_chg")),
                    "net_amount_yi": _num(row.get("net_amount")),
                    "net_buy_amount_yi": _num(row.get("net_buy_amount")),
                    "net_sell_amount_yi": _num(row.get("net_sell_amount")),
                    "lead_stock": row.get("lead_stock"),
                    "company_num": _num(row.get("company_num")),
                    "raw": _json(row),
                }
            )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {qname(settings, 'moneyflow_concept')} WHERE trade_date=%s", (date_value,))
        concept_count = upsert_rows(
            conn,
            table=qname(settings, "moneyflow_concept"),
            columns=[
                "trade_date",
                "theme_name",
                "source",
                "pct_chg",
                "net_amount_yi",
                "net_buy_amount_yi",
                "net_sell_amount_yi",
                "lead_stock",
                "company_num",
                "raw",
            ],
            rows=concept_rows,
            conflict_columns=["trade_date", "source", "theme_name"],
        )
        conn.commit()
    metrics["moneyflow_concept_rows"] = concept_count
    metrics["moneyflow_layer_warnings"] = warnings
    return metrics


def update_trading_auxiliary(settings: Settings, fetcher: TushareFetcher, trade_date: str) -> dict[str, Any]:
    data_cfg = settings.section("data")
    if not bool(data_cfg.get("trading_auxiliary_enabled", True)):
        return {"trading_auxiliary_skipped": True}

    metrics: dict[str, Any] = {"trading_auxiliary_skipped": False}
    metrics["moneyflow_stock_rows"] = sync_moneyflow_stock_from_daily(settings, trade_date)

    if bool(data_cfg.get("limit_lhb_enabled", True)):
        try:
            metrics.update(fetch_limit_events(settings, fetcher, trade_date))
        except Exception as exc:
            LOGGER.warning("limit events update failed: %s", exc)
            metrics["limit_error"] = str(exc)
        try:
            metrics.update(fetch_lhb(settings, fetcher, trade_date))
        except Exception as exc:
            LOGGER.warning("lhb update failed: %s", exc)
            metrics["lhb_error"] = str(exc)

    if bool(data_cfg.get("market_moneyflow_enabled", True)):
        try:
            metrics.update(fetch_moneyflow_layers(settings, fetcher, trade_date))
        except Exception as exc:
            LOGGER.warning("moneyflow layers update failed: %s", exc)
            metrics["moneyflow_layer_error"] = str(exc)

    return metrics


def _volume_state(row: pd.Series, cfg: dict[str, Any]) -> str:
    ratio5 = row.get("volume_ratio_5")
    pct_chg = row.get("pct_chg")
    if pd.isna(ratio5) or pd.isna(pct_chg):
        return ""
    heavy = float(cfg.get("heavy_volume_ratio", 1.5))
    shrink = float(cfg.get("shrink_volume_ratio", 0.7))
    if float(ratio5) >= heavy:
        return "放量上涨" if float(pct_chg) > 0 else "放量下跌"
    if float(ratio5) <= shrink:
        return "缩量上涨" if float(pct_chg) > 0 else "缩量回调"
    return "量能正常"


def _technical_score(row: pd.Series, cfg: dict[str, Any], scoring: dict[str, Any] | None = None) -> tuple[float, str, list[str], list[str]]:
    scoring = scoring or {}
    score = float(scoring.get("base", 50.0))
    tags: list[str] = []
    risks: list[str] = []
    close = row.get("adj_close")
    ma5, ma10, ma20, ma60 = row.get("ma5"), row.get("ma10"), row.get("ma20"), row.get("ma60")
    if pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20):
        if ma5 > ma10 > ma20:
            score += float(scoring.get("bullish_alignment", 18.0))
            tags.append("多头排列")
        elif ma5 < ma10 < ma20:
            score += float(scoring.get("bearish_alignment", -22.0))
            risks.append("空头排列")
    if pd.notna(close) and pd.notna(ma20):
        if close > ma20:
            score += float(scoring.get("above_ma20", 8.0))
            tags.append("站上MA20")
        else:
            score += float(scoring.get("below_ma20", -10.0))
            risks.append("跌破MA20")
    if pd.notna(close) and pd.notna(ma60):
        if close > ma60:
            score += float(scoring.get("above_ma60", 6.0))
            tags.append("站上MA60")
        else:
            score += float(scoring.get("below_ma60", -5.0))
    if pd.notna(row.get("macd_hist")):
        if row.get("macd_hist") > 0:
            score += float(scoring.get("macd_hist_positive", 7.0))
            tags.append("MACD偏强")
        else:
            score += float(scoring.get("macd_hist_nonpositive", -5.0))
    if pd.notna(row.get("bias5")) and row.get("bias5") >= float(cfg.get("overheat_bias5", 8.0)):
        score += float(scoring.get("overheat_bias_penalty", -14.0))
        risks.append("短线乖离过大")
    if pd.notna(row.get("rsi14")) and row.get("rsi14") >= float(cfg.get("high_rsi", 80.0)):
        score += float(scoring.get("high_rsi_penalty", -10.0))
        risks.append("RSI过热")
    if "多头排列" in tags and pd.notna(row.get("high20")) and pd.notna(close) and close >= row.get("high20") * 0.995:
        phase = "breakout"
    elif "多头排列" in tags:
        phase = "uptrend"
    elif "空头排列" in risks or "跌破MA20" in risks:
        phase = "weakening"
    else:
        phase = "sideways"
    return round(_bounded(score), 2), phase, tags, risks


def _price_volume_score(
    row: pd.Series,
    amount_yi: float | None,
    volume_state: str,
    scoring: dict[str, Any] | None = None,
) -> tuple[float, list[str], list[str]]:
    scoring = scoring or {}
    score = float(scoring.get("base", 50.0))
    tags: list[str] = []
    risks: list[str] = []
    pct = _num(row.get("pct_chg")) or 0.0
    turnover = _num(row.get("turnover_rate")) or 0.0
    ratio = _num(row.get("volume_ratio_5")) or _num(row.get("volume_ratio")) or 0.0
    pct_limit = float(scoring.get("pct_chg_limit", 20.0))
    score += max(min(pct, pct_limit), -pct_limit) * float(scoring.get("pct_chg_weight", 1.5))
    score += min(max(amount_yi or 0.0, 0.0), float(scoring.get("amount_yi_limit", 120.0))) * float(scoring.get("amount_yi_weight", 0.16))
    score += min(max(turnover, 0.0), float(scoring.get("turnover_rate_limit", 30.0))) * float(scoring.get("turnover_rate_weight", 0.55))
    score += min(max(ratio, 0.0), float(scoring.get("volume_ratio_limit", 8.0))) * float(scoring.get("volume_ratio_weight", 2.0))
    if volume_state == "放量上涨":
        tags.append("放量上涨")
        score += float(scoring.get("heavy_up_bonus", 8.0))
    elif volume_state == "放量下跌":
        risks.append("放量下跌")
        score += float(scoring.get("heavy_down_penalty", -10.0))
    elif volume_state == "缩量回调":
        tags.append("缩量回调")
    elif volume_state == "缩量上涨":
        risks.append("缩量上涨")
        score += float(scoring.get("shrink_up_penalty", -4.0))
    return round(_bounded(score), 2), tags, risks


def _moneyflow_score(row: pd.Series, scoring: dict[str, Any] | None = None) -> tuple[float, list[str], list[str]]:
    scoring = scoring or {}
    net_yi = _wan_to_yi(row.get("net_mf_amount"))
    rate = _num(row.get("net_mf_rate"))
    if net_yi is None and rate is None:
        return float(scoring.get("base", 50.0)), [], []
    score = float(scoring.get("base", 50.0))
    tags: list[str] = []
    risks: list[str] = []
    if net_yi is not None:
        limit = float(scoring.get("net_mf_amount_yi_limit", 8.0))
        score += max(min(net_yi, limit), -limit) * float(scoring.get("net_mf_amount_yi_weight", 3.0))
    if rate is not None:
        limit = float(scoring.get("net_mf_rate_limit", 12.0))
        score += max(min(rate, limit), -limit) * float(scoring.get("net_mf_rate_weight", 1.2))
    if (net_yi or 0) > 0 or (rate or 0) > 0:
        tags.append("资金净流入")
    if (net_yi or 0) < 0 or (rate or 0) < 0:
        risks.append("资金净流出")
    return round(_bounded(score), 2), tags, risks


def _limit_score(limit_row: dict[str, Any] | None, scoring: dict[str, Any] | None = None) -> tuple[float, str, bool, bool, bool, list[str], list[str]]:
    scoring = scoring or {}
    if not limit_row:
        return float(scoring.get("default", 50.0)), "", False, False, False, [], []
    limit_type = str(limit_row.get("limit_type") or "")
    limit_times = _num(limit_row.get("limit_times")) or 0.0
    open_times = _num(limit_row.get("open_times")) or 0.0
    tags: list[str] = []
    risks: list[str] = []
    if limit_type == "U":
        score = (
            float(scoring.get("limit_up_base", 82.0))
            + min(limit_times, float(scoring.get("limit_times_limit", 5.0))) * float(scoring.get("limit_times_weight", 3.0))
            + min(open_times, float(scoring.get("open_times_limit", 5.0))) * float(scoring.get("open_times_penalty", -2.0))
        )
        tags.append("涨停")
        if limit_times >= 2:
            tags.append(f"{int(limit_times)}连板")
        return round(_bounded(score), 2), "limit_up", True, False, False, tags, risks
    if limit_type == "D":
        risks.append("跌停")
        return float(scoring.get("limit_down", 15.0)), "limit_down", False, True, False, tags, risks
    if limit_type == "Z":
        risks.append("炸板")
        return float(scoring.get("broken_board", 42.0)), "broken_board", False, False, True, tags, risks
    return float(scoring.get("default", 50.0)), "", False, False, False, tags, risks


def _lhb_score(lhb_row: dict[str, Any] | None, scoring: dict[str, Any] | None = None) -> tuple[float, list[str], list[str]]:
    scoring = scoring or {}
    if not lhb_row:
        return float(scoring.get("default", 50.0)), [], []
    net = _num(lhb_row.get("lhb_net_buy_yi")) or 0.0
    inst = _num(lhb_row.get("institution_net_buy_yi")) or 0.0
    north = _num(lhb_row.get("northbound_net_buy_yi")) or 0.0
    amount_rate = _num(lhb_row.get("amount_rate")) or 0.0
    net_limit = float(scoring.get("net_buy_yi_limit", 8.0))
    inst_limit = float(scoring.get("institution_net_buy_yi_limit", 5.0))
    north_limit = float(scoring.get("northbound_net_buy_yi_limit", 5.0))
    score = float(scoring.get("default", 50.0))
    score += max(min(net, net_limit), -net_limit) * float(scoring.get("net_buy_yi_weight", 2.2))
    score += max(min(inst, inst_limit), -inst_limit) * float(scoring.get("institution_net_buy_yi_weight", 3.0))
    score += max(min(north, north_limit), -north_limit) * float(scoring.get("northbound_net_buy_yi_weight", 2.0))
    score += max(min(amount_rate, float(scoring.get("amount_rate_limit", 30.0))), 0.0) * float(scoring.get("amount_rate_weight", 0.18))
    tags: list[str] = []
    risks: list[str] = []
    if net > 0:
        tags.append("龙虎榜净买")
    elif net < 0:
        risks.append("龙虎榜净卖")
    if inst > 0:
        tags.append("机构净买")
    if north > 0:
        tags.append("陆股通净买")
    return round(_bounded(score), 2), tags, risks


def _stock_signal_columns() -> list[str]:
    return [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "concepts",
        "close",
        "pct_chg",
        "amount_yi",
        "turnover_rate",
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "bias5",
        "rsi14",
        "macd",
        "macd_signal",
        "macd_hist",
        "vol_ma5",
        "vol_ma20",
        "prev_vol_ma5",
        "prev_vol_ma20",
        "volume_ratio_5",
        "volume_ratio_20",
        "high20",
        "low20",
        "volume_ratio",
        "technical_score",
        "price_volume_score",
        "moneyflow_score",
        "limit_score",
        "lhb_score",
        "total_signal_score",
        "signal_level",
        "trend_phase",
        "volume_state",
        "limit_status",
        "is_limit_up",
        "is_limit_down",
        "is_broken_board",
        "limit_times",
        "open_times",
        "first_limit_time",
        "last_limit_time",
        "net_mf_amount",
        "net_mf_amount_yi",
        "net_mf_rate",
        "lhb_net_buy_yi",
        "institution_net_buy_yi",
        "northbound_net_buy_yi",
        "lhb_reason",
        "tags",
        "risk_flags",
        "reason",
        "data_quality",
    ]


def _build_stock_signal_rows(
    settings: Settings,
    df: pd.DataFrame,
    current: pd.DataFrame,
    limit_rows: list[dict[str, Any]],
    lhb_rows: list[dict[str, Any]],
    formula_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    cfg = merged_signal_config(settings, formula_spec)
    scoring = formula_section(formula_spec, "stock_signal_scoring")
    technical_scoring = formula_section(scoring, "technical")
    price_volume_scoring = formula_section(scoring, "price_volume")
    moneyflow_scoring = formula_section(scoring, "moneyflow")
    limit_scoring = formula_section(scoring, "limit")
    lhb_scoring = formula_section(scoring, "lhb")
    total_weights = formula_section(scoring, "total_weights")
    levels = formula_section(scoring, "levels")
    meta = _stock_meta()
    limit_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in limit_rows:
        key = (str(item.get("trade_date")), str(item.get("ts_code") or ""))
        existing = limit_by_key.get(key)
        if not existing or str(item.get("limit_type")) == "U":
            limit_by_key[key] = item
    lhb_by_key = {(str(item.get("trade_date")), str(item.get("ts_code") or "")): item for item in lhb_rows}
    history_frame = df[["trade_date", "ts_code"]].copy()
    history_frame["trade_date_key"] = history_frame["trade_date"].astype(str)
    history_frame["ts_code_key"] = history_frame["ts_code"].astype(str)
    history_frame = history_frame.sort_values(["ts_code_key", "trade_date_key"])
    history_frame["history_days"] = history_frame.groupby("ts_code_key").cumcount() + 1
    history_counts = dict(
        zip(
            zip(history_frame["trade_date_key"], history_frame["ts_code_key"]),
            history_frame["history_days"].astype(int),
        )
    )
    min_history = int(settings.section("signals").get("min_history_days", 60))

    rows: list[dict[str, Any]] = []
    for _, row in current.iterrows():
        ts_code = str(row.get("ts_code") or "")
        date_value = str(row.get("trade_date"))
        stock_meta = meta.get(ts_code, {})
        amount_yi = _thousand_yuan_to_yi(row.get("amount"))
        volume_state = _volume_state(row, cfg)
        technical_score, trend_phase, tech_tags, tech_risks = _technical_score(row, cfg, technical_scoring)
        pv_score, pv_tags, pv_risks = _price_volume_score(row, amount_yi, volume_state, price_volume_scoring)
        mf_score, mf_tags, mf_risks = _moneyflow_score(row, moneyflow_scoring)
        limit_row = limit_by_key.get((date_value, ts_code))
        lhb_row = lhb_by_key.get((date_value, ts_code))
        limit_score, limit_status, is_up, is_down, is_broken, limit_tags, limit_risks = _limit_score(limit_row, limit_scoring)
        lhb_score, lhb_tags, lhb_risks = _lhb_score(lhb_row, lhb_scoring)
        total = (
            technical_score * float(total_weights.get("technical_score", 0.45))
            + pv_score * float(total_weights.get("price_volume_score", 0.25))
            + mf_score * float(total_weights.get("moneyflow_score", 0.15))
            + limit_score * float(total_weights.get("limit_score", 0.10))
            + lhb_score * float(total_weights.get("lhb_score", 0.05))
        )
        total = round(_bounded(total), 2)
        if total >= float(levels.get("strong", 78.0)):
            level = "strong"
        elif total >= float(levels.get("watch", 62.0)):
            level = "watch"
        elif total <= float(levels.get("risk", 35.0)):
            level = "risk"
        else:
            level = "neutral"
        tags = list(dict.fromkeys([*tech_tags, *pv_tags, *mf_tags, *limit_tags, *lhb_tags]))
        risks = list(dict.fromkeys([*tech_risks, *pv_risks, *mf_risks, *limit_risks, *lhb_risks]))
        reason_parts = []
        if tags:
            reason_parts.append("、".join(tags[:5]))
        if risks:
            reason_parts.append("风险：" + "、".join(risks[:4]))
        if not reason_parts:
            reason_parts.append("个股交易信号暂不突出")
        history_days = int(history_counts.get((date_value, ts_code), 0))
        data_quality = {
            "history_days": history_days,
            "enough_history": history_days >= min_history,
            "has_moneyflow": pd.notna(row.get("net_mf_amount")),
            "has_limit_event": bool(limit_row),
            "has_lhb": bool(lhb_row),
            "scope": "all_a",
        }
        rows.append(
            {
                "trade_date": date_value,
                "ts_code": ts_code,
                "name": stock_meta.get("name") or ts_code,
                "industry": stock_meta.get("industry"),
                "concepts": stock_meta.get("concepts"),
                "close": _clean(row.get("close")),
                "pct_chg": _clean(row.get("pct_chg")),
                "amount_yi": amount_yi,
                "turnover_rate": _clean(row.get("turnover_rate")),
                "ma5": _clean(row.get("ma5")),
                "ma10": _clean(row.get("ma10")),
                "ma20": _clean(row.get("ma20")),
                "ma60": _clean(row.get("ma60")),
                "bias5": _clean(row.get("bias5")),
                "rsi14": _clean(row.get("rsi14")),
                "macd": _clean(row.get("macd")),
                "macd_signal": _clean(row.get("macd_signal")),
                "macd_hist": _clean(row.get("macd_hist")),
                "vol_ma5": _clean(row.get("vol_ma5")),
                "vol_ma20": _clean(row.get("vol_ma20")),
                "prev_vol_ma5": _clean(row.get("prev_vol_ma5")),
                "prev_vol_ma20": _clean(row.get("prev_vol_ma20")),
                "volume_ratio_5": _clean(row.get("volume_ratio_5")),
                "volume_ratio_20": _clean(row.get("volume_ratio_20")),
                "high20": _clean(row.get("high20")),
                "low20": _clean(row.get("low20")),
                "volume_ratio": _clean(row.get("volume_ratio_5")),
                "technical_score": technical_score,
                "price_volume_score": pv_score,
                "moneyflow_score": mf_score,
                "limit_score": limit_score,
                "lhb_score": lhb_score,
                "total_signal_score": total,
                "signal_level": level,
                "trend_phase": trend_phase,
                "volume_state": volume_state,
                "limit_status": limit_status,
                "is_limit_up": is_up,
                "is_limit_down": is_down,
                "is_broken_board": is_broken,
                "limit_times": _clean(limit_row.get("limit_times")) if limit_row else None,
                "open_times": _clean(limit_row.get("open_times")) if limit_row else None,
                "first_limit_time": limit_row.get("first_limit_time") if limit_row else "",
                "last_limit_time": limit_row.get("last_limit_time") if limit_row else "",
                "net_mf_amount": _clean(row.get("net_mf_amount")),
                "net_mf_amount_yi": _wan_to_yi(row.get("net_mf_amount")),
                "net_mf_rate": _clean(row.get("net_mf_rate")),
                "lhb_net_buy_yi": _clean(lhb_row.get("lhb_net_buy_yi")) if lhb_row else None,
                "institution_net_buy_yi": _clean(lhb_row.get("institution_net_buy_yi")) if lhb_row else None,
                "northbound_net_buy_yi": _clean(lhb_row.get("northbound_net_buy_yi")) if lhb_row else None,
                "lhb_reason": lhb_row.get("primary_reason") if lhb_row else "",
                "tags": _json(tags),
                "risk_flags": _json(risks),
                "reason": "；".join(reason_parts),
                "data_quality": _json(data_quality),
            }
        )
    return rows


def refresh_stock_signal_daily(settings: Settings, trade_date: str | None = None) -> dict[str, Any]:
    formula_spec = load_formula_spec(settings)
    cfg = merged_signal_config(settings, formula_spec)
    date_value = _to_date(trade_date) if trade_date else None
    with connect() as conn, conn.cursor() as cur:
        if date_value is None:
            cur.execute(f"SELECT max(trade_date) AS d FROM {qname(settings, 'daily_bars')}")
            row = cur.fetchone()
            date_value = str(row["d"]) if row and row["d"] else ""
        if not date_value:
            raise RuntimeError("No trade_date available for stock_signal_daily")
        history_days = int(cfg.get("stock_signal_history_trading_days", 180))
        history_start = _history_start_date(settings, date_value, history_days)
        history_filter = ""
        params: list[Any] = [date_value]
        if history_start:
            history_filter = "AND b.trade_date >= %s"
            params.append(history_start)
        cur.execute(
            f"""
            SELECT b.*, db.turnover_rate, db.volume_ratio, mf.net_mf_amount,
                   CASE WHEN b.amount IS NOT NULL AND b.amount <> 0 THEN mf.net_mf_amount * 1000.0 / b.amount ELSE NULL END AS net_mf_rate
            FROM {qname(settings, 'daily_bars')} b
            LEFT JOIN {qname(settings, 'daily_basic')} db
              ON db.ts_code=b.ts_code AND db.trade_date=b.trade_date
            LEFT JOIN {qname(settings, 'moneyflow_daily')} mf
              ON mf.ts_code=b.ts_code AND mf.trade_date=b.trade_date
            WHERE b.trade_date <= %s
              {history_filter}
              AND b.adj_close IS NOT NULL
            ORDER BY b.ts_code, b.trade_date
            """,
            tuple(params),
        )
        bar_rows = cur.fetchall()
        cur.execute(f"SELECT * FROM {qname(settings, 'limit_events')} WHERE trade_date=%s", (date_value,))
        limit_rows = [dict(row) for row in cur.fetchall()]
        cur.execute(f"SELECT * FROM {qname(settings, 'lhb_stocks')} WHERE trade_date=%s", (date_value,))
        lhb_rows = [dict(row) for row in cur.fetchall()]

    df = pd.DataFrame(bar_rows)
    if df.empty:
        raise RuntimeError("No daily bars available for stock_signal_daily")
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "pct_chg",
        "vol",
        "amount",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "turnover_rate",
        "volume_ratio",
        "net_mf_amount",
        "net_mf_rate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_indicators(df, formula_spec)
    current = df[df["trade_date"].astype(str) == str(date_value)].copy()
    if current.empty:
        raise RuntimeError(f"No current daily bars for stock_signal_daily {date_value}")

    rows = _build_stock_signal_rows(settings, df, current, limit_rows, lhb_rows, formula_spec)
    columns = _stock_signal_columns()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {qname(settings, 'stock_signal_daily')} WHERE trade_date=%s", (date_value,))
        count = upsert_rows(
            conn,
            table=qname(settings, "stock_signal_daily"),
            columns=columns,
            rows=rows,
            conflict_columns=["trade_date", "ts_code"],
        )
        conn.commit()
    return {"stock_signal_daily_rows": count, "stock_signal_trade_date": date_value}


def refresh_stock_signal_daily_range(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    start_value = _to_date(start_date)
    end_value = _to_date(end_date)
    if not start_value or not end_value or start_value > end_value:
        raise RuntimeError(f"Invalid stock_signal_daily range: {start_date} to {end_date}")

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            WITH target_dates AS (
                SELECT cal_date AS trade_date
                FROM {qname(settings, 'trade_calendar')}
                WHERE is_open=true AND cal_date BETWEEN %s AND %s
            ),
            daily_counts AS (
                SELECT trade_date, count(*) AS n
                FROM {qname(settings, 'daily_bars')}
                WHERE trade_date BETWEEN %s AND %s AND adj_close IS NOT NULL
                GROUP BY trade_date
            ),
            signal_counts AS (
                SELECT trade_date, count(*) AS n
                FROM {qname(settings, 'stock_signal_daily')}
                WHERE trade_date BETWEEN %s AND %s
                GROUP BY trade_date
            )
            SELECT d.trade_date,
                   coalesce(dc.n, 0) AS daily_rows,
                   coalesce(sc.n, 0) AS signal_rows
            FROM target_dates d
            LEFT JOIN daily_counts dc ON dc.trade_date=d.trade_date
            LEFT JOIN signal_counts sc ON sc.trade_date=d.trade_date
            ORDER BY d.trade_date
            """,
            (start_value, end_value, start_value, end_value, start_value, end_value),
        )
        date_rows = [dict(row) for row in cur.fetchall()]

    target_dates: list[str] = []
    skipped_complete = 0
    skipped_no_bars = 0
    expected_rows = 0
    for row in date_rows:
        date_text = str(row["trade_date"])
        daily_rows = int(row.get("daily_rows") or 0)
        signal_rows = int(row.get("signal_rows") or 0)
        if daily_rows <= 0:
            skipped_no_bars += 1
            continue
        if not force and signal_rows >= daily_rows:
            skipped_complete += 1
            continue
        target_dates.append(date_text)
        expected_rows += daily_rows

    if not target_dates:
        return {
            "stock_signal_range_start": start_value,
            "stock_signal_range_end": end_value,
            "stock_signal_target_dates": len(date_rows),
            "stock_signal_refreshed_dates": 0,
            "stock_signal_skipped_complete_dates": skipped_complete,
            "stock_signal_skipped_no_bars": skipped_no_bars,
            "stock_signal_daily_rows": 0,
        }

    formula_spec = load_formula_spec(settings)
    cfg = merged_signal_config(settings, formula_spec)
    history_days = int(cfg.get("stock_signal_history_trading_days", 180))
    history_start = _history_start_date(settings, target_dates[0], history_days)
    history_start = history_start or target_dates[0]

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT b.*, db.turnover_rate, db.volume_ratio, mf.net_mf_amount,
                   CASE WHEN b.amount IS NOT NULL AND b.amount <> 0 THEN mf.net_mf_amount * 1000.0 / b.amount ELSE NULL END AS net_mf_rate
            FROM {qname(settings, 'daily_bars')} b
            LEFT JOIN {qname(settings, 'daily_basic')} db
              ON db.ts_code=b.ts_code AND db.trade_date=b.trade_date
            LEFT JOIN {qname(settings, 'moneyflow_daily')} mf
              ON mf.ts_code=b.ts_code AND mf.trade_date=b.trade_date
            WHERE b.trade_date BETWEEN %s AND %s
              AND b.adj_close IS NOT NULL
            ORDER BY b.ts_code, b.trade_date
            """,
            (history_start, target_dates[-1]),
        )
        bar_rows = cur.fetchall()
        cur.execute(
            f"SELECT * FROM {qname(settings, 'limit_events')} WHERE trade_date BETWEEN %s AND %s",
            (target_dates[0], target_dates[-1]),
        )
        limit_rows = [dict(row) for row in cur.fetchall()]
        cur.execute(
            f"SELECT * FROM {qname(settings, 'lhb_stocks')} WHERE trade_date BETWEEN %s AND %s",
            (target_dates[0], target_dates[-1]),
        )
        lhb_rows = [dict(row) for row in cur.fetchall()]

    df = pd.DataFrame(bar_rows)
    if df.empty:
        raise RuntimeError(f"No daily bars available for stock_signal_daily range {start_value} to {end_value}")
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "pct_chg",
        "vol",
        "amount",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "turnover_rate",
        "volume_ratio",
        "net_mf_amount",
        "net_mf_rate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_indicators(df, formula_spec)
    target_set = set(target_dates)
    current = df[df["trade_date"].astype(str).isin(target_set)].copy()
    if current.empty:
        raise RuntimeError(f"No current daily bars for stock_signal_daily range {start_value} to {end_value}")

    rows = _build_stock_signal_rows(settings, df, current, limit_rows, lhb_rows, formula_spec)
    columns = _stock_signal_columns()
    with connect() as conn:
        with conn.cursor() as cur:
            for date_text in target_dates:
                cur.execute(f"DELETE FROM {qname(settings, 'stock_signal_daily')} WHERE trade_date=%s", (date_text,))
        count = upsert_rows(
            conn,
            table=qname(settings, "stock_signal_daily"),
            columns=columns,
            rows=rows,
            conflict_columns=["trade_date", "ts_code"],
        )
        conn.commit()

    return {
        "stock_signal_range_start": start_value,
        "stock_signal_range_end": end_value,
        "stock_signal_target_dates": len(date_rows),
        "stock_signal_refreshed_dates": len(target_dates),
        "stock_signal_skipped_complete_dates": skipped_complete,
        "stock_signal_skipped_no_bars": skipped_no_bars,
        "stock_signal_expected_rows": expected_rows,
        "stock_signal_daily_rows": count,
    }


def refresh_theme_signal_daily(settings: Settings, trade_date: str | None = None) -> dict[str, Any]:
    formula_spec = load_formula_spec(settings)
    scoring = formula_section(formula_spec, "theme_signal_scoring")
    industry_heat_cfg = formula_section(scoring, "industry_heat")
    industry_momentum_cfg = formula_section(scoring, "industry_momentum")
    concept_heat_cfg = formula_section(scoring, "concept_heat")
    concept_momentum_cfg = formula_section(scoring, "concept_momentum")
    levels = formula_section(scoring, "levels")
    date_value = _to_date(trade_date) if trade_date else None
    if date_value is None:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT max(trade_date) AS d FROM {qname(settings, 'daily_bars')}")
            row = cur.fetchone()
            date_value = str(row["d"]) if row and row["d"] else ""
    if not date_value:
        return {"theme_signal_daily_rows": 0}
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {qname(settings, 'moneyflow_industry')} WHERE trade_date=%s", (date_value,))
        industry_flow = [dict(row) for row in cur.fetchall()]
        cur.execute(f"SELECT * FROM {qname(settings, 'moneyflow_concept')} WHERE trade_date=%s", (date_value,))
        concept_flow = [dict(row) for row in cur.fetchall()]
        cur.execute(
            f"""
            SELECT industry, limit_type, count(*) AS n
            FROM {qname(settings, 'limit_events')}
            WHERE trade_date=%s AND industry IS NOT NULL AND industry <> ''
            GROUP BY industry, limit_type
            """,
            (date_value,),
        )
        limit_counts = [dict(row) for row in cur.fetchall()]
        cur.execute(
            f"""
            SELECT industry, ts_code, name, total_signal_score
            FROM {qname(settings, 'stock_signal_daily')}
            WHERE trade_date=%s AND signal_level IN ('strong', 'watch')
            ORDER BY total_signal_score DESC NULLS LAST
            """,
            (date_value,),
        )
        strong_stocks = [dict(row) for row in cur.fetchall()]

    limit_by_industry: dict[str, dict[str, int]] = {}
    for row in limit_counts:
        industry = str(row.get("industry") or "")
        limit_type = str(row.get("limit_type") or "")
        limit_by_industry.setdefault(industry, {})[limit_type] = int(row.get("n") or 0)

    top_stocks_by_industry: dict[str, list[dict[str, Any]]] = {}
    for row in strong_stocks:
        industry = str(row.get("industry") or "")
        if not industry:
            continue
        bucket = top_stocks_by_industry.setdefault(industry, [])
        if len(bucket) < 8:
            bucket.append(
                {
                    "ts_code": row.get("ts_code"),
                    "name": row.get("name"),
                    "score": row.get("total_signal_score"),
                }
            )

    rows: list[dict[str, Any]] = []
    for row in industry_flow:
        theme = str(row.get("theme_name") or "")
        if not theme:
            continue
        limit_info = limit_by_industry.get(theme, {})
        limit_up = int(limit_info.get("U", 0))
        broken = int(limit_info.get("Z", 0))
        strong_count = len(top_stocks_by_industry.get(theme, []))
        net = _num(row.get("net_amount_yi")) or 0.0
        pct = _num(row.get("pct_chg")) or 0.0
        heat_net_limit = float(industry_heat_cfg.get("net_amount_yi_limit", 50.0))
        heat_pct_limit = float(industry_heat_cfg.get("pct_chg_limit", 10.0))
        momentum_pct_limit = float(industry_momentum_cfg.get("pct_chg_limit", 10.0))
        heat = _bounded(
            float(industry_heat_cfg.get("base", 50.0))
            + max(min(net, heat_net_limit), -heat_net_limit) * float(industry_heat_cfg.get("net_amount_yi_weight", 0.8))
            + max(min(pct, heat_pct_limit), -heat_pct_limit) * float(industry_heat_cfg.get("pct_chg_weight", 2.0))
            + limit_up * float(industry_heat_cfg.get("limit_up_count_weight", 3.0))
            + strong_count * float(industry_heat_cfg.get("strong_stock_count_weight", 2.0))
        )
        momentum = _bounded(
            float(industry_momentum_cfg.get("base", 50.0))
            + max(min(pct, momentum_pct_limit), -momentum_pct_limit) * float(industry_momentum_cfg.get("pct_chg_weight", 3.0))
            + limit_up * float(industry_momentum_cfg.get("limit_up_count_weight", 2.0))
            + broken * float(industry_momentum_cfg.get("broken_count_penalty", -1.5))
        )
        rows.append(
            {
                "trade_date": date_value,
                "theme_type": "industry",
                "theme_name": theme,
                "source": row.get("source"),
                "pct_chg": row.get("pct_chg"),
                "net_amount_yi": row.get("net_amount_yi"),
                "limit_up_count": limit_up,
                "broken_count": broken,
                "strong_stock_count": strong_count,
                "top_stocks": _json(top_stocks_by_industry.get(theme, [])),
                "related_concepts": _json([]),
                "heat_score": round(heat, 2),
                "momentum_score": round(momentum, 2),
                "persistence_days": 1,
                "signal_level": "strong" if heat >= float(levels.get("strong", 78.0)) else "watch" if heat >= float(levels.get("watch", 62.0)) else "risk" if heat <= float(levels.get("risk", 35.0)) else "neutral",
                "reason": f"行业资金净额{net:.2f}亿，涨停{limit_up}家，强个股{strong_count}只",
            }
        )

    for row in concept_flow:
        theme = str(row.get("theme_name") or "")
        if not theme:
            continue
        net = _num(row.get("net_amount_yi")) or 0.0
        pct = _num(row.get("pct_chg")) or 0.0
        heat_net_limit = float(concept_heat_cfg.get("net_amount_yi_limit", 50.0))
        heat_pct_limit = float(concept_heat_cfg.get("pct_chg_limit", 10.0))
        momentum_pct_limit = float(concept_momentum_cfg.get("pct_chg_limit", 10.0))
        heat = _bounded(
            float(concept_heat_cfg.get("base", 50.0))
            + max(min(net, heat_net_limit), -heat_net_limit) * float(concept_heat_cfg.get("net_amount_yi_weight", 0.8))
            + max(min(pct, heat_pct_limit), -heat_pct_limit) * float(concept_heat_cfg.get("pct_chg_weight", 2.0))
        )
        momentum = _bounded(
            float(concept_momentum_cfg.get("base", 50.0))
            + max(min(pct, momentum_pct_limit), -momentum_pct_limit) * float(concept_momentum_cfg.get("pct_chg_weight", 3.0))
        )
        rows.append(
            {
                "trade_date": date_value,
                "theme_type": "concept",
                "theme_name": theme,
                "source": row.get("source"),
                "pct_chg": row.get("pct_chg"),
                "net_amount_yi": row.get("net_amount_yi"),
                "limit_up_count": 0,
                "broken_count": 0,
                "strong_stock_count": 0,
                "top_stocks": _json([]),
                "related_concepts": _json([]),
                "heat_score": round(heat, 2),
                "momentum_score": round(momentum, 2),
                "persistence_days": 1,
                "signal_level": "strong" if heat >= float(levels.get("strong", 78.0)) else "watch" if heat >= float(levels.get("watch", 62.0)) else "risk" if heat <= float(levels.get("risk", 35.0)) else "neutral",
                "reason": f"概念资金净额{net:.2f}亿，涨跌幅{pct:.2f}%",
            }
        )

    columns = [
        "trade_date",
        "theme_type",
        "theme_name",
        "source",
        "pct_chg",
        "net_amount_yi",
        "limit_up_count",
        "broken_count",
        "strong_stock_count",
        "top_stocks",
        "related_concepts",
        "heat_score",
        "momentum_score",
        "persistence_days",
        "signal_level",
        "reason",
    ]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {qname(settings, 'theme_signal_daily')} WHERE trade_date=%s", (date_value,))
        count = upsert_rows(
            conn,
            table=qname(settings, "theme_signal_daily"),
            columns=columns,
            rows=rows,
            conflict_columns=["trade_date", "theme_type", "theme_name", "source"],
        )
        conn.commit()
    return {"theme_signal_daily_rows": count, "theme_signal_trade_date": date_value}


def refresh_final_signal_layers(settings: Settings, trade_date: str | None = None) -> dict[str, Any]:
    stock_metrics = refresh_stock_signal_daily(settings, trade_date)
    theme_metrics = refresh_theme_signal_daily(settings, stock_metrics.get("stock_signal_trade_date") or trade_date)
    return {**stock_metrics, **theme_metrics}
