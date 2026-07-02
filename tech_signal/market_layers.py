from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .config import Settings
from .db import connect, qname, upsert_rows
from .tushare_fetcher import TushareFetcher


LOGGER = logging.getLogger(__name__)


DEFAULT_A_SHARE_INDEXES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "899050.BJ": "北证50",
}

DEFAULT_GLOBAL_INDEXES = [
    {"region": "US", "index_code": "DJI", "index_name": "道琼斯"},
    {"region": "US", "index_code": "SPX", "index_name": "标普500"},
    {"region": "US", "index_code": "IXIC", "index_name": "纳斯达克"},
    {"region": "Europe", "index_code": "GDAXI", "index_name": "德国DAX"},
    {"region": "Europe", "index_code": "FCHI", "index_name": "法国CAC40"},
    {"region": "Europe", "index_code": "FTSE", "index_name": "英国富时100"},
    {"region": "Asia", "index_code": "N225", "index_name": "日经225"},
    {"region": "Asia", "index_code": "KS11", "index_name": "韩国KOSPI"},
]


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


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _parse_json_list(value: object) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _latest_daily_bar_trade_date(settings: Settings) -> str:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT max(trade_date) AS d FROM {qname(settings, 'daily_bars')}")
        row = cur.fetchone()
        return row["d"].strftime("%Y%m%d") if row and row["d"] else ""


def _latest_open_trade_date(settings: Settings) -> str:
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


def _start_date(end_date: str, days: int) -> str:
    end_dt = datetime.strptime(_date_text(end_date), "%Y%m%d")
    return (end_dt - timedelta(days=days)).strftime("%Y%m%d")


def _open_trade_dates_between(settings: Settings, start_date: str, end_date: str) -> list[str]:
    start_value = _to_date(start_date)
    end_value = _to_date(end_date)
    if not start_value or not end_value:
        raise RuntimeError(f"Invalid trade date range: {start_date} to {end_date}")
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


def _has_daily_bars(settings: Settings, trade_date: str) -> bool:
    date_value = _to_date(trade_date)
    if not date_value:
        return False
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT count(*) AS n
            FROM {qname(settings, 'daily_bars')}
            WHERE trade_date=%s AND adj_close IS NOT NULL
            """,
            (date_value,),
        )
        row = cur.fetchone()
        return bool(row and int(row["n"] or 0) > 0)


def _stock_signal_count(settings: Settings, trade_date: str) -> int:
    date_value = _to_date(trade_date)
    if not date_value:
        return 0
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) AS n FROM {qname(settings, 'stock_signal_daily')} WHERE trade_date=%s",
            (date_value,),
        )
        row = cur.fetchone()
        return int(row["n"] or 0) if row else 0


def _a_share_indexes(settings: Settings) -> dict[str, str]:
    configured = settings.section("indexes").get("a_share")
    if isinstance(configured, dict) and configured:
        return {str(code): str(name) for code, name in configured.items()}
    return DEFAULT_A_SHARE_INDEXES


def _global_indexes(settings: Settings) -> list[dict[str, str]]:
    configured = settings.section("indexes").get("global")
    if isinstance(configured, list) and configured:
        out = []
        for row in configured:
            if not isinstance(row, dict):
                continue
            code = str(row.get("index_code") or "").strip()
            if not code:
                continue
            out.append(
                {
                    "region": str(row.get("region") or ""),
                    "index_code": code,
                    "index_name": str(row.get("index_name") or code),
                }
            )
        if out:
            return out
    return DEFAULT_GLOBAL_INDEXES


def _latest_row(df: pd.DataFrame, target_date: str) -> dict[str, Any] | None:
    if df is None or df.empty or "trade_date" not in df.columns:
        return None
    data = df.copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data = data[data["trade_date"] <= target_date].sort_values("trade_date", ascending=False)
    if data.empty:
        return None
    return data.iloc[0].to_dict()


def refresh_index_daily(settings: Settings, trade_date: str | None = None) -> dict[str, Any]:
    target = _date_text(trade_date) if trade_date else _latest_daily_bar_trade_date(settings)
    if not target:
        raise RuntimeError("No daily_bars trade_date available for index_daily refresh")
    fetcher = TushareFetcher(settings)
    start = _start_date(target, int(settings.section("indexes").get("a_share_lookback_days", 20)))
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    now = datetime.now()
    for code, name in _a_share_indexes(settings).items():
        df = fetcher._call(
            f"index_daily:{code}",
            lambda c=code: fetcher.pro.index_daily(ts_code=c, start_date=start, end_date=target),
        )
        latest = _latest_row(df, target)
        if not latest:
            warnings.append(f"index_daily {code} returned no row <= {target}")
            continue
        actual_date = _date_text(latest.get("trade_date"))
        if actual_date != target:
            warnings.append(f"index_daily {code} latest row {actual_date}, target {target}")
        rows.append(
            {
                "trade_date": _to_date(actual_date),
                "index_code": code,
                "index_name": name,
                "open": _num(latest.get("open")),
                "high": _num(latest.get("high")),
                "low": _num(latest.get("low")),
                "close": _num(latest.get("close")),
                "pre_close": _num(latest.get("pre_close")),
                "change": _num(latest.get("change")),
                "pct_chg": _num(latest.get("pct_chg")),
                "vol": _num(latest.get("vol")),
                "amount": _num(latest.get("amount")),
                "source": "tushare.index_daily",
                "updated_at": now,
            }
        )
    columns = [
        "trade_date",
        "index_code",
        "index_name",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "source",
        "updated_at",
    ]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "index_daily"),
            columns=columns,
            rows=rows,
            conflict_columns=["trade_date", "index_code"],
        )
        conn.commit()
    return {
        "trade_date": _to_date(target),
        "index_daily_rows": count,
        "index_daily_target_rows": len(_a_share_indexes(settings)),
        "index_daily_warnings": warnings,
    }


def refresh_index_daily_range(settings: Settings, start_date: str, end_date: str) -> dict[str, Any]:
    start = _date_text(start_date)
    end = _date_text(end_date)
    if len(start) != 8 or len(end) != 8 or start > end:
        raise RuntimeError(f"Invalid index_daily backfill range: {start_date} to {end_date}")

    fetcher = TushareFetcher(settings)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    now = datetime.now()
    for code, name in _a_share_indexes(settings).items():
        df = fetcher._call(
            f"index_daily:{code}:{start}-{end}",
            lambda c=code: fetcher.pro.index_daily(ts_code=c, start_date=start, end_date=end),
        )
        if df is None or df.empty:
            warnings.append(f"index_daily {code} returned no rows for {start}-{end}")
            continue
        for _, raw in df.iterrows():
            actual_date = _date_text(raw.get("trade_date"))
            if len(actual_date) != 8:
                continue
            rows.append(
                {
                    "trade_date": _to_date(actual_date),
                    "index_code": code,
                    "index_name": name,
                    "open": _num(raw.get("open")),
                    "high": _num(raw.get("high")),
                    "low": _num(raw.get("low")),
                    "close": _num(raw.get("close")),
                    "pre_close": _num(raw.get("pre_close")),
                    "change": _num(raw.get("change")),
                    "pct_chg": _num(raw.get("pct_chg")),
                    "vol": _num(raw.get("vol")),
                    "amount": _num(raw.get("amount")),
                    "source": "tushare.index_daily",
                    "updated_at": now,
                }
            )
    columns = [
        "trade_date",
        "index_code",
        "index_name",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "source",
        "updated_at",
    ]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "index_daily"),
            columns=columns,
            rows=rows,
            conflict_columns=["trade_date", "index_code"],
        )
        conn.commit()
    return {
        "index_daily_backfill_start": _to_date(start),
        "index_daily_backfill_end": _to_date(end),
        "index_daily_rows": count,
        "index_daily_source_rows": len(rows),
        "index_daily_target_indexes": len(_a_share_indexes(settings)),
        "index_daily_warnings": warnings,
    }


def _global_data_status(region: str, target_date: str, market_date: str) -> str:
    if not market_date:
        return "missing"
    target_dt = datetime.strptime(target_date, "%Y%m%d")
    market_dt = datetime.strptime(market_date, "%Y%m%d")
    gap_days = (target_dt - market_dt).days
    if region == "Asia":
        return "same_day_or_intraday" if gap_days == 0 else "prior_close_or_stale"
    if gap_days < 0:
        return "future_date_check"
    if gap_days <= 4:
        return "latest_close"
    return "holiday_or_stale"


def refresh_global_index_daily(settings: Settings, trade_date: str | None = None) -> dict[str, Any]:
    target = _date_text(trade_date) if trade_date else _latest_open_trade_date(settings)
    if not target:
        raise RuntimeError("No trade calendar date available for global_index_daily refresh")
    fetcher = TushareFetcher(settings)
    lookback_days = int(settings.section("indexes").get("global_lookback_days", 45))
    start = _start_date(target, lookback_days)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    now = datetime.now()
    for spec in _global_indexes(settings):
        region = spec["region"]
        code = spec["index_code"]
        name = spec["index_name"]
        df = fetcher._call(
            f"index_global:{code}",
            lambda c=code: fetcher.pro.index_global(ts_code=c, start_date=start, end_date=target),
        )
        latest = _latest_row(df, target)
        if not latest:
            warnings.append(f"index_global {code} returned no row <= {target}")
            continue
        market_date = _date_text(latest.get("trade_date"))
        rows.append(
            {
                "trade_date": _to_date(target),
                "market_date": _to_date(market_date),
                "region": region,
                "index_code": code,
                "index_name": name,
                "open": _num(latest.get("open")),
                "high": _num(latest.get("high")),
                "low": _num(latest.get("low")),
                "close": _num(latest.get("close")),
                "pre_close": _num(latest.get("pre_close")),
                "change": _num(latest.get("change")),
                "pct_chg": _num(latest.get("pct_chg")),
                "source": "tushare.index_global",
                "data_status": _global_data_status(region, target, market_date),
                "updated_at": now,
            }
        )
    columns = [
        "trade_date",
        "market_date",
        "region",
        "index_code",
        "index_name",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "source",
        "data_status",
        "updated_at",
    ]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "global_index_daily"),
            columns=columns,
            rows=rows,
            conflict_columns=["trade_date", "index_code"],
        )
        conn.commit()
    return {
        "trade_date": _to_date(target),
        "global_index_daily_rows": count,
        "global_index_daily_target_rows": len(_global_indexes(settings)),
        "global_index_daily_warnings": warnings,
    }


def refresh_global_index_daily_range(settings: Settings, start_date: str, end_date: str) -> dict[str, Any]:
    start = _date_text(start_date)
    end = _date_text(end_date)
    if len(start) != 8 or len(end) != 8 or start > end:
        raise RuntimeError(f"Invalid global_index_daily backfill range: {start_date} to {end_date}")

    target_dates = _open_trade_dates_between(settings, start, end)
    if not target_dates:
        raise RuntimeError(f"No open trade dates for global_index_daily backfill: {start}-{end}")

    fetcher = TushareFetcher(settings)
    lookback_days = int(settings.section("indexes").get("global_lookback_days", 45))
    source_start = _start_date(start, lookback_days)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    now = datetime.now()

    for spec in _global_indexes(settings):
        region = spec["region"]
        code = spec["index_code"]
        name = spec["index_name"]
        df = fetcher._call(
            f"index_global:{code}:{source_start}-{end}",
            lambda c=code: fetcher.pro.index_global(ts_code=c, start_date=source_start, end_date=end),
        )
        if df is None or df.empty:
            warnings.append(f"index_global {code} returned no rows for {source_start}-{end}")
            continue
        data = df.copy()
        data["trade_date"] = data["trade_date"].astype(str).str.replace("-", "").str[:8]
        data = data[data["trade_date"].str.len() == 8].sort_values("trade_date")
        records = data.to_dict("records")
        if not records:
            warnings.append(f"index_global {code} has no valid trade_date rows for {source_start}-{end}")
            continue
        pos = 0
        latest: dict[str, Any] | None = None
        for target in target_dates:
            while pos < len(records) and str(records[pos].get("trade_date")) <= target:
                latest = records[pos]
                pos += 1
            if not latest:
                warnings.append(f"index_global {code} has no row <= {target}")
                continue
            market_date = _date_text(latest.get("trade_date"))
            rows.append(
                {
                    "trade_date": _to_date(target),
                    "market_date": _to_date(market_date),
                    "region": region,
                    "index_code": code,
                    "index_name": name,
                    "open": _num(latest.get("open")),
                    "high": _num(latest.get("high")),
                    "low": _num(latest.get("low")),
                    "close": _num(latest.get("close")),
                    "pre_close": _num(latest.get("pre_close")),
                    "change": _num(latest.get("change")),
                    "pct_chg": _num(latest.get("pct_chg")),
                    "source": "tushare.index_global",
                    "data_status": _global_data_status(region, target, market_date),
                    "updated_at": now,
                }
            )

    columns = [
        "trade_date",
        "market_date",
        "region",
        "index_code",
        "index_name",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "source",
        "data_status",
        "updated_at",
    ]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "global_index_daily"),
            columns=columns,
            rows=rows,
            conflict_columns=["trade_date", "index_code"],
        )
        conn.commit()
    return {
        "global_index_daily_backfill_start": _to_date(start),
        "global_index_daily_backfill_end": _to_date(end),
        "global_index_daily_target_dates": len(target_dates),
        "global_index_daily_rows": count,
        "global_index_daily_source_rows": len(rows),
        "global_index_daily_target_indexes": len(_global_indexes(settings)),
        "global_index_daily_warnings": warnings,
    }


def _bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _split_concepts(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = []
    for sep in ["|", "、", ",", "，", ";", "；"]:
        text = text.replace(sep, "|")
    for item in text.split("|"):
        item = item.strip()
        if item and item not in parts:
            parts.append(item)
    return parts


def _theme_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("theme_name") or "").strip()
        if not name:
            continue
        existing = out.get(name)
        if existing is None or float(row.get("heat_score") or 0) > float(existing.get("heat_score") or 0):
            out[name] = row
    return out


def _matched_themes(row: dict[str, Any], themes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    names = []
    industry = str(row.get("industry") or "").strip()
    if industry:
        names.append(industry)
    for name in _split_concepts(row.get("concepts")):
        if name not in names:
            names.append(name)
    matched = [themes[name] for name in names if name in themes]
    matched.sort(key=lambda item: float(item.get("heat_score") or 0), reverse=True)
    return matched[:5]


def _leader_score(row: dict[str, Any], matched_themes: list[dict[str, Any]]) -> tuple[float, list[str], list[str]]:
    pct = float(row.get("pct_chg") or 0)
    amount_yi = float(row.get("amount_yi") or 0)
    turnover = float(row.get("turnover_rate") or 0)
    volume_ratio = float(row.get("volume_ratio") or 0)
    total_signal = float(row.get("total_signal_score") or 50)
    lhb_net = float(row.get("lhb_net_buy_yi") or 0)
    inst = float(row.get("institution_net_buy_yi") or 0)
    north = float(row.get("northbound_net_buy_yi") or 0)
    limit_times = float(row.get("limit_times") or 0)

    score = 35.0
    score += max(min(pct, 20), -20) * 1.6
    score += min(max(amount_yi, 0), 300) * 0.08
    score += min(max(turnover, 0), 30) * 0.55
    score += min(max(volume_ratio, 0), 8) * 1.8
    score += max(min(total_signal - 50, 40), -30) * 0.35
    if row.get("is_limit_up"):
        score += 16 + min(limit_times, 5) * 3.5
    elif row.get("is_broken_board"):
        score += 3
    score += min(max(lhb_net, 0), 10) * 1.4
    score += min(max(inst, 0), 5) * 2.0
    score += min(max(north, 0), 5) * 1.6
    if matched_themes:
        heat = max(float(item.get("heat_score") or 0) for item in matched_themes)
        score += max(heat - 60, 0) * 0.25

    reasons = [
        f"涨跌幅{pct:.2f}%",
        f"成交额{amount_yi:.2f}亿",
        f"换手{turnover:.2f}%",
        f"量比{volume_ratio:.2f}",
        f"个股信号{total_signal:.1f}",
    ]
    if row.get("is_limit_up"):
        reasons.append(f"涨停/连板{limit_times:.0f}")
    if row.get("is_broken_board"):
        reasons.append("炸板")
    if lhb_net > 0:
        reasons.append(f"龙虎榜净买{lhb_net:.2f}亿")
    if inst > 0:
        reasons.append(f"机构净买{inst:.2f}亿")
    if north > 0:
        reasons.append(f"陆股通净买{north:.2f}亿")
    if matched_themes:
        reasons.append("主题热度：" + "、".join(str(item.get("theme_name")) for item in matched_themes[:3]))

    risks = [str(x) for x in _parse_json_list(row.get("risk_flags")) if x]
    if row.get("is_broken_board") and "炸板" not in risks:
        risks.append("炸板")
    if turnover >= 25 and "高换手" not in risks:
        risks.append("高换手")
    if str(row.get("volume_state") or "") == "放量下跌" and "放量下跌" not in risks:
        risks.append("放量下跌")
    return round(_bounded(score, 0, 120), 2), reasons, risks


def refresh_dragon_leader_daily(settings: Settings, trade_date: str | None = None) -> dict[str, Any]:
    date_value = _to_date(trade_date) if trade_date else None
    with connect() as conn, conn.cursor() as cur:
        if not date_value:
            cur.execute(f"SELECT max(trade_date) AS d FROM {qname(settings, 'stock_signal_daily')}")
            row = cur.fetchone()
            date_value = str(row["d"]) if row and row["d"] else ""
        if not date_value:
            raise RuntimeError("No stock_signal_daily trade_date available for dragon leader refresh")
        cur.execute(f"SELECT * FROM {qname(settings, 'stock_signal_daily')} WHERE trade_date=%s", (date_value,))
        stock_rows = [dict(row) for row in cur.fetchall()]
        cur.execute(f"SELECT * FROM {qname(settings, 'theme_signal_daily')} WHERE trade_date=%s", (date_value,))
        theme_rows = [dict(row) for row in cur.fetchall()]

    cfg = settings.section("dragon_leader")
    min_amount_yi = float(cfg.get("min_amount_yi", 20.0))
    top_n = int(cfg.get("top_n", 300))
    themes = _theme_lookup(theme_rows)
    candidates: list[dict[str, Any]] = []
    now = datetime.now()
    for row in stock_rows:
        if float(row.get("amount_yi") or 0) < min_amount_yi:
            continue
        matched = _matched_themes(row, themes)
        score, reasons, risks = _leader_score(row, matched)
        level = "strong" if score >= 88 else "watch" if score >= 68 else "candidate"
        candidates.append(
            {
                "trade_date": date_value,
                "ts_code": row.get("ts_code"),
                "name": row.get("name"),
                "industry": row.get("industry"),
                "concepts": row.get("concepts"),
                "pct_chg": row.get("pct_chg"),
                "amount_yi": row.get("amount_yi"),
                "turnover_rate": row.get("turnover_rate"),
                "volume_ratio": row.get("volume_ratio"),
                "limit_status": row.get("limit_status"),
                "is_limit_up": bool(row.get("is_limit_up")),
                "is_broken_board": bool(row.get("is_broken_board")),
                "limit_times": row.get("limit_times"),
                "lhb_net_buy_yi": row.get("lhb_net_buy_yi"),
                "institution_net_buy_yi": row.get("institution_net_buy_yi"),
                "northbound_net_buy_yi": row.get("northbound_net_buy_yi"),
                "theme_names": _json([item.get("theme_name") for item in matched]),
                "leader_score": score,
                "leader_rank": 0,
                "leader_level": level,
                "reason": "；".join(reasons),
                "risk_flags": _json(risks),
                "updated_at": now,
            }
        )
    candidates.sort(key=lambda item: (float(item.get("leader_score") or 0), float(item.get("amount_yi") or 0)), reverse=True)
    candidates = candidates[:top_n]
    for rank, row in enumerate(candidates, 1):
        row["leader_rank"] = rank

    columns = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "concepts",
        "pct_chg",
        "amount_yi",
        "turnover_rate",
        "volume_ratio",
        "limit_status",
        "is_limit_up",
        "is_broken_board",
        "limit_times",
        "lhb_net_buy_yi",
        "institution_net_buy_yi",
        "northbound_net_buy_yi",
        "theme_names",
        "leader_score",
        "leader_rank",
        "leader_level",
        "reason",
        "risk_flags",
        "updated_at",
    ]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {qname(settings, 'dragon_leader_daily')} WHERE trade_date=%s", (date_value,))
        count = upsert_rows(
            conn,
            table=qname(settings, "dragon_leader_daily"),
            columns=columns,
            rows=candidates,
            conflict_columns=["trade_date", "ts_code"],
        )
        conn.commit()
    return {
        "trade_date": date_value,
        "dragon_leader_daily_rows": count,
        "dragon_leader_candidates_scanned": len(stock_rows),
    }


def refresh_dragon_leader_daily_range(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    generate_missing_signals: bool = True,
) -> dict[str, Any]:
    start = _date_text(start_date)
    end = _date_text(end_date)
    if len(start) != 8 or len(end) != 8 or start > end:
        raise RuntimeError(f"Invalid dragon_leader_daily backfill range: {start_date} to {end_date}")

    dates = _open_trade_dates_between(settings, start, end)
    warnings: list[str] = []
    skipped_no_bars: list[str] = []
    skipped_no_signals: list[str] = []
    generated_signal_dates = 0
    refreshed_dates = 0
    total_rows = 0

    refresh_final_signal_layers = None
    if generate_missing_signals:
        from .trading_signals import refresh_final_signal_layers as _refresh_final_signal_layers

        refresh_final_signal_layers = _refresh_final_signal_layers

    for idx, trade_date in enumerate(dates, 1):
        date_value = _to_date(trade_date)
        if not date_value:
            continue
        if not _has_daily_bars(settings, trade_date):
            skipped_no_bars.append(date_value)
            continue
        try:
            if generate_missing_signals and _stock_signal_count(settings, trade_date) == 0:
                assert refresh_final_signal_layers is not None
                refresh_final_signal_layers(settings, trade_date)
                generated_signal_dates += 1
            if _stock_signal_count(settings, trade_date) == 0:
                skipped_no_signals.append(date_value)
                continue
            metrics = refresh_dragon_leader_daily(settings, trade_date)
            refreshed_dates += 1
            total_rows += int(metrics.get("dragon_leader_daily_rows") or 0)
            LOGGER.info(
                "dragon leader backfill %s/%s %s rows=%s",
                idx,
                len(dates),
                date_value,
                metrics.get("dragon_leader_daily_rows"),
            )
        except Exception as exc:
            warnings.append(f"{date_value}: {type(exc).__name__}: {exc}")
            LOGGER.exception("dragon leader backfill failed for %s", date_value)

    return {
        "dragon_leader_backfill_start": _to_date(start),
        "dragon_leader_backfill_end": _to_date(end),
        "dragon_leader_target_dates": len(dates),
        "dragon_leader_refreshed_dates": refreshed_dates,
        "dragon_leader_generated_signal_dates": generated_signal_dates,
        "dragon_leader_rows": total_rows,
        "dragon_leader_skipped_no_bars": len(skipped_no_bars),
        "dragon_leader_skipped_no_signals": len(skipped_no_signals),
        "dragon_leader_warning_count": len(warnings),
        "dragon_leader_warning_samples": warnings[:20],
    }


def refresh_market_structure_layers(settings: Settings, trade_date: str | None = None) -> dict[str, Any]:
    index_metrics = refresh_index_daily(settings, trade_date=trade_date)
    global_metrics = refresh_global_index_daily(settings, trade_date=trade_date or index_metrics.get("trade_date"))
    return {**index_metrics, **global_metrics}


def backfill_market_structure_layers(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    include_dragon_leaders: bool = True,
) -> dict[str, Any]:
    index_metrics = refresh_index_daily_range(settings, start_date, end_date)
    global_metrics = refresh_global_index_daily_range(settings, start_date, end_date)
    dragon_metrics: dict[str, Any] = {}
    if include_dragon_leaders:
        dragon_metrics = refresh_dragon_leader_daily_range(settings, start_date, end_date)
    return {**index_metrics, **global_metrics, **dragon_metrics}
