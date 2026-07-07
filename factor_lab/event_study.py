from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from tech_signal.config import Settings
from tech_signal.db import connect, init_schema, qname, upsert_rows

from .factor_definitions import EVENT_DEFINITIONS, normalize_date
from .factor_evaluator import (
    _clean_float,
    _drawdown_stats,
    _format_pct,
    _load_forward_returns,
    _market_regime_dates,
    report_dir,
)


EVENT_HORIZONS = [1, 3, 5, 10, 20]


def _load_events(settings: Settings, start_date: str, end_date: str) -> pd.DataFrame:
    start_value = normalize_date(start_date)
    end_value = normalize_date(end_date)
    frames: list[pd.DataFrame] = []
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT trade_date, ts_code,
                   CASE
                     WHEN lhb_net_buy_yi > 0 THEN 'lhb_net_buy_positive'
                     ELSE NULL
                   END AS event_name,
                   'lhb' AS event_group,
                   '' AS event_subtype
            FROM {qname(settings, 'lhb_stocks')}
            WHERE trade_date BETWEEN %s AND %s AND lhb_net_buy_yi > 0
            UNION ALL
            SELECT trade_date, ts_code, 'lhb_institution_buy', 'lhb', ''
            FROM {qname(settings, 'lhb_stocks')}
            WHERE trade_date BETWEEN %s AND %s AND institution_net_buy_yi > 0
            UNION ALL
            SELECT trade_date, ts_code, 'lhb_northbound_buy', 'lhb', ''
            FROM {qname(settings, 'lhb_stocks')}
            WHERE trade_date BETWEEN %s AND %s AND northbound_net_buy_yi > 0
            UNION ALL
            SELECT trade_date, ts_code, 'lhb_buy_amount_ratio_high', 'lhb', ''
            FROM {qname(settings, 'lhb_stocks')}
            WHERE trade_date BETWEEN %s AND %s AND amount_rate >= 20
            UNION ALL
            SELECT trade_date, ts_code, 'lhb_reason', 'lhb', coalesce(nullif(primary_reason, ''), 'unknown')
            FROM {qname(settings, 'lhb_stocks')}
            WHERE trade_date BETWEEN %s AND %s
            """,
            (
                start_value,
                end_value,
                start_value,
                end_value,
                start_value,
                end_value,
                start_value,
                end_value,
                start_value,
                end_value,
            ),
        )
        frames.append(pd.DataFrame(cur.fetchall()))

        cur.execute(
            f"""
            SELECT trade_date, ts_code,
                   CASE
                     WHEN limit_type='U' AND coalesce(limit_times, 1) <= 1 THEN 'limit_first_board'
                     WHEN limit_type='U' AND coalesce(limit_times, 1) >= 2 THEN 'limit_multi_board'
                     WHEN limit_type='Z' THEN 'broken_board'
                     WHEN limit_type='D' THEN 'limit_down'
                     ELSE NULL
                   END AS event_name,
                   CASE WHEN limit_type='D' THEN 'risk' ELSE 'sentiment' END AS event_group,
                   coalesce(industry, '') AS event_subtype
            FROM {qname(settings, 'limit_events')}
            WHERE trade_date BETWEEN %s AND %s
              AND limit_type IN ('U', 'D', 'Z')
            UNION ALL
            SELECT trade_date, ts_code, 'limit_resealed', 'sentiment', coalesce(industry, '')
            FROM {qname(settings, 'limit_events')}
            WHERE trade_date BETWEEN %s AND %s AND limit_type='U' AND coalesce(open_times, 0) > 0
            UNION ALL
            SELECT trade_date, ts_code, 'limit_strong_seal', 'sentiment', coalesce(industry, '')
            FROM {qname(settings, 'limit_events')}
            WHERE trade_date BETWEEN %s AND %s
              AND limit_type='U'
              AND amount_yi IS NOT NULL
              AND fd_amount_yi IS NOT NULL
              AND fd_amount_yi / nullif(amount_yi, 0) >= 0.08
            """,
            (start_value, end_value, start_value, end_value, start_value, end_value),
        )
        frames.append(pd.DataFrame(cur.fetchall()))

        cur.execute(
            f"""
            WITH mf AS (
                SELECT trade_date, ts_code, net_mf_amount,
                       sum(CASE WHEN net_mf_amount > 0 THEN 1 ELSE 0 END)
                       OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS positive_3
                FROM {qname(settings, 'moneyflow_stock')}
                WHERE trade_date BETWEEN %s::date - interval '10 days' AND %s
            )
            SELECT trade_date, ts_code, 'moneyflow_consecutive_3' AS event_name, 'moneyflow' AS event_group, '' AS event_subtype
            FROM mf
            WHERE trade_date BETWEEN %s AND %s AND positive_3 >= 3
            """,
            (start_value, end_value, start_value, end_value),
        )
        frames.append(pd.DataFrame(cur.fetchall()))

    events = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if events.empty:
        return events
    events = events.dropna(subset=["event_name"]).drop_duplicates(["trade_date", "ts_code", "event_name", "event_subtype"])
    events["trade_date"] = pd.to_datetime(events["trade_date"]).dt.date
    return events


def run_event_study(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    horizons: Iterable[int] = EVENT_HORIZONS,
) -> dict[str, Any]:
    init_schema(settings)
    horizon_list = [int(item) for item in horizons]
    events = _load_events(settings, start_date, end_date)
    if events.empty:
        write_event_study_report(settings, start_date, end_date, [])
        return {"start_date": normalize_date(start_date), "end_date": normalize_date(end_date), "event_rows": 0, "event_study_rows": 0}
    returns = _load_forward_returns(settings, start_date, end_date, horizon_list)
    regimes = _market_regime_dates(returns)
    market_returns = {}
    for horizon in horizon_list:
        col = f"fwd_return_{horizon}"
        market_returns[horizon] = returns.groupby("trade_date")[col].mean().rename(f"market_return_{horizon}")
    base = events.merge(returns, on=["trade_date", "ts_code"], how="inner")
    rows: list[dict[str, Any]] = []
    for horizon in horizon_list:
        ret_col = f"fwd_return_{horizon}"
        frame = base.merge(market_returns[horizon], on="trade_date", how="left")
        frame["excess_return"] = frame[ret_col] - frame[f"market_return_{horizon}"]
        for (event_name, event_group, event_subtype), group in frame.groupby(["event_name", "event_group", "event_subtype"], dropna=False):
            for regime, dates in regimes.items():
                subset = group if regime == "all" else group[group["trade_date"].isin(dates)]
                subset = subset.dropna(subset=[ret_col])
                if subset.empty:
                    continue
                daily = subset.groupby("trade_date")[ret_col].mean()
                avg_drawdown, max_drawdown = _drawdown_stats(daily)
                rows.append(
                    {
                        "event_name": str(event_name),
                        "event_group": str(event_group),
                        "event_subtype": str(event_subtype or "")[:120],
                        "start_date": normalize_date(start_date),
                        "end_date": normalize_date(end_date),
                        "horizon_days": horizon,
                        "market_regime": regime,
                        "sample_count": int(len(subset)),
                        "avg_return": _clean_float(subset[ret_col].mean()),
                        "avg_excess_return": _clean_float(subset["excess_return"].mean()),
                        "win_rate": _clean_float((subset[ret_col] > 0).mean()),
                        "avg_drawdown": avg_drawdown,
                        "max_drawdown": max_drawdown,
                    }
                )
    columns = [
        "event_name",
        "event_group",
        "event_subtype",
        "start_date",
        "end_date",
        "horizon_days",
        "market_regime",
        "sample_count",
        "avg_return",
        "avg_excess_return",
        "win_rate",
        "avg_drawdown",
        "max_drawdown",
    ]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "factor_event_study"),
            columns=columns,
            rows=rows,
            conflict_columns=["event_name", "event_subtype", "start_date", "end_date", "horizon_days", "market_regime"],
        )
        conn.commit()
    path = write_event_study_report(settings, start_date, end_date, rows)
    return {
        "start_date": normalize_date(start_date),
        "end_date": normalize_date(end_date),
        "raw_event_rows": int(len(events)),
        "event_study_rows": count,
        "report_file": str(path),
    }


def write_event_study_report(settings: Settings, start_date: str, end_date: str, rows: list[dict[str, Any]]) -> Path:
    path = report_dir(settings) / f"event_study_{normalize_date(end_date).replace('-', '')}.md"
    data = pd.DataFrame(rows)
    lines = [
        f"# Event Study {normalize_date(start_date)} to {normalize_date(end_date)}",
        "",
        "收益口径：事件在 trade_date 盘后确认，下一交易日复权开盘买入，持有 N 个交易日，按复权收盘退出。",
        "龙虎榜、资金流、涨跌停均按盘后可知信息处理，默认只能用于次日交易。",
        "",
    ]
    if data.empty:
        lines.append("No event-study rows.")
    else:
        subset = data[(data["horizon_days"] == 5) & (data["market_regime"] == "all") & (data["event_subtype"] == "")].copy()
        if subset.empty:
            subset = data[(data["horizon_days"] == 5) & (data["market_regime"] == "all")].copy()
        subset["avg_excess_return"] = pd.to_numeric(subset["avg_excess_return"], errors="coerce")
        lines.extend(["## Best Events, 5-Day Horizon", ""])
        for row in subset.sort_values("avg_excess_return", ascending=False).head(15).to_dict("records"):
            lines.append(
                f"- {row['event_name']} ({row['event_group']}): samples={row['sample_count']}, "
                f"avg={_format_pct(row['avg_return'])}, excess={_format_pct(row['avg_excess_return'])}, "
                f"win={_format_pct(row['win_rate'])}, mdd={_format_pct(row['max_drawdown'])}"
            )
        lines.extend(["", "## Weak / Negative Events", ""])
        for row in subset.sort_values("avg_excess_return", ascending=True).head(12).to_dict("records"):
            lines.append(
                f"- {row['event_name']} ({row['event_group']}): samples={row['sample_count']}, "
                f"avg={_format_pct(row['avg_return'])}, excess={_format_pct(row['avg_excess_return'])}"
            )
        reason_rows = data[(data["event_name"] == "lhb_reason") & (data["horizon_days"] == 5) & (data["market_regime"] == "all")].copy()
        if not reason_rows.empty:
            reason_rows["avg_excess_return"] = pd.to_numeric(reason_rows["avg_excess_return"], errors="coerce")
            lines.extend(["", "## LHB Reasons", ""])
            for row in reason_rows.sort_values("avg_excess_return", ascending=False).head(20).to_dict("records"):
                lines.append(
                    f"- {row['event_subtype']}: samples={row['sample_count']}, excess={_format_pct(row['avg_excess_return'])}, win={_format_pct(row['win_rate'])}"
                )
    lines.extend(["", "## Event Definitions", ""])
    for item in EVENT_DEFINITIONS:
        lines.append(f"- {item.name} ({item.group}): {item.description}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

