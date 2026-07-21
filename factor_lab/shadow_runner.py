from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tech_signal.config import Settings
from tech_signal.db import connect, init_schema, qname, upsert_rows

from .backtest_engine import _load_trade_returns, _score_frame
from .factor_definitions import normalize_date
from .factor_evaluator import _clean_float, report_dir
from .factor_evaluator import build_factor_daily
from .short_strength import SHORT_MODEL_NAME, SHORT_TRACKING_HORIZONS, build_short_strength_shadow_rows
from .trend_pure import TREND_MODEL_NAME, TREND_TRACKING_HORIZONS, build_trend_pure_shadow_rows
from .weight_optimizer import load_model_weights, optimize_weights


@dataclass(frozen=True)
class ShadowModelSpec:
    model_name: str
    top_n: int
    hold_days: int


DEFAULT_SHADOW_MODELS = (
    ShadowModelSpec(TREND_MODEL_NAME, 20, 5),
    ShadowModelSpec(TREND_MODEL_NAME, 30, 5),
    ShadowModelSpec(TREND_MODEL_NAME, 20, 10),
    ShadowModelSpec(TREND_MODEL_NAME, 30, 10),
)
TRACKING_HORIZONS = (1, 3, 5, 10, 20)


def _latest_signal_date(settings: Settings) -> str:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT max(trade_date)::text AS d FROM {qname(settings, 'stock_signal_daily')}")
        row = cur.fetchone()
    if not row or not row["d"]:
        raise RuntimeError("No stock_signal_daily trade_date available for factor shadow run")
    return str(row["d"])


def _open_date_floor(settings: Settings, end_date: str, lookback: int) -> str:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT min(cal_date)::text AS d
            FROM (
                SELECT cal_date
                FROM {qname(settings, 'trade_calendar')}
                WHERE is_open=true AND cal_date <= %s
                ORDER BY cal_date DESC
                LIMIT %s
            ) x
            """,
            (normalize_date(end_date), int(lookback)),
        )
        row = cur.fetchone()
    return str(row["d"]) if row and row["d"] else normalize_date(end_date)


def _factor_rows_available(settings: Settings, trade_date: str) -> bool:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT count(DISTINCT factor_name) AS factor_count, count(*) AS row_count
            FROM {qname(settings, 'factor_daily')}
            WHERE trade_date=%s
            """,
            (normalize_date(trade_date),),
        )
        row = cur.fetchone()
    return int(row["factor_count"] or 0) >= 10 and int(row["row_count"] or 0) > 0


def _ensure_factor_rows(settings: Settings, trade_date: str) -> dict[str, Any]:
    if _factor_rows_available(settings, trade_date):
        return {"status": "reused", "trade_date": normalize_date(trade_date)}
    return build_factor_daily(settings, trade_date, trade_date)


def _latest_weight_meta(
    settings: Settings,
    model_name: str,
    horizon_days: int,
    as_of_date: str,
) -> dict[str, Any] | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT as_of_date::text AS as_of_date,
                   train_start::text AS train_start,
                   train_end::text AS train_end,
                   count(*) AS row_count
            FROM {qname(settings, 'model_weight_history')}
            WHERE model_name=%s
              AND horizon_days=%s
              AND as_of_date <= %s
            GROUP BY as_of_date, train_start, train_end
            ORDER BY as_of_date DESC, train_end DESC NULLS LAST
            LIMIT 1
            """,
            (model_name, int(horizon_days), normalize_date(as_of_date)),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _ensure_weights(
    settings: Settings,
    *,
    model_name: str,
    horizon_days: int,
    as_of_date: str,
    research_start_date: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    meta = _latest_weight_meta(settings, model_name, horizon_days, as_of_date)
    if meta is None:
        optimize_weights(
            settings,
            research_start_date,
            as_of_date,
            as_of_date=as_of_date,
            horizon_days=horizon_days,
        )
        meta = _latest_weight_meta(settings, model_name, horizon_days, as_of_date)
    if meta is None:
        raise RuntimeError(f"No shadow weights for {model_name} hold {horizon_days} as of {as_of_date}")
    weights = load_model_weights(
        settings,
        model_name,
        str(meta["as_of_date"]),
        horizon_days=horizon_days,
        train_start=str(meta["train_start"]) if meta.get("train_start") else None,
        train_end=str(meta["train_end"]) if meta.get("train_end") else None,
        fallback=False,
    )
    if not weights:
        raise RuntimeError(f"Empty shadow weights for {model_name} hold {horizon_days} as of {meta['as_of_date']}")
    return weights, meta


def _load_production_ranks(settings: Settings, trade_date: str) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT trade_date, ts_code, name, industry, concepts, close, pct_chg,
                   total_signal_score AS production_score,
                   row_number() OVER (ORDER BY total_signal_score DESC NULLS LAST, ts_code) AS production_rank
            FROM {qname(settings, 'stock_signal_daily')}
            WHERE trade_date=%s
            """,
            (normalize_date(trade_date),),
        )
        rows = cur.fetchall()
    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError(f"No stock_signal_daily rows for {trade_date}")
    for column in ["production_score", "production_rank", "close", "pct_chg"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _reason_for_row(row: pd.Series, production_top_n: int) -> str:
    factor_rank = row.get("factor_rank")
    production_rank = row.get("production_rank")
    comparison_type = str(row.get("comparison_type") or "")
    if comparison_type == "confluence":
        return f"生产排名{int(production_rank)} + 因子模型排名{int(factor_rank)}"
    if comparison_type == "factor_only":
        if pd.notna(production_rank):
            return f"因子模型排名{int(factor_rank)}，生产排名{int(production_rank)}，未进生产Top{production_top_n}"
        return f"因子模型排名{int(factor_rank)}，生产无有效排名"
    if comparison_type == "production_only":
        if pd.notna(factor_rank):
            return f"生产排名{int(production_rank)}，因子模型排名{int(factor_rank)}，未进模型Top{int(row['top_n'])}"
        return f"生产排名{int(production_rank)}，因子模型无有效排名"
    return ""


def _build_candidates_for_model(
    settings: Settings,
    *,
    trade_date: str,
    spec: ShadowModelSpec,
    production_top_n: int,
    research_start_date: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    production = _load_production_ranks(settings, trade_date)
    if spec.model_name == TREND_MODEL_NAME:
        return build_trend_pure_shadow_rows(
            settings,
            trade_date=trade_date,
            top_n=spec.top_n,
            hold_days=spec.hold_days,
            production=production,
            production_top_n=production_top_n,
        )
    if spec.model_name == SHORT_MODEL_NAME:
        return build_short_strength_shadow_rows(
            settings,
            trade_date=trade_date,
            top_n=spec.top_n,
            hold_days=spec.hold_days,
            production=production,
            production_top_n=production_top_n,
        )
    weights, weight_meta = _ensure_weights(
        settings,
        model_name=spec.model_name,
        horizon_days=spec.hold_days,
        as_of_date=trade_date,
        research_start_date=research_start_date,
    )
    scored = _score_frame(settings, trade_date, trade_date, weights)
    scored = scored.sort_values(["trade_date", "factor_score", "ts_code"], ascending=[True, False, True]).copy()
    scored["factor_rank"] = range(1, len(scored) + 1)
    merged = scored.merge(
        production[
            [
                "ts_code",
                "production_score",
                "production_rank",
                "name",
                "industry",
                "concepts",
            ]
        ],
        on="ts_code",
        how="outer",
        suffixes=("", "_prod"),
    )
    for column in ["name", "industry", "concepts"]:
        prod_col = f"{column}_prod"
        if prod_col in merged.columns:
            merged[column] = merged[column].fillna(merged[prod_col])
            merged = merged.drop(columns=[prod_col])

    model_top = merged[merged["factor_rank"] <= spec.top_n].copy()
    production_top_codes = set(production[production["production_rank"] <= production_top_n]["ts_code"].astype(str))
    model_top_codes = set(model_top["ts_code"].astype(str))
    union_codes = model_top_codes | production_top_codes
    candidates = merged[merged["ts_code"].astype(str).isin(union_codes)].copy()

    candidates["is_model_pick"] = candidates["ts_code"].astype(str).isin(model_top_codes)
    candidates["is_production_pick"] = candidates["ts_code"].astype(str).isin(production_top_codes)
    candidates["comparison_type"] = "factor_only"
    candidates.loc[candidates["is_model_pick"] & candidates["is_production_pick"], "comparison_type"] = "confluence"
    candidates.loc[(~candidates["is_model_pick"]) & candidates["is_production_pick"], "comparison_type"] = "production_only"
    candidates["model_name"] = spec.model_name
    candidates["top_n"] = spec.top_n
    candidates["hold_days"] = spec.hold_days
    candidates["reason"] = candidates.apply(lambda row: _reason_for_row(row, production_top_n), axis=1)

    rows: list[dict[str, Any]] = []
    for row in candidates.sort_values(["comparison_type", "factor_rank"], na_position="last").to_dict("records"):
        rows.append(
            {
                "trade_date": normalize_date(trade_date),
                "model_name": spec.model_name,
                "top_n": spec.top_n,
                "hold_days": spec.hold_days,
                "ts_code": row.get("ts_code"),
                "name": row.get("name"),
                "industry": row.get("industry"),
                "concepts": row.get("concepts"),
                "factor_rank": int(row["factor_rank"]) if pd.notna(row.get("factor_rank")) else None,
                "factor_score": _clean_float(row.get("factor_score")),
                "production_rank": int(row["production_rank"]) if pd.notna(row.get("production_rank")) else None,
                "production_score": _clean_float(row.get("production_score")),
                "comparison_type": row.get("comparison_type"),
                "is_model_pick": bool(row.get("is_model_pick")),
                "is_production_pick": bool(row.get("is_production_pick")),
                "reason": row.get("reason"),
            }
        )
    return rows, {
        "model_name": spec.model_name,
        "top_n": spec.top_n,
        "hold_days": spec.hold_days,
        "weight_as_of_date": weight_meta.get("as_of_date"),
        "weight_train_start": weight_meta.get("train_start"),
        "weight_train_end": weight_meta.get("train_end"),
        "candidate_rows": len(rows),
        "model_pick_rows": len(model_top_codes),
        "production_top_rows": len(production_top_codes),
    }


def _clear_shadow_rows(
    settings: Settings,
    *,
    trade_date: str,
    model_specs: tuple[ShadowModelSpec, ...],
) -> None:
    active_model_names = sorted({spec.model_name for spec in model_specs})
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {qname(settings, 'factor_shadow_candidates')} WHERE trade_date=%s AND NOT (model_name = ANY(%s))",
            (normalize_date(trade_date), active_model_names),
        )
        cur.execute(
            f"DELETE FROM {qname(settings, 'factor_shadow_tracking')} WHERE signal_date=%s AND NOT (model_name = ANY(%s))",
            (normalize_date(trade_date), active_model_names),
        )
        for spec in model_specs:
            cur.execute(
                f"""
                DELETE FROM {qname(settings, 'factor_shadow_candidates')}
                WHERE trade_date=%s AND model_name=%s AND top_n=%s AND hold_days=%s
                """,
                (normalize_date(trade_date), spec.model_name, int(spec.top_n), int(spec.hold_days)),
            )
            cur.execute(
                f"""
                DELETE FROM {qname(settings, 'factor_shadow_tracking')}
                WHERE signal_date=%s AND model_name=%s AND top_n=%s AND hold_days=%s
                """,
                (normalize_date(trade_date), spec.model_name, int(spec.top_n), int(spec.hold_days)),
            )
        conn.commit()


def _upsert_candidates(settings: Settings, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    columns = [
        "trade_date",
        "model_name",
        "top_n",
        "hold_days",
        "ts_code",
        "name",
        "industry",
        "concepts",
        "factor_rank",
        "factor_score",
        "production_rank",
        "production_score",
        "comparison_type",
        "is_model_pick",
        "is_production_pick",
        "reason",
    ]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "factor_shadow_candidates"),
            columns=columns,
            rows=rows,
            conflict_columns=["trade_date", "model_name", "top_n", "hold_days", "ts_code"],
        )
        conn.commit()
    return count


def _load_recent_candidates(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    model_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    params: list[Any] = [normalize_date(start_date), normalize_date(end_date)]
    model_filter = ""
    if model_names:
        model_filter = "AND model_name = ANY(%s)"
        params.append(list(model_names))
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT trade_date, model_name, top_n, hold_days, ts_code, comparison_type
            FROM {qname(settings, 'factor_shadow_candidates')}
            WHERE trade_date BETWEEN %s AND %s
            {model_filter}
            """,
            tuple(params),
        )
        rows = cur.fetchall()
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.date
    data["top_n"] = pd.to_numeric(data["top_n"], errors="coerce").astype("Int64")
    data["hold_days"] = pd.to_numeric(data["hold_days"], errors="coerce").astype("Int64")
    return data


def _load_current_prices(settings: Settings, start_date: str, end_date: str, ts_codes: list[str]) -> pd.DataFrame:
    codes = sorted({str(code) for code in ts_codes if code})
    if not codes:
        return pd.DataFrame()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT ON (ts_code)
                   ts_code,
                   trade_date AS current_trade_date,
                   adj_close AS current_price
            FROM {qname(settings, 'daily_bars')}
            WHERE trade_date BETWEEN %s AND %s
              AND ts_code = ANY(%s)
              AND adj_close IS NOT NULL
            ORDER BY ts_code, trade_date DESC
            """,
            (normalize_date(start_date), normalize_date(end_date), codes),
        )
        rows = cur.fetchall()
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data["current_trade_date"] = pd.to_datetime(data["current_trade_date"]).dt.date
    data["current_price"] = pd.to_numeric(data["current_price"], errors="coerce")
    return data


def _open_date_index(settings: Settings, start_date: str, end_date: str) -> dict[Any, int]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT cal_date
            FROM {qname(settings, 'trade_calendar')}
            WHERE is_open=true AND cal_date BETWEEN %s AND %s
            ORDER BY cal_date
            """,
            (normalize_date(start_date), normalize_date(end_date)),
        )
        rows = cur.fetchall()
    return {row["cal_date"]: idx for idx, row in enumerate(rows)}


def _elapsed_open_days(open_index: dict[Any, int], entry_date: Any, current_date: Any) -> int | None:
    if pd.isna(entry_date) or pd.isna(current_date):
        return None
    entry_key = pd.to_datetime(entry_date).date()
    current_key = pd.to_datetime(current_date).date()
    if entry_key not in open_index or current_key not in open_index:
        return None
    if open_index[current_key] < open_index[entry_key]:
        return None
    return int(open_index[current_key] - open_index[entry_key] + 1)


def _horizons_for_model(model_name: str) -> tuple[int, ...]:
    if model_name == TREND_MODEL_NAME:
        return TREND_TRACKING_HORIZONS
    if model_name == SHORT_MODEL_NAME:
        return SHORT_TRACKING_HORIZONS
    return TRACKING_HORIZONS


def _update_tracking(
    settings: Settings,
    *,
    end_date: str,
    lookback_days: int,
    model_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    start_date = _open_date_floor(settings, end_date, lookback_days)
    candidates = _load_recent_candidates(settings, start_date, end_date, model_names=model_names)
    if candidates.empty:
        return {"tracking_start_date": start_date, "tracking_rows": 0}

    current_prices = _load_current_prices(settings, start_date, end_date, candidates["ts_code"].astype(str).tolist())
    open_index = _open_date_index(settings, start_date, end_date)
    rows: list[dict[str, Any]] = []
    for horizon in TRACKING_HORIZONS:
        horizon_candidates = candidates[
            candidates["model_name"].astype(str).map(lambda name: horizon in _horizons_for_model(name))
        ].copy()
        if horizon_candidates.empty:
            continue
        returns = _load_trade_returns(settings, start_date, end_date, horizon)
        if returns.empty:
            continue
        returns = returns[
            [
                "trade_date",
                "ts_code",
                "entry_date",
                "entry_price",
                "exit_date",
                "exit_price",
                "future_return",
            ]
        ].copy()
        merged = horizon_candidates.merge(returns, on=["trade_date", "ts_code"], how="left")
        if not current_prices.empty:
            merged = merged.merge(current_prices, on="ts_code", how="left")
        else:
            merged["current_trade_date"] = None
            merged["current_price"] = None
        for row in merged.to_dict("records"):
            is_complete = pd.notna(row.get("future_return"))
            entry_date = row.get("entry_date") if pd.notna(row.get("entry_date")) else None
            entry_price = _clean_float(row.get("entry_price"))
            exit_date = row.get("exit_date") if pd.notna(row.get("exit_date")) else None
            exit_price = _clean_float(row.get("exit_price"))
            current_date = None
            current_price = None
            current_return = None
            elapsed_days = None
            if is_complete:
                current_date = exit_date
                current_price = exit_price
                current_return = _clean_float(row.get("future_return"))
                elapsed_days = int(horizon)
            elif entry_date is not None and entry_price is not None and pd.notna(row.get("current_trade_date")) and pd.notna(row.get("current_price")):
                latest_date = pd.to_datetime(row.get("current_trade_date")).date()
                entry_date_value = pd.to_datetime(entry_date).date()
                if latest_date >= entry_date_value:
                    current_date = latest_date
                    current_price = _clean_float(row.get("current_price"))
                    if current_price is not None and entry_price:
                        current_return = _clean_float(float(current_price) / float(entry_price) - 1.0)
                    elapsed_days = _elapsed_open_days(open_index, entry_date, latest_date)
            rows.append(
                {
                    "signal_date": row.get("trade_date"),
                    "model_name": row.get("model_name"),
                    "top_n": int(row.get("top_n")),
                    "hold_days": int(row.get("hold_days")),
                    "ts_code": row.get("ts_code"),
                    "horizon_days": horizon,
                    "comparison_type": row.get("comparison_type"),
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "return_pct": _clean_float(row.get("future_return")),
                    "current_trade_date": current_date,
                    "current_price": current_price,
                    "current_return_pct": current_return,
                    "elapsed_days": elapsed_days,
                    "is_complete": bool(is_complete),
                }
            )

    if not rows:
        return {"tracking_start_date": start_date, "tracking_rows": 0}
    columns = [
        "signal_date",
        "model_name",
        "top_n",
        "hold_days",
        "ts_code",
        "horizon_days",
        "comparison_type",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "return_pct",
        "current_trade_date",
        "current_price",
        "current_return_pct",
        "elapsed_days",
        "is_complete",
    ]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "factor_shadow_tracking"),
            columns=columns,
            rows=rows,
            conflict_columns=["signal_date", "model_name", "top_n", "hold_days", "ts_code", "horizon_days"],
        )
        conn.commit()
    return {
        "tracking_start_date": start_date,
        "tracking_rows": count,
        "tracking_signal_dates": int(candidates["trade_date"].nunique()),
    }


def _format_number(value: Any, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def _format_pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return ""


def _query_df(settings: Settings, sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def write_shadow_report(
    settings: Settings,
    *,
    trade_date: str,
    production_top_n: int,
    tracking_start_date: str,
    model_names: tuple[str, ...] | None = None,
) -> str:
    date_value = normalize_date(trade_date)
    compact = date_value.replace("-", "")
    active_model_names = list(model_names or [])
    model_filter = "AND model_name = ANY(%s)" if active_model_names else ""
    candidate_model_filter = "AND c.model_name = ANY(%s)" if active_model_names else ""
    active_detail_model_filter = "AND t.model_name = ANY(%s)" if active_model_names else ""
    candidates = _query_df(
        settings,
        f"""
        SELECT c.*,
               t.current_trade_date,
               t.current_return_pct,
               t.elapsed_days,
               t.return_pct AS complete_return_pct,
               t.is_complete AS tracking_complete
        FROM {qname(settings, 'factor_shadow_candidates')} c
        LEFT JOIN {qname(settings, 'factor_shadow_tracking')} t
          ON t.signal_date=c.trade_date
         AND t.model_name=c.model_name
         AND t.top_n=c.top_n
         AND t.hold_days=c.hold_days
         AND t.ts_code=c.ts_code
         AND t.horizon_days=c.hold_days
        WHERE c.trade_date=%s
        {candidate_model_filter}
        ORDER BY c.model_name, c.top_n, c.hold_days, c.comparison_type, c.factor_rank NULLS LAST, c.production_rank NULLS LAST
        """,
        tuple([date_value] + ([active_model_names] if active_model_names else [])),
    )
    tracking = _query_df(
        settings,
        f"""
        SELECT model_name, top_n, hold_days, comparison_type, horizon_days,
               count(*) FILTER (WHERE is_complete) AS complete_count,
               avg(return_pct) FILTER (WHERE is_complete) AS avg_return,
               avg((return_pct > 0)::int) FILTER (WHERE is_complete) AS win_rate,
               count(*) FILTER (WHERE NOT is_complete AND current_return_pct IS NOT NULL) AS active_count,
               max(current_trade_date) FILTER (WHERE NOT is_complete AND current_return_pct IS NOT NULL) AS current_trade_date,
               avg(current_return_pct) FILTER (WHERE NOT is_complete AND current_return_pct IS NOT NULL) AS avg_current_return,
               avg((current_return_pct > 0)::int) FILTER (WHERE NOT is_complete AND current_return_pct IS NOT NULL) AS current_win_rate
        FROM {qname(settings, 'factor_shadow_tracking')}
        WHERE signal_date BETWEEN %s AND %s
        {model_filter}
        GROUP BY model_name, top_n, hold_days, comparison_type, horizon_days
        ORDER BY model_name, top_n, hold_days, comparison_type, horizon_days
        """,
        tuple([normalize_date(tracking_start_date), date_value] + ([active_model_names] if active_model_names else [])),
    )
    active_details = _query_df(
        settings,
        f"""
        SELECT t.signal_date,
               t.model_name,
               t.top_n,
               t.hold_days,
               t.horizon_days,
               t.comparison_type,
               c.ts_code,
               c.name,
               c.industry,
               c.factor_rank,
               c.production_rank,
               t.entry_date,
               t.current_trade_date,
               t.elapsed_days,
               t.current_return_pct
        FROM {qname(settings, 'factor_shadow_tracking')} t
        JOIN {qname(settings, 'factor_shadow_candidates')} c
          ON c.trade_date=t.signal_date
         AND c.model_name=t.model_name
         AND c.top_n=t.top_n
         AND c.hold_days=t.hold_days
         AND c.ts_code=t.ts_code
        WHERE t.signal_date BETWEEN %s AND %s
          AND t.horizon_days=t.hold_days
          AND t.is_complete=false
          AND t.current_return_pct IS NOT NULL
        {active_detail_model_filter}
        ORDER BY t.signal_date DESC, t.model_name, t.top_n, t.hold_days,
                 t.comparison_type, c.factor_rank NULLS LAST, c.production_rank NULLS LAST
        LIMIT 600
        """,
        tuple([normalize_date(tracking_start_date), date_value] + ([active_model_names] if active_model_names else [])),
    )
    days = _query_df(
        settings,
        f"""
        SELECT count(DISTINCT trade_date) AS shadow_days
        FROM {qname(settings, 'factor_shadow_candidates')}
        WHERE trade_date <= %s
        {model_filter}
        """,
        tuple([date_value] + ([active_model_names] if active_model_names else [])),
    )
    shadow_days = int(days.iloc[0]["shadow_days"] or 0) if not days.empty else 0

    path = report_dir(settings) / f"factor_shadow_{compact}.md"
    lines = [
        f"# Factor Shadow {date_value}",
        "",
        "研究影子运行输出，不修改 `stock_signal_daily` 生产评分权重。",
        "",
        f"- 生产对比口径：`stock_signal_daily.total_signal_score` Top{production_top_n}",
        "- 交易口径：收盘后出信号，下一交易日复权开盘买入；trend_pure_v1 跟踪 3/5/10/20 日；回测/跟踪收益使用板块涨跌停、一字跌停顺延退出和成交额容量约束。",
        "- `current_return` 为未到期样本按当前最新复权收盘价计算的浮动收益；`complete_return` 为走完对应持有窗口后的正式结算收益。",
        "- 当前默认影子模型：只跑纯技术趋势 `trend_pure_v1`；旧的 event_adjusted_v1、walk_forward_v1、short_strength_v1 保留为历史研究命令，不再进入每日默认影子名单。",
        f"- 已累计影子交易日：{shadow_days}",
        "",
    ]
    if candidates.empty:
        lines.append("- No shadow candidates generated.")
    else:
        candidates["factor_rank"] = pd.to_numeric(candidates["factor_rank"], errors="coerce")
        candidates["production_rank"] = pd.to_numeric(candidates["production_rank"], errors="coerce")
        for (model_name, top_n, hold_days), group in candidates.groupby(["model_name", "top_n", "hold_days"]):
            lines.extend(["", f"## {model_name} Top{top_n} Hold{hold_days}", ""])
            counts = group["comparison_type"].value_counts().to_dict()
            lines.append(
                f"- 共振 {int(counts.get('confluence', 0))}；模型强/生产弱 {int(counts.get('factor_only', 0))}；生产强/模型弱 {int(counts.get('production_only', 0))}"
            )
            for section_name, comparison_type in [
                ("生产评分强 + 因子模型强", "confluence"),
                ("模型强但生产未进Top", "factor_only"),
                ("生产Top但模型未进Top", "production_only"),
            ]:
                subset = group[group["comparison_type"] == comparison_type].copy()
                lines.extend(["", f"### {section_name}", ""])
                if subset.empty:
                    lines.append("- 无")
                    continue
                subset = subset.sort_values(["factor_rank", "production_rank"], na_position="last").head(30)
                lines.append("| code | name | industry | factor_rank | prod_rank | factor_score | prod_score | asof | current_return | complete_return | reason |")
                lines.append("|---|---|---|---:|---:|---:|---:|---|---:|---:|---|")
                for row in subset.to_dict("records"):
                    current_date = row.get("current_trade_date")
                    current_date_text = "" if pd.isna(current_date) else str(current_date)
                    lines.append(
                        f"| {row.get('ts_code', '')} | {row.get('name', '') or ''} | {row.get('industry', '') or ''} | "
                        f"{'' if pd.isna(row.get('factor_rank')) else int(row.get('factor_rank'))} | "
                        f"{'' if pd.isna(row.get('production_rank')) else int(row.get('production_rank'))} | "
                        f"{_format_number(row.get('factor_score'))} | {_format_number(row.get('production_score'))} | "
                        f"{current_date_text} | {_format_pct(row.get('current_return_pct'))} | {_format_pct(row.get('complete_return_pct'))} | "
                        f"{row.get('reason', '') or ''} |"
                    )
    lines.extend(["", "## 跟踪收益汇总", ""])
    if tracking.empty:
        lines.append("- 暂无可汇总的跟踪记录。")
    else:
        lines.append("| model | top | hold | type | horizon | complete | avg_return | win_rate | active | asof | current_return | current_win |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|")
        for row in tracking.to_dict("records"):
            current_date = row.get("current_trade_date")
            current_date_text = "" if pd.isna(current_date) else str(current_date)
            lines.append(
                f"| {row.get('model_name')} | {row.get('top_n')} | {row.get('hold_days')} | {row.get('comparison_type')} | "
                f"{row.get('horizon_days')} | {row.get('complete_count')} | {_format_pct(row.get('avg_return'))} | {_format_pct(row.get('win_rate'))} | "
                f"{row.get('active_count')} | {current_date_text} | {_format_pct(row.get('avg_current_return'))} | {_format_pct(row.get('current_win_rate'))} |"
            )
    lines.extend(["", "## 持仓中浮动收益明细", ""])
    if active_details.empty:
        lines.append("- 暂无已买入但未到期的影子样本。")
    else:
        lines.append("| signal | model | top | hold | type | code | name | factor_rank | prod_rank | entry | asof | days | current_return |")
        lines.append("|---|---|---:|---:|---|---|---|---:|---:|---|---|---:|---:|")
        for row in active_details.to_dict("records"):
            lines.append(
                f"| {row.get('signal_date')} | {row.get('model_name')} | {row.get('top_n')} | {row.get('hold_days')} | "
                f"{row.get('comparison_type')} | {row.get('ts_code')} | {row.get('name') or ''} | "
                f"{'' if pd.isna(row.get('factor_rank')) else int(row.get('factor_rank'))} | "
                f"{'' if pd.isna(row.get('production_rank')) else int(row.get('production_rank'))} | "
                f"{row.get('entry_date') or ''} | {row.get('current_trade_date') or ''} | "
                f"{'' if pd.isna(row.get('elapsed_days')) else int(row.get('elapsed_days'))} | "
                f"{_format_pct(row.get('current_return_pct'))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def run_shadow_pipeline(
    settings: Settings,
    *,
    trade_date: str | None = None,
    production_top_n: int = 30,
    tracking_lookback_days: int = 30,
    research_start_date: str = "2020-01-02",
    model_specs: tuple[ShadowModelSpec, ...] = DEFAULT_SHADOW_MODELS,
) -> dict[str, Any]:
    init_schema(settings)
    date_value = normalize_date(trade_date or _latest_signal_date(settings))
    active_model_names = tuple(sorted({spec.model_name for spec in model_specs}))
    if any(spec.model_name != TREND_MODEL_NAME for spec in model_specs):
        factor_metrics = _ensure_factor_rows(settings, date_value)
    else:
        factor_metrics = {"status": "skipped", "reason": "trend_pure_v1 uses daily_bars/daily_basic only"}
    all_rows: list[dict[str, Any]] = []
    model_metrics: list[dict[str, Any]] = []
    _clear_shadow_rows(settings, trade_date=date_value, model_specs=model_specs)
    for spec in model_specs:
        rows, metrics = _build_candidates_for_model(
            settings,
            trade_date=date_value,
            spec=spec,
            production_top_n=production_top_n,
            research_start_date=research_start_date,
        )
        all_rows.extend(rows)
        model_metrics.append(metrics)
    candidate_count = _upsert_candidates(settings, all_rows)
    tracking_metrics = _update_tracking(
        settings,
        end_date=date_value,
        lookback_days=tracking_lookback_days,
        model_names=active_model_names,
    )
    report_file = write_shadow_report(
        settings,
        trade_date=date_value,
        production_top_n=production_top_n,
        tracking_start_date=str(tracking_metrics.get("tracking_start_date") or date_value),
        model_names=active_model_names,
    )
    return {
        "shadow_trade_date": date_value,
        "shadow_factor_metrics": factor_metrics,
        "shadow_models": model_metrics,
        "shadow_candidate_rows": candidate_count,
        "shadow_tracking": tracking_metrics,
        "shadow_report_file": report_file,
    }
