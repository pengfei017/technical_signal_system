from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .config import Settings
from .secrets import get_secret


@dataclass(frozen=True)
class PostgresSettings:
    host: str
    port: int
    dbname: str
    user: str
    password: str


def load_postgres_settings() -> PostgresSettings:
    return PostgresSettings(
        host=get_secret("POSTGRES_HOST", "127.0.0.1"),
        port=int(get_secret("POSTGRES_PORT", "5432")),
        dbname=get_secret("POSTGRES_DB", "codex_research"),
        user=get_secret("POSTGRES_USER", "postgres"),
        password=get_secret("POSTGRES_PASSWORD", ""),
    )


def connect(*, autocommit: bool = False) -> psycopg.Connection:
    settings = load_postgres_settings()
    conn = psycopg.connect(
        host=settings.host,
        port=settings.port,
        dbname=settings.dbname,
        user=settings.user,
        password=settings.password,
        autocommit=autocommit,
        row_factory=dict_row,
    )
    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'Asia/Shanghai'")
        cur.execute("SET client_encoding = 'UTF8'")
    return conn


def qname(settings: Settings, table: str) -> str:
    return f"{settings.schema}.{table}"


def init_schema(settings: Settings) -> None:
    schema = settings.schema
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.signal_runs (
                run_id text PRIMARY KEY,
                started_at timestamptz NOT NULL DEFAULT now(),
                finished_at timestamptz,
                status text NOT NULL,
                task text NOT NULL,
                trade_date date,
                message text,
                metrics jsonb NOT NULL DEFAULT '{{}}'::jsonb
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.trade_calendar (
                cal_date date PRIMARY KEY,
                is_open boolean NOT NULL,
                pretrade_date date,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.daily_bars (
                ts_code text NOT NULL,
                trade_date date NOT NULL,
                open numeric,
                high numeric,
                low numeric,
                close numeric,
                pre_close numeric,
                change numeric,
                pct_chg numeric,
                vol numeric,
                amount numeric,
                adj_factor numeric,
                adj_open numeric,
                adj_high numeric,
                adj_low numeric,
                adj_close numeric,
                source text NOT NULL DEFAULT 'tushare',
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (ts_code, trade_date)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.daily_basic (
                ts_code text NOT NULL,
                trade_date date NOT NULL,
                turnover_rate numeric,
                turnover_rate_f numeric,
                volume_ratio numeric,
                pe numeric,
                pe_ttm numeric,
                pb numeric,
                ps numeric,
                ps_ttm numeric,
                dv_ratio numeric,
                dv_ttm numeric,
                total_share numeric,
                float_share numeric,
                free_share numeric,
                total_mv numeric,
                circ_mv numeric,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (ts_code, trade_date)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.moneyflow_daily (
                ts_code text NOT NULL,
                trade_date date NOT NULL,
                buy_sm_amount numeric,
                sell_sm_amount numeric,
                buy_md_amount numeric,
                sell_md_amount numeric,
                buy_lg_amount numeric,
                sell_lg_amount numeric,
                buy_elg_amount numeric,
                sell_elg_amount numeric,
                net_mf_amount numeric,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (ts_code, trade_date)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.moneyflow_stock (
                ts_code text NOT NULL,
                trade_date date NOT NULL,
                name text,
                close numeric,
                pct_chg numeric,
                amount_yi numeric,
                turnover_rate numeric,
                buy_sm_amount numeric,
                sell_sm_amount numeric,
                buy_md_amount numeric,
                sell_md_amount numeric,
                buy_lg_amount numeric,
                sell_lg_amount numeric,
                buy_elg_amount numeric,
                sell_elg_amount numeric,
                net_mf_amount numeric,
                net_mf_amount_yi numeric,
                net_mf_rate numeric,
                source text NOT NULL DEFAULT 'tushare.moneyflow',
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (ts_code, trade_date)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.moneyflow_market (
                trade_date date PRIMARY KEY,
                pct_change_sh numeric,
                pct_change_sz numeric,
                net_amount_yi numeric,
                net_amount_rate numeric,
                buy_elg_amount_yi numeric,
                buy_lg_amount_yi numeric,
                buy_md_amount_yi numeric,
                buy_sm_amount_yi numeric,
                source text NOT NULL DEFAULT 'tushare.moneyflow_mkt_dc',
                raw jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.moneyflow_industry (
                trade_date date NOT NULL,
                theme_name text NOT NULL,
                source text NOT NULL,
                pct_chg numeric,
                net_amount_yi numeric,
                net_buy_amount_yi numeric,
                net_sell_amount_yi numeric,
                lead_stock text,
                rank numeric,
                raw jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, source, theme_name)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.moneyflow_concept (
                trade_date date NOT NULL,
                theme_name text NOT NULL,
                source text NOT NULL,
                pct_chg numeric,
                net_amount_yi numeric,
                net_buy_amount_yi numeric,
                net_sell_amount_yi numeric,
                lead_stock text,
                company_num numeric,
                raw jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, source, theme_name)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.index_daily (
                trade_date date NOT NULL,
                index_code text NOT NULL,
                index_name text,
                open numeric,
                high numeric,
                low numeric,
                close numeric,
                pre_close numeric,
                change numeric,
                pct_chg numeric,
                vol numeric,
                amount numeric,
                source text NOT NULL DEFAULT 'tushare.index_daily',
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, index_code)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.global_index_daily (
                trade_date date NOT NULL,
                market_date date NOT NULL,
                region text NOT NULL,
                index_code text NOT NULL,
                index_name text,
                open numeric,
                high numeric,
                low numeric,
                close numeric,
                pre_close numeric,
                change numeric,
                pct_chg numeric,
                source text NOT NULL DEFAULT 'tushare.index_global',
                data_status text NOT NULL DEFAULT '',
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, index_code)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.limit_events (
                trade_date date NOT NULL,
                ts_code text NOT NULL,
                name text,
                industry text,
                limit_type text NOT NULL,
                close numeric,
                pct_chg numeric,
                amount_yi numeric,
                turnover_rate numeric,
                fd_amount_yi numeric,
                first_limit_time text,
                last_limit_time text,
                open_times numeric,
                limit_times numeric,
                source text NOT NULL DEFAULT 'tushare.limit_list_d',
                raw jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, ts_code, limit_type)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.limit_market_stats (
                trade_date date PRIMARY KEY,
                limit_up_count integer NOT NULL DEFAULT 0,
                limit_down_count integer NOT NULL DEFAULT 0,
                broken_count integer NOT NULL DEFAULT 0,
                broken_rate numeric,
                max_board integer NOT NULL DEFAULT 0,
                limit_up_industry_distribution jsonb NOT NULL DEFAULT '[]'::jsonb,
                previous_limit_positive jsonb NOT NULL DEFAULT '[]'::jsonb,
                previous_limit_negative jsonb NOT NULL DEFAULT '[]'::jsonb,
                source text,
                warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.lhb_stocks (
                trade_date date NOT NULL,
                ts_code text NOT NULL,
                name text,
                close numeric,
                pct_change numeric,
                turnover_rate numeric,
                amount_yi numeric,
                lhb_amount_yi numeric,
                lhb_net_buy_yi numeric,
                net_rate numeric,
                amount_rate numeric,
                institution_net_buy_yi numeric,
                northbound_net_buy_yi numeric,
                broker_seat_net_buy_yi numeric,
                top_count integer NOT NULL DEFAULT 0,
                primary_reason text,
                reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
                top_seats jsonb NOT NULL DEFAULT '[]'::jsonb,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, ts_code)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.lhb_seats (
                trade_date date NOT NULL,
                ts_code text NOT NULL,
                name text,
                exalter text NOT NULL,
                seat_type text NOT NULL,
                buy_yi numeric,
                sell_yi numeric,
                net_buy_yi numeric,
                reason text NOT NULL DEFAULT '',
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, ts_code, exalter, reason)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.signal_universe (
                ts_code text PRIMARY KEY,
                name text,
                source_types text[] NOT NULL DEFAULT ARRAY[]::text[],
                is_watchlist boolean NOT NULL DEFAULT false,
                is_candidate boolean NOT NULL DEFAULT false,
                candidate_status text,
                codex_rating text,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.technical_signals (
                ts_code text NOT NULL,
                trade_date date NOT NULL,
                name text,
                close numeric,
                pct_chg numeric,
                amount numeric,
                turnover_rate numeric,
                ma5 numeric,
                ma10 numeric,
                ma20 numeric,
                ma60 numeric,
                bias5 numeric,
                rsi14 numeric,
                macd numeric,
                macd_signal numeric,
                macd_hist numeric,
                vol_ma5 numeric,
                vol_ma20 numeric,
                prev_vol_ma5 numeric,
                prev_vol_ma20 numeric,
                volume_ratio_5 numeric,
                volume_ratio_20 numeric,
                volume_state text,
                trend_phase text,
                signal_level text,
                signal_score numeric,
                tags jsonb NOT NULL DEFAULT '[]'::jsonb,
                risk_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
                reason text,
                data_quality jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (ts_code, trade_date)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.latest_signals (
                ts_code text PRIMARY KEY,
                trade_date date NOT NULL,
                name text,
                close numeric,
                pct_chg numeric,
                trend_phase text,
                signal_level text,
                signal_score numeric,
                volume_state text,
                volume_ratio_5 numeric,
                volume_ratio_20 numeric,
                tags jsonb NOT NULL DEFAULT '[]'::jsonb,
                risk_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
                reason text,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.stock_signal_daily (
                trade_date date NOT NULL,
                ts_code text NOT NULL,
                name text,
                industry text,
                concepts text,
                close numeric,
                pct_chg numeric,
                amount_yi numeric,
                turnover_rate numeric,
                volume_ratio numeric,
                technical_score numeric,
                price_volume_score numeric,
                moneyflow_score numeric,
                limit_score numeric,
                lhb_score numeric,
                total_signal_score numeric,
                signal_level text,
                trend_phase text,
                volume_state text,
                limit_status text,
                is_limit_up boolean NOT NULL DEFAULT false,
                is_limit_down boolean NOT NULL DEFAULT false,
                is_broken_board boolean NOT NULL DEFAULT false,
                limit_times numeric,
                open_times numeric,
                first_limit_time text,
                last_limit_time text,
                net_mf_amount numeric,
                net_mf_amount_yi numeric,
                net_mf_rate numeric,
                lhb_net_buy_yi numeric,
                institution_net_buy_yi numeric,
                northbound_net_buy_yi numeric,
                lhb_reason text,
                tags jsonb NOT NULL DEFAULT '[]'::jsonb,
                risk_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
                reason text,
                data_quality jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, ts_code)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.theme_signal_daily (
                trade_date date NOT NULL,
                theme_type text NOT NULL,
                theme_name text NOT NULL,
                source text,
                pct_chg numeric,
                net_amount_yi numeric,
                limit_up_count integer NOT NULL DEFAULT 0,
                broken_count integer NOT NULL DEFAULT 0,
                strong_stock_count integer NOT NULL DEFAULT 0,
                top_stocks jsonb NOT NULL DEFAULT '[]'::jsonb,
                related_concepts jsonb NOT NULL DEFAULT '[]'::jsonb,
                heat_score numeric,
                momentum_score numeric,
                persistence_days integer,
                signal_level text,
                reason text,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, theme_type, theme_name, source)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.dragon_leader_daily (
                trade_date date NOT NULL,
                ts_code text NOT NULL,
                name text,
                industry text,
                concepts text,
                pct_chg numeric,
                amount_yi numeric,
                turnover_rate numeric,
                volume_ratio numeric,
                limit_status text,
                is_limit_up boolean NOT NULL DEFAULT false,
                is_broken_board boolean NOT NULL DEFAULT false,
                limit_times numeric,
                lhb_net_buy_yi numeric,
                institution_net_buy_yi numeric,
                northbound_net_buy_yi numeric,
                theme_names jsonb NOT NULL DEFAULT '[]'::jsonb,
                leader_score numeric,
                leader_rank integer,
                leader_level text,
                reason text,
                risk_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (trade_date, ts_code)
            )
            """
        )
        cur.execute(f"ALTER TABLE {schema}.technical_signals ADD COLUMN IF NOT EXISTS prev_vol_ma5 numeric")
        cur.execute(f"ALTER TABLE {schema}.technical_signals ADD COLUMN IF NOT EXISTS prev_vol_ma20 numeric")
        cur.execute(f"ALTER TABLE {schema}.technical_signals ADD COLUMN IF NOT EXISTS volume_ratio_5 numeric")
        cur.execute(f"ALTER TABLE {schema}.technical_signals ADD COLUMN IF NOT EXISTS volume_ratio_20 numeric")
        cur.execute(f"ALTER TABLE {schema}.technical_signals ADD COLUMN IF NOT EXISTS volume_state text")
        cur.execute(f"ALTER TABLE {schema}.latest_signals ADD COLUMN IF NOT EXISTS volume_state text")
        cur.execute(f"ALTER TABLE {schema}.latest_signals ADD COLUMN IF NOT EXISTS volume_ratio_5 numeric")
        cur.execute(f"ALTER TABLE {schema}.latest_signals ADD COLUMN IF NOT EXISTS volume_ratio_20 numeric")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_daily_bars_date ON {schema}.daily_bars(trade_date)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_daily_basic_date ON {schema}.daily_basic(trade_date)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_index_daily_date ON {schema}.index_daily(trade_date)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_global_index_daily_date ON {schema}.global_index_daily(trade_date, market_date)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_signals_date ON {schema}.technical_signals(trade_date)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_signals_level ON {schema}.technical_signals(signal_level)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_stock_signal_date_score ON {schema}.stock_signal_daily(trade_date, total_signal_score DESC)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_stock_signal_level ON {schema}.stock_signal_daily(signal_level)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_theme_signal_date_score ON {schema}.theme_signal_daily(trade_date, heat_score DESC)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_dragon_leader_date_rank ON {schema}.dragon_leader_daily(trade_date, leader_rank)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_limit_events_date_type ON {schema}.limit_events(trade_date, limit_type)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_lhb_stocks_date_net ON {schema}.lhb_stocks(trade_date, lhb_net_buy_yi DESC)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{schema}_moneyflow_stock_date ON {schema}.moneyflow_stock(trade_date)")
        conn.commit()


def upsert_rows(
    conn: psycopg.Connection,
    *,
    table: str,
    columns: list[str],
    rows: Iterable[dict[str, Any]],
    conflict_columns: list[str],
    update_columns: list[str] | None = None,
    batch_size: int = 2000,
) -> int:
    rows_list = list(rows)
    if not rows_list:
        return 0
    update_columns = update_columns or [c for c in columns if c not in conflict_columns]
    placeholders = ", ".join(["%s"] * len(columns))
    col_sql = ", ".join(columns)
    conflict_sql = ", ".join(conflict_columns)
    update_sql = ", ".join(f"{col}=EXCLUDED.{col}" for col in update_columns)
    sql = (
        f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_sql}) DO UPDATE SET {update_sql}"
    )
    total = 0
    with conn.cursor() as cur:
        for idx in range(0, len(rows_list), batch_size):
            chunk = rows_list[idx : idx + batch_size]
            values = [tuple(row.get(col) for col in columns) for row in chunk]
            cur.executemany(sql, values)
            total += len(chunk)
    return total
