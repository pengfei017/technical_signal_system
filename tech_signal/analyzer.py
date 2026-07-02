from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .config import Settings
from .db import connect, qname, upsert_rows
from .formula_spec import load_formula_spec, merged_signal_config
from .indicators import add_indicators
from .trading_signals import refresh_final_signal_layers


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


def _score_row(row: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    tags: list[str] = []
    risks: list[str] = []
    score = 50.0
    reason_parts: list[str] = []

    ma5, ma10, ma20, ma60 = row.get("ma5"), row.get("ma10"), row.get("ma20"), row.get("ma60")
    close = row.get("adj_close")
    volume_ratio_5 = row.get("volume_ratio_5")
    volume_ratio_20 = row.get("volume_ratio_20")
    volume_state = _volume_state(row, cfg)
    bias5 = row.get("bias5")
    rsi14 = row.get("rsi14")
    macd_hist = row.get("macd_hist")
    high20 = row.get("high20")

    if pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20):
        if ma5 > ma10 > ma20:
            score += 18
            tags.append("多头排列")
            reason_parts.append("MA5>MA10>MA20，短中期趋势向上")
        elif ma5 < ma10 < ma20:
            score -= 22
            risks.append("空头排列")
            reason_parts.append("MA5<MA10<MA20，趋势结构偏弱")

    if pd.notna(ma20) and pd.notna(close):
        if close > ma20:
            score += 8
            tags.append("站上MA20")
        else:
            score -= 10
            risks.append("跌破MA20")

    if pd.notna(ma60) and pd.notna(close):
        if close > ma60:
            score += 6
            tags.append("站上MA60")
        else:
            score -= 5

    if pd.notna(high20) and pd.notna(close) and pd.notna(volume_ratio_20):
        if close >= high20 * 0.995 and float(volume_ratio_20) >= float(cfg.get("breakout_volume_ratio", 1.3)):
            score += 20
            tags.append("放量突破")
            reason_parts.append(f"接近或突破20日高点且量能放大（约前20日均量{float(volume_ratio_20):.1f}倍）")

    if pd.notna(ma10) and pd.notna(ma20) and pd.notna(close) and pd.notna(volume_ratio_20):
        near_ma10 = abs(close / ma10 - 1.0) <= 0.025
        near_ma20 = abs(close / ma20 - 1.0) <= 0.035
        volume_shrink = float(volume_ratio_20) <= float(cfg.get("pullback_volume_ratio", 0.9))
        if ma5 > ma10 > ma20 and (near_ma10 or near_ma20) and volume_shrink:
            score += 16
            tags.append("缩量回踩")
            reason_parts.append(f"上升趋势中缩量回踩均线附近（约前20日均量{float(volume_ratio_20):.1f}倍）")

    if volume_state == "放量下跌":
        score -= 6
        risks.append("放量下跌")
        if pd.notna(volume_ratio_5):
            reason_parts.append(f"放量下跌，成交量约前5日均量{float(volume_ratio_5):.1f}倍")
    elif volume_state == "缩量上涨":
        score -= 3
        risks.append("缩量上涨")
        if pd.notna(volume_ratio_5):
            reason_parts.append(f"缩量上涨，成交量约前5日均量{float(volume_ratio_5):.1f}倍，上攻动能偏弱")

    if pd.notna(macd_hist):
        if macd_hist > 0:
            score += 7
            tags.append("MACD偏强")
        else:
            score -= 5

    if pd.notna(bias5) and bias5 >= float(cfg.get("overheat_bias5", 8.0)):
        score -= 14
        risks.append("短线乖离过大")
        reason_parts.append(f"5日乖离率约{bias5:.1f}%，短线追高风险上升")

    if pd.notna(rsi14) and rsi14 >= float(cfg.get("high_rsi", 80.0)):
        score -= 10
        risks.append("RSI过热")

    if pd.notna(row.get("net_mf_amount")) and row.get("net_mf_amount") > 0:
        score += 4
        tags.append("资金净流入")
    elif pd.notna(row.get("net_mf_amount")) and row.get("net_mf_amount") < 0:
        score -= 4

    score = max(0.0, min(100.0, score))
    if score >= 78:
        level = "strong"
    elif score >= 62:
        level = "watch"
    elif score <= 35:
        level = "risk"
    else:
        level = "neutral"

    if "放量突破" in tags:
        phase = "breakout"
    elif "缩量回踩" in tags:
        phase = "pullback"
    elif "多头排列" in tags:
        phase = "uptrend"
    elif "空头排列" in risks or "跌破MA20" in risks:
        phase = "weakening"
    else:
        phase = "sideways"

    if not reason_parts:
        if tags:
            reason_parts.append("；".join(tags[:3]))
        elif risks:
            reason_parts.append("；".join(risks[:3]))
        else:
            reason_parts.append("量价结构暂未出现强信号")

    return {
        "signal_score": round(score, 2),
        "signal_level": level,
        "trend_phase": phase,
        "tags": tags,
        "risk_flags": risks,
        "volume_state": volume_state,
        "reason": "；".join(reason_parts),
    }


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def compute_signals(settings: Settings, trade_date: str | None = None) -> dict[str, int | str]:
    with connect() as conn, conn.cursor() as cur:
        if trade_date is None:
            cur.execute(f"SELECT max(trade_date) AS d FROM {qname(settings, 'daily_bars')}")
            row = cur.fetchone()
            trade_date = str(row["d"]) if row and row["d"] else ""
        if not trade_date:
            raise RuntimeError("No trade_date available in daily_bars")
        cur.execute(f"SELECT ts_code, name FROM {qname(settings, 'signal_universe')} ORDER BY ts_code")
        universe = cur.fetchall()
        codes = [row["ts_code"] for row in universe]
        name_map = {row["ts_code"]: row["name"] for row in universe}
        if not codes:
            raise RuntimeError("Signal universe is empty")
        cur.execute(
            f"""
            SELECT b.*, db.turnover_rate, mf.net_mf_amount
            FROM {qname(settings, 'daily_bars')} b
            LEFT JOIN {qname(settings, 'daily_basic')} db
              ON db.ts_code=b.ts_code AND db.trade_date=b.trade_date
            LEFT JOIN {qname(settings, 'moneyflow_daily')} mf
              ON mf.ts_code=b.ts_code AND mf.trade_date=b.trade_date
            WHERE b.ts_code = ANY(%s)
              AND b.trade_date <= %s
              AND b.adj_close IS NOT NULL
            ORDER BY b.ts_code, b.trade_date
            """,
            (codes, trade_date),
        )
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No bar data for signal universe")
    for col in [
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
        "net_mf_amount",
        "prev_vol_ma5",
        "prev_vol_ma20",
        "volume_ratio_5",
        "volume_ratio_20",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    formula_spec = load_formula_spec(settings)
    signal_cfg = merged_signal_config(settings, formula_spec)
    df = add_indicators(df, formula_spec)
    current = df[df["trade_date"].astype(str) == trade_date].copy()
    if current.empty:
        raise RuntimeError(f"No current rows for {trade_date}")

    min_history = int(settings.section("signals").get("min_history_days", 60))
    history_counts = df.groupby("ts_code")["trade_date"].count().to_dict()
    out_rows = []
    latest_rows = []
    for _, row in current.iterrows():
        scored = _score_row(row, signal_cfg)
        ts_code = row["ts_code"]
        data_quality = {
            "history_days": int(history_counts.get(ts_code, 0)),
            "enough_history": int(history_counts.get(ts_code, 0)) >= min_history,
            "has_moneyflow": pd.notna(row.get("net_mf_amount")),
        }
        base = {
            "ts_code": ts_code,
            "trade_date": trade_date,
            "name": name_map.get(ts_code, ts_code),
            "close": row.get("close"),
            "pct_chg": row.get("pct_chg"),
            "amount": row.get("amount"),
            "turnover_rate": row.get("turnover_rate"),
            "ma5": row.get("ma5"),
            "ma10": row.get("ma10"),
            "ma20": row.get("ma20"),
            "ma60": row.get("ma60"),
            "bias5": row.get("bias5"),
            "rsi14": row.get("rsi14"),
            "macd": row.get("macd"),
            "macd_signal": row.get("macd_signal"),
            "macd_hist": row.get("macd_hist"),
            "vol_ma5": row.get("vol_ma5"),
            "vol_ma20": row.get("vol_ma20"),
            "prev_vol_ma5": row.get("prev_vol_ma5"),
            "prev_vol_ma20": row.get("prev_vol_ma20"),
            "volume_ratio_5": row.get("volume_ratio_5"),
            "volume_ratio_20": row.get("volume_ratio_20"),
            "volume_state": scored["volume_state"],
            "trend_phase": scored["trend_phase"],
            "signal_level": scored["signal_level"],
            "signal_score": scored["signal_score"],
            "tags": _json(scored["tags"]),
            "risk_flags": _json(scored["risk_flags"]),
            "reason": scored["reason"],
            "data_quality": _json(data_quality),
        }
        out_rows.append(base)
        latest_rows.append(
            {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "name": base["name"],
                "close": base["close"],
                "pct_chg": base["pct_chg"],
                "trend_phase": base["trend_phase"],
                "signal_level": base["signal_level"],
                "signal_score": base["signal_score"],
                "volume_state": base["volume_state"],
                "volume_ratio_5": base["volume_ratio_5"],
                "volume_ratio_20": base["volume_ratio_20"],
                "tags": base["tags"],
                "risk_flags": base["risk_flags"],
                "reason": base["reason"],
            }
        )

    signal_cols = [
        "ts_code",
        "trade_date",
        "name",
        "close",
        "pct_chg",
        "amount",
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
        "volume_state",
        "trend_phase",
        "signal_level",
        "signal_score",
        "tags",
        "risk_flags",
        "reason",
        "data_quality",
    ]
    latest_cols = [
        "ts_code",
        "trade_date",
        "name",
        "close",
        "pct_chg",
        "trend_phase",
        "signal_level",
        "signal_score",
        "volume_state",
        "volume_ratio_5",
        "volume_ratio_20",
        "tags",
        "risk_flags",
        "reason",
    ]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {qname(settings, 'technical_signals')} WHERE trade_date=%s AND NOT (ts_code = ANY(%s))",
                (trade_date, codes),
            )
            cur.execute(
                f"DELETE FROM {qname(settings, 'latest_signals')} WHERE NOT (ts_code = ANY(%s))",
                (codes,),
            )
        signals = upsert_rows(
            conn,
            table=qname(settings, "technical_signals"),
            columns=signal_cols,
            rows=out_rows,
            conflict_columns=["ts_code", "trade_date"],
        )
        latest = upsert_rows(
            conn,
            table=qname(settings, "latest_signals"),
            columns=latest_cols,
            rows=latest_rows,
            conflict_columns=["ts_code"],
        )
        conn.commit()
    final_metrics = refresh_final_signal_layers(settings, trade_date)
    return {"trade_date": trade_date, "signals": signals, "latest": latest, **final_metrics}
