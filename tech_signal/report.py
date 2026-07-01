from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .db import connect, qname


def _fmt_num(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _parse_json(value: object) -> list[str] | dict[str, Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return []


def _tags_text(value: object) -> str:
    parsed = _parse_json(value)
    if isinstance(parsed, list):
        return "、".join(str(x) for x in parsed[:5]) or "-"
    return "-"


def _volume_text(row: dict[str, Any]) -> str:
    state = str(row.get("volume_state") or "").strip()
    ratio = _fmt_num(row.get("volume_ratio_5"), 2)
    if state and ratio != "-":
        return f"{state} {ratio}x"
    return state or "-"


def _section(title: str, rows: list[dict[str, Any]], max_rows: int) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.extend(["暂无。", ""])
        return lines
    lines.append("| 股票 | 收盘 | 涨跌幅 | 分数 | 阶段 | 量能 | 标签/风险 | 说明 |")
    lines.append("|---|---:|---:|---:|---|---|---|---|")
    for row in rows[:max_rows]:
        lines.append(
            "| {name} `{ts_code}` | {close} | {pct}% | {score} | {phase} | {volume} | {tags} | {reason} |".format(
                name=row.get("name") or row.get("ts_code"),
                ts_code=row.get("ts_code"),
                close=_fmt_num(row.get("close")),
                pct=_fmt_num(row.get("pct_chg")),
                score=_fmt_num(row.get("signal_score")),
                phase=row.get("trend_phase") or "-",
                volume=_volume_text(row),
                tags=_tags_text(row.get("tags") or row.get("risk_flags")),
                reason=str(row.get("reason") or "").replace("|", "/"),
            )
        )
    lines.append("")
    return lines


def generate_report(settings: Settings, trade_date: str | None = None) -> Path:
    top_n = int(settings.section("report").get("top_n", 30))
    with connect() as conn, conn.cursor() as cur:
        if trade_date is None:
            cur.execute(f"SELECT max(trade_date) AS d FROM {qname(settings, 'technical_signals')}")
            row = cur.fetchone()
            trade_date = str(row["d"]) if row and row["d"] else ""
        if not trade_date:
            raise RuntimeError("No technical signal trade_date available")
        cur.execute(
            f"""
            SELECT *
            FROM {qname(settings, 'technical_signals')}
            WHERE trade_date=%s
            ORDER BY signal_score DESC NULLS LAST, pct_chg DESC NULLS LAST
            """,
            (trade_date,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        cur.execute(f"SELECT count(*) AS n FROM {qname(settings, 'daily_bars')}")
        bars_count = cur.fetchone()["n"]
        cur.execute(f"SELECT count(DISTINCT ts_code) AS n FROM {qname(settings, 'daily_bars')}")
        stock_count = cur.fetchone()["n"]
        cur.execute(f"SELECT count(DISTINCT trade_date) AS n FROM {qname(settings, 'daily_bars')}")
        date_count = cur.fetchone()["n"]

    strong = [r for r in rows if r.get("signal_level") == "strong"]
    watch = [r for r in rows if r.get("signal_level") == "watch"]
    pullback = [r for r in rows if r.get("trend_phase") == "pullback"]
    breakout = [r for r in rows if r.get("trend_phase") == "breakout"]
    risk = [r for r in rows if r.get("signal_level") == "risk"]

    lines = [
        f"# 技术信号日报 | {trade_date}",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 本次信号股票数：{len(rows)}",
        f"- 日线库覆盖：{stock_count} 只，{date_count} 个交易日，{bars_count} 行",
        f"- 运行模式：混合模式；技术指标使用前复权价格；当前仍为影子运行，未接入主投研系统。",
        "",
        "## 总览",
        "",
        f"- 强趋势：{len(strong)} 只",
        f"- 观察：{len(watch)} 只",
        f"- 缩量回踩：{len(pullback)} 只",
        f"- 放量突破：{len(breakout)} 只",
        f"- 风险/转弱：{len(risk)} 只",
        "",
    ]
    lines.extend(_section("强趋势", strong, top_n))
    lines.extend(_section("放量突破", breakout, top_n))
    lines.extend(_section("缩量回踩", pullback, top_n))
    lines.extend(_section("观察", watch, top_n))
    lines.extend(_section("风险/转弱", risk, top_n))

    safe_date = trade_date.replace("-", "")
    path = settings.reports_dir / f"{safe_date}_technical_signal_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
