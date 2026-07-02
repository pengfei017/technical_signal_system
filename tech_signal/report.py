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


def _stock_signal_section(rows: list[dict[str, Any]], max_rows: int) -> list[str]:
    lines = ["## 个股交易信号 Top", ""]
    if not rows:
        return lines + ["暂无。", ""]
    lines.append("| 股票 | 分数 | 级别 | 涨跌幅 | 成交额 | 量能 | 涨停状态 | 资金净流入 | 龙虎榜净买 | 标签/风险 |")
    lines.append("|---|---:|---|---:|---:|---|---|---:|---:|---|")
    for row in rows[:max_rows]:
        lines.append(
            "| {name} `{ts_code}` | {score} | {level} | {pct}% | {amount}亿 | {volume} | {limit_status} | {mf}亿 | {lhb}亿 | {tags} |".format(
                name=row.get("name") or row.get("ts_code"),
                ts_code=row.get("ts_code"),
                score=_fmt_num(row.get("total_signal_score")),
                level=row.get("signal_level") or "-",
                pct=_fmt_num(row.get("pct_chg")),
                amount=_fmt_num(row.get("amount_yi")),
                volume=row.get("volume_state") or "-",
                limit_status=row.get("limit_status") or "-",
                mf=_fmt_num(row.get("net_mf_amount_yi")),
                lhb=_fmt_num(row.get("lhb_net_buy_yi")),
                tags=_tags_text(row.get("tags") or row.get("risk_flags")),
            )
        )
    lines.append("")
    return lines


def _theme_section(rows: list[dict[str, Any]], max_rows: int) -> list[str]:
    lines = ["## 主题交易热度 Top", ""]
    if not rows:
        return lines + ["暂无。", ""]
    lines.append("| 类型 | 主题 | 热度 | 动量 | 涨跌幅 | 资金净额 | 涨停数 | 强个股数 | 说明 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows[:max_rows]:
        lines.append(
            "| {typ} | {name} | {heat} | {momentum} | {pct}% | {net}亿 | {up} | {strong} | {reason} |".format(
                typ=row.get("theme_type") or "-",
                name=row.get("theme_name") or "-",
                heat=_fmt_num(row.get("heat_score")),
                momentum=_fmt_num(row.get("momentum_score")),
                pct=_fmt_num(row.get("pct_chg")),
                net=_fmt_num(row.get("net_amount_yi")),
                up=row.get("limit_up_count") or 0,
                strong=row.get("strong_stock_count") or 0,
                reason=str(row.get("reason") or "").replace("|", "/"),
            )
        )
    lines.append("")
    return lines


def _limit_stats_section(row: dict[str, Any] | None) -> list[str]:
    lines = ["## 短线生态统计", ""]
    if not row:
        return lines + ["暂无。", ""]
    broken_rate = "-" if row.get("broken_rate") is None else f"{float(row.get('broken_rate')):.2f}%"
    lines.extend(
        [
            f"- 涨停家数：{row.get('limit_up_count', 0)}",
            f"- 跌停家数：{row.get('limit_down_count', 0)}",
            f"- 炸板家数：{row.get('broken_count', 0)}",
            f"- 炸板率：{broken_rate}",
            f"- 连板高度：{row.get('max_board', 0)}板",
            "",
        ]
    )
    return lines


def _lhb_section(rows: list[dict[str, Any]], max_rows: int) -> list[str]:
    lines = ["## 龙虎榜确认信号", ""]
    if not rows:
        return lines + ["暂无。", ""]
    lines.append("| 股票 | 涨跌幅 | 龙虎榜净买 | 机构净买 | 陆股通净买 | 上榜原因 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in rows[:max_rows]:
        lines.append(
            "| {name} `{ts_code}` | {pct}% | {net}亿 | {inst}亿 | {north}亿 | {reason} |".format(
                name=row.get("name") or row.get("ts_code"),
                ts_code=row.get("ts_code"),
                pct=_fmt_num(row.get("pct_change")),
                net=_fmt_num(row.get("lhb_net_buy_yi")),
                inst=_fmt_num(row.get("institution_net_buy_yi")),
                north=_fmt_num(row.get("northbound_net_buy_yi")),
                reason=str(row.get("primary_reason") or "").replace("|", "/")[:60],
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
        cur.execute(
            f"""
            SELECT *
            FROM {qname(settings, 'stock_signal_daily')}
            WHERE trade_date=%s
            ORDER BY total_signal_score DESC NULLS LAST, amount_yi DESC NULLS LAST
            LIMIT %s
            """,
            (trade_date, top_n),
        )
        stock_signal_rows = [dict(row) for row in cur.fetchall()]
        cur.execute(
            f"""
            SELECT *
            FROM {qname(settings, 'theme_signal_daily')}
            WHERE trade_date=%s
            ORDER BY heat_score DESC NULLS LAST, momentum_score DESC NULLS LAST
            LIMIT %s
            """,
            (trade_date, top_n),
        )
        theme_rows = [dict(row) for row in cur.fetchall()]
        cur.execute(f"SELECT * FROM {qname(settings, 'limit_market_stats')} WHERE trade_date=%s", (trade_date,))
        limit_stats = cur.fetchone()
        cur.execute(
            f"""
            SELECT *
            FROM {qname(settings, 'lhb_stocks')}
            WHERE trade_date=%s
            ORDER BY lhb_net_buy_yi DESC NULLS LAST
            LIMIT %s
            """,
            (trade_date, top_n),
        )
        lhb_rows = [dict(row) for row in cur.fetchall()]
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
        f"- 个股交易信号表：{len(stock_signal_rows)} 只 Top 样本",
        f"- 主题交易热度表：{len(theme_rows)} 个 Top 样本",
        "",
    ]
    lines.extend(_stock_signal_section(stock_signal_rows, top_n))
    lines.extend(_theme_section(theme_rows, top_n))
    lines.extend(_limit_stats_section(dict(limit_stats) if limit_stats else None))
    lines.extend(_lhb_section(lhb_rows, top_n))
    lines.extend(_section("强趋势", strong, top_n))
    lines.extend(_section("放量突破", breakout, top_n))
    lines.extend(_section("缩量回踩", pullback, top_n))
    lines.extend(_section("观察", watch, top_n))
    lines.extend(_section("风险/转弱", risk, top_n))

    safe_date = trade_date.replace("-", "")
    path = settings.reports_dir / f"{safe_date}_technical_signal_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
