from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tech_signal.config import Settings
from tech_signal.db import connect, init_schema, qname, upsert_rows

from .factor_definitions import FACTOR_DEFINITIONS, factor_group, normalize_date
from .factor_evaluator import _clean_float, _format_pct, report_dir


BASE_GROUP_WEIGHTS = {
    "trend": 0.25,
    "volume": 0.20,
    "moneyflow": 0.20,
    "sentiment": 0.15,
    "lhb": 0.10,
    "risk": -0.10,
    "reversal": 0.05,
    "relative_strength": 0.05,
}


def _split_dates(start_date: str, end_date: str) -> dict[str, str]:
    start = pd.to_datetime(normalize_date(start_date))
    end = pd.to_datetime(normalize_date(end_date))
    span = max((end - start).days, 1)
    train_end = start + pd.Timedelta(days=int(span * 0.6))
    validation_end = start + pd.Timedelta(days=int(span * 0.8))
    return {
        "train_start": start.date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "validation_start": (train_end + pd.Timedelta(days=1)).date().isoformat(),
        "validation_end": validation_end.date().isoformat(),
        "test_start": (validation_end + pd.Timedelta(days=1)).date().isoformat(),
        "test_end": end.date().isoformat(),
    }


def _normalize_abs(weights: dict[str, float]) -> dict[str, float]:
    total = sum(abs(value) for value in weights.values())
    if total <= 0:
        return weights
    return {name: value / total for name, value in weights.items()}


def baseline_weights() -> dict[str, float]:
    by_group: dict[str, list[str]] = {}
    for item in FACTOR_DEFINITIONS:
        by_group.setdefault(item.group, []).append(item.name)
    weights: dict[str, float] = {}
    for group, names in by_group.items():
        group_weight = BASE_GROUP_WEIGHTS.get(group, 0.0)
        if not names:
            continue
        for name in names:
            weights[name] = group_weight / len(names)
    return _normalize_abs(weights)


def _load_performance(settings: Settings, start_date: str, end_date: str, horizon_days: int) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT *
            FROM {qname(settings, 'factor_performance')}
            WHERE start_date=%s
              AND end_date=%s
              AND horizon_days=%s
              AND market_regime='all'
            """,
            (normalize_date(start_date), normalize_date(end_date), horizon_days),
        )
        rows = cur.fetchall()
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    for column in ["rank_ic_mean", "long_short_return", "win_rate", "max_drawdown"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def performance_weights(settings: Settings, start_date: str, end_date: str, horizon_days: int = 5) -> dict[str, float]:
    data = _load_performance(settings, start_date, end_date, horizon_days)
    if data.empty:
        return baseline_weights()
    weights: dict[str, float] = {}
    for row in data.itertuples():
        rank_ic = float(row.rank_ic_mean) if pd.notna(row.rank_ic_mean) else 0.0
        long_short = float(row.long_short_return) if pd.notna(row.long_short_return) else 0.0
        win_rate = float(row.win_rate) if pd.notna(row.win_rate) else 0.5
        drawdown = abs(float(row.max_drawdown)) if pd.notna(row.max_drawdown) else 0.0
        raw = rank_ic * 0.70 + long_short * 2.0 + (win_rate - 0.5) * 0.08
        raw = raw / (1.0 + drawdown * 5.0)
        weights[str(row.factor_name)] = raw
    if sum(abs(value) for value in weights.values()) <= 1e-12:
        return baseline_weights()
    return _normalize_abs(weights)


def _load_correlations(settings: Settings, start_date: str, end_date: str) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT factor_a, factor_b, correlation
            FROM {qname(settings, 'factor_correlation')}
            WHERE start_date=%s AND end_date=%s
            """,
            (normalize_date(start_date), normalize_date(end_date)),
        )
        rows = cur.fetchall()
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    data["correlation"] = pd.to_numeric(data["correlation"], errors="coerce")
    return data


def decorrelated_weights(settings: Settings, start_date: str, end_date: str, horizon_days: int = 5) -> dict[str, float]:
    weights = performance_weights(settings, start_date, end_date, horizon_days)
    correlations = _load_correlations(settings, start_date, end_date)
    if correlations.empty:
        return weights
    adjusted = weights.copy()
    high = correlations[correlations["correlation"].abs() >= 0.75].copy()
    for row in high.itertuples():
        left = str(row.factor_a)
        right = str(row.factor_b)
        if left not in adjusted or right not in adjusted:
            continue
        if abs(adjusted[left]) <= abs(adjusted[right]):
            adjusted[left] *= 0.5
        else:
            adjusted[right] *= 0.5
    return _normalize_abs(adjusted)


def optimize_weights(
    settings: Settings,
    start_date: str,
    end_date: str,
    *,
    as_of_date: str | None = None,
    horizon_days: int = 5,
) -> dict[str, Any]:
    init_schema(settings)
    as_of = normalize_date(as_of_date or end_date)
    split = _split_dates(start_date, end_date)
    methods = {
        "baseline_v1": ("manual_baseline", baseline_weights()),
        "ic_weighted_v1": ("ic_rankic_drawdown_weighted", performance_weights(settings, start_date, end_date, horizon_days)),
        "decorrelated_v1": ("decorrelated_ic_weighted", decorrelated_weights(settings, start_date, end_date, horizon_days)),
    }
    rows: list[dict[str, Any]] = []
    for model_name, (method, weights) in methods.items():
        for factor_name, weight in weights.items():
            rows.append(
                {
                    "model_name": model_name,
                    "as_of_date": as_of,
                    "factor_name": factor_name,
                    "factor_group": factor_group(factor_name),
                    "weight": _clean_float(weight) or 0.0,
                    "method": method,
                    "train_start": split["train_start"],
                    "train_end": split["train_end"],
                    "validation_start": split["validation_start"],
                    "validation_end": split["validation_end"],
                }
            )
    columns = [
        "model_name",
        "as_of_date",
        "factor_name",
        "factor_group",
        "weight",
        "method",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
    ]
    with connect() as conn:
        count = upsert_rows(
            conn,
            table=qname(settings, "model_weight_history"),
            columns=columns,
            rows=rows,
            conflict_columns=["model_name", "as_of_date", "factor_name", "method"],
        )
        conn.commit()
    path = write_weight_report(settings, start_date, end_date, rows)
    return {
        "as_of_date": as_of,
        "model_count": len(methods),
        "weight_rows": count,
        "report_file": str(path),
        **split,
    }


def load_model_weights(settings: Settings, model_name: str, as_of_date: str) -> dict[str, float]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT factor_name, weight
            FROM {qname(settings, 'model_weight_history')}
            WHERE model_name=%s AND as_of_date=%s
            """,
            (model_name, normalize_date(as_of_date)),
        )
        rows = cur.fetchall()
    weights = {str(row["factor_name"]): float(row["weight"]) for row in rows}
    return weights or baseline_weights()


def write_weight_report(settings: Settings, start_date: str, end_date: str, rows: list[dict[str, Any]]) -> Path:
    path = report_dir(settings) / f"weight_suggestion_{normalize_date(end_date).replace('-', '')}.md"
    data = pd.DataFrame(rows)
    lines = [
        f"# Weight Suggestion {normalize_date(start_date)} to {normalize_date(end_date)}",
        "",
        "研究模块输出，不改生产评分。",
        "",
        "## Summary",
        "",
        "- baseline_v1: manual group baseline.",
        "- ic_weighted_v1: IC/RankIC/win-rate/drawdown weighted.",
        "- decorrelated_v1: high-correlation pairs above 0.75 are downweighted.",
        "",
    ]
    if not data.empty:
        for model in ["baseline_v1", "ic_weighted_v1", "decorrelated_v1"]:
            subset = data[data["model_name"] == model].copy()
            subset["abs_weight"] = subset["weight"].abs()
            lines.extend([f"## {model}", ""])
            for row in subset.sort_values("abs_weight", ascending=False).head(20).to_dict("records"):
                lines.append(f"- {row['factor_name']} ({row['factor_group']}): {row['weight']:.4f}")
            lines.append("")
    lines.extend(
        [
            "## Production View",
            "",
            "- Current output is calibration evidence only.",
            "- Do not replace `stock_signal_daily` weights before more out-of-sample and live shadow validation.",
            "- Watch especially for unstable sign flips, short sample LHB factors, and duplicated trend/volume signals.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

