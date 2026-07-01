from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .db import connect, qname, upsert_rows


@dataclass
class UniverseMember:
    ts_code: str
    name: str
    source_types: set[str] = field(default_factory=set)
    is_watchlist: bool = False
    is_candidate: bool = False
    candidate_status: str = ""
    codex_rating: str = ""


def _normalize_rating(value: object) -> str:
    return str(value or "").strip().upper()


def _normalize_status(value: object) -> str:
    return str(value or "").strip().lower()


def load_all_a_stocks() -> list[dict[str, str]]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT ts_code, name
            FROM stock_master
            WHERE COALESCE(list_status, 'L') = 'L'
              AND ts_code ~ '^[0-9]{6}\\.(SZ|SH|BJ)$'
            ORDER BY ts_code
            """
        )
        return [dict(row) for row in cur.fetchall()]


def load_focus_universe(settings: Settings) -> list[UniverseMember]:
    universe_cfg = settings.section("universe")
    candidate_statuses = {_normalize_status(x) for x in universe_cfg.get("watch_candidate_statuses", [])}
    candidate_ratings = {_normalize_rating(x) for x in universe_cfg.get("watch_candidate_ratings", [])}
    members: dict[str, UniverseMember] = {}

    def add_member(ts_code: str, name: str, source_type: str, **kwargs: object) -> None:
        if not ts_code:
            return
        item = members.get(ts_code)
        if item is None:
            item = UniverseMember(ts_code=ts_code, name=name or ts_code)
            members[ts_code] = item
        if name and not item.name:
            item.name = name
        item.source_types.add(source_type)
        if kwargs.get("is_watchlist"):
            item.is_watchlist = True
        if kwargs.get("is_candidate"):
            item.is_candidate = True
            item.candidate_status = str(kwargs.get("candidate_status") or item.candidate_status or "")
            item.codex_rating = str(kwargs.get("codex_rating") or item.codex_rating or "")

    with connect() as conn, conn.cursor() as cur:
        if universe_cfg.get("include_watchlist", True):
            cur.execute(
                """
                SELECT sm.ts_code, COALESCE(sm.name, w.name) AS name
                FROM watchlist_stocks w
                JOIN stock_master sm
                  ON sm.symbol = w.ticker
                  OR sm.ts_code = w.ticker
                  OR sm.name = w.name
                WHERE COALESCE(w.status, 'active') <> 'inactive'
                  AND sm.ts_code ~ '^[0-9]{6}\\.(SZ|SH|BJ)$'
                """
            )
            for row in cur.fetchall():
                add_member(row["ts_code"], row["name"], "watchlist", is_watchlist=True)

        cur.execute(
            """
            SELECT
                COALESCE(NULLIF(w.ts_code, ''), sm.ts_code) AS ts_code,
                COALESCE(w.name, sm.name) AS name,
                w.status,
                w.codex_rating
            FROM watch_candidates w
            LEFT JOIN stock_master sm
              ON sm.ts_code = w.ts_code OR sm.name = w.name
            WHERE COALESCE(w.status, '') <> ''
               OR COALESCE(w.codex_rating, '') <> ''
            """
        )
        for row in cur.fetchall():
            status = _normalize_status(row.get("status"))
            rating = _normalize_rating(row.get("codex_rating"))
            if status not in candidate_statuses and rating not in candidate_ratings:
                continue
            add_member(
                row.get("ts_code") or "",
                row.get("name") or "",
                "watch_candidate",
                is_candidate=True,
                candidate_status=row.get("status") or "",
                codex_rating=row.get("codex_rating") or "",
            )

    return sorted(members.values(), key=lambda x: x.ts_code)


def sync_signal_universe(settings: Settings, members: list[UniverseMember]) -> int:
    rows = [
        {
            "ts_code": item.ts_code,
            "name": item.name,
            "source_types": sorted(item.source_types),
            "is_watchlist": item.is_watchlist,
            "is_candidate": item.is_candidate,
            "candidate_status": item.candidate_status,
            "codex_rating": item.codex_rating,
        }
        for item in members
    ]
    columns = [
        "ts_code",
        "name",
        "source_types",
        "is_watchlist",
        "is_candidate",
        "candidate_status",
        "codex_rating",
    ]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {qname(settings, 'signal_universe')}")
        count = upsert_rows(
            conn,
            table=qname(settings, "signal_universe"),
            columns=columns,
            rows=rows,
            conflict_columns=["ts_code"],
        )
        conn.commit()
        return count
