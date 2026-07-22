"""Pure filter/sort/paginate helper for the file-backed persona profile list (P3a).

Personas have no FTS and no tags, so this is a straight in-memory pass over the
(≤100) profiles the scope service returns — no DB, no backend edit.
"""

from __future__ import annotations

from typing import Any


def _modified_key(p: dict[str, Any]) -> str:
    for key in ("updated_at", "last_modified", "modified_at"):
        if p.get(key):
            return str(p[key])
    return str(p.get("created_at") or "")


_SORTS = {
    "name_asc": (lambda p: str(p.get("name") or "").lower(), False),
    "created_desc": (lambda p: str(p.get("created_at") or ""), True),
    "modified_desc": (_modified_key, True),
}


def page_persona_profiles(
    profiles: list[dict[str, Any]],
    *,
    search_term: str | None,
    sort_key: str,
    offset: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return ``(page_rows, filtered_total)`` for the persona list.

    Search is a case-insensitive substring over name + description. ``sort_key``
    outside ``_SORTS`` (e.g. "relevance") falls back to "name_asc".
    """
    rows = list(profiles or [])
    term = (search_term or "").strip().lower()
    if term:
        rows = [
            p for p in rows
            if term in str(p.get("name") or "").lower()
            or term in str(p.get("description") or "").lower()
        ]
    filtered_total = len(rows)
    key, reverse = _SORTS.get(sort_key, _SORTS["name_asc"])
    rows = sorted(rows, key=key, reverse=reverse)
    start = max(0, offset)
    return rows[start:start + page_size], filtered_total
