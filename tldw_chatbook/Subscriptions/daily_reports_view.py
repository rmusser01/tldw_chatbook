"""Read-only aggregation of watchlist briefings into Daily Report rows.

A "Daily Report" is a briefing (ADR-079). This module is the thin derivation
layer between the `briefings` tables and the Artifacts screen's Reports slot:
no writes, no new tables, no caching. Callers own thread discipline -- call
from a worker thread or wrap in `asyncio.to_thread`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from tldw_chatbook.Subscriptions.briefing_audio import audio_file_path_is_safe

_STATUS_MARKERS = {
    "complete": "",
    "empty": " (empty)",
    "failed": " (failed)",
    "generating": " (writing…)",
}

#: Placeholder shown when a briefing carries no parseable timestamp.
_UNKNOWN_TIME = "unknown time"


def format_report_timestamp(value: Any) -> str:
    """Render a stored briefing timestamp in the viewer's local timezone.

    Briefings carry ``created_at`` in two shapes depending on the write path:
    the DB default ``CURRENT_TIMESTAMP`` yields a naive UTC
    ``"YYYY-MM-DD HH:MM:SS"`` string, while the demo/accept path writes an
    aware microsecond ISO string (``datetime.now(timezone.utc).isoformat()``,
    e.g. ``"2026-09-05T23:10:20.123456+00:00"``). Both parse through
    :func:`datetime.fromisoformat`; a naive result is treated as UTC. The
    result is local-zone, minute precision -- matching the Watchlists
    screen's ``_local_schedule_time`` -- so the Reports list and preview show
    a human time instead of raw microsecond ISO (TASK-31803).

    Args:
        value: A ``datetime``, an ISO/SQLite timestamp string, or ``None``.

    Returns:
        ``"YYYY-MM-DD HH:MM"`` (plus the local zone abbreviation when the
        platform supplies one) in local time, or ``"unknown time"`` for a
        missing value and the raw text for an unparseable one.
    """
    if value is None:
        return _UNKNOWN_TIME
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return _UNKNOWN_TIME
        candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            # Unparseable: surface the raw text rather than invent a time.
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()


def list_recent_reports(db: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    """Recent briefings across all watchlists, newest first, UI-shaped.

    Args:
        db: The single ``SubscriptionsDB`` instance.
        limit: Maximum rows.

    Returns:
        One dict per briefing with keys ``id``, ``watchlist_id``,
        ``watchlist_name``, ``status``, ``created_at``, ``item_count``,
        ``model_used``, ``has_audio``, ``audio_file_path`` (non-None only for
        a complete audio row whose path passes the safety guard), ``label``.
    """
    rows = db.list_recent_briefings(limit=limit)
    return [_to_report_row(row) for row in rows]


def _to_report_row(row: Mapping[str, Any]) -> dict[str, Any]:
    audio_path = row.get("latest_audio_file_path")
    has_audio = bool(audio_path) and audio_file_path_is_safe(audio_path)
    return {
        "id": int(row["briefing_id"]),
        "watchlist_id": int(row["watchlist_id"]),
        "watchlist_name": str(
            row.get("watchlist_name") or f"Watchlist {row['watchlist_id']}"
        ),
        "status": str(row.get("status") or ""),
        "created_at": row.get("created_at"),
        "item_count": row.get("item_count") or 0,
        "model_used": row.get("model_used"),
        "has_audio": has_audio,
        "audio_file_path": str(audio_path) if has_audio else None,
        "label": _label(row),
    }


def _label(row: Mapping[str, Any]) -> str:
    name = row.get("watchlist_name") or f"Watchlist {row.get('watchlist_id')}"
    marker = _STATUS_MARKERS.get(
        str(row.get("status") or ""), f" ({row.get('status')})"
    )
    audio = " · audio" if row.get("complete_audio_count") else ""
    when = format_report_timestamp(row.get("created_at"))
    return f"{name} — {when}{marker}{audio}"
