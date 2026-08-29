"""Read-only aggregation of watchlist briefings into Daily Report rows.

A "Daily Report" is a briefing (ADR-079). This module is the thin derivation
layer between the `briefings` tables and the Artifacts screen's Reports slot:
no writes, no new tables, no caching. Callers own thread discipline -- call
from a worker thread or wrap in `asyncio.to_thread`.
"""

from __future__ import annotations

from typing import Any, Mapping

from tldw_chatbook.Subscriptions.briefing_audio import audio_file_path_is_safe

_STATUS_MARKERS = {
    "complete": "",
    "empty": " (empty)",
    "failed": " (failed)",
    "generating": " (writing…)",
}


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
    return f"{name} — {row.get('created_at', '')}{marker}{audio}"
