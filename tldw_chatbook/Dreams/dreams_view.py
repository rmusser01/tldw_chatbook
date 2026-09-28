"""Read-only aggregation of Dreams stories into Artifacts "Dream" rows.

Mirrors :mod:`tldw_chatbook.Subscriptions.daily_reports_view`'s role in the
Artifacts screen: a thin derivation layer between the ``dream_stories`` /
``dreams_collections`` tables and the list pane -- no writes, no UI imports,
no caching. Callers own thread discipline; call from a worker thread.

Ruling R2 (binding): rows are FULL ``dream_stories`` rows -- Task 7's detail
modal reads ``body``/``url``/``snippet``/``matched_topics``/``query``/
``event_date``/``location`` straight off them -- plus the shaping fields
``label`` (title truncated for the list) and ``collection_date``.

The recent-collections read (for the failed-cycle synthetic rows) goes
through ``DreamsDB.connection()`` -- the documented single-statement read
seam -- because Phase 1's ``DreamsDB`` ships no ``list_recent_collections``
method and Task 6 may not modify the DB layer. It is one SELECT, read-only,
and off-thread friendly like everything else here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: List-label truncation for story headlines (the full ``title`` column
#: survives on the row for Task 7's modal; only the list label is capped).
LABEL_MAX_LENGTH = 60


def list_recent_dreams(dreams_db: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    """Recent Dreams stories, newest collection first, UI-shaped.

    Ordering follows ``DreamsDB.list_recent_stories``: collection
    ``local_date`` descending, kept stories first within a collection, then
    insert order. Each ``failed`` collection contributes one synthetic row
    ahead of its stories so a bad cycle stays visible even when it produced
    no story rows at all.

    Args:
        dreams_db: The single ``DreamsDB`` instance.
        limit: Maximum rows returned (the merged newest-first stream is
            sliced to this).

    Returns:
        One dict per story: every ``dream_stories`` column plus
        ``collection_date`` and ``label``; failed collections instead (or
        additionally) contribute
        ``{"label": f"Cycle {date}: failed", "status": "failed",
        "kind": "unknown", "collection_date": date, "synthetic": True}``.
    """
    from tldw_chatbook.DB.Dreams_DB import clamp_limit

    limit = clamp_limit(limit)
    stories_by_date: dict[str, list[dict[str, Any]]] = {}
    for story in dreams_db.list_recent_stories(limit=limit):
        shaped = dict(story)
        collection_date = str(story["local_date"])
        shaped["collection_date"] = collection_date
        shaped["label"] = _label(story)
        stories_by_date.setdefault(collection_date, []).append(shaped)

    rows: list[dict[str, Any]] = []
    for collection in _recent_collections(dreams_db, limit):
        collection_date = str(collection["local_date"])
        if str(collection.get("status") or "") == "failed":
            rows.append(
                {
                    "label": f"Cycle {collection_date}: failed",
                    "status": "failed",
                    "kind": "unknown",
                    "collection_date": collection_date,
                    "synthetic": True,
                }
            )
        rows.extend(stories_by_date.pop(collection_date, []))
    # Story dates the collections window missed (only possible when the two
    # reads straddle a concurrent cycle commit) keep their rows, newest
    # first by insertion order.
    for dated_stories in stories_by_date.values():
        rows.extend(dated_stories)
    return rows[:limit]


def format_dream_row(story: Mapping[str, Any]) -> str:
    """One list-row label: ``"> Dream: <label>"`` plus the kept badge.

    Matches the Reports row idiom on the Artifacts screen (``"> Report:
    {label}"`` + ``" · kept"``).

    Args:
        story: One ``list_recent_dreams`` row.

    Returns:
        The row's display text (render it literally, never as markup).
    """
    label = f"> Dream: {story.get('label', '')}"
    if story.get("kept"):
        label += " · kept"
    return label


def _label(story: Mapping[str, Any]) -> str:
    return str(story.get("title") or "")[:LABEL_MAX_LENGTH]


def _recent_collections(dreams_db: Any, limit: int) -> list[dict[str, Any]]:
    """Newest collections first (read-only, via the connection() seam)."""
    with dreams_db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM dreams_collections ORDER BY local_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
