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

Phase 2 Task 5 adds two reads on the same terms: ``list_tracked_updates``
(the Tracked group's rows, derived from ``list_tracked_items`` plus each
question item's latest run) and the ``tracked`` flag on story rows whose id
is an ACTIVE tracked origin (one parameterized SELECT through the same
``connection()`` seam, folded into this module's single read hop).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: List-label truncation for story headlines (the full ``title`` column
#: survives on the row for Task 7's modal; only the list label is capped).
LABEL_MAX_LENGTH = 60

#: How many tracked updates the Artifacts "Tracked" group surfaces.
TRACKED_UPDATES_LIMIT = 5


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
        ``collection_date``, ``label``, and ``tracked`` (True when an
        ACTIVE tracked item names this story as its origin); failed
        collections instead (or additionally) contribute
        ``{"label": f"Cycle {date}: failed", "status": "failed",
        "kind": "unknown", "collection_date": date, "synthetic": True}``.
    """
    from tldw_chatbook.DB.Dreams_DB import clamp_limit

    limit = clamp_limit(limit)
    tracked_origins = _active_tracked_origins(dreams_db)
    stories_by_date: dict[str, list[dict[str, Any]]] = {}
    for story in dreams_db.list_recent_stories(limit=limit):
        shaped = dict(story)
        collection_date = str(story["local_date"])
        shaped["collection_date"] = collection_date
        shaped["label"] = _label(story)
        shaped["tracked"] = int(story["id"]) in tracked_origins
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


def list_tracked_updates(
    dreams_db: Any, *, limit: int = TRACKED_UPDATES_LIMIT
) -> list[dict[str, Any]]:
    """Active tracked items as Tracked-group rows, changed-first.

    Question items carry ``last_run_status`` from their latest run (any
    status; ``None`` when the item has never been checked). Page items
    always carry ``None`` -- their dispositions live in the Subscriptions
    DB and surface through the Watchlists notifications pane, which the
    label's ``(alerts → Watchlists)`` pointer says out loud.

    Ordering: rows whose latest run is ``changed`` first, then
    ``event_date`` ascending (``None`` last), then ``created_at``
    descending (the stable sort rides ``list_tracked_items``' order).

    Args:
        dreams_db: The single ``DreamsDB`` instance.
        limit: Maximum rows returned.

    Returns:
        Rows shaped ``{id: tracked_item_id, label, mechanism, intent,
        status, last_checked, last_run_status, event_date,
        synthetic: False}``.
    """
    from tldw_chatbook.DB.Dreams_DB import clamp_limit

    limit = clamp_limit(limit)
    rows: list[dict[str, Any]] = []
    for item in dreams_db.list_tracked_items("active"):
        mechanism = str(item.get("mechanism") or "")
        last_run_status: str | None = None
        if mechanism == "question":
            runs = dreams_db.list_recent_track_runs(int(item["id"]), limit=1)
            last_run_status = str(runs[0]["status"]) if runs else None
        label = f"Tracking: {item.get('query_template') or 'page'}"
        if mechanism == "page":
            label += " (alerts → Watchlists)"
        rows.append(
            {
                "id": int(item["id"]),
                "label": label,
                "mechanism": mechanism,
                "intent": str(item.get("intent") or ""),
                "status": str(item.get("status") or ""),
                "last_checked": item.get("last_checked"),
                "last_run_status": last_run_status,
                "event_date": item.get("event_date"),
                "synthetic": False,
            }
        )
    rows.sort(
        key=lambda row: (
            # A changed latest run outranks every date consideration.
            0 if row["last_run_status"] == "changed" else 1,
            # event_date ascending, undated items last.
            1 if row["event_date"] is None else 0,
            str(row["event_date"] or ""),
        )
    )
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


def format_tracked_row(update: Mapping[str, Any]) -> str:
    """One Tracked-group row label, with the surfaced run status.

    Same idiom as :func:`format_dream_row` (the Reports/Dreams rows):
    ``"> Tracked: <label>"`` plus ``" · <last_run_status>"`` when the item
    has a run to show -- a ``changed`` disposition is the group's headline,
    so it rides the row.

    Args:
        update: One ``list_tracked_updates`` row.

    Returns:
        The row's display text (render it literally, never as markup).
    """
    label = f"> Tracked: {update.get('label', '')}"
    if update.get("last_run_status"):
        label += f" · {update['last_run_status']}"
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


def _active_tracked_origins(dreams_db: Any) -> set[int]:
    """Story ids named as origins by ACTIVE tracked items.

    One parameterized SELECT through the ``connection()`` seam; retired or
    paused items do not badge their origin story. Callers fold this into
    the same read hop as the stories themselves.
    """
    with dreams_db.connection() as conn:
        rows = conn.execute(
            "SELECT origin_story_id FROM dream_tracked_items"
            " WHERE status = 'active'",
            (),
        ).fetchall()
    return {
        int(row["origin_story_id"])
        for row in rows
        if row["origin_story_id"] is not None
    }
