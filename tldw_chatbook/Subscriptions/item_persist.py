"""Single persistence path for scraped subscription items.

Replaces two divergent INSERT statements that wrote disjoint column sets:
``Subscriptions_DB`` wrote the change/dedup fields but dropped run linkage and
status, while ``LocalWatchlistsService`` did the reverse. Neither wrote body
text. Both callers (``Subscriptions_DB._add_subscription_item`` and
``LocalWatchlistsService._upsert_subscription_items``) now route through
:func:`persist_subscription_item`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


#: The vocabulary of ``content_kind``/``content_format``, named once.
#:
#: TASK-1343: producers used to write neither field, so
#: ``content_pane.render_for`` fell through to the article renderer for every
#: item including site changes. They now write both -- and since an invalid
#: pairing *raises* out of ``persist_subscription_item``, inside a scheduled
#: fetch, the producers import these names from here (the module that owns the
#: rule) rather than repeat string literals that a typo would turn into a
#: run-time failure. This module has no heavy imports of its own, so it is safe
#: for any producer to import at module scope.
CONTENT_KIND_ARTICLE = "article"
CONTENT_KIND_CHANGE = "change"
CONTENT_FORMAT_TEXT = "text"
CONTENT_FORMAT_MARKDOWN = "markdown"
CONTENT_FORMAT_DIFF = "diff"

_VALID_PAIRINGS = {
    (CONTENT_KIND_ARTICLE, CONTENT_FORMAT_TEXT),
    (CONTENT_KIND_ARTICLE, CONTENT_FORMAT_MARKDOWN),
    (CONTENT_KIND_CHANGE, CONTENT_FORMAT_DIFF),
}


def _json_or_none(value: Any) -> str | None:
    """Serialize a value to JSON, collapsing an empty container to ``None``.

    ``categories``, ``enclosures``, ``extracted_data``, and ``alert_matches``
    are always either ``None``, a list, or a dict -- never a plausible
    falsy scalar like ``0`` or ``False``. So rather than a general
    truthiness check (which would also collapse those scalars), this keys
    specifically on container emptiness: an empty list or dict serializes
    to ``None`` just like a missing value, matching the old DB-side
    write path's behavior. Storing the literal string ``"[]"`` instead
    would make it indistinguishable from a real, non-empty payload to
    readers that treat "any string" as meaningful, e.g.
    ``AggregationEngine._parse_categories``, which splits every string on
    commas and would otherwise manufacture a phantom ``["[]"]`` category.
    """
    if value is None:
        return None
    if hasattr(value, "__len__") and len(value) == 0:
        return None
    return json.dumps(value)


def _validate_content_pairing(kind: Any, fmt: Any) -> None:
    """Reject impossible kind/format combinations at the persist boundary."""
    if kind is None and fmt is None:
        return
    if (kind, fmt) not in _VALID_PAIRINGS:
        raise ValueError(
            f"invalid content_kind/content_format pairing: {kind!r}/{fmt!r}. "
            f"Valid pairings: {sorted(_VALID_PAIRINGS)}"
        )


def persist_subscription_item(
    conn: Any,
    subscription_id: int,
    item: Mapping[str, Any],
    run_id: int | None,
    now: str,
) -> int:
    """Insert or update one item, writing the full column set.

    Existing ``reviewed``, ``ignored``, and ``ingested`` statuses are
    preserved across re-fetches, since each reflects a deliberate user
    action on that item. Anything else (including ``error``) resets to
    ``new`` — a successful re-fetch should clear a prior error.

    Args:
        conn: An open connection inside a transaction.
        subscription_id: Owning source id.
        item: Normalized item mapping.
        run_id: Run that produced this item, if any.
        now: ISO-8601 timestamp for created_at/updated_at.

    Returns:
        The id of the inserted or updated row.

    Raises:
        ValueError: If content_kind and content_format are an invalid pairing.
    """
    content_kind = item.get("content_kind")
    content_format = item.get("content_format")
    _validate_content_pairing(content_kind, content_format)

    cursor = conn.execute(
        """
        INSERT INTO subscription_items (
            subscription_id, url, title, content, content_kind, content_format,
            content_hash, published_date, author, categories, enclosures,
            extracted_data, status, run_id, alert_matches, canonical_url,
            previous_hash, change_percentage, diff_summary, change_type,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(subscription_id, url, content_hash) DO UPDATE SET
            title = excluded.title,
            content = excluded.content,
            content_kind = excluded.content_kind,
            content_format = excluded.content_format,
            published_date = excluded.published_date,
            author = excluded.author,
            categories = excluded.categories,
            enclosures = excluded.enclosures,
            extracted_data = excluded.extracted_data,
            run_id = excluded.run_id,
            alert_matches = excluded.alert_matches,
            canonical_url = excluded.canonical_url,
            previous_hash = excluded.previous_hash,
            change_percentage = excluded.change_percentage,
            diff_summary = excluded.diff_summary,
            change_type = excluded.change_type,
            status = CASE
                WHEN subscription_items.status IN ('reviewed', 'ignored', 'ingested')
                THEN subscription_items.status
                ELSE 'new'
            END,
            updated_at = excluded.updated_at
        RETURNING id
        """,
        (
            subscription_id,
            item.get("url"),
            item.get("title"),
            item.get("content"),
            content_kind,
            content_format,
            item.get("content_hash"),
            item.get("published_date"),
            item.get("author"),
            _json_or_none(item.get("categories")),
            _json_or_none(item.get("enclosures")),
            _json_or_none(item.get("extracted_data")),
            "new",
            run_id,
            _json_or_none(item.get("alert_matches")),
            item.get("canonical_url"),
            item.get("previous_hash"),
            item.get("change_percentage"),
            item.get("diff_summary"),
            item.get("change_type"),
            now,
            now,
        ),
    )
    return cursor.fetchone()[0]
