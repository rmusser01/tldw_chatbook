"""Builders for the exact seven-key Library Media browse summary row.

The browse contract in ``tldw_chatbook/Library/library_media_state.py``
(``_MEDIA_SUMMARY_KEYS`` / ``validate_media_browse_items``) is an exact key
set, so every fake and shape test that hands rows to the validator has to
move whenever the contract does. These builders are the single place tests
spell the shape out, so the next contract move touches one file.

``has_analysis`` is projected in SQL (the newest ``DocumentVersions`` row's
analysis text); ``reviewed`` is not a media-DB fact -- the projection leaves
it ``None`` and the screen decorates it from the active review set. Keyword
match reasons are a per-query side channel, deliberately NOT an eighth key.
"""

from __future__ import annotations

from typing import Any

DEFAULT_UPDATED_AT = "2026-08-16T00:00:00+00:00"

#: Distinguishes "argument omitted" from an explicit ``None``, which is a
#: legal projected value for title and updated_at.
_UNSET: Any = object()


def summary_row(
    *,
    id: int | str,
    title: str | None = _UNSET,
    media_type: str | None = "article",
    updated_at: str | None = _UNSET,
    has_analysis: bool = False,
    reviewed: bool | None = None,
    backing_media_id: int | None = None,
) -> dict[str, Any]:
    """Build one exact seven-key browse summary row.

    Args:
        id: The backing media id, or the canonical ``local:media:<id>``
            stable id. Either form fills both identity keys.
        title: Row title; omit for ``"Media <backing id>"``.
        media_type: The row's media type (``None`` is a legal projection).
        updated_at: ISO timestamp; omit for ``DEFAULT_UPDATED_AT``.
        has_analysis: Whether the newest version carries analysis text.
        reviewed: ``None`` (no active review set), or the set-local mark.
        backing_media_id: Overrides the id derived from ``id`` -- for
            negative tests that need the two identity keys to disagree.

    Returns:
        A fresh mutable dict carrying exactly the seven contract keys.
    """
    derived = int(str(id).removeprefix("local:media:"))
    backing = derived if backing_media_id is None else backing_media_id
    return {
        "id": f"local:media:{derived}",
        "backing_media_id": backing,
        "title": f"Media {derived}" if title is _UNSET else title,
        "media_type": media_type,
        "updated_at": DEFAULT_UPDATED_AT if updated_at is _UNSET else updated_at,
        "has_analysis": has_analysis,
        "reviewed": reviewed,
    }


def summary_rows(n: int, *, start: int = 1, **overrides: Any) -> list[dict[str, Any]]:
    """Build ``n`` seven-key rows with ids ``start .. start + n - 1``."""
    return [summary_row(id=index, **overrides) for index in range(start, start + n)]
