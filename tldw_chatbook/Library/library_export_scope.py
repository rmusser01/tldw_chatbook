"""Pure scope resolution for Library chatbook bulk export.

The Library canvases render capped snapshots of each source
(``LIBRARY_SOURCE_PAGE_SIZES`` in ``UI/Screens/library_screen.py``: notes
100, media 50, conversations 50 rows). Resolving a bulk export from those
rendered snapshots would silently truncate any library larger than the
page size. This module -- and the ``get_all_*`` DB methods it calls
(``Client_Media_DB_v2.MediaDatabase.get_all_active_media_ids``,
``DB.ChaChaNotes_DB.CharactersRAGDB.get_all_conversation_ids``/
``get_all_note_ids``) -- deliberately never reads a rendered snapshot: every
count/resolve call issues a fresh, uncapped id query against the database.

Pure module: stdlib + ``Chatbooks.chatbook_models.ContentType`` + type hints
only. DB handles are passed in by the caller and never constructed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from tldw_chatbook.Chatbooks.chatbook_models import ContentType

_VALID_KINDS = ("everything", "media", "conversations", "notes")

# Sentinel used by the Library media canvas's "no filter" select option.
_UNFILTERED_MEDIA_TYPE_SENTINEL = "All"


@dataclass(frozen=True)
class ExportScope:
    """What a Library chatbook export should include.

    Attributes:
        kind: One of "everything", "media", "conversations", "notes".
        media_type: Only meaningful when ``kind == "media"``; a specific
            media ``type`` column value to filter to. Ignored for every
            other ``kind``. ``None`` and the Library media canvas's "no
            filter" sentinel ``"All"`` both mean unfiltered -- every active
            media item is in scope.
    """

    kind: str
    media_type: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"Unknown export scope kind: {self.kind!r}. Expected one of {_VALID_KINDS}."
            )


class MediaIdSource(Protocol):
    """The subset of ``Client_Media_DB_v2.MediaDatabase`` this module needs."""

    def get_all_active_media_ids(self, media_type: str | None = None) -> list[int]: ...


class ChaChaNotesIdSource(Protocol):
    """The subset of ``DB.ChaChaNotes_DB.CharactersRAGDB`` this module needs."""

    def get_all_conversation_ids(self) -> list[str]: ...

    def get_all_note_ids(self) -> list[str]: ...


def _effective_media_type(scope: ExportScope) -> str | None:
    """Return the media ``type`` filter to apply, or ``None`` for unfiltered.

    Only a ``kind="media"`` scope ever applies a type filter -- for every
    other scope, ``scope.media_type`` is meaningless and ignored (an
    "everything" export always includes every active media item). The
    Library media canvas's "no filter" sentinel ``"All"`` is normalized to
    ``None``, matching ``ExportScope.media_type``'s documented contract.
    """
    if scope.kind != "media":
        return None
    if scope.media_type in (None, _UNFILTERED_MEDIA_TYPE_SENTINEL):
        return None
    return scope.media_type


def count_export_scope(
    scope: ExportScope,
    media_db: MediaIdSource,
    chachanotes_db: ChaChaNotesIdSource,
) -> dict[str, int]:
    """Count every item in ``scope`` per source, with no page cap.

    Always returns all three keys ("media", "conversations", "notes") so
    the export form can render a stable three-source summary; a source
    outside ``scope`` reports 0 rather than being omitted from the dict.
    """
    counts = {"media": 0, "conversations": 0, "notes": 0}
    if scope.kind in ("everything", "media"):
        counts["media"] = len(media_db.get_all_active_media_ids(_effective_media_type(scope)))
    if scope.kind in ("everything", "conversations"):
        counts["conversations"] = len(chachanotes_db.get_all_conversation_ids())
    if scope.kind in ("everything", "notes"):
        counts["notes"] = len(chachanotes_db.get_all_note_ids())
    return counts


def resolve_export_selections(
    scope: ExportScope,
    media_db: MediaIdSource,
    chachanotes_db: ChaChaNotesIdSource,
) -> dict[ContentType, list[str]]:
    """Resolve every id in ``scope`` into a ``ChatbookCreator`` content-selection dict.

    Issues a fresh, uncapped id query per in-scope source -- never reads a
    rendered/capped Library snapshot (see module docstring).

    Ids are ``str(int(...))`` for media (``ChatbookCreator._collect_media``
    calls ``int(media_id)`` on each entry before looking it up) and native
    id strings for conversations/notes (already UUID strings in the DB).

    A ``ContentType`` key is present only when its source is in ``scope``
    *and* resolves at least one id: a source outside ``scope`` is never
    queried at all, and an in-scope source with zero matches is omitted
    rather than included as an empty list. This keeps
    ``ChatbookCreator.create_chatbook``'s ``if ContentType.X in
    content_selections`` guards -- and the caller's
    ``ContentType.MEDIA in selections`` -> ``include_media`` decision --
    correct without extra empty-list special-casing downstream.
    """
    selections: dict[ContentType, list[str]] = {}
    if scope.kind in ("everything", "media"):
        media_ids = [
            str(int(media_id))
            for media_id in media_db.get_all_active_media_ids(_effective_media_type(scope))
        ]
        if media_ids:
            selections[ContentType.MEDIA] = media_ids
    if scope.kind in ("everything", "conversations"):
        conversation_ids = list(chachanotes_db.get_all_conversation_ids())
        if conversation_ids:
            selections[ContentType.CONVERSATION] = conversation_ids
    if scope.kind in ("everything", "notes"):
        note_ids = list(chachanotes_db.get_all_note_ids())
        if note_ids:
            selections[ContentType.NOTE] = note_ids
    return selections


def export_scope_label(scope: ExportScope, counts: Mapping[str, int]) -> str:
    """Build the export form's scope summary line.

    Examples:
        "Everything: 128 media · 542 conversations · 87 notes"
        "Media (type: video) · 12 items"
        "Media · 12 items"
        "Conversations · 542 items"
        "Notes · 87 items"
    """
    if scope.kind == "everything":
        return (
            f"Everything: {counts.get('media', 0)} media · "
            f"{counts.get('conversations', 0)} conversations · "
            f"{counts.get('notes', 0)} notes"
        )
    if scope.kind == "media":
        media_type = _effective_media_type(scope)
        if media_type is not None:
            return f"Media (type: {media_type}) · {counts.get('media', 0)} items"
        return f"Media · {counts.get('media', 0)} items"
    if scope.kind == "conversations":
        return f"Conversations · {counts.get('conversations', 0)} items"
    return f"Notes · {counts.get('notes', 0)} items"
