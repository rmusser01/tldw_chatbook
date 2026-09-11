"""Pure scope resolution for Library chatbook bulk export.

The Library canvases render capped snapshots of each source
(``LIBRARY_SOURCE_PAGE_SIZES`` in ``UI/Screens/library_screen.py``: notes
100, media 50, conversations 50, prompts 50 rows). Resolving a bulk export from those
rendered snapshots would silently truncate any library larger than the
page size. This module -- and the ``get_all_*`` DB methods it calls
(``Client_Media_DB_v2.MediaDatabase.get_all_active_media_ids``,
``DB.ChaChaNotes_DB.CharactersRAGDB.get_all_conversation_ids``/
``get_all_note_ids``, ``DB.Prompts_DB.PromptsDatabase.get_all_active_prompt_ids``)
-- deliberately never reads a rendered snapshot: every count/resolve call
issues a fresh, uncapped id query against the database.

Pure module: stdlib + ``Chatbooks.chatbook_models.ContentType`` +
``Library.library_media_state``'s backing-id coercion + type hints only. DB
handles are passed in by the caller and never constructed here.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Protocol

from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.Library.library_media_state import library_media_int_backing_id

_VALID_KINDS = ("everything", "media", "conversations", "notes", "prompts")

# Sentinel used by the Library media canvas's "no filter" select option.
_UNFILTERED_MEDIA_TYPE_SENTINEL = "All"

_KIND_TO_CONTENT_TYPE = {
    "media": ContentType.MEDIA,
    "conversations": ContentType.CONVERSATION,
    "notes": ContentType.NOTE,
    "prompts": ContentType.PROMPT,
}


@dataclass(frozen=True)
class ExportScope:
    """What a Library chatbook export should include.

    Attributes:
        kind: One of "everything", "media", "conversations", "notes", "prompts".
        media_type: Only meaningful when ``kind == "media"``; a specific
            media ``type`` column value to filter to. Ignored for every
            other ``kind``. ``None`` and the Library media canvas's "no
            filter" sentinel ``"All"`` both mean unfiltered -- every active
            media item is in scope.
        ids: An explicit subset of ids to export, overriding a whole-source
            query. Only meaningful for a single-source ``kind`` ("media",
            "conversations", "notes", "prompts") -- raises if set with
            ``kind="everything"``. When non-empty, every resolver returns
            these ids directly without querying the database -- media ids
            first normalized to their bare backing id (task-32232, see
            ``_media_selection_id``), since select mode carries canonical
            ``local:media:<n>`` display ids.
    """

    kind: str
    media_type: str | None = None
    ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"Unknown export scope kind: {self.kind!r}. Expected one of {_VALID_KINDS}."
            )
        if self.ids and self.kind == "everything":
            raise ValueError(
                "ExportScope.ids may only scope a single source, not 'everything'."
            )


class MediaIdSource(Protocol):
    """The subset of ``Client_Media_DB_v2.MediaDatabase`` this module needs."""

    def get_all_active_media_ids(self, media_type: str | None = None) -> list[int]: ...


class ChaChaNotesIdSource(Protocol):
    """The subset of ``DB.ChaChaNotes_DB.CharactersRAGDB`` this module needs."""

    def get_all_conversation_ids(self) -> list[str]: ...

    def get_all_note_ids(self) -> list[str]: ...


class PromptIdSource(Protocol):
    """The subset of ``DB.Prompts_DB.PromptsDatabase`` this module needs."""

    def get_all_active_prompt_ids(self) -> list[int]: ...


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


def _media_selection_id(media_id: str) -> str:
    """Normalize one selected media id to the bare backing id the collector parses.

    task-32232: Media select mode carries CANONICAL display ids
    (``local:media:<n>``), while ``ChatbookCreator._collect_media`` parses
    its selection with ``int(media_id)`` inside a broad ``except`` -- so an
    uncoerced selection logged ``invalid literal for int()`` per item and
    wrote a bundle containing README + ``content_items: []`` while the run
    reported success. The whole-source branch below already normalized with
    ``str(int(...))``; this is the same normalization for the explicit-ids
    branch, done through the single backing-id owner
    (``library_media_int_backing_id``) rather than a second parser.

    An id carrying no backing id is deliberately passed through UNCHANGED
    rather than dropped: dropping it would shrink the selection silently
    (the exact failure mode this fixes), while keeping it lets the
    creator's "a non-empty selection collected nothing" guard report an
    honest failure.
    """
    backing_id = library_media_int_backing_id(media_id)
    return str(backing_id) if backing_id is not None else str(media_id)


def count_export_scope(
    scope: ExportScope,
    media_db: MediaIdSource,
    chachanotes_db: ChaChaNotesIdSource,
    prompts_db: PromptIdSource | None,
) -> dict[str, int]:
    """Count every item in ``scope`` per source, with no page cap.

    Always returns all four keys ("media", "conversations", "notes",
    "prompts") so the export form can render a stable four-source summary; a source
    outside ``scope`` reports 0 rather than being omitted from the dict. When
    the optional Prompt source is unavailable, its count remains 0 without
    suppressing healthy sources in an ``everything`` export.
    """
    counts = {"media": 0, "conversations": 0, "notes": 0, "prompts": 0}
    if scope.ids:
        counts[scope.kind] = len(scope.ids)
        return counts
    if scope.kind in ("everything", "media"):
        counts["media"] = len(
            media_db.get_all_active_media_ids(_effective_media_type(scope))
        )
    if scope.kind in ("everything", "conversations"):
        counts["conversations"] = len(chachanotes_db.get_all_conversation_ids())
    if scope.kind in ("everything", "notes"):
        counts["notes"] = len(chachanotes_db.get_all_note_ids())
    if prompts_db is not None and scope.kind in ("everything", "prompts"):
        counts["prompts"] = len(prompts_db.get_all_active_prompt_ids())
    return counts


# How many titles the export canvas lists before it summarises the rest.
# Owned here, beside the query that applies it as a LIMIT, and imported by
# ``library_export_state`` -- one number, not two that can drift.
_CONTENTS_PREVIEW_LIMIT = 20


class MediaContentSource(Protocol):
    """The read seam ``preview_export_scope`` needs off ``MediaDatabase``."""

    def execute_query(self, query: str, params: tuple = ...) -> Any:
        """Execute one parameterised read and return its cursor.

        Args:
            query: A single SQL statement with ``?`` placeholders.
            params: The values to bind, positionally.

        Returns:
            The ``sqlite3.Cursor`` the statement produced. The caller owns
            it and must close it (``preview_export_scope`` does, via
            ``contextlib.closing``).

        Raises:
            Exception: Whatever the implementation raises for a bad
                statement, a missing table, or a closed connection --
                ``MediaDatabase`` raises its own ``DatabaseError``.
        """
        ...


class ExportPreview(NamedTuple):
    """What the bundle will hold, as far as one pre-write query can say.

    task-32353 AC#2: the export canvas asked for a destination and a name
    and then wrote a bundle nobody had seen the contents of. This is the
    "before you press it" half -- read on the counts worker, beside the
    counts, never on the UI thread.

    Attributes:
        titles: The in-scope media items' titles, in export (id) order.
        approx_bytes: Those items' stored content in bytes, or ``None``
            when the scope's media set is not enumerable up front (an
            ``everything`` export spans four sources, only one of which
            this query can size). ``None`` is what drives the canvas's
            honest "size known once it runs" copy -- never a zero
            standing in for "unknown", and a real zero (an empty scope)
            is reported as ``0``.
        item_count: How many ACTIVE rows the scope resolved to, or
            ``None`` when it was not enumerable. Not the same number as
            ``count_export_scope``'s: that one trusts ``len(scope.ids)``
            for an explicit selection, while this one applies the
            deleted/trashed filter the collector will apply, so a
            selection whose item was trashed underneath counts here as
            what the archive will actually hold.
    """

    titles: tuple[str, ...] = ()
    approx_bytes: int | None = None
    item_count: int | None = None


def preview_export_scope(
    scope: ExportScope, media_db: MediaContentSource | None
) -> ExportPreview:
    """Read the in-scope media items' titles and total stored size.

    Only a ``kind="media"`` scope has an enumerable, sizeable item set --
    every other kind returns the empty preview so the canvas says "size
    known once it runs" rather than guessing.

    What the byte total means: ``ChatbookCreator._collect_media`` writes
    each item's ``content`` column TWICE before compression -- once as
    ``media_<id>.txt`` and again inside a metadata JSON that embeds the
    same column under a ``"content"`` key, alongside summary/prompt/
    provenance. This counts that column ONCE, so it is a deliberate lower
    bound on the pre-compression payload, not the pre-compression total.
    That is why the canvas says "about". Do not "correct" it against the
    doubled figure: the archive is then zipped, and the user's own
    reference point is how much of their library is going in.

    One statement, not three: ``COUNT(*) OVER ()`` and ``SUM(...) OVER
    ()`` are evaluated over every row the ``WHERE`` matched, BEFORE the
    ``LIMIT`` trims the rows returned -- so a single cursor yields the
    true count, the true byte total, and just the handful of titles the
    canvas renders. That also means the three facts come from one
    snapshot (a concurrent write cannot land between them) and there is
    one cursor to close, deterministically, rather than two left to the
    garbage collector (Qodo #2/#3 on PR #2601).

    Raises rather than degrading: the quiet-degrade AND its log line live
    in ``LibraryExportController._compute_library_export_preview``,
    exactly where ``count_export_scope``'s live
    (``_compute_library_export_counts``). A silent ``except`` here would
    make the one path in this feature that can fail in production fail
    invisibly -- a canvas stuck on "size known once it runs" with no
    trace in the log.

    Args:
        scope: What this export will include.
        media_db: The media database read seam, or ``None``.

    Returns:
        The active-row count and byte total plus at most
        ``_CONTENTS_PREVIEW_LIMIT + 1`` titles, or ``ExportPreview()``
        when the scope is not sizeable.

    Raises:
        Exception: Whatever the media seam raises -- a missing table, a
            selection past SQLite's bound-parameter limit, a changed
            seam. The controller wrapper logs and degrades.
    """
    if scope.kind != "media" or media_db is None:
        return ExportPreview()
    where = ["deleted = 0", "is_trash = 0"]
    params: list[Any] = []
    if scope.ids:
        selected = [_media_selection_id(value) for value in scope.ids]
        where.append(f"id IN ({','.join('?' * len(selected))})")
        params.extend(selected)
    else:
        media_type = _effective_media_type(scope)
        if media_type is not None:
            where.append("type = ?")
            params.append(media_type)
    # LENGTH(CAST(... AS BLOB)) is the UTF-8 BYTE count; a bare LENGTH()
    # on TEXT counts characters and under-reports any non-ASCII library.
    # One row past the render limit is all the caller needs to know the
    # list was truncated; the true remainder comes from ``item_count``.
    query = (
        # No COALESCE around the SUM: SQLite rejects it in window
        # position. SUM skips NULL content (it contributes no bytes), and
        # an all-NULL scope yields NULL, which the `or 0` below floors.
        "SELECT title, COUNT(*) OVER () AS items, "
        "SUM(LENGTH(CAST(content AS BLOB))) OVER () AS size "
        f"FROM Media WHERE {' AND '.join(where)} "
        "ORDER BY id ASC LIMIT ?"
    )
    with closing(
        media_db.execute_query(
            query, (*params, _CONTENTS_PREVIEW_LIMIT + 1)
        )
    ) as cursor:
        rows = cursor.fetchall()
    if not rows:
        # An empty scope is a KNOWN zero, not an unknown -- the canvas
        # renders the two differently.
        return ExportPreview((), 0, 0)
    return ExportPreview(
        tuple(str(row["title"] or "Untitled") for row in rows),
        int(rows[0]["size"] or 0),
        int(rows[0]["items"] or 0),
    )


def resolve_export_selections(
    scope: ExportScope,
    media_db: MediaIdSource,
    chachanotes_db: ChaChaNotesIdSource,
    prompts_db: PromptIdSource | None,
) -> dict[ContentType, list[str]]:
    """Resolve every id in ``scope`` into a ``ChatbookCreator`` content-selection dict.

    Issues a fresh, uncapped id query per in-scope source -- never reads a
    rendered/capped Library snapshot (see module docstring).

    Ids are ``str(int(...))`` for media and Prompts (their collectors parse
    integer source ids) and native id strings for conversations/notes
    (already UUID strings in the DB).

    A ``ContentType`` key is present only when its source is in ``scope``
    *and* resolves at least one id: a source outside ``scope`` is never
    queried at all, and an in-scope source with zero matches is omitted
    rather than included as an empty list. This keeps
    ``ChatbookCreator.create_chatbook``'s ``if ContentType.X in
    content_selections`` guards -- and the caller's
    ``ContentType.MEDIA in selections`` -> ``include_media`` decision --
    correct without extra empty-list special-casing downstream. An unavailable
    optional Prompt source is likewise omitted while other sources resolve.
    """
    if scope.ids:
        if scope.kind == "media":
            return {ContentType.MEDIA: [_media_selection_id(i) for i in scope.ids]}
        return {_KIND_TO_CONTENT_TYPE[scope.kind]: list(scope.ids)}
    selections: dict[ContentType, list[str]] = {}
    if scope.kind in ("everything", "media"):
        media_ids = [
            str(int(media_id))
            for media_id in media_db.get_all_active_media_ids(
                _effective_media_type(scope)
            )
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
    if prompts_db is not None and scope.kind in ("everything", "prompts"):
        prompt_ids = [str(value) for value in prompts_db.get_all_active_prompt_ids()]
        if prompt_ids:
            selections[ContentType.PROMPT] = prompt_ids
    return selections


def _count_phrase(count: int, singular: str) -> str:
    """Return "1 note" / "2 notes" -- every export noun pluralises the same way.

    task-32221: only the Prompts branch used to count properly, so a
    one-note export read "1 notes" (critique #9 row 18). "media" is already
    plural, so its noun is "media item" rather than a bare "media".
    """
    return f"{count} {singular}" if count == 1 else f"{count} {singular}s"


def export_scope_label(scope: ExportScope, counts: Mapping[str, int]) -> str:
    """Build the export form's scope summary line.

    The widest scope names all four portable Library sources. Skills and
    collections remain outside Chatbook export.

    Examples:
        "Everything: 128 media items · 542 conversations · 87 notes · 13 prompts"
        "Media (type: video) · 12 items"
        "Media · 1 item"
        "Conversations · 542 items"
        "Notes · 87 items"
    """
    if scope.ids:
        selected = counts.get(scope.kind, len(scope.ids))
        return f"Selected {scope.kind} · {_count_phrase(selected, 'item')}"
    if scope.kind == "everything":
        return (
            f"Everything: {_count_phrase(counts.get('media', 0), 'media item')} · "
            f"{_count_phrase(counts.get('conversations', 0), 'conversation')} · "
            f"{_count_phrase(counts.get('notes', 0), 'note')} · "
            f"{_count_phrase(counts.get('prompts', 0), 'prompt')}"
        )
    if scope.kind == "media":
        media_type = _effective_media_type(scope)
        media_phrase = _count_phrase(counts.get("media", 0), "item")
        if media_type is not None:
            return f"Media (type: {media_type}) · {media_phrase}"
        return f"Media · {media_phrase}"
    if scope.kind == "conversations":
        return f"Conversations · {_count_phrase(counts.get('conversations', 0), 'item')}"
    if scope.kind == "notes":
        return f"Notes · {_count_phrase(counts.get('notes', 0), 'item')}"
    return f"Prompts · {_count_phrase(counts.get('prompts', 0), 'item')}"
