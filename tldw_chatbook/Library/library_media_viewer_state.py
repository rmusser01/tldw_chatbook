"""Pure display-state contract for the Library media viewer canvas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tldw_chatbook.Workspaces.conversation_browser_state import format_console_relative_age

_ID_KEYS = ("id", "media_id", "uuid")
_TYPE_KEYS = ("type", "media_type")
_INGESTED_KEYS = ("ingestion_date", "last_modified")

_EMPTY_EDIT_FIELDS: dict[str, str] = {"title": "", "author": "", "url": "", "keywords": ""}


@dataclass(frozen=True)
class LibraryMediaHighlightRow:
    """One reading highlight row in the Library media viewer's highlights section.

    Attributes:
        highlight_id: Stable id of the highlight, as returned by
            ``media_reading_scope_service.list_highlights``/``create_highlight``.
        quote: The highlighted quote text.
        note: Optional note attached to the highlight, or "" when absent.
        color: Optional highlight color, or "" when absent.
        display_text: Ready-to-render text for the row (quote, plus a
            "Color: .../Note: ..." line when either is present).
    """

    highlight_id: str
    quote: str
    note: str
    color: str
    display_text: str


@dataclass(frozen=True)
class LibraryMediaViewerState:
    """Pure display state for the Library media viewer canvas.

    Attributes:
        media_id: Stable id of the viewed media item, or "" when empty.
        title: Media title, or "" when absent.
        metadata_lines: Ordered, ready-to-render metadata lines (Type is
            always present; Author/URL/Keywords/Ingested appear only when
            their source data is present).
        content: Full content/transcript text, or "" when none.
        analysis: Analysis content text, or "" when none.
        has_content: Whether ``content`` is non-blank.
        has_analysis: Whether ``analysis`` is non-blank.
        version: Optimistic-locking version from the detail row, or None.
        edit_fields: Current values for the edit form, keyed by
            "title"/"author"/"url"/"keywords".
        read_later: Whether the item is currently saved for read-it-later,
            sourced from the detail's ``is_read_it_later`` flag (as set by
            ``LocalMediaReadingService._enrich_with_read_it_later_state``).
    """

    media_id: str
    title: str
    metadata_lines: tuple[str, ...]
    content: str
    analysis: str
    has_content: bool
    has_analysis: bool
    version: int | None
    edit_fields: dict[str, str]
    read_later: bool


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_present_text(detail: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = detail.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _keywords_text(detail: Mapping[str, Any]) -> str:
    keywords = detail.get("keywords")
    if keywords is None:
        return ""
    if isinstance(keywords, str):
        return keywords.strip()
    if isinstance(keywords, Sequence):
        items = [str(item).strip() for item in keywords if str(item).strip()]
        return ", ".join(items)
    return ""


def _version(detail: Mapping[str, Any]) -> int | None:
    value = detail.get("version")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty_state() -> LibraryMediaViewerState:
    return LibraryMediaViewerState(
        media_id="",
        title="",
        metadata_lines=(),
        content="",
        analysis="",
        has_content=False,
        has_analysis=False,
        version=None,
        edit_fields=dict(_EMPTY_EDIT_FIELDS),
        read_later=False,
    )


def _latest_version_analysis_text(detail: Mapping[str, Any]) -> str:
    """Return the newest document version's analysis text, or "".

    Local media detail rows never carry ``analysis_content`` at the top
    level -- it lives on ``DocumentVersions`` rows only (see
    ``Client_Media_DB_v2.create_document_version``). ``get_media_item``'s
    ``versions`` list is already ordered newest-first
    (``get_all_document_versions`` sorts ``ORDER BY version_number DESC``),
    so the first entry's ``analysis_content`` is the current analysis --
    including intentionally blank when the latest version cleared it.

    Args:
        detail: A ``get_media_item`` detail mapping, possibly carrying a
            ``versions`` list.

    Returns:
        The latest version's stripped analysis text, or "" when there are
        no versions or the newest one has none.
    """
    versions = detail.get("versions")
    if not isinstance(versions, Sequence) or isinstance(versions, (str, bytes)):
        return ""
    for version in versions:
        if isinstance(version, Mapping):
            return _text(version.get("analysis_content"))
    return ""


def build_library_media_viewer_state(
    detail: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> LibraryMediaViewerState:
    """Build the Library media viewer canvas display state.

    Args:
        detail: A ``get_media_item`` detail mapping (the Media row plus a
            ``keywords`` list and ``content``), or None/non-mapping when no
            media item is loaded yet. Tolerated to have missing/None fields.
        now: Reference time for the "Ingested" relative-age label; defaults
            to the current UTC time.

    Returns:
        Immutable viewer state: title, ordered metadata lines, content,
        analysis, presence flags, version, and edit-form field values.
    """
    if not isinstance(detail, Mapping):
        return _empty_state()

    reference_now = now if now is not None else datetime.now(timezone.utc)

    media_id = _first_present_text(detail, _ID_KEYS)
    title = _text(detail.get("title"))
    media_type = _first_present_text(detail, _TYPE_KEYS) or "unknown"
    author = _text(detail.get("author"))
    url = _text(detail.get("url"))
    keywords_text = _keywords_text(detail)
    ingested_raw = _first_present_text(detail, _INGESTED_KEYS)
    ingested_age = (
        format_console_relative_age(ingested_raw, now=reference_now) if ingested_raw else ""
    )

    lines: list[str] = [f"Type: {media_type}"]
    if author:
        lines.append(f"Author: {author}")
    if url:
        lines.append(f"URL: {url}")
    if keywords_text:
        lines.append(f"Keywords: {keywords_text}")
    if ingested_age:
        lines.append(f"Ingested: {ingested_age}")

    content = _text(detail.get("content"))
    analysis = _text(detail.get("analysis_content")) or _latest_version_analysis_text(detail)
    read_later = bool(detail.get("is_read_it_later"))

    return LibraryMediaViewerState(
        media_id=media_id,
        title=title,
        metadata_lines=tuple(lines),
        content=content,
        analysis=analysis,
        has_content=bool(content),
        has_analysis=bool(analysis),
        version=_version(detail),
        edit_fields={
            "title": title,
            "author": author,
            "url": url,
            "keywords": keywords_text,
        },
        read_later=read_later,
    )


def find_content_matches(content: str | None, query: str | None) -> tuple[int, ...]:
    """Find the 0-based line indices of lines containing ``query`` in ``content``.

    Matching is case-insensitive and a line is reported at most once even
    when the query occurs multiple times on it. This is the pure core of
    the Library media viewer's in-content search -- the widget/screen layer
    is responsible for scrolling to (and optionally highlighting) the
    resulting line indices.

    Args:
        content: Full content/transcript text to search within. Tolerated
            to be None/blank, which yields no matches.
        query: Search text to look for. Tolerated to be None/blank, which
            yields no matches.

    Returns:
        Ordered (ascending) line indices of matching lines, or an empty
        tuple when either ``content`` or ``query`` is blank, or there are
        no matches.
    """
    if not content or not query:
        return ()
    needle = query.lower()
    return tuple(
        index
        for index, line in enumerate(content.split("\n"))
        if needle in line.lower()
    )


def _highlight_id_text(highlight: Mapping[str, Any]) -> str:
    value = highlight.get("id")
    if value is None:
        return ""
    return str(value)


def build_library_media_highlight_rows(
    highlights: Sequence[Mapping[str, Any]] | None,
) -> tuple[LibraryMediaHighlightRow, ...]:
    """Build the Library media viewer's highlight rows from raw highlight dicts.

    Args:
        highlights: Highlight mappings as returned by
            ``media_reading_scope_service.list_highlights`` (each with at
            least ``id``/``quote``, optionally ``note``/``color``). Tolerated
            to be None, or to contain non-mapping/blank-quote entries, which
            are skipped.

    Returns:
        Immutable, ready-to-render highlight rows in the given order.
    """
    rows: list[LibraryMediaHighlightRow] = []
    for highlight in highlights or ():
        if not isinstance(highlight, Mapping):
            continue
        quote = _text(highlight.get("quote"))
        if not quote:
            continue
        note = _text(highlight.get("note"))
        color = _text(highlight.get("color"))
        lines = [f"“{quote}”"]
        extras = []
        if color:
            extras.append(f"Color: {color}")
        if note:
            extras.append(f"Note: {note}")
        if extras:
            lines.append(" · ".join(extras))
        rows.append(
            LibraryMediaHighlightRow(
                highlight_id=_highlight_id_text(highlight),
                quote=quote,
                note=note,
                color=color,
                display_text="\n".join(lines),
            )
        )
    return tuple(rows)
