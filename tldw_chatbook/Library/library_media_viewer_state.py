"""Pure display-state contract for the Library media viewer canvas."""

from __future__ import annotations

import io
import itertools
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tldw_chatbook.Workspaces.conversation_browser_state import (
    format_console_relative_age,
)

_ID_KEYS = ("id", "media_id", "uuid")
_TYPE_KEYS = ("type", "media_type")
# local_file_ingestion.py maps BOTH .md/.markdown and .txt/.rst/.csv/.log to
# the single "plaintext" media type, and Obsidian imports use
# "obsidian_note" -- neither type alone proves the content is markdown, so
# LIB-13's "Rendered by default" decision also requires a content sniff
# (``looks_like_markdown_content``) before defaulting to the rendered view.
_MARKDOWN_MEDIA_TYPES = frozenset({"plaintext", "markdown", "obsidian_note"})
_ATX_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_TABLE_SEPARATOR_ROW_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$")

# task-2858 review fix: ``looks_like_markdown_content`` runs on the UI thread
# inside ``build_library_media_viewer_state`` every time LibraryScreen opens
# an item with ``include_content=True`` -- an unbounded ``splitlines()`` scan
# over a large transcript/document costs a full line-list allocation plus a
# full-content regex sweep just to pick a *default* view. These caps bound
# that cost to a small, constant-size prefix of the content. Semantics: this
# only changes the DEFAULT view -- a document whose first markdown marker
# sits beyond the sniff window will now default to Raw instead of Rendered,
# but the Raw/Rendered toggle remains available, so the content is never
# hidden or altered, only the initial view choice is heuristic. Acceptable
# by design.
MAX_MARKDOWN_SNIFF_CHARS = 32_000
MAX_MARKDOWN_SNIFF_LINES = 200
# Match the media list's temporal field + label (library_media_state._UPDATED_KEYS)
# so the same item reads the same "Updated: <age>" in the list and the viewer.
_UPDATED_KEYS = ("last_modified", "ingestion_date", "date", "updated_at")

_EMPTY_EDIT_FIELDS: dict[str, str] = {
    "title": "",
    "author": "",
    "url": "",
    "keywords": "",
}


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
            always present; Author/URL/Keywords/Updated appear only when
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
        media_type: The raw type string (``type``/``media_type`` on the
            detail), or "unknown" when absent -- the same value shown on
            the "Type: ..." metadata line.
        is_markdown: Whether the Content section should default to the
            Rendered (Markdown) view rather than Raw -- true only when
            ``media_type`` is one of the types local ingestion can
            plausibly tag a markdown file with AND ``content`` actually
            contains markdown syntax (see ``looks_like_markdown_content``).
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
    media_type: str
    is_markdown: bool
    backend: str
    canonical_id: str
    original_source: str
    stored_representation: str


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


def looks_like_markdown_content(content: str) -> bool:
    """Return True when ``content`` contains at least one line of real markdown syntax.

    Narrows the ambiguous ``"plaintext"``/``"obsidian_note"`` media types
    (which cover genuinely-markdown files alongside plain .txt/.csv/.log)
    down to the files LIB-13 is actually about: an ATX heading (``# ...``
    through ``###### ...``), a fenced code block, or a GFM table separator
    row (e.g. ``| --- | --- |``). A bare ``---`` thematic break does not
    count on its own -- only a separator row that also has at least one
    ``|`` reads as an actual table.

    The scan is bounded to the first ``MAX_MARKDOWN_SNIFF_CHARS`` characters
    and ``MAX_MARKDOWN_SNIFF_LINES`` lines of ``content`` (see that
    constant's comment) -- this is a "pick a default view" heuristic, not
    an exhaustive search, so a marker that only appears past the sniff
    window is not found and the item defaults to Raw instead of Rendered.

    Args:
        content: The media item's stored content/transcript text.

    Returns:
        True when ``content`` looks like markdown within the bounded
        sniff window, else False.
    """
    if not content:
        return False
    sniff = content[:MAX_MARKDOWN_SNIFF_CHARS]
    lines = itertools.islice(io.StringIO(sniff, newline=None), MAX_MARKDOWN_SNIFF_LINES)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _ATX_HEADING_RE.match(stripped):
            return True
        if stripped.startswith("```"):
            return True
        if _TABLE_SEPARATOR_ROW_RE.match(stripped):
            return True
    return False


def _is_markdown_media(media_type: str, content: str) -> bool:
    """Combine the media-type allowlist with the content sniff (see
    ``looks_like_markdown_content``) into the single "default to Rendered"
    decision.
    """
    if media_type.strip().lower() not in _MARKDOWN_MEDIA_TYPES:
        return False
    return looks_like_markdown_content(content)


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
        media_type="",
        is_markdown=False,
        backend="local",
        canonical_id="",
        original_source="",
        stored_representation="No stored content",
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
    arrival_note: str = "",
    backend: str = "local",
    canonical_id: str = "",
) -> LibraryMediaViewerState:
    """Build the Library media viewer canvas display state.

    Args:
        detail: A ``get_media_item`` detail mapping (the Media row plus a
            ``keywords`` list and ``content``), or None/non-mapping when no
            media item is loaded yet. Tolerated to have missing/None fields.
        now: Reference time for the "Updated" relative-age label; defaults
            to the current UTC time.
        arrival_note: One-shot context line rendered FIRST in the metadata
            lines (task-2223: e.g. reaching this item via a dedup-matched
            ingest row); empty renders nothing extra.
        backend: Provenance backend displayed by Reader Info.
        canonical_id: Stable backend-qualified id. When omitted, it is
            derived from ``backend`` and the detail's media id.

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
    updated_raw = _first_present_text(detail, _UPDATED_KEYS)
    updated_age = (
        format_console_relative_age(updated_raw, now=reference_now)
        if updated_raw
        else ""
    )

    content = _text(detail.get("content"))
    is_markdown = _is_markdown_media(media_type, content)

    lines: list[str] = []
    # (task-2223) One-shot arrival context, e.g. reaching this item via a
    # dedup-matched ingest row -- rendered first so the "why am I here"
    # is answered before the metadata.
    if arrival_note:
        lines.append(arrival_note)
    # task-4023 AC#7: an item the viewer renders as Markdown must not
    # introduce itself as "Type: plaintext" -- say what the user is
    # looking at, while still naming the stored type honestly.
    if is_markdown and media_type == "plaintext":
        lines.append("Type: markdown (stored as plaintext)")
    else:
        lines.append(f"Type: {media_type}")
    if author:
        lines.append(f"Author: {author}")
    if url and not url.startswith("local://"):
        # The synthetic "local://media/{id}" placeholder (and any other
        # local:// scheme value) is an internal identifier, not a
        # user-meaningful link -- hide it from the metadata lines while
        # still prefilling it in the edit form's URL field (L3).
        lines.append(f"URL: {url}")
    if keywords_text:
        lines.append(f"Keywords: {keywords_text}")
    if updated_age:
        lines.append(f"Updated: {updated_age}")

    analysis = _text(detail.get("analysis_content")) or _latest_version_analysis_text(
        detail
    )
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
        media_type=media_type,
        is_markdown=is_markdown,
        backend=backend,
        canonical_id=canonical_id
        or (
            f"{backend}:media:{media_id.removeprefix('media-')}" if media_id else ""
        ),
        original_source=url,
        stored_representation=("Complete stored text" if content else "No stored content"),
    )


def find_content_matches(content: str | None, query: str | None) -> tuple[int, ...]:
    """Find the 0-based line indices of lines containing ``query`` in ``content``.

    Matching is case-insensitive and a line is reported at most once even
    when the query occurs multiple times on it. This is the pure core of
    the Library media viewer's in-content search -- the widget/screen layer
    is responsible for scrolling to (and optionally highlighting) the
    resulting line indices.

    The query is stripped before matching so a padded query (e.g. a
    trailing space from the search box) counts and scrolls to the same
    lines the viewer highlights -- the highlighter strips too, so both
    stay in lockstep regardless of what the caller passes.

    Args:
        content: Full content/transcript text to search within. Tolerated
            to be None/blank, which yields no matches.
        query: Search text to look for. Tolerated to be None/blank (or
            whitespace-only), which yields no matches.

    Returns:
        Ordered (ascending) line indices of matching lines, or an empty
        tuple when either ``content`` or the stripped ``query`` is blank,
        or there are no matches.
    """
    if not content or not query:
        return ()
    needle = query.strip().lower()
    if not needle:
        return ()
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


def detail_analysis_text(detail: Mapping[str, Any]) -> str:
    """Public read of the newest version's analysis text (see the private helper)."""
    return _latest_version_analysis_text(detail)


def analysis_find_unavailable_reason(
    *, mode: str, analysis: str, generating: bool, editing: bool
) -> str:
    """Why Find cannot open on the Analysis tab right now, or "" when it can.

    Qodo on #2378: Find opens the bar for the tab being read, and the
    Analysis tab composes its bar only around analysis text. With no text
    (or while generating / editing) there is nothing to mount, so the
    gesture must be disabled with a reason instead of silently arming
    ``find_open``. The Read tab always has a body to search.

    Args:
        mode: The Reader mode (``"read"``, ``"analysis"``, ...).
        analysis: The current analysis text, or "".
        generating: Whether an analysis is being generated.
        editing: Whether the analysis edit form is open.

    Returns:
        The user-facing reason, or "" when Find is available.
    """
    if mode != "analysis":
        return ""
    if generating:
        return "Analysis is still generating."
    if editing:
        return "Finish editing the analysis first."
    if not analysis.strip():
        return "No analysis to search yet."
    return ""
