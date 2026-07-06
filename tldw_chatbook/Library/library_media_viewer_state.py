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
    )


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
    analysis = _text(detail.get("analysis_content"))

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
    )
