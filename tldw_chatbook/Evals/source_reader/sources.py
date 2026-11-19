"""Bounded, revision-pinned Library text for the fixture-only reader experiment."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from tldw_chatbook.Library.library_tool_contract import (
    DISPLAY_NAME_MAX_BYTES,
    MAX_RESULT_BYTES,
    LibraryToolError,
    parse_cursor,
    parse_public_id,
    serialized_size,
)
from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService

MAX_SOURCES = 6
MAX_SOURCE_BYTES = 256 * 1024
PAGE_CHARS = 8000
PACKET_CHARS = 4000
PACKET_OVERLAP = 200
MAX_PACKETS = 64


class SourceReaderError(ValueError):
    """A bounded status code, without source content or backend error messages."""

    def __init__(self, code: str) -> None:
        self.code = (
            code
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
            else "source_reader_error"
        )
        super().__init__(self.code)


@dataclass(frozen=True)
class Selection:
    """Host-selected source identity and the revision the host authorized."""

    source_id: str
    revision: str


@dataclass(frozen=True)
class Source:
    """Complete text captured from one selected Library source revision."""

    source_id: str
    revision: str
    title: str = field(repr=False)
    text: str = field(repr=False)


@dataclass(frozen=True)
class Packet:
    """An unchanged source span with a zero-based Unicode-codepoint start."""

    packet_id: str
    source_id: str
    revision: str
    start: int
    text: str = field(repr=False)


def assemble_sources(
    service: LocalLibraryToolService,
    selections: Iterable[Selection],
    source_ids: Iterable[str],
    *,
    direct_enabled: bool = True,
) -> tuple[Source, ...]:
    """Read the complete selected source set or fail without a partial result.

    Args:
        service: Existing direct Library read service.
        selections: Host-owned manifest of source IDs and expected revisions.
        source_ids: Distinct requested IDs within that manifest, in output order.
        direct_enabled: Whether direct Library access is currently admitted.

    Returns:
        Immutable complete source texts, rechecked after all reads finish.

    Raises:
        SourceReaderError: Invalid admission, scope, source, paging, or size.
    """
    if direct_enabled is not True:
        raise SourceReaderError("direct_access_disabled")
    selected = _selected_sources(selections, source_ids)
    sources: list[Source] = []
    retained_bytes = 0
    for selection in selected:
        pages: list[str] = []
        cursor = None
        start = 0
        total = None
        title = None
        while True:
            result = _read_page(service, selection, cursor, start, total)
            content = result["content"]
            page_text = content["text"]
            retained_bytes += _text_bytes(page_text)
            if retained_bytes > MAX_SOURCE_BYTES:
                raise SourceReaderError("source_budget_exceeded")
            if title is None:
                title = result["item"]["title"]
            elif title != result["item"]["title"]:
                raise SourceReaderError("content_changed")
            # Retain only after the complete page and aggregate size are checked.
            pages.append(page_text)
            total = content["total_chars"]
            if not content["has_more"]:
                break
            start = content["end"]
            cursor = content["next_cursor"]
        text = "".join(pages)
        if not text.strip():
            raise SourceReaderError("empty_source")
        sources.append(Source(selection.source_id, selection.revision, title, text))

    # This captures individual source revisions, not an atomic multi-item snapshot.
    # Rechecking after every item has been read catches earlier items changing
    # while a later item was assembled. The service also rechecks active access.
    for selection, source in zip(selected, sources):
        result = _read_page(service, selection, None, 0, len(source.text))
        if result["item"]["title"] != source.title or not source.text.startswith(
            result["content"]["text"]
        ):
            raise SourceReaderError("content_changed")
    return tuple(sources)


def _selected_sources(
    selections: Iterable[Selection], source_ids: Iterable[str]
) -> tuple[Selection, ...]:
    try:
        manifest = tuple(selections)
        requested = tuple(source_ids)
    except TypeError:
        raise SourceReaderError("invalid_scope") from None
    if not 1 <= len(requested) <= MAX_SOURCES:
        raise SourceReaderError("invalid_scope")
    by_id: dict[str, Selection] = {}
    for selection in manifest:
        if not isinstance(selection, Selection):
            raise SourceReaderError("invalid_scope")
        _validate_identity(selection.source_id, selection.revision)
        previous = by_id.get(selection.source_id)
        if previous is not None and previous != selection:
            raise SourceReaderError("invalid_scope")
        by_id[selection.source_id] = selection
    selected: list[Selection] = []
    seen: set[str] = set()
    for source_id in requested:
        try:
            parse_public_id(source_id, expected_type="media")
        except LibraryToolError:
            raise SourceReaderError("invalid_scope") from None
        if source_id not in by_id or source_id in seen:
            raise SourceReaderError("invalid_scope")
        seen.add(source_id)
        selected.append(by_id[source_id])
    return tuple(selected)


def _validate_identity(source_id: str, revision: str) -> None:
    try:
        parse_public_id(source_id, expected_type="media")
    except LibraryToolError:
        raise SourceReaderError("invalid_scope") from None
    if not isinstance(revision, str) or not revision.strip() or len(revision) > 128:
        raise SourceReaderError("invalid_scope")


def _text_bytes(text: str) -> int:
    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError:
        raise SourceReaderError("invalid_source_response") from None


def _read_page(
    service: LocalLibraryToolService,
    selection: Selection,
    cursor: str | None,
    start: int,
    total: int | None,
) -> Mapping:
    try:
        result = service.invoke(
            "library_get_media",
            {"id": selection.source_id, "cursor": cursor, "max_chars": PAGE_CHARS},
        )
    except Exception:  # noqa: BLE001 - scrub backend exceptions at this boundary.
        raise SourceReaderError("source_unavailable") from None
    if not isinstance(result, Mapping):
        raise SourceReaderError("invalid_source_response")
    if "error" in result:
        error = result["error"]
        code = error.get("code") if isinstance(error, Mapping) else None
        raise SourceReaderError(
            "content_changed" if code == "content_changed" else "source_unavailable"
        )
    item = result.get("item")
    content = result.get("content")
    if not isinstance(item, Mapping) or not isinstance(content, Mapping):
        raise SourceReaderError("invalid_source_response")
    if (
        item.get("id") != selection.source_id
        or item.get("type") != "media"
        or not isinstance(item.get("title"), str)
        or _text_bytes(item["title"]) > DISPLAY_NAME_MAX_BYTES
        or not isinstance(content.get("text"), str)
        or not isinstance(content.get("revision"), str)
    ):
        raise SourceReaderError("invalid_source_response")
    if content["revision"] != selection.revision:
        raise SourceReaderError("content_changed")
    text = content["text"]
    if (
        any(
            type(content.get(key)) is not int
            for key in ("start", "end", "returned_chars", "total_chars")
        )
        or content["start"] != start
        or content["end"] != start + len(text)
        or content["returned_chars"] != len(text)
        or content["total_chars"] < content["end"]
        or len(text) > PAGE_CHARS
        or type(content.get("has_more")) is not bool
        or content["has_more"] != (content["end"] < content["total_chars"])
    ):
        raise SourceReaderError("invalid_source_response")
    if total is not None and content["total_chars"] != total:
        raise SourceReaderError("content_changed")
    if content["has_more"]:
        if not text:
            raise SourceReaderError("non_progressing_source")
        try:
            state = parse_cursor(content.get("next_cursor"))
        except LibraryToolError:
            raise SourceReaderError("invalid_source_response") from None
        if (
            state["item"] != selection.source_id
            or state["rev"] != selection.revision
            or type(state["off"]) is not int
            or state["off"] != content["end"]
            or set(state) != {"v", "item", "rev", "off"}
        ):
            raise SourceReaderError("invalid_source_response")
    elif content.get("next_cursor") is not None:
        raise SourceReaderError("invalid_source_response")
    try:
        size = serialized_size(result)
    except (TypeError, ValueError, OverflowError):
        raise SourceReaderError("invalid_source_response") from None
    if size > MAX_RESULT_BYTES:
        raise SourceReaderError("invalid_source_response")
    return result


def pack_sources(sources: Iterable[Source]) -> tuple[Packet, ...]:
    """Split complete sources into bounded, unchanged overlapping spans.

    Args:
        sources: Complete sources from the host assembly boundary.

    Returns:
        Packets in source order, with deterministic invocation-local IDs.

    Raises:
        SourceReaderError: Empty sources or more than 64 required packets.
    """
    packets: list[Packet] = []
    for source in sources:
        if not source.text.strip():
            raise SourceReaderError("empty_source")
        start = 0
        while True:
            if len(packets) >= MAX_PACKETS:
                raise SourceReaderError("packet_budget_exceeded")
            text = source.text[start : start + PACKET_CHARS]
            packets.append(
                Packet(
                    f"p{len(packets) + 1}",
                    source.source_id,
                    source.revision,
                    start,
                    text,
                )
            )
            if start + len(text) == len(source.text):
                break
            start += PACKET_CHARS - PACKET_OVERLAP
    if not packets:
        raise SourceReaderError("empty_source")
    return tuple(packets)
