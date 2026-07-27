"""Map a Library ingest submission onto the server ingest-jobs API.

The Library ingest canvas has only ever run locally: ``build_library_ingest_state``
accepts a ``runtime_source`` and, per its own docstring, uses it for nothing but
a "ingest runs on Local" quiet line. Server-backed ingestion lives in a separate
window that task-684 retires, so its capability has to arrive here first.

This module is deliberately the *pure* half of that: it turns a source plus the
canvas's option snapshot into the keyword arguments
``ServerMediaReadingService.submit_ingest_jobs`` already accepts. No I/O, no
widgets, no app instance -- so the request shape is pinned by unit tests before
any routing exists.

Keep this stdlib-plus-Library-only; the heavy ``tldw_api`` schemas are built by
the service, not here.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from tldw_chatbook.Library.ingest_capabilities import get_capabilities, get_type_group
from tldw_chatbook.Library.library_ingest_jobs import DEFAULT_CHUNK_SIZE
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    FileIngestionError,
    classify_ingest_source,
    is_http_url,
)


class ServerIngestUnsupported(Exception):
    """Raised when a source has no server ingest-jobs equivalent.

    Separate from ``FileIngestionError`` so callers can tell "this app cannot
    ingest that at all" from "the *server* backend cannot, but local can".
    """


#: The only ``media_type`` values the server's ingest-jobs endpoint accepts.
#:
#: Established by submitting to a live server, NOT from its OpenAPI spec: the
#: spec types ``media_type`` as a bare string, and the real set is enforced by a
#: runtime validator whose rejection reads "Input should be 'video', 'audio',
#: 'document', 'pdf' or 'ebook'". Guarded by a test, since nothing in the
#: generated client pins it.
SERVER_ACCEPTED_MEDIA_TYPES: frozenset[str] = frozenset(
    {"video", "audio", "document", "pdf", "ebook"}
)

#: What ``classify_ingest_source``/``detect_file_type`` call a source, mapped to
#: one of :data:`SERVER_ACCEPTED_MEDIA_TYPES`.
#:
#: An earlier version of this table sent ``plaintext`` and ``xml``, inferred
#: from the retired standalone ingest window's own form dispatch. That
#: dispatch described *that window's* form, not this endpoint's contract, and
#: the server rejects both -- so every plain-text server ingest would have
#: failed validation. The server has no text/markup type at all: its document
#: extractor takes them.
SERVER_MEDIA_TYPE_BY_LOCAL_TYPE: dict[str, str] = {
    "pdf": "pdf",
    "document": "document",
    "ebook": "ebook",
    "audio": "audio",
    "video": "video",
    "plaintext": "document",
    "html": "document",
    "xml": "document",
}

#: Local types that are real capabilities but belong to a different server
#: endpoint, mapped to the reason shown to the user.
_ELSEWHERE: dict[str, str] = {
    "article": (
        "A web page is clipped rather than ingested as a media file; that runs "
        "through the web-clipper endpoint, not the ingest-jobs API."
    ),
}


def server_media_type_for(source: str) -> str:
    """Return the server media type for ``source``.

    Args:
        source: A local file path or an http(s) URL.

    Returns:
        The server's media type string.

    Raises:
        ServerIngestUnsupported: When the source has no server ingest-jobs
            equivalent -- either this app cannot ingest it at all, or the
            capability lives behind a different endpoint.
    """
    if not source or not source.strip():
        raise ServerIngestUnsupported("No source was given.")

    source = source.strip()
    try:
        local_type = classify_ingest_source(source)
    except FileIngestionError as exc:
        raise ServerIngestUnsupported(str(exc)) from exc

    reason = _ELSEWHERE.get(local_type)
    if reason:
        raise ServerIngestUnsupported(reason)

    media_type = SERVER_MEDIA_TYPE_BY_LOCAL_TYPE.get(local_type)
    if media_type is None:
        raise ServerIngestUnsupported(
            f"The server backend has no handler for {local_type!r} sources."
        )
    return media_type


def _coerce_int(value: Any, fallback: int) -> int:
    """Return ``value`` as an int, falling back rather than raising.

    Option values arrive from the canvas's form echo, where numbers are display
    *text*; a half-typed field must not be able to break a submission.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def _generic_default(name: str, fallback: Any) -> Any:
    """Return the ``generic`` group's declared default for ``name``."""
    for field_spec in get_capabilities("generic").fields:
        if field_spec.name == name:
            return field_spec.default
    return fallback


def build_server_ingest_kwargs(
    source: str,
    *,
    options: Mapping[str, Mapping[str, Any]],
    title: str = "",
    author: str = "",
    keywords: Iterable[str] = (),
    perform_analysis: bool | None = None,
) -> dict[str, Any]:
    """Build the keyword arguments for a server ingest-jobs submission.

    Mirrors what the local path already does with the same inputs, so the two
    backends cannot disagree about what the user asked for: chunking comes from
    the ``generic`` group (the single declaration of those defaults), and the
    detected type group's own options travel with it.

    Args:
        source: A local file path or an http(s) URL.
        options: The canvas's per-type option snapshot, keyed by type group.
        title: Optional title, per-file.
        author: Optional author, batch metadata.
        keywords: Keywords, batch metadata.
        perform_analysis: Whether to analyse after ingest; ``None`` defers to
            the ``generic`` group's declared default.

    Returns:
        Keyword arguments for ``ServerMediaReadingService.submit_ingest_jobs``.

    Raises:
        ServerIngestUnsupported: When ``source`` has no server equivalent.
    """
    media_type = server_media_type_for(source)
    source = source.strip()

    generic = dict(options.get("generic") or {})
    chunk_enabled = bool(generic.get("chunk", _generic_default("chunk", True)))

    kwargs: dict[str, Any] = {
        "media_type": media_type,
        "perform_chunking": chunk_enabled,
        "chunk_size": _coerce_int(
            generic.get("chunk_size", _generic_default("chunk_size", DEFAULT_CHUNK_SIZE)),
            DEFAULT_CHUNK_SIZE,
        ),
        "chunk_overlap": _coerce_int(
            generic.get("chunk_overlap", _generic_default("chunk_overlap", 100)), 100
        ),
    }

    if is_http_url(source):
        kwargs["urls"] = [source]
    else:
        kwargs["file_paths"] = [source]

    keyword_list = [k for k in (kw.strip() for kw in keywords) if k]
    if keyword_list:
        kwargs["keywords"] = keyword_list
    if title.strip():
        kwargs["title"] = title.strip()
    if author.strip():
        kwargs["author"] = author.strip()

    resolved_analysis = (
        bool(generic.get("analyze", _generic_default("analyze", False)))
        if perform_analysis is None
        else bool(perform_analysis)
    )
    kwargs["perform_analysis"] = resolved_analysis

    # Only the detected group's own options are meaningful to this submission;
    # forwarding another group's would ask the server to transcribe a PDF.
    group = get_type_group(source)
    for name, value in (options.get(group) or {}).items():
        if group == "generic" and name in {
            "analyze",
            "chunk",
            "chunk_size",
            "chunk_overlap",
        }:
            continue
        kwargs[name] = value

    return kwargs
