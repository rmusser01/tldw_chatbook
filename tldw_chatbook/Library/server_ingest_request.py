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

from pathlib import Path
from stat import S_ISREG
from typing import Any, Iterable, Mapping

from tldw_chatbook.Library.ingest_capabilities import (
    generic_option_default,
    get_type_group,
)
from tldw_chatbook.Library.library_ingest_jobs import DEFAULT_CHUNK_SIZE
from tldw_chatbook.Library.web_clip_request import is_web_clip_source
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
    # task-2857 review (round 3): "rather than ingested as a media file" was
    # plain user copy and is now "imported"; "the ingest-jobs API" stays --
    # it names the server's actual endpoint family (``submit_ingest_jobs``/
    # ``list_media_ingest_jobs``/``cancel_media_ingest_jobs_batch`` on
    # ``ServerMediaReadingService``, established against a live server, not
    # a Library UI label this task governs).
    "article": (
        "A web page is clipped rather than imported as a media file; that runs "
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


def empty_source_refusal(source: str) -> str | None:
    """Return why a 0-byte local file is refused before it is sent.

    (task-14910) The forecast counts every 0-byte staged file as a certain
    failure on BOTH backends. Locally that is verified -- the parse chain
    raises ``EmptySourceIngestError`` before any write. On the server path
    it was an unearned claim: the app built kwargs for the empty file and
    sent it, handing the outcome to a server this process cannot inspect
    -- the very reason the same forecast line admits "server tooling isn't
    checked from here".

    Refusing here is what makes the claim true rather than lucky, and it
    is the better outcome anyway: a 0-byte file is almost certainly a
    mistake, the reason is already known locally, and stating it takes no
    round trip. Both the forecast-side predicate
    (:func:`server_ingest_refusal`) and the submit-side builder
    (:func:`build_server_ingest_kwargs`) call THIS function, so what the
    canvas promises and what the queue records cannot drift.

    Args:
        source: A local file path or an http(s) URL.

    Returns:
        The refusal reason -- the exact message the failed job row will
        carry -- or ``None`` when the source is not a 0-byte local file.
        A URL has no local size to measure and is never called empty; nor
        is a path that cannot be statted, mirroring the pre-flight's own
        ``_statted_size`` (an unreadable file is not a 0 B one).
    """
    text = str(source or "").strip()
    if not text or is_http_url(text):
        return None
    try:
        path = Path(text).expanduser()
        info = path.stat()
    except (OSError, ValueError, RuntimeError):
        return None
    # ``S_ISREG`` because a DIRECTORY can stat at 0 bytes on some
    # filesystems, and "this folder is empty" is a different diagnosis
    # with a different recovery (the ingest gate owns that one).
    if info.st_size != 0 or not S_ISREG(info.st_mode):
        return None
    name = path.name or text
    return f"{name} is empty; there was nothing to send."


def server_ingest_refusal(source: str) -> str | None:
    """Return why a SERVER-targeted submission will refuse ``source``.

    (task-14827) The forecast must ask the backend it is actually
    targeting, and the two backends refuse different sets: local
    unsupported-ness is ``get_type_group(...) == UNSUPPORTED_GROUP``,
    while the server ALSO refuses everything it has no media type for
    (images, deliberately left server-unmapped by task-3307). Forecasting
    the server's outcome from the local verdict said "will skip" for
    files ``_submit_server_ingest_job`` records as permanent FAILURES.

    Mirrors the routing ``submit_library_ingest_job`` performs in server
    mode rather than re-deriving it: a page is handed to the clipper
    BEFORE :func:`build_server_ingest_kwargs` is ever asked, so the
    ingest-jobs API having no media type for one is not a refusal of that
    import. Everything else is asked the same question the submit path
    asks, so the two cannot drift.

    Args:
        source: A local file path or an http(s) URL.

    Returns:
        The refusal reason -- the exact message the failed job row will
        carry -- or ``None`` when the server path will accept the source.
    """
    text = str(source or "").strip()
    if not text:
        return "No source was given."
    if is_web_clip_source(text):
        return None
    # (task-14910) Emptiness outranks the type mapping: a 0-byte .png is
    # counted by the forecast as an EMPTY failure (the pre-flight pulls
    # 0-byte files out before classifying them), so the row it produces
    # must give the empty reason rather than the image one.
    empty = empty_source_refusal(text)
    if empty is not None:
        return empty
    try:
        server_media_type_for(text)
    except ServerIngestUnsupported as exc:
        return str(exc)
    return None


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
    """Return the ``generic`` group's declared default for ``name``.

    (task-3301) Delegates to the shared schema accessor so the server and
    local paths can never disagree about an untouched form's defaults.
    """
    return generic_option_default(name, fallback)


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
        ServerIngestUnsupported: When ``source`` has no server equivalent,
            or (task-14910) when it is a 0-byte file -- there is nothing
            in it to send, and only a refusal here makes the forecast's
            "will fail" a fact about this process rather than a guess
            about the server's.
    """
    empty = empty_source_refusal(source)
    if empty is not None:
        raise ServerIngestUnsupported(empty)
    media_type = server_media_type_for(source)
    source = source.strip()

    generic = dict(options.get("generic") or {})
    chunk_enabled = bool(generic.get("chunk", _generic_default("chunk", True)))

    kwargs: dict[str, Any] = {
        "media_type": media_type,
        "perform_chunking": chunk_enabled,
        "chunk_size": _coerce_int(
            generic.get(
                "chunk_size", _generic_default("chunk_size", DEFAULT_CHUNK_SIZE)
            ),
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
