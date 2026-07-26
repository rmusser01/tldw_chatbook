"""Map a Library ingest submission of a web page onto a server clip request.

The counterpart to ``server_ingest_request`` for pages rather than files. The
ingest-jobs API has no media type for a web page -- ``server_media_type_for``
refuses one on purpose -- because clipping is a different endpoint with a
different shape, so the two mappings stay separate rather than growing a branch.

Two things about that endpoint drive this module's design, both established by
driving a live server (task-684.3):

- **It is synchronous.** It returns the extracted content directly, with no job
  or batch id, so a clip has nothing to poll. A clip is recorded as a local
  registry job that settles when the call returns -- unlike a server *file*
  ingest, which really does produce a remote job.
- **It reports no media id.** ``media_ids`` comes back absent, so a finished clip
  cannot link to the row the server made, exactly as with remote ingest jobs.

Pure mapping only: no UI, no I/O, so the request shape is testable without a
server.
"""

from __future__ import annotations

from typing import Any, Mapping

from tldw_chatbook.Library.ingest_capabilities import MULTI_PAGE_SCRAPE_METHODS
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    classify_ingest_source,
    is_http_url,
)

#: The server's ``ScrapeMethod`` enum, read from its OpenAPI document. Sending
#: anything else is rejected by a runtime validator -- the same trap as the
#: ingest-jobs ``media_type``, whose accepted set is likewise absent from the
#: spec's type (see ``server_ingest_request``).
SERVER_SCRAPE_METHODS = frozenset(
    {"individual", "sitemap", "url_level", "recursive_scraping"}
)

#: Default page size for chunking a clipped page, matching the capability
#: schema's ``generic`` group so a clip chunks like every other import.
_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 100


class NotAWebClipSource(ValueError):
    """Raised when a source is not a web page this can clip."""


def is_web_clip_source(source: str) -> bool:
    """Report whether ``source`` is a web page rather than a file or media URL.

    A media URL (a video host, an audio link) belongs to the ingest-jobs path,
    which can download and transcode it; only a page belongs to the clipper.

    Args:
        source: A local path or http(s) URL.

    Returns:
        ``True`` for an http(s) URL the pipeline classifies as an article.
    """
    text = str(source or "").strip()
    if not text or not is_http_url(text):
        return False
    try:
        return classify_ingest_source(text) == "article"
    except Exception:
        # An unclassifiable source is not a clip candidate; the caller's normal
        # unsupported-source handling reports it.
        return False


def _coerce_int(value: Any, fallback: int) -> int:
    """Read an int from a form value, falling back rather than raising.

    Number fields round-trip through display text, so a partially typed or
    cleared input arrives as a string. A bad value must not abort a submission
    the user has already confirmed.
    """
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def build_web_clip_kwargs(
    source: str,
    *,
    options: Mapping[str, Mapping[str, Any]],
    title: str | None = None,
    author: str | None = None,
    keywords: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the kwargs for ``ServerMediaReadingService.ingest_web_content``.

    Targets that method's real signature deliberately: a fake shaped by an
    assumption about the seam has validated a wrong call in this feature more
    than once (see ``server_ingest_request``).

    Args:
        source: The page URL to clip.
        options: The canvas's per-group option snapshot. Chunking is read from
            ``generic`` and scope from ``web``, matching where the capability
            schema declares them.
        title: Optional title override.
        author: Optional author override.
        keywords: Optional keywords.

    Returns:
        Keyword arguments for the clip call.

    Raises:
        NotAWebClipSource: If ``source`` is empty or is not a clippable page.
    """
    text = str(source or "").strip()
    if not text:
        raise NotAWebClipSource("No source was given to clip.")
    if not is_web_clip_source(text):
        raise NotAWebClipSource(
            f"{text!r} is not a web page; a file or media URL is ingested, not clipped."
        )

    generic = dict(options.get("generic") or {})
    web = dict(options.get("web") or {})

    scrape_method = str(web.get("scrape_method") or "individual").strip()
    if scrape_method not in SERVER_SCRAPE_METHODS:
        scrape_method = "individual"

    perform_chunking = bool(generic.get("chunk", True))
    kwargs: dict[str, Any] = {
        # The endpoint takes a list; the canvas submits one source per job, so a
        # clip is always a single-element list rather than a batch.
        "urls": [text],
        "scrape_method": scrape_method,
        "perform_analysis": bool(generic.get("analyze", False)),
        "perform_chunking": perform_chunking,
    }

    if perform_chunking:
        kwargs["chunk_size"] = _coerce_int(
            generic.get("chunk_size"), _DEFAULT_CHUNK_SIZE
        )
        kwargs["chunk_overlap"] = _coerce_int(
            generic.get("chunk_overlap"), _DEFAULT_CHUNK_OVERLAP
        )

    # Page and depth limits only mean anything when more than the given page is
    # fetched; sending them for a single-page clip would imply a crawl.
    if scrape_method in MULTI_PAGE_SCRAPE_METHODS:
        kwargs["max_pages"] = _coerce_int(web.get("max_pages"), 3)
        kwargs["max_depth"] = _coerce_int(web.get("max_depth"), 3)

    if title:
        kwargs["titles"] = [str(title)]
    if author:
        kwargs["authors"] = [str(author)]
    if keywords:
        kwargs["keywords"] = [str(keyword) for keyword in keywords]

    return kwargs


def clip_failure_reason(response: Any) -> str | None:
    """Return why a clip failed, or ``None`` when it succeeded.

    The endpoint answers 200 with a body describing the outcome, so a successful
    HTTP call does not by itself mean a page was captured: a per-result
    ``extraction_successful`` of ``False`` is a failed clip reported as a success
    at the transport level. Recording that as ``done`` would repeat the empty
    ingest the local pipeline had to be guarded against (task-677).

    Args:
        response: The clip response, model- or dict-shaped.

    Returns:
        A human-readable reason, or ``None`` if the clip produced content.
    """

    def field(payload: Any, name: str) -> Any:
        if isinstance(payload, Mapping):
            return payload.get(name)
        return getattr(payload, name, None)

    if response is None:
        return "The server returned no response."

    status = str(field(response, "status") or "").strip().lower()
    if status and status not in ("success", "ok", "completed"):
        message = field(response, "message")
        return str(message or f"The server reported {status!r}.")

    results = field(response, "results") or []
    if not results:
        return str(field(response, "message") or "The server captured nothing.")

    for result in results:
        if field(result, "extraction_successful") is False:
            url = field(result, "url") or "the page"
            return f"Nothing could be extracted from {url}."

    return None
