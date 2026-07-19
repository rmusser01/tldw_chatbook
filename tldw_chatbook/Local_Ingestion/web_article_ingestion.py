"""Fetch a URL and extract its article text for Library ingest.

Sync + dependency-light (httpx + trafilatura, no browser) so it runs inside
the spawn-based parse pool worker exactly like the file-parsing branches.
Heavy imports (httpx/trafilatura) are deferred inside the function.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from .local_file_ingestion import PermanentIngestError, canonicalize_url

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB body guard
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
_BOILERPLATE = re.compile(
    r"^\s*(subscribe now|sign up|share on \w+|follow us|newsletter|read more|"
    r"thanks for reading|advertisement)\b.*$",
    re.IGNORECASE,
)


def _strip_boilerplate(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _BOILERPLATE.match(ln)]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def extract_article_for_ingest(url: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch a URL and extract its readable article text for Library ingest.

    Runs synchronously inside the spawn parse-pool worker (``httpx`` and
    ``trafilatura`` are imported inside the function so the module scope stays
    stdlib-only). The response is streamed and aborted the moment it crosses
    the size cap -- an over-size or binary body is never fully buffered.
    ``trafilatura`` extracts the main article text, boilerplate lines are
    stripped, and the returned ``url`` is the canonicalized post-redirect URL.

    Args:
        url: The http(s) URL to fetch. Expected to be syntactically validated
            by the caller (the UI ``validate_url`` gate); a malformed or
            unsupported URL still fails permanently here.
        options: Ingest options. ``title`` and ``author`` are used as
            fallbacks when the page supplies none.

    Returns:
        A ``result`` dict with keys ``content`` (boilerplate-stripped article
        text), ``title``, ``author``, ``keywords``, ``chunks``, ``analysis``,
        ``metadata`` (trafilatura extras plus ``ingestion_method``), and
        ``url`` (the canonical post-redirect URL) -- the shape consumed by the
        shared ingest tail in :func:`parse_local_file_for_ingest`.

    Raises:
        PermanentIngestError: For a failure that recurs identically on retry --
            an invalid/unsupported-protocol URL, a redirect loop, a DNS
            resolution failure, a non-408/429 4xx status, a non-HTML
            content-type, an over-size response, empty extraction, or a
            missing ``trafilatura`` dependency.
        httpx.HTTPError: For a retryable transport failure (network error,
            timeout, 5xx, or 408/429) -- propagated unwrapped so the parse
            worker classifies the job as retryable.
    """
    import httpx

    try:
        with httpx.Client(follow_redirects=True, timeout=30.0,
                          headers={"User-Agent": _UA, "Accept": "text/html,*/*"}) as client:
            with client.stream("GET", url) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status in _RETRYABLE_STATUS:
                        raise                                    # retryable
                    raise PermanentIngestError(f"URL fetch failed ({status}) for {url}") from exc
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and "html" not in ctype and "xml" not in ctype:
                    raise PermanentIngestError(f"URL is not a web page (content-type {ctype!r}): {url}")
                # Stream the body, aborting the instant the running total
                # crosses the cap -- never buffer an unbounded/hostile response.
                collected = bytearray()
                for chunk in resp.iter_bytes():
                    collected += chunk
                    if len(collected) > _MAX_BYTES:
                        raise PermanentIngestError(f"URL response too large (>{_MAX_BYTES} bytes): {url}")
                final_url = str(resp.url)
                encoding = resp.encoding or "utf-8"
            body = bytes(collected).decode(encoding, errors="replace")
    except httpx.InvalidURL as exc:
        # Not an httpx.HTTPError subclass -- must be caught separately.
        raise PermanentIngestError(f"Invalid URL: {url}") from exc
    except httpx.HTTPError as exc:
        # network/timeout/5xx -> retryable; unsupported protocol, redirect
        # loop, or a DNS-resolution failure -> permanent (identical on retry).
        import socket
        if isinstance(exc, (httpx.UnsupportedProtocol, httpx.TooManyRedirects)):
            raise PermanentIngestError(f"URL could not be fetched ({type(exc).__name__}): {url}") from exc
        if isinstance(getattr(exc, "__cause__", None), socket.gaierror):
            raise PermanentIngestError(f"URL host could not be resolved: {url}") from exc
        raise  # retryable transport error

    try:
        import trafilatura
    except ImportError as exc:
        raise PermanentIngestError(
            "Web article ingest requires 'trafilatura' -- install with "
            "pip install tldw_chatbook[websearch]"
        ) from exc

    content = trafilatura.extract(body, include_comments=False, include_tables=False)
    content = _strip_boilerplate(content or "")
    if not content:
        raise PermanentIngestError(
            f"Couldn't extract article content -- the page may require JavaScript (not supported): {url}"
        )

    meta = trafilatura.extract_metadata(body)
    md = {}
    if meta is not None:
        md = {k: getattr(meta, k, None) for k in ("sitename", "hostname", "language", "date")}
    md["ingestion_method"] = "web_article"

    return {
        "content": content,
        "title": (getattr(meta, "title", None) or options.get("title") or url),
        "author": (getattr(meta, "author", None) or options.get("author") or "Unknown"),
        "keywords": [],
        "chunks": [],
        "analysis": "",
        "metadata": md,
        "url": canonicalize_url(final_url),
    }
