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
    import httpx

    try:
        with httpx.Client(follow_redirects=True, timeout=30.0,
                          headers={"User-Agent": _UA, "Accept": "text/html,*/*"}) as client:
            resp = client.get(url)
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
            body = resp.text
            if len(body.encode("utf-8", errors="ignore")) > _MAX_BYTES:
                raise PermanentIngestError(f"URL response too large (>{_MAX_BYTES} bytes): {url}")
            final_url = str(resp.url)
    except httpx.HTTPError as exc:
        # network/timeout -> retryable; a DNS resolution failure -> permanent.
        import socket
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
