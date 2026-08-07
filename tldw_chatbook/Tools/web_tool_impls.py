"""Sync cores for web_* agent tools.

The SSRF guard below is written fresh for tldw_chatbook, using tldw_server's
tldw_Server_API/app/core/Web_Scraping/outbound_policy.py @ 5605b9d9906322c2e6b5342b48c391ae674d315e
(https://github.com/rmusser01/tldw_server, GPL-3.0-only) as the requirements
checklist — see re-plan spec §2.1 (2026-08-05).

web_fetch ports the *behaviors* (manual redirect loop with per-hop policy
checks, bounded streaming reads, per-domain rate limiting, TTL response cache,
trafilatura extraction with tag-strip fallback) of tldw_server's
tldw_Server_API/app/core/MCP_unified/modules/implementations/web_fetch_module.py,
web_rate_limit.py and web_cache.py @ 5605b9d9906322c2e6b5342b48c391ae674d315e —
rewritten as a small sync function, not a line port.
"""

import codecs
import html as html_lib
import importlib.util
import ipaddress
import re
import socket
import time
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx
from loguru import logger

from .local_tool_impls import LocalToolError

# XXE hardening for attacker-controlled sitemap XML (mirrors
# tldw_chatbook/Subscriptions/security.py's pattern): defusedxml is an
# optional dep (websearch/ebook/subscriptions extras), so fall back to the
# stdlib parser rather than hard-failing this core module when it's absent.
# defusedxml.ElementTree re-exports xml.etree.ElementTree's ParseError object
# unchanged (verified: `xET.ParseError is xml.etree.ElementTree.ParseError`
# == True across both branches). defusedxml's own refusals (EntitiesForbidden
# etc.) are a SEPARATE hierarchy that subclasses ValueError, not ParseError,
# so `_parse_sitemap` catches `(xET.ParseError, ValueError)` to cover both a
# malformed document and a hardening refusal with the same structured
# [crawl-failed] error, regardless of which library is in use.
try:
    import defusedxml.ElementTree as xET
except ImportError:
    import xml.etree.ElementTree as xET

    logger.warning(
        "defusedxml not available, using standard xml.etree for sitemap parsing. "
        "Install defusedxml for better security."
    )

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Non-public ranges ipaddress does not flag on its own: CGNAT/shared space
# (RFC 6598 — Tailscale tailnets, carrier gear) is NOT private per
# ipaddress.is_private on Python 3.12, and 192.0.0.0/24 (IETF protocol
# assignments, RFC 6890) is only partially covered across versions.
_BLOCKED_EXTRA_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("192.0.0.0/24"),
)


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:  # ::ffff:127.0.0.1 -> check the embedded v4 address
        ip = mapped
    if any(ip in net for net in _BLOCKED_EXTRA_NETWORKS):
        return False
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def validate_outbound_url(url: str) -> str:
    """Return ``url`` if it's safe to fetch; raise LocalToolError otherwise.

    Checks: scheme allowlist (http/https only), host resolves, and EVERY
    resolved IP is public (private/loopback/link-local/multicast/reserved/
    unspecified/CGNAT refused). Literal IPs are checked directly without DNS.

    Called for the initial URL AND every redirect hop. DNS-rebinding caveat:
    resolution happens again inside the HTTP client, so a hostile DNS server
    could return a public IP here and a private one at connection time;
    re-validating every redirect hop bounds that window. Pinning the
    connection to the validated IP is deliberately out of scope.

    Raises:
        LocalToolError: if the scheme is disallowed, the URL is malformed
            (bad port, bad IPv6 brackets), the host is missing or does not
            resolve, or any resolved IP is not public.
    """
    try:
        parts = urlsplit(url.strip())
        port = parts.port  # ValueError on out-of-range/non-numeric ports
    except ValueError as exc:
        raise LocalToolError(f"URL is malformed: {url!r}") from exc
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise LocalToolError(f"URL scheme not allowed (http/https only): {url!r}")
    host = parts.hostname
    if not host:
        raise LocalToolError(f"URL has no host: {url!r}")
    try:
        ipaddress.ip_address(host)  # literal IP: check directly
        candidates = [host]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, port or (443 if parts.scheme == "https" else 80),
                                       proto=socket.IPPROTO_TCP)
        except (socket.gaierror, UnicodeError, OSError) as exc:
            raise LocalToolError(f"host does not resolve: {host!r}") from exc
        candidates = [info[4][0] for info in infos]
    if not candidates or not all(_is_public_ip(ip) for ip in candidates):
        raise LocalToolError(f"host resolves to a private/internal address: {host!r}")
    return url


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------

FETCH_MAX_REDIRECTS = 5
FETCH_TIMEOUT_SECONDS = 30.0
FETCH_MAX_BYTES = 1 * 1024 * 1024          # default cap
FETCH_HARD_MAX_BYTES = 5 * 1024 * 1024     # absolute ceiling for max_bytes arg
FETCH_CACHE_TTL_SECONDS = 900.0
FETCH_CACHE_MAX_ENTRIES = 256
RATE_LIMIT_INTERVAL_SECONDS = 1.0          # per-domain min interval
PDF_MAX_BYTES = 20 * 1024 * 1024  # refusal threshold, never a truncation (spec §1)

_USER_AGENT = "tldw-chatbook-web-fetch/1.0"
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# Content types surfaced as text. Anything else is refused so the tool never
# returns binary blobs through the text tool contract.
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_PLAIN_TYPES = frozenset({
    "text/plain", "text/markdown", "application/json",
    "application/xml", "text/xml", "application/ld+json",
})

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t ]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")

# Module-level state (per-process). Cleared by _reset_state_for_tests().
# Keyed by (url, effective max_bytes): a small-cap fetch must not poison a
# later full-cap call. Bounded because web_crawl bulk-loads it (spec §1).
_fetch_cache: dict[tuple[str, int], tuple[float, str]] = {}
_domain_last_fetch: dict[str, float] = {}

# Test seam: tests set this to an httpx.MockTransport.
_transport: "httpx.BaseTransport | None" = None


def _reset_state_for_tests() -> None:
    """Clear the module-level fetch cache and rate-limit state."""
    _fetch_cache.clear()
    _domain_last_fetch.clear()


def _cache_put(key: tuple[str, int], text: str) -> None:
    """Insert into cache, evicting earliest-expiry entry if at capacity."""
    if key not in _fetch_cache and len(_fetch_cache) >= FETCH_CACHE_MAX_ENTRIES:
        oldest = min(_fetch_cache, key=lambda k: _fetch_cache[k][0])
        _fetch_cache.pop(oldest)
    _fetch_cache[key] = (time.monotonic() + FETCH_CACHE_TTL_SECONDS, text)


def _validate_hop(url: str) -> None:
    """validate_outbound_url with structured reason prefixes."""
    try:
        validate_outbound_url(url)
    except LocalToolError as exc:
        msg = str(exc)
        reason = "ssrf" if ("private" in msg or "internal" in msg) else "invalid-url"
        raise LocalToolError(f"[{reason}] {msg}") from exc


def _enforce_rate_limit(host: str) -> None:
    """Sleep until the per-domain minimum interval has elapsed.

    Raises LocalToolError("rate-limited") if sleeping did not advance the
    clock (pathological/frozen clock) instead of spinning forever.
    """
    last = _domain_last_fetch.get(host)
    now = time.monotonic()
    if last is not None:
        wait = RATE_LIMIT_INTERVAL_SECONDS - (now - last)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
            if now - last < RATE_LIMIT_INTERVAL_SECONDS:
                raise LocalToolError(
                    f"[rate-limited] per-domain interval not respected for {host!r}"
                )
    _domain_last_fetch[host] = now


_PDF_MAGIC = b"%PDF-"


def _fetch_once(
    client: httpx.Client,
    url: str,
    max_bytes: int,
    *,
    pdf_max_bytes: "int | None" = None,
    html_only: bool = False,
) -> tuple[int, httpx.Headers, bytes, bool, bool]:
    """One GET with a bounded streaming read; redirects are NOT followed.

    Returns (status, headers, body, truncated, is_pdf). The read cap is
    decided MID-STREAM (spec §1): a response identified as PDF — by header
    or by a %PDF- prefix sniff on the first >=5 buffered bytes — reads up
    to ``pdf_max_bytes`` instead of ``max_bytes``, because a byte-truncated
    PDF is unparseable. ``html_only`` (web_crawl) stops the body read once
    the type is KNOWN (after the sniff buffer) and it is not HTML — either
    the sniff resolved ``is_pdf`` True, or a non-empty declared type is not
    an HTML type. A response that sniffs as non-PDF and declares HTML (or
    nothing) keeps reading for the caller's later ``<html`` sniff.
    """
    with client.stream("GET", url) as response:
        status = response.status_code
        if status in _REDIRECT_STATUSES:
            return status, response.headers, b"", False, False
        declared = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        chunks: list[bytes] = []
        downloaded = 0
        is_pdf: "bool | None" = True if declared == "application/pdf" else None
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            downloaded += len(chunk)
            if is_pdf is None and downloaded >= len(_PDF_MAGIC):
                is_pdf = b"".join(chunks)[: len(_PDF_MAGIC)] == _PDF_MAGIC
            if html_only and is_pdf is not None and (is_pdf or (declared and declared not in _HTML_TYPES)):
                break  # crawl only needs the type; don't drain the body
            cap = pdf_max_bytes if (is_pdf and pdf_max_bytes is not None) else max_bytes
            if downloaded > cap:
                break  # overshoot by at most one chunk; sliced below
        if is_pdf is None:  # body shorter than the magic prefix
            is_pdf = b"".join(chunks)[: len(_PDF_MAGIC)] == _PDF_MAGIC
        body = b"".join(chunks)
        cap = pdf_max_bytes if (is_pdf and pdf_max_bytes is not None) else max_bytes
        truncated = len(body) > cap
        if truncated:
            body = body[:cap]
        return status, response.headers, body, truncated, is_pdf


def _decode_body(body: bytes, content_type: str) -> str:
    """Decode using the charset advertised in the content-type header."""
    charset = "utf-8"
    if "charset=" in content_type.lower():
        candidate = content_type.lower().split("charset=", 1)[1].split(";", 1)[0].strip().strip('"')
        try:
            codecs.lookup(candidate)
            charset = candidate
        except (LookupError, ValueError):
            charset = "utf-8"
    return body.decode(charset, errors="replace")


def _strip_tags(html: str) -> str:
    """Fallback extraction: drop script/style, strip tags, collapse whitespace."""
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", without_scripts)
    text = html_lib.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def _extract_text(body: bytes, content_type: str) -> str:
    """Extract readable text; trafilatura for HTML, raw for plain types."""
    main_type = content_type.split(";", 1)[0].strip().lower()
    text = _decode_body(body, content_type)

    if main_type in _HTML_TYPES or (not main_type and "<html" in text.lower()):
        try:
            import trafilatura  # local import: heavy, keep module import cheap

            extracted = trafilatura.extract(
                text, include_comments=False, include_tables=False, include_images=False
            )
        except Exception:  # extractor failures fall back to tag strip
            extracted = None
        if extracted and extracted.strip():
            return extracted.strip()
        stripped = _strip_tags(text)
        if stripped:
            return stripped
        raise LocalToolError("[empty-content] no readable content could be extracted")

    if main_type and main_type not in _PLAIN_TYPES:
        raise LocalToolError(f"[empty-content] unsupported content type: {main_type}")
    cleaned = text.strip()
    if not cleaned:
        raise LocalToolError("[empty-content] response body was empty")
    return cleaned


def _pymupdf_available() -> bool:
    """Cheap availability probe (no import): the 20 MB PDF read ceiling and
    the [missing-dep] refusal must be decided before downloading, and
    optional_deps.check_dependency() eagerly imports the module — wrong cost
    for the fetch hot path."""
    return importlib.util.find_spec("pymupdf") is not None


def _extract_pdf_text(body: bytes, max_bytes: int) -> str:
    """Ephemeral PDF text extraction: bytes in, text out, nothing on disk.

    Stops the page loop as soon as accumulated text passes ``max_bytes`` —
    a 20 MB PDF can be thousands of pages and the tail is about to be
    thrown away anyway (spec §1).
    """
    try:
        import pymupdf  # local import: optional heavy dep (pdf extra)
    except ImportError as exc:
        raise LocalToolError(
            "[missing-dep] PDF support requires pymupdf — pip install tldw_chatbook[pdf]"
        ) from exc
    try:
        doc = pymupdf.open(stream=body, filetype="pdf")
    except Exception as exc:
        raise LocalToolError(f"[pdf-error] could not parse PDF: {exc}") from exc
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise LocalToolError("[pdf-error] PDF is encrypted")
        total_pages = doc.page_count
        parts: list[str] = []
        size = 0
        processed = 0
        for page in doc:
            text = page.get_text()
            processed += 1
            if text.strip():
                parts.append(text.strip())
                size += len(text.encode("utf-8"))
            if size > max_bytes:
                break
    except LocalToolError:
        raise
    except Exception as exc:  # damaged page trees surface mid-iteration
        raise LocalToolError(f"[pdf-error] could not parse PDF: {exc}") from exc
    finally:
        doc.close()
    joined = "\n\n".join(parts)
    if not joined:
        raise LocalToolError(
            "[empty-content] PDF contains no extractable text (scanned document?) "
            "— use media ingestion with OCR"
        )
    if size > max_bytes or processed < total_pages:
        raw = joined.encode("utf-8")[:max_bytes]
        joined = raw.decode("utf-8", errors="ignore") + (
            f"\n\n[... truncated: extracted text exceeded max_bytes={max_bytes}; "
            f"processed {processed} of {total_pages} pages ...]"
        )
    return joined


def web_fetch(url: str, *, max_bytes: int = FETCH_MAX_BYTES) -> str:
    """Fetch ``url`` and return extracted text (trafilatura for HTML, PyMuPDF for PDF).

    SSRF-guarded per hop (validate_outbound_url), redirect-capped,
    rate-limited per domain, cached in-memory by (url, max_bytes) key
    (256-entry bound, earliest-expiry eviction, FETCH_CACHE_TTL_SECONDS expiry).

    HTML/plain-text extracted via trafilatura with fallback tag-strip;
    script/style tags removed. Result ends with a truncation marker when
    capped at max_bytes (default FETCH_MAX_BYTES=1 MiB).

    PDF detection: declared type "application/pdf" or %PDF- magic sniff.
    Text extracted via PyMuPDF if available. When pymupdf is installed, PDFs
    respect a 20 MB hardened ceiling (PDF_MAX_BYTES, never truncated);
    when unavailable, PDFs are refused before download. Extracted text is
    truncated per-page if total exceeds max_bytes; the result includes page count.

    All failures raise LocalToolError with structured reasons:
        - "invalid-url", "ssrf", "redirect-limit", "timeout",
          "http-<status>", "rate-limited", "fetch-failed" (general fetch)
        - "empty-content" (unextractable HTML/text/PDF)
        - "missing-dep" (PDF requires pymupdf extra)
        - "pdf-error" (PyMuPDF extraction or encryption failure)
        - "too-large" (PDF exceeds 20 MB ceiling when pymupdf available)

    Raises:
        LocalToolError: on invalid/SSRF URLs, redirect overflow, timeouts,
            HTTP error statuses, rate limiting, PDF processing errors, or
            unextractable content.
    """
    if not isinstance(url, str) or not url.strip():
        raise LocalToolError("[invalid-url] url must be a non-empty string")
    url = url.strip()
    try:
        max_bytes = max(1, min(int(max_bytes), FETCH_HARD_MAX_BYTES))
    except (TypeError, ValueError) as exc:
        raise LocalToolError(f"[invalid-url] max_bytes must be an integer: {max_bytes!r}") from exc

    cached = _fetch_cache.get((url, max_bytes))
    if cached is not None:
        expires_at, text = cached
        if time.monotonic() < expires_at:
            _validate_hop(url)  # re-check policy on cache hits (cheap, no body)
            return text
        _fetch_cache.pop((url, max_bytes), None)

    client = httpx.Client(
        follow_redirects=False,
        timeout=FETCH_TIMEOUT_SECONDS,
        headers={"User-Agent": _USER_AGENT},
        transport=_transport,
        # trust_env=False: with HTTP(S)_PROXY set, the proxy does its own DNS
        # and connects on our behalf, bypassing the SSRF guard entirely.
        trust_env=False,
    )
    try:
        current_url = url
        for _hop in range(FETCH_MAX_REDIRECTS + 1):
            # Policy re-checked on EVERY hop: a permitted URL must not be able
            # to redirect into private/denied address space.
            _validate_hop(current_url)
            _enforce_rate_limit(urlsplit(current_url).hostname or "unknown")
            status, headers, body, truncated, is_pdf = _fetch_once(
                client, current_url, max_bytes,
                pdf_max_bytes=PDF_MAX_BYTES if _pymupdf_available() else None,
            )
            if status in _REDIRECT_STATUSES:
                location = headers.get("location")
                if not location:
                    raise LocalToolError(f"[http-{status}] redirect without a Location header")
                current_url = urljoin(current_url, location)
                continue
            break
        else:
            raise LocalToolError(
                f"[redirect-limit] exceeded {FETCH_MAX_REDIRECTS} redirects for {url!r}"
            )
    except httpx.TimeoutException as exc:
        raise LocalToolError(
            f"[timeout] fetch timed out after {FETCH_TIMEOUT_SECONDS}s: {url!r}"
        ) from exc
    except httpx.InvalidURL as exc:  # NOT an HTTPError subclass — catch explicitly
        raise LocalToolError(f"[invalid-url] {exc}") from exc
    except httpx.HTTPError as exc:
        raise LocalToolError(f"[fetch-failed] {exc}") from exc
    finally:
        client.close()

    if status >= 400:
        raise LocalToolError(f"[http-{status}] upstream returned status {status} for {url!r}")

    if is_pdf:
        if not _pymupdf_available():
            # Decided BEFORE the size check: pdf_max_bytes was already None
            # for this fetch (above), so `truncated` reflects the ordinary
            # max_bytes cap, not the 20 MB ceiling — a [too-large] verdict
            # here would be meaningless (and dishonest about the cap used).
            raise LocalToolError(
                "[missing-dep] PDF support requires pymupdf — pip install tldw_chatbook[pdf]"
            )
        if truncated:  # hit the 20 MB PDF ceiling: refuse, never truncate bytes
            raise LocalToolError(
                f"[too-large] PDF exceeds {PDF_MAX_BYTES // (1024 * 1024)} MB — "
                "use media ingestion for large documents"
            )
        text = _extract_pdf_text(body, max_bytes)
    else:
        text = _extract_text(body, headers.get("content-type", ""))
        if truncated:
            text += f"\n\n[... truncated: response exceeded max_bytes={max_bytes} ...]"
    _cache_put((url, max_bytes), text)
    return text


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

SEARCH_DEFAULT_ENGINE = "duckduckgo"
SEARCH_ENGINES = ("google", "bing", "duckduckgo", "brave", "kagi", "tavily", "searx")
SEARCH_DEFAULT_RESULT_COUNT = 5
SEARCH_MAX_RESULT_COUNT = 10
# Byte budgets (re-plan spec §2.2), matching the provider's byte-based
# 32 KiB result fitting: per-result bound and a total cap comfortably under
# it, so provider fitting never triggers even for multibyte (CJK) content.
SEARCH_RESULT_MAX_BYTES = 4 * 1024
SEARCH_TOTAL_MAX_BYTES = 24 * 1024

_TRUNCATED_MARKER = "… [truncated]"


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    """Cap ``text`` at ``max_bytes`` of UTF-8, never splitting a codepoint."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore") + _TRUNCATED_MARKER


def web_search(
    query: str,
    *,
    search_engine: str = SEARCH_DEFAULT_ENGINE,
    result_count: int = SEARCH_DEFAULT_RESULT_COUNT,
) -> str:
    """Run a web search and return bounded, formatted results as text.

    Delegates to ``Web_Scraping.WebSearch_APIs.perform_websearch`` with the
    legacy ``Tools/web_search_tool.py`` config-default wiring (country US,
    English in/out, moderate safesearch, no advanced filters). Each result
    block is bounded to SEARCH_RESULT_MAX_BYTES and the whole output to
    SEARCH_TOTAL_MAX_BYTES (both UTF-8 byte budgets), so the provider's
    32 KiB byte fitting never triggers on search output. Backend failures
    and error envelopes return an error string rather than raising (legacy
    tool contract); only invalid arguments raise LocalToolError.

    Raises:
        LocalToolError: if ``query`` is empty.
    """
    if not isinstance(query, str) or not query.strip():
        raise LocalToolError("[invalid-args] query must be a non-empty string")
    query = query.strip()
    # Coerced like result_count below: garbage input degrades to the default.
    if isinstance(search_engine, str) and search_engine.strip():
        engine = search_engine.strip().lower()
    else:
        engine = SEARCH_DEFAULT_ENGINE
    try:
        count = int(result_count)
    except (TypeError, ValueError):
        count = SEARCH_DEFAULT_RESULT_COUNT
    if count < 1 or count > SEARCH_MAX_RESULT_COUNT:
        count = SEARCH_DEFAULT_RESULT_COUNT

    # Local import: WebSearch_APIs pulls the config/metrics stack; keep this
    # module cheap to import and let tests monkeypatch the source attribute.
    from ..Web_Scraping.WebSearch_APIs import perform_websearch

    try:
        results = perform_websearch(
            search_engine=engine,
            search_query=query,
            content_country="US",
            search_lang="en",
            output_lang="en",
            result_count=count,
            date_range=None,
            safesearch="moderate",
            site_blacklist=None,
            exactTerms=None,
            excludeTerms=None,
            filter=None,
            geolocation=None,
            search_result_language=None,
            sort_results_by=None,
        )
    except Exception as exc:  # noqa: BLE001 — backend failure is a result string, not an exception
        logger.warning(f"web_search backend failure via {engine!r}: {exc}")
        return f"[search-failed] web search via {engine!r} failed: {exc}"

    if not isinstance(results, dict):
        return (
            f"No results found or unexpected response format from {engine!r} "
            f"(raw: {str(results)[:500]})"
        )
    # A well-formed envelope can still carry a failure: surface THAT reason.
    reason = results.get("processing_error") or results.get("error")
    if reason:
        return f"[search-failed] web search via {engine!r} reported an error: {reason}"
    if not isinstance(results.get("results"), list):
        return (
            f"No results found or unexpected response format from {engine!r} "
            f"(raw: {str(results)[:500]})"
        )
    items = [item if isinstance(item, dict) else {} for item in results["results"][:count]]
    if not items:
        return f"No results found for {query!r} via {engine!r}."

    blocks: list[str] = []
    total_bytes = 0
    for i, item in enumerate(items, 1):
        # Real standardized shape (process_web_search_results): body text is
        # top-level "content"; "snippet" lives under metadata. Accept both.
        snippet = item.get("snippet") or item.get("content") or "No description available"
        block = (
            f"{i}. {item.get('title') or 'No title'}\n"
            f"   URL: {item.get('url') or ''}\n"
            f"   {snippet}"
        )
        block = _truncate_to_bytes(block, SEARCH_RESULT_MAX_BYTES)
        block_bytes = len(block.encode("utf-8"))
        if total_bytes + block_bytes > SEARCH_TOTAL_MAX_BYTES:
            blocks.append("… [further results omitted: total size cap reached]")
            break
        blocks.append(block)
        total_bytes += block_bytes
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# web_crawl (spec 2026-08-06 §2)
# ---------------------------------------------------------------------------

CRAWL_DEFAULT_MAX_PAGES = 20
CRAWL_MAX_PAGES_CEILING = 40
CRAWL_DEFAULT_MAX_DEPTH = 2
CRAWL_MAX_DEPTH_CEILING = 5
CRAWL_DEADLINE_SECONDS = 120.0
CRAWL_PAGE_TIMEOUT_SECONDS = 10.0   # per page; a hung page must not eat the crawl
CRAWL_EXCERPT_MAX_CHARS = 200
CRAWL_RESULT_MAX_BYTES = 24 * 1024
CRAWL_BLOCK_MAX_BYTES = 1024
CRAWL_MAX_LINKS_PER_PAGE = 500      # frontier bound: cap links enqueued FROM one page
SITEMAP_MAX_BYTES = 5 * 1024 * 1024
SITEMAP_MAX_CHILDREN = 20           # cap child sitemaps actually fetched from an index

_CRAWL_USER_AGENT = "tldw-chatbook-web-crawl/1.0"

_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


class _CrawlLinkParser(HTMLParser):
    """Collect <a href>, <base href>, and <title> text from one page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.base_href: "str | None" = None
        self.title: str = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        elif tag == "base" and self.base_href is None:
            href = dict(attrs).get("href")
            if href:
                self.base_href = href
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            # Bounded: an unclosed <title> would otherwise concatenate the
            # rest of the page's text data unboundedly (handle_data fires
            # once per chunk of text). The excerpt/title only ever needs a
            # display-sized prefix.
            self.title = (self.title + data)[:512]


def _crawl_host(url: str) -> str:
    """Lowercased host with a leading ``www.`` folded; '' when absent/bad."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _normalize_crawl_url(url: str) -> str:
    """Visited-set identity: scheme+folded host+path+query, no fragment.

    On malformed URLs (bad port, invalid IPv6, etc.), returns the input unchanged
    for stable visited-set identity; downstream egress guard rejects them as invalid.
    """
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        port = f":{parts.port}" if parts.port else ""
        path = parts.path or "/"
        query = f"?{parts.query}" if parts.query else ""
        return f"{parts.scheme.lower()}://{host}{port}{path}{query}"
    except ValueError:
        # Malformed URL (e.g., bad port, invalid IPv6): return unchanged
        return url


def _coerce_budget(value, default: int, ceiling: int) -> int:
    """v1 argument style: garbage degrades to the default, range clamps."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(result, ceiling))


def _parse_sitemap(xml_bytes: bytes) -> tuple[list[str], list[str]]:
    """Return (page_urls, child_sitemap_urls) from a urlset/sitemapindex.

    Sitemaps without the sitemaps.org namespace (some generators emit
    these) are also accepted: the namespaced findall is tried first, and
    an un-namespaced `.//loc` fallback runs only when it comes back empty.
    """
    try:
        root = xET.fromstring(xml_bytes)
    # defusedxml's refusals (EntitiesForbidden, etc.) subclass ValueError,
    # not xET.ParseError, so a hardening refusal must be caught here too —
    # otherwise it escapes as a raw, untyped exception instead of the
    # structured [crawl-failed] every other sitemap failure produces.
    except (xET.ParseError, ValueError) as exc:
        raise LocalToolError(f"[crawl-failed] sitemap could not be parsed: {exc}") from exc
    locs = [
        loc.text.strip()
        for loc in root.findall(f".//{_SITEMAP_NS}loc")
        if loc.text and loc.text.strip()
    ]
    if not locs:
        locs = [
            loc.text.strip()
            for loc in root.findall(".//loc")
            if loc.text and loc.text.strip()
        ]
    if root.tag.rsplit("}", 1)[-1] == "sitemapindex":
        return [], locs
    return locs, []


def _format_crawl_result(pages: list[dict], failed: int, blocked: int, stop_reason: str) -> str:
    blocks: list[str] = []
    total = 0
    for i, page in enumerate(pages, 1):
        if page["marker"]:
            block = f"{i}. {page['marker']}\n   URL: {page['url']}"
        else:
            block = f"{i}. {page['title'] or 'No title'}\n   URL: {page['url']}"
            if page["excerpt"]:
                block += f"\n   {page['excerpt']}"
        block = _truncate_to_bytes(block, CRAWL_BLOCK_MAX_BYTES)
        block_bytes = len(block.encode("utf-8"))
        if total + block_bytes > CRAWL_RESULT_MAX_BYTES:
            blocks.append("… [further pages omitted: total size cap reached]")
            break
        blocks.append(block)
        total += block_bytes
    footer = f"Crawled {len(pages)} pages ({failed} failed, {blocked} blocked). Stopped: {stop_reason}."
    return "\n\n".join(blocks + [footer]) if blocks else footer


class _CrawlDeadline(Exception):
    """Internal: the wall-clock budget expired mid-fetch."""


def _crawl_fetch_page(
    client: httpx.Client,
    url: str,
    deadline: float,
    *,
    max_bytes: int = FETCH_MAX_BYTES,
    html_only: bool = True,
) -> tuple[str, "httpx.Headers", bytes, bool, bool]:
    """Guarded, rate-limited GET with the crawl's redirect loop.

    Returns (final_url, headers, body, truncated, is_pdf). Checks the
    deadline between redirect hops — one page's full chain must not
    overshoot the crawl budget by minutes (spec §2). ``is_pdf`` is
    ``_fetch_once``'s sniff result and MUST be consulted by callers before
    treating a response as HTML: a PDF mislabeled (or unlabeled) as
    text/html must not fall through to HTML extraction, or its raw bytes
    become the page excerpt and its garbage "extracted text" warm-writes
    the shared web_fetch cache (see web_crawl's marker branch).
    """
    current = url
    for _hop in range(FETCH_MAX_REDIRECTS + 1):
        if time.monotonic() >= deadline:
            raise _CrawlDeadline()
        _validate_hop(current)
        _enforce_rate_limit(urlsplit(current).hostname or "unknown")
        try:
            status, headers, body, truncated, is_pdf = _fetch_once(
                client, current, max_bytes, html_only=html_only
            )
        except httpx.TimeoutException as exc:
            raise LocalToolError(f"[timeout] fetch timed out: {current!r}") from exc
        except httpx.InvalidURL as exc:
            raise LocalToolError(f"[invalid-url] {exc}") from exc
        except httpx.HTTPError as exc:
            raise LocalToolError(f"[fetch-failed] {exc}") from exc
        if status in _REDIRECT_STATUSES:
            location = headers.get("location")
            if not location:
                raise LocalToolError(f"[http-{status}] redirect without a Location header")
            try:
                current = urljoin(current, location)
            except ValueError as exc:
                raise LocalToolError(f"[invalid-url] malformed redirect Location: {location!r}") from exc
            continue
        if status >= 400:
            raise LocalToolError(f"[http-{status}] upstream returned status {status} for {current!r}")
        return current, headers, body, truncated, is_pdf
    raise LocalToolError(f"[redirect-limit] exceeded {FETCH_MAX_REDIRECTS} redirects for {url!r}")


def _seed_from_sitemap(
    client: httpx.Client,
    sitemap_url: str,
    scope_host: str,
    max_pages: int,
    deadline: float,
) -> tuple[list[str], bool]:
    """Collect up to max_pages same-host page URLs from a sitemap.

    Sitemap fetches are discovery overhead — they do NOT consume the page
    budget; the deadline bounds a pathological index (spec §2). Host rules:
    child sitemaps must share sitemap_url's host; page URLs must share the
    crawl scope host.

    Returns (urls, children_capped): children_capped is True when the
    SITEMAP_MAX_CHILDREN break below fires, so the caller can report an
    honest stop reason instead of claiming the sitemap was exhausted when
    children were actually left unfetched.
    """
    final_url, _headers, body, truncated, _is_pdf = _crawl_fetch_page(
        client, sitemap_url, deadline, max_bytes=SITEMAP_MAX_BYTES, html_only=False
    )
    if truncated:
        raise LocalToolError(f"[crawl-failed] sitemap exceeds {SITEMAP_MAX_BYTES} bytes: {sitemap_url!r}")
    page_urls, children = _parse_sitemap(body)
    sitemap_host = _crawl_host(final_url)

    urls: list[str] = []
    seen: set[str] = set()

    def take(candidates: list[str]) -> None:
        for candidate in candidates:
            if len(urls) >= max_pages:
                return
            if _crawl_host(candidate) != scope_host:
                continue
            norm = _normalize_crawl_url(candidate)
            if norm in seen:
                continue
            seen.add(norm)
            urls.append(candidate)

    take(page_urls)
    children_fetched = 0
    children_capped = False
    for child in children:
        if len(urls) >= max_pages or time.monotonic() >= deadline:
            break
        if _crawl_host(child) != sitemap_host:
            continue
        if children_fetched >= SITEMAP_MAX_CHILDREN:
            # Amplification guard: a pathological same-host index (~119
            # children measured in the review, ~600 MB at 5 MiB each) is
            # bounded here instead of relying solely on the deadline.
            children_capped = True
            break
        children_fetched += 1
        try:
            _f, _h, child_body, child_truncated, _is_pdf = _crawl_fetch_page(
                client, child, deadline, max_bytes=SITEMAP_MAX_BYTES, html_only=False
            )
        except (LocalToolError, _CrawlDeadline):
            continue
        if child_truncated:
            continue
        try:
            child_pages, _nested = _parse_sitemap(child_body)  # one level: nested indexes ignored
        except LocalToolError:
            continue
        take(child_pages)
    return urls, children_capped


def web_crawl(
    url: str,
    *,
    max_pages: int = CRAWL_DEFAULT_MAX_PAGES,
    max_depth: int = CRAWL_DEFAULT_MAX_DEPTH,
    sitemap_url: "str | None" = None,
) -> str:
    """Same-host breadth-first crawl returning a bounded page list.

    Each listed page carries URL, title, and a short excerpt; the model is
    expected to follow up with web_fetch on pages that matter (spec §2).
    Every URL is egress-guarded; budgets bound fetch ATTEMPTS; a wall-clock
    deadline bounds the whole crawl. Ephemeral: no database writes.

    When ``sitemap_url`` is given, sitemap mode replaces link-discovery BFS:
    the page list comes from the sitemap (urlset, or a one-level
    sitemapindex); ``max_depth`` is ignored and links on seeded pages are
    not expanded.

    Raises:
        LocalToolError: [invalid-args] for a bad url/host or a blank
            sitemap_url; [crawl-failed] when the START url cannot be
            fetched in BFS mode, or when the sitemap itself cannot be
            fetched/parsed (per-page failures inside the crawl are
            results, counted in the footer).
    """
    if not isinstance(url, str) or not url.strip():
        raise LocalToolError("[invalid-args] url must be a non-empty string")
    url = url.strip()
    max_pages = _coerce_budget(max_pages, CRAWL_DEFAULT_MAX_PAGES, CRAWL_MAX_PAGES_CEILING)
    max_depth = _coerce_budget(max_depth, CRAWL_DEFAULT_MAX_DEPTH, CRAWL_MAX_DEPTH_CEILING)
    scope_host = _crawl_host(url)
    if not scope_host:
        raise LocalToolError(f"[invalid-args] url has no host: {url!r}")

    deadline = time.monotonic() + CRAWL_DEADLINE_SECONDS
    queue: "deque[tuple[str, int]]" = deque([(url, 0)])
    visited = {_normalize_crawl_url(url)}
    pages: list[dict] = []
    failed = blocked = 0
    attempts = 0
    stop_reason = "no more links within depth"

    client = httpx.Client(
        follow_redirects=False,
        timeout=CRAWL_PAGE_TIMEOUT_SECONDS,
        headers={"User-Agent": _CRAWL_USER_AGENT},
        transport=_transport,
        trust_env=False,
    )
    try:
        expand_links = sitemap_url is None
        if sitemap_url is not None:
            if not isinstance(sitemap_url, str) or not sitemap_url.strip():
                raise LocalToolError("[invalid-args] sitemap_url must be a non-empty string")
            try:
                seeded, children_capped = _seed_from_sitemap(
                    client, sitemap_url.strip(), scope_host, max_pages, deadline
                )
            except _CrawlDeadline:
                seeded = []
                children_capped = False
            except LocalToolError as exc:
                if "[crawl-failed]" in str(exc):
                    raise
                raise LocalToolError(f"[crawl-failed] sitemap could not be fetched: {exc}") from exc
            queue = deque((u, 0) for u in seeded)
            visited = {_normalize_crawl_url(u) for u in seeded}
            # Three non-exceptional paths can leave `seeded` short/empty for a
            # reason other than "the sitemap was fully consulted": the clock
            # ran out (root fetch's _CrawlDeadline, caught above, and the
            # child-sitemap loop's plain `break` on time.monotonic() >=
            # deadline both leave the clock past the deadline, read back
            # here), or the SITEMAP_MAX_CHILDREN break fired and left child
            # sitemaps unfetched (children_capped). Check the clock first:
            # a deadline that also capped children is reported as the
            # deadline, since that's the harder budget the caller hit.
            if time.monotonic() >= deadline:
                stop_reason = "deadline reached"
            elif children_capped:
                stop_reason = "sitemap child budget reached"
            else:
                stop_reason = "sitemap exhausted"

        while queue:
            if attempts >= max_pages:
                stop_reason = "page budget reached"
                break
            if time.monotonic() >= deadline:
                stop_reason = "deadline reached"
                break
            current, depth = queue.popleft()
            is_start = attempts == 0
            attempts += 1
            try:
                final_url, headers, body, truncated, is_pdf = _crawl_fetch_page(client, current, deadline)
            except _CrawlDeadline:
                stop_reason = "deadline reached"
                break
            except LocalToolError as exc:
                if is_start and sitemap_url is None:
                    raise LocalToolError(f"[crawl-failed] start URL could not be fetched: {exc}") from exc
                # Prefix check, not substring: _validate_hop puts the reason
                # at position 0 of ITS message, but a URL echoed into an
                # unrelated error (e.g. an http-404 message quoting the
                # failing URL) can contain the literal text "[ssrf]"
                # anywhere in the string without being an SSRF refusal.
                if str(exc).startswith("[ssrf]"):
                    blocked += 1
                else:
                    failed += 1
                continue
            final_norm = _normalize_crawl_url(final_url)
            if final_norm in visited and final_norm != _normalize_crawl_url(current):
                continue  # a previously-fetched URL already redirected here
            visited.add(final_norm)

            ctype = (headers.get("content-type") or "").split(";", 1)[0].strip().lower()
            # is_pdf (the %PDF- sniff) takes priority over the declared
            # content-type: a PDF mislabeled as text/html (or unlabeled)
            # must never fall through to HTML extraction below, or its raw
            # bytes become the excerpt and its "extracted text" warm-writes
            # the shared web_fetch cache with binary garbage. Matches the
            # spec's own detection rule ("the sniff wins over the declared
            # type" §1): when is_pdf is true the marker is always
            # "[application/pdf]", regardless of what the server claimed;
            # only a genuinely non-PDF, non-HTML response is labeled with
            # its own declared type. `ctype` is guaranteed non-empty on the
            # else side (the `or` branch below required it truthy to enter
            # this block at all), so no `ctype or ...` fallback is needed.
            if is_pdf or (ctype and ctype not in _HTML_TYPES):
                marker = "[application/pdf]" if is_pdf else f"[{ctype}]"
                pages.append({"url": final_url, "title": "", "excerpt": "", "marker": marker})
                continue

            html = _decode_body(body, headers.get("content-type", ""))
            parser = _CrawlLinkParser()
            try:
                parser.feed(html)
                parser.close()
            except Exception:  # noqa: BLE001 — keep whatever was collected
                pass
            try:
                full_text = _extract_text(body, headers.get("content-type", ""))
            except LocalToolError:
                full_text = ""
            if full_text:
                # Parity with web_fetch: a body already sliced to FETCH_MAX_BYTES
                # must carry the same marker web_fetch would have appended, or a
                # default web_fetch() cache hit silently hands back a cut page.
                cache_text = full_text
                if truncated:
                    cache_text += f"\n\n[... truncated: response exceeded max_bytes={FETCH_MAX_BYTES} ...]"
                _cache_put((final_url, FETCH_MAX_BYTES), cache_text)
            pages.append({
                "url": final_url,
                "title": parser.title.strip(),
                "excerpt": full_text[:CRAWL_EXCERPT_MAX_CHARS].strip(),
                "marker": None,
            })

            # Expansion: same-host pages only, within the depth budget. A page
            # that redirected off-host is listed but its links are not followed.
            # Sitemap mode is discovery-complete: links are never expanded.
            if not expand_links:
                continue
            if depth >= max_depth or _crawl_host(final_url) != scope_host:
                continue
            base = final_url
            if parser.base_href:
                try:
                    base = urljoin(final_url, parser.base_href)
                except ValueError:
                    base = final_url  # malformed <base href>: fall back to the page URL
            enqueued_from_page = 0
            for href in parser.links:
                if enqueued_from_page >= CRAWL_MAX_LINKS_PER_PAGE:
                    # Frontier bound: a hostile/spammy page must not grow
                    # visited/queue without limit (52,975 entries measured
                    # from one page in the review).
                    break
                try:
                    absolute = urljoin(base, href)
                    scheme = urlsplit(absolute).scheme.lower()
                except ValueError:
                    continue  # malformed href (e.g. unbalanced IPv6 brackets): skip it
                if scheme not in _ALLOWED_SCHEMES:
                    continue
                if _crawl_host(absolute) != scope_host:
                    continue
                norm = _normalize_crawl_url(absolute)
                if norm in visited:
                    continue
                visited.add(norm)
                queue.append((absolute, depth + 1))
                enqueued_from_page += 1
    finally:
        client.close()

    return _format_crawl_result(pages, failed, blocked, stop_reason)
