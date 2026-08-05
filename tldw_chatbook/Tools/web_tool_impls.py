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
import ipaddress
import re
import socket
import time
from urllib.parse import urljoin, urlsplit

import httpx
from loguru import logger

from .local_tool_impls import LocalToolError

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
RATE_LIMIT_INTERVAL_SECONDS = 1.0          # per-domain min interval

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
_fetch_cache: dict[str, tuple[float, str]] = {}
_domain_last_fetch: dict[str, float] = {}

# Test seam: tests set this to an httpx.MockTransport.
_transport: "httpx.BaseTransport | None" = None


def _reset_state_for_tests() -> None:
    """Clear the module-level fetch cache and rate-limit state."""
    _fetch_cache.clear()
    _domain_last_fetch.clear()


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


def _fetch_once(client: httpx.Client, url: str, max_bytes: int) -> tuple[int, httpx.Headers, bytes, bool]:
    """One GET with a bounded streaming read; redirects are NOT followed."""
    with client.stream("GET", url) as response:
        status = response.status_code
        if status in _REDIRECT_STATUSES:
            return status, response.headers, b"", False
        chunks: list[bytes] = []
        downloaded = 0
        truncated = False
        for chunk in response.iter_bytes():
            remaining = max_bytes - downloaded
            if remaining <= 0:
                truncated = True
                break
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated = True
                break
            chunks.append(chunk)
            downloaded += len(chunk)
        return status, response.headers, b"".join(chunks), truncated


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


def web_fetch(url: str, *, max_bytes: int = FETCH_MAX_BYTES) -> str:
    """Fetch ``url`` and return extracted text (trafilatura), byte-capped.

    SSRF-guarded per hop (validate_outbound_url), redirect-capped,
    rate-limited per domain, cached in-memory for FETCH_CACHE_TTL_SECONDS.
    Result ends with a truncation marker when capped. All failures raise
    LocalToolError with structured reasons ("invalid-url", "ssrf",
    "redirect-limit", "timeout", "http-<status>", "too-large", "rate-limited").

    Raises:
        LocalToolError: on invalid/SSRF URLs, redirect overflow, timeouts,
            HTTP error statuses, rate limiting, or unextractable content.
    """
    if not isinstance(url, str) or not url.strip():
        raise LocalToolError("[invalid-url] url must be a non-empty string")
    url = url.strip()
    try:
        max_bytes = max(1, min(int(max_bytes), FETCH_HARD_MAX_BYTES))
    except (TypeError, ValueError) as exc:
        raise LocalToolError(f"[invalid-url] max_bytes must be an integer: {max_bytes!r}") from exc

    cached = _fetch_cache.get(url)
    if cached is not None:
        expires_at, text = cached
        if time.monotonic() < expires_at:
            _validate_hop(url)  # re-check policy on cache hits (cheap, no body)
            return text
        _fetch_cache.pop(url, None)

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
            status, headers, body, truncated = _fetch_once(client, current_url, max_bytes)
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

    text = _extract_text(body, headers.get("content-type", ""))
    if truncated:
        text += f"\n\n[... truncated: response exceeded max_bytes={max_bytes} ...]"
    _fetch_cache[url] = (time.monotonic() + FETCH_CACHE_TTL_SECONDS, text)
    return text


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

SEARCH_DEFAULT_ENGINE = "duckduckgo"
SEARCH_ENGINES = ("google", "bing", "duckduckgo", "brave", "kagi", "tavily", "searx")
SEARCH_DEFAULT_RESULT_COUNT = 5
SEARCH_MAX_RESULT_COUNT = 10
SEARCH_RESULT_MAX_CHARS = 4 * 1024      # per-result bound (re-plan spec §2.2)
SEARCH_TOTAL_MAX_CHARS = 24 * 1024      # total cap, under the provider's 32 KiB fit


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
    block is bounded to SEARCH_RESULT_MAX_CHARS and the whole output to
    SEARCH_TOTAL_MAX_CHARS, so the provider's 32 KiB result fitting never
    triggers on search output. Backend failures return an error string
    rather than raising (legacy tool contract); only invalid arguments
    raise LocalToolError.

    Raises:
        LocalToolError: if ``query`` is empty.
    """
    if not isinstance(query, str) or not query.strip():
        raise LocalToolError("[invalid-args] query must be a non-empty string")
    query = query.strip()
    engine = (search_engine or SEARCH_DEFAULT_ENGINE).strip().lower()
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

    if not isinstance(results, dict) or not isinstance(results.get("results"), list):
        return (
            f"No results found or unexpected response format from {engine!r} "
            f"(raw: {str(results)[:500]})"
        )
    items = [item if isinstance(item, dict) else {} for item in results["results"][:count]]
    if not items:
        return f"No results found for {query!r} via {engine!r}."

    blocks: list[str] = []
    total = 0
    for i, item in enumerate(items, 1):
        block = (
            f"{i}. {item.get('title') or 'No title'}\n"
            f"   URL: {item.get('url') or ''}\n"
            f"   {item.get('snippet') or 'No description available'}"
        )
        if len(block) > SEARCH_RESULT_MAX_CHARS:
            block = block[:SEARCH_RESULT_MAX_CHARS] + "… [truncated]"
        if total + len(block) > SEARCH_TOTAL_MAX_CHARS:
            blocks.append("… [further results omitted: total size cap reached]")
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)
