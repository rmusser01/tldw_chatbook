"""Sync HTTP shim for the ported image adapters, backed by the app-wide SSRF policy.

Provides the exact surface the server's http_client exposed to Image_Generation,
backed by httpx.Client. Egress is enforced by ``Utils/egress.py`` (the app-wide
SSRF policy, task-498): a URL is allowed iff every resolved IP is public and
not a cloud metadata endpoint, OR its hostname is in the caller-supplied
``trusted_origins`` set. Callers pass ``trusted_origins`` for hosts derived
from a user-configured backend ``base_url`` (e.g. a local SwarmUI/sd.cpp
server); URLs extracted from a remote API's response body (image links from
OpenRouter/Novita/ModelStudio) must NOT be trusted and are fully enforced.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse
import httpx
from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
from tldw_chatbook.Utils import egress


def _int_env(name: str, default: int) -> int:
    """Parse an int environment variable, falling back to ``default``.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is unset or not an int.

    Returns:
        The parsed int, or ``default`` on a missing/invalid value (never raises).
    """
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


DEFAULT_MAX_REDIRECTS = _int_env("HTTP_MAX_REDIRECTS", 5)
_DEFAULT_TIMEOUT = 120.0


def _validate_egress_or_raise(url: str, *, trusted_origins: frozenset = frozenset()) -> None:
    """Reject a URL the adapters must not fetch, per the app-wide egress policy.

    Delegates to ``Utils/egress.py``: non-http(s) schemes, private/link-local
    ranges, and cloud metadata endpoints are blocked, except that a hostname
    in ``trusted_origins`` may resolve to a private/link-local IP (metadata
    endpoints are always blocked regardless of trust).

    Args:
        url: The absolute URL about to be requested.
        trusted_origins: Hostnames the caller has already established as
            user-intended (e.g. a configured backend ``base_url``'s host).

    Raises:
        ImageGenerationError: If the URL is blocked by the egress policy.
    """
    try:
        egress.check_url_or_raise(url, trusted_origins=trusted_origins)
    except egress.EgressBlockedError as exc:
        raise ImageGenerationError(f"Refusing blocked URL ({exc.reason}): {url!r}") from exc


def _resolve_redirect_url(base: str, location: str) -> str:
    """Resolve a redirect ``Location`` against the request URL."""
    return urljoin(base, location)


@dataclass(frozen=True)
class URLPolicyResult:
    allowed: bool
    reason: str | None = None


def evaluate_url_policy(
    url: str,
    *,
    allowed_hosts: set[str] | None = None,
    trusted_origins: frozenset = frozenset(),
) -> URLPolicyResult:
    """Decide whether ``url`` may be fetched: egress policy, then an optional allowlist.

    Args:
        url: The absolute URL to evaluate.
        allowed_hosts: If given, ``url``'s host must equal or be a subdomain of
            one of these; if empty/None, any host that clears the egress
            policy is allowed.
        trusted_origins: Hostnames the caller has already established as
            user-intended (see ``_validate_egress_or_raise``).

    Returns:
        A ``URLPolicyResult`` with ``allowed`` and an optional ``reason``.
    """
    decision = egress.evaluate_url_policy(url, trusted_origins=trusted_origins)
    if not decision.allowed:
        return URLPolicyResult(False, decision.reason)
    if not allowed_hosts:
        return URLPolicyResult(True, None)
    host = decision.host or (urlparse(url).hostname or "").lower()
    if any(host == h or host.endswith("." + h) for h in allowed_hosts):
        return URLPolicyResult(True, None)
    return URLPolicyResult(False, f"host {host!r} not in allowlist")


def create_client(timeout: float | None = None, *, follow_redirects: bool = False) -> httpx.Client:
    """Build an ``httpx.Client`` for the image adapters.

    Redirects are NOT auto-followed by default: ``fetch_json`` and
    ``fetch_image_bytes`` run their own per-hop-validated redirect loops, because
    blindly following a redirect would bypass the egress guard.

    Args:
        timeout: Per-request timeout in seconds (defaults to 120s).
        follow_redirects: Whether httpx auto-follows redirects. Default False.

    Returns:
        A configured ``httpx.Client``.
    """
    return httpx.Client(
        timeout=_DEFAULT_TIMEOUT if timeout is None else timeout,
        follow_redirects=follow_redirects,
        max_redirects=DEFAULT_MAX_REDIRECTS,
    )


def fetch_json(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    json: Any = None,
    params: dict | None = None,
    cookies: dict | None = None,
    timeout: float | None = None,
    trusted_origins: frozenset = frozenset(),
) -> Any:
    """Issue a JSON HTTP request, validating the egress URL on every hop.

    Redirects are followed manually so ``_validate_egress_or_raise`` re-runs on
    each ``Location`` — a blindly-followed redirect could reach a disallowed
    host/scheme and defeat the egress guard. ``headers``/``cookies`` are also
    re-evaluated per hop: ``Authorization``/``Cookie``/``Proxy-Authorization``
    are stripped on any hop whose origin (scheme + host + effective port)
    differs from the original request's origin — a same-host HTTPS->HTTP
    downgrade or a same-host different-port redirect is treated as
    cross-origin too — so a redirect cannot be used to exfiltrate
    credentials to a different (still-public, so not SSRF-blocked) origin.
    ``params`` is only applied to the first hop — the redirected URL already
    carries whatever query the server encoded into ``Location``.

    Args:
        method: HTTP method.
        url: Absolute request URL (validated before each hop).
        headers: Optional request headers; credential headers are dropped on
            cross-origin hops (see above).
        json: Optional JSON body.
        params: Optional query params (first hop only).
        cookies: Optional cookies; dropped on cross-origin hops.
        timeout: Per-request timeout in seconds.
        trusted_origins: Hostnames trusted to resolve to a private/internal
            IP (e.g. a configured backend ``base_url``'s host). Leave empty
            for URLs sourced from a remote API's response body.

    Returns:
        The parsed JSON response body.

    Raises:
        ImageGenerationError: On a blocked URL (see ``_validate_egress_or_raise``),
            a redirect without a ``Location``, or exceeding the redirect cap.
    """
    current = url
    with create_client(timeout=timeout) as client:
        for hop in range(DEFAULT_MAX_REDIRECTS + 1):
            _validate_egress_or_raise(current, trusted_origins=trusted_origins)
            # A redirect to a DIFFERENT origin (scheme+host+effective-port)
            # must not carry credentials -- public hosts (unlike private
            # ones) are not blocked by the SSRF policy, so this is the only
            # guard against a compromised/malicious backend exfiltrating our
            # Authorization/Cookie via a same-host scheme downgrade, a
            # different-port redirect, or a redirect to an outright
            # different host. Mirrors Utils.egress.same_origin, the same
            # rule the app's other egress consumers already apply.
            same_origin = egress.same_origin(url, current)
            resp = client.request(
                method,
                current,
                headers=egress._hop_headers(headers, same_origin),
                json=json,
                params=params if hop == 0 else None,
                cookies=cookies if same_origin else None,
            )
            if resp.is_redirect:
                location = resp.headers.get("location") or resp.headers.get("Location")
                if not location:
                    raise ImageGenerationError("request failed: redirect without location")
                current = _resolve_redirect_url(str(resp.url), str(location))
                continue
            resp.raise_for_status()
            return resp.json()
    raise ImageGenerationError("request failed: too many redirects")


def _positive_byte_limit(max_bytes: int | None) -> int | None:
    """Normalize a byte cap: ``None``/non-int/non-positive values mean "no cap".

    Relocated from ``Image_Generation/adapters/image_format_utils.py`` (task-1
    fix round 1) so both ``fetch_bytes_via_post`` (this module) and
    ``fetch_image_bytes`` (image_format_utils, which imports it back from
    here) share one implementation instead of duplicating the cap logic.
    """
    if max_bytes is None:
        return None
    try:
        limit = int(max_bytes)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def _reject_declared_oversize(headers: Any, max_bytes: int | None) -> None:
    """Reject a response whose declared ``Content-Length`` already exceeds the cap.

    Runs before any body bytes are read, so a server that honestly declares
    an oversized body is rejected without buffering it at all. A missing or
    unparsable ``Content-Length`` is not itself an error here -- the running
    total check in ``_read_stream_with_limit`` is the backstop for a body
    that is unbounded or lies about its declared size.
    """
    limit = _positive_byte_limit(max_bytes)
    if limit is None:
        return
    content_length = headers.get("content-length") or headers.get("Content-Length")
    if content_length is None:
        return
    try:
        declared_size = int(str(content_length).strip())
    except ValueError:
        return
    if declared_size > limit:
        raise ImageGenerationError("image content too large")


def _read_stream_with_limit(chunks: Any, max_bytes: int | None) -> bytes:
    """Read a byte-chunk iterator, aborting mid-stream once the running total exceeds the cap.

    This is the backstop against a body that is unbounded or that lies about
    its declared ``Content-Length`` (or omits it): each chunk is only
    buffered after confirming the running total is still within the cap, so
    a body that blows the cap is never fully read into memory.
    """
    limit = _positive_byte_limit(max_bytes)
    total = 0
    parts: list[bytes] = []
    for chunk in chunks:
        if not chunk:
            continue
        total += len(chunk)
        if limit is not None and total > limit:
            raise ImageGenerationError("image content too large")
        parts.append(chunk)
    return b"".join(parts)


def fetch_bytes_via_post(
    url: str,
    *,
    headers: dict | None = None,
    json: Any = None,
    timeout: float | None = None,
    trusted_origins: frozenset = frozenset(),
    max_bytes: int = 32 * 1024 * 1024,
) -> tuple[bytes, str]:
    """Issue a POST request and return the raw response body, not parsed JSON.

    For backends that return image bytes directly from a POST (e.g. Fireworks'
    image-generation endpoint) rather than a JSON envelope carrying a URL or
    base64 payload. Mirrors ``fetch_json``'s manual, per-hop-validated
    redirect loop: egress is re-validated on every hop,
    ``Authorization``/``Cookie``/``Proxy-Authorization`` are stripped on any
    hop whose origin differs from the original request's (see ``fetch_json``'s
    docstring for the full same-origin rationale), and redirects are never
    auto-followed by the transport.

    Unlike ``fetch_json``, the final hop's response is streamed
    (``client.stream()``) rather than eagerly buffered: ``httpx`` has no
    built-in body-size limit, so reading the whole body via ``resp.content``
    before checking it against ``max_bytes`` would let a hostile or broken
    endpoint force an unbounded in-memory buffer regardless of the cap. Two
    guards apply instead, mirroring ``image_format_utils.fetch_image_bytes``
    (whose ``_reject_declared_oversize``/``_read_stream_with_limit`` helpers
    live in this module and are imported back by that module, so both GET and
    POST byte-fetch paths share one implementation):

    1. ``_reject_declared_oversize`` rejects a response whose declared
       ``Content-Length`` already exceeds ``max_bytes`` -- before a single
       chunk is read.
    2. ``_read_stream_with_limit`` reads the body chunk-by-chunk, aborting as
       soon as the running total exceeds ``max_bytes`` -- the backstop for a
       body with no (or an inaccurate) ``Content-Length``.

    Args:
        url: Absolute request URL (validated before each hop).
        headers: Optional request headers; credential headers are dropped on
            cross-origin hops (see ``fetch_json``).
        json: Optional JSON body, re-sent on every hop.
        timeout: Per-request timeout in seconds. ``None`` uses
            ``_DEFAULT_TIMEOUT``; an explicit ``0`` is honored as given
            (fail-fast), not silently replaced by the default.
        trusted_origins: Hostnames trusted to resolve to a private/internal
            IP (e.g. a configured backend ``base_url``'s host). Leave empty
            for URLs sourced from a remote API's response body.
        max_bytes: Maximum allowed response body size in bytes. A response
            body larger than this raises ``ImageGenerationError`` naming the
            cap -- the body is never silently truncated and returned partial,
            nor fully buffered in memory before being rejected.

    Returns:
        A ``(body_bytes, content_type_header_value)`` tuple.

    Raises:
        ImageGenerationError: On a blocked URL (see
            ``_validate_egress_or_raise``), a redirect without a
            ``Location``, exceeding the redirect cap, or the response body
            exceeding ``max_bytes`` (declared via ``Content-Length`` or
            discovered mid-stream).
    """
    current = url
    with create_client(timeout=timeout) as client:
        for hop in range(DEFAULT_MAX_REDIRECTS + 1):
            _validate_egress_or_raise(current, trusted_origins=trusted_origins)
            # Same credential-exfiltration rationale as fetch_json: a
            # redirect to a still-public (so not SSRF-blocked) different
            # origin must not carry Authorization/Cookie/Proxy-Authorization
            # along with it.
            same_origin = egress.same_origin(url, current)
            with client.stream(
                "POST",
                current,
                headers=egress._hop_headers(headers, same_origin),
                json=json,
            ) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location") or resp.headers.get("Location")
                    if not location:
                        raise ImageGenerationError("request failed: redirect without location")
                    current = _resolve_redirect_url(str(resp.url), str(location))
                    continue
                resp.raise_for_status()
                try:
                    _reject_declared_oversize(resp.headers, max_bytes)
                    content = _read_stream_with_limit(resp.iter_bytes(), max_bytes)
                except ImageGenerationError as exc:
                    raise ImageGenerationError(
                        f"response body exceeds max_bytes cap ({max_bytes} bytes)"
                    ) from exc
                return content, resp.headers.get("content-type", "")
    raise ImageGenerationError("request failed: too many redirects")
