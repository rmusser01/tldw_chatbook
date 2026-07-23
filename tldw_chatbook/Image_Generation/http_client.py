"""Sync HTTP shim + light egress guard for the ported image adapters.

Provides the exact surface the server's http_client exposed to Image_Generation,
backed by httpx.Client. Full SSRF hardening is deferred to task-498; this guard
rejects non-http(s) schemes, enforces a redirect cap, and re-validates every
redirect hop, while staying permissive for user-configured (incl. local) backend
base URLs.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse
import httpx
from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError


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


def _validate_egress_or_raise(url: str) -> None:
    """Reject a URL the adapters must not fetch (light Phase-1 guard).

    Args:
        url: The absolute URL about to be requested.

    Raises:
        ImageGenerationError: If the scheme is not http/https.
    """
    scheme = (urlparse(url).scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ImageGenerationError(f"Refusing non-http(s) URL: {url!r}")
    # task-498: private/link-local/metadata range blocking for API-returned URLs goes here.


def _resolve_redirect_url(base: str, location: str) -> str:
    """Resolve a redirect ``Location`` against the request URL."""
    return urljoin(base, location)


@dataclass(frozen=True)
class URLPolicyResult:
    allowed: bool
    reason: str | None = None


def evaluate_url_policy(url: str, *, allowed_hosts: set[str] | None = None) -> URLPolicyResult:
    """Decide whether ``url`` may be fetched, optionally against a host allowlist.

    Args:
        url: The absolute URL to evaluate (scheme is always validated).
        allowed_hosts: If given, ``url``'s host must equal or be a subdomain of
            one of these; if empty/None, any http(s) host is allowed.

    Returns:
        A ``URLPolicyResult`` with ``allowed`` and an optional ``reason``.

    Raises:
        ImageGenerationError: If the scheme is not http/https.
    """
    _validate_egress_or_raise(url)
    if not allowed_hosts:
        return URLPolicyResult(True, None)
    host = (urlparse(url).hostname or "").lower()
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
        timeout=timeout or _DEFAULT_TIMEOUT,
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
) -> Any:
    """Issue a JSON HTTP request, validating the egress URL on every hop.

    Redirects are followed manually so ``_validate_egress_or_raise`` re-runs on
    each ``Location`` — a blindly-followed redirect could reach a disallowed
    host/scheme and defeat the egress guard.

    Args:
        method: HTTP method.
        url: Absolute request URL (validated before each hop).
        headers: Optional request headers.
        json: Optional JSON body.
        params: Optional query params.
        cookies: Optional cookies.
        timeout: Per-request timeout in seconds.

    Returns:
        The parsed JSON response body.

    Raises:
        ImageGenerationError: On a non-http(s) URL, a redirect without a
            ``Location``, or exceeding the redirect cap.
    """
    current = url
    with create_client(timeout=timeout) as client:
        for _ in range(DEFAULT_MAX_REDIRECTS + 1):
            _validate_egress_or_raise(current)
            resp = client.request(
                method, current, headers=headers, json=json, params=params, cookies=cookies
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
