"""Sync HTTP shim + light egress guard for the ported image adapters.

Provides the exact surface the server's http_client exposed to Image_Generation,
backed by httpx.Client. Full SSRF hardening is deferred to task-485; this guard
only rejects non-http(s) schemes and enforces a redirect cap, staying permissive
for user-configured (incl. local) backend base URLs.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import httpx
from loguru import logger
from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

DEFAULT_MAX_REDIRECTS = int(os.getenv("HTTP_MAX_REDIRECTS", "5"))
_DEFAULT_TIMEOUT = 120.0


def _validate_egress_or_raise(url: str) -> None:
    scheme = (urlparse(url).scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ImageGenerationError(f"Refusing non-http(s) URL: {url!r}")
    # task-485: private/link-local/metadata range blocking for API-returned URLs goes here.


def _resolve_redirect_url(base: str, location: str) -> str:
    return urljoin(base, location)


@dataclass(frozen=True)
class URLPolicyResult:
    allowed: bool
    reason: str | None = None


def evaluate_url_policy(url: str, *, allowed_hosts: set[str] | None = None) -> URLPolicyResult:
    _validate_egress_or_raise(url)
    if not allowed_hosts:
        return URLPolicyResult(True, None)
    host = (urlparse(url).hostname or "").lower()
    if any(host == h or host.endswith("." + h) for h in allowed_hosts):
        return URLPolicyResult(True, None)
    return URLPolicyResult(False, f"host {host!r} not in allowlist")


def create_client(timeout: float | None = None) -> httpx.Client:
    return httpx.Client(
        timeout=timeout or _DEFAULT_TIMEOUT,
        follow_redirects=True,
        max_redirects=DEFAULT_MAX_REDIRECTS,
    )


def fetch_json(method, url, *, headers=None, json=None, params=None, cookies=None, timeout=None):
    _validate_egress_or_raise(url)
    with create_client(timeout=timeout) as client:
        resp = client.request(method, url, headers=headers, json=json, params=params, cookies=cookies)
        resp.raise_for_status()
        return resp.json()
