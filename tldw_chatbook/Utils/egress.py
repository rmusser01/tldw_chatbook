"""App-wide egress policy for outbound URL fetching (SSRF protection).

One rule: a URL is allowed iff every resolved IP is public and not a cloud
metadata endpoint, OR its hostname is in ``trusted_origins`` (a host the USER
explicitly typed/configured), OR its hostname is in the ``[web_security]
allowed_hosts`` config allowlist. Metadata endpoints are stricter: blocked
even for trusted origins; only the config allowlist overrides them.

Shared pipeline code must NEVER auto-trust its own input URL — trust is
seeded only at boundaries where user intent is known and threaded down
(see Docs/superpowers/specs/2026-07-23-web-fetch-hardening-design.md).

Non-goals (documented residual risk): DNS-rebinding IP pinning (we
resolve-and-check; the HTTP client re-resolves to connect), proxy-aware
policy (env-var proxies keep working; the target URL is what's validated),
and DNS caching (OS resolver caches suffice).
"""

from __future__ import annotations

import asyncio
import ipaddress
import json as _json
import socket
from dataclasses import dataclass, field
from typing import Iterable, List
from urllib.parse import urljoin, urlparse

from loguru import logger

from ..config import get_cli_setting
from ..Metrics.metrics_logger import log_counter

# Cloud metadata endpoints: blocked even for trusted origins.
_METADATA_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure IPv4
        ipaddress.ip_address("fd00:ec2::254"),  # AWS IPv6
        ipaddress.ip_address("100.100.100.200"),  # Alibaba Cloud
    }
)
METADATA_HOSTNAMES = frozenset({"metadata.google.internal", "metadata.azure.com"})

MAX_REDIRECT_HOPS = 10
MAX_FETCH_BYTES_PAGE = 10 * 1024 * 1024
MAX_FETCH_BYTES_SITEMAP = 50 * 1024 * 1024  # sitemap protocol allows 50MB uncompressed
MAX_FETCH_BYTES_GITHUB_FILE = 20 * 1024 * 1024
MAX_FETCH_BYTES_MEDIA = 500 * 1024 * 1024


class EgressBlockedError(Exception):
    """URL blocked by the egress policy (SSRF guard)."""

    def __init__(self, url: str, reason: str, detail: str = ""):
        self.url = url
        self.reason = reason
        self.detail = detail
        super().__init__(
            f"Egress blocked ({reason}) for {url}"
            + (f": {detail}" if detail else "")
            + " [remedy: add the host to [web_security] allowed_hosts in"
            " config.toml, or set [web_security] enabled = false]"
        )


@dataclass(frozen=True)
class EgressDecision:
    allowed: bool
    reason: str  # "ok" | "scheme" | "metadata" | "private" | "dns_failure" | "disabled"
    host: str
    resolved_ips: tuple = ()


def _resolve(host: str) -> List[str]:
    """Resolve every A/AAAA record for ``host`` (test seam — monkeypatched)."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


async def _resolve_async(host: str) -> List[str]:
    """Async resolution via the event loop (test seam — monkeypatched)."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


def _config_enabled() -> bool:
    value = get_cli_setting("web_security", "enabled", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def _config_allowed_hosts() -> frozenset:
    value = get_cli_setting("web_security", "allowed_hosts", [])
    if not isinstance(value, (list, tuple, set)):
        return frozenset()
    return frozenset(str(h).strip().lower() for h in value if str(h).strip())


def _classify_ip(ip_str: str) -> str:
    """Classify one resolved IP: "metadata" | "private" | "public"."""
    ip = ipaddress.ip_address(ip_str)
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if ip in _METADATA_IPS:
        return "metadata"
    return "public" if ip.is_global else "private"


def _blocked(url: str, reason: str, host: str, detail: str = "") -> EgressDecision:
    logger.warning(f"Egress blocked ({reason}): {url} {detail}".rstrip())
    log_counter("egress_blocked", labels={"reason": reason})
    return EgressDecision(allowed=False, reason=reason, host=host)


def _pre_resolution(url: str, trusted_origins: frozenset):
    """Checks that need no DNS. Returns EgressDecision, or the host to resolve."""
    if not _config_enabled():
        logger.debug(f"Egress check disabled by [web_security] for {url}")
        log_counter("egress_check_skipped")
        return EgressDecision(allowed=True, reason="disabled", host="")
    try:
        parsed = urlparse(url)
        host = parsed.hostname
    except ValueError:
        return _blocked(url, "scheme", "", "unparseable URL")
    if parsed.scheme not in ("http", "https") or not host:
        return _blocked(url, "scheme", host or "", "only http/https with a host")
    h = host.lower()
    allowed_hosts = _config_allowed_hosts()
    if h in allowed_hosts:
        return EgressDecision(allowed=True, reason="ok", host=h)
    if h in METADATA_HOSTNAMES:
        return _blocked(url, "metadata", h, "cloud metadata hostname")
    try:
        ipaddress.ip_address(h)
    except ValueError:
        return h  # hostname — caller resolves and calls _post_resolution
    # IP-literal host (incl. bracketed IPv6): classify directly, no DNS.
    return _post_resolution(url, h, (h,), trusted_origins)


def _post_resolution(
    url: str, host: str, ips: Iterable[str], trusted_origins: frozenset
) -> EgressDecision:
    ips = tuple(ips)
    classes = {_classify_ip(ip) for ip in ips}
    if "metadata" in classes:
        return _blocked(url, "metadata", host, f"resolves to metadata IP ({ips})")
    if host in trusted_origins:
        return EgressDecision(allowed=True, reason="ok", host=host, resolved_ips=ips)
    if "private" in classes:
        return _blocked(url, "private", host, f"resolves to private IP ({ips})")
    return EgressDecision(allowed=True, reason="ok", host=host, resolved_ips=ips)


def _normalize_trusted(trusted_origins) -> frozenset:
    return frozenset(str(h).strip().lower() for h in (trusted_origins or ()) if h)


def evaluate_url_policy(url: str, *, trusted_origins=frozenset()) -> EgressDecision:
    """Evaluate the egress policy for ``url`` (sync — blocking DNS).

    Never call from an asyncio event loop; use
    :func:`evaluate_url_policy_async` there.
    """
    trusted = _normalize_trusted(trusted_origins)
    pre = _pre_resolution(url, trusted)
    if isinstance(pre, EgressDecision):
        return pre
    try:
        ips = _resolve(pre)
    except (OSError, socket.gaierror) as exc:
        return _blocked(url, "dns_failure", pre, str(exc))
    return _post_resolution(url, pre, ips, trusted)


async def evaluate_url_policy_async(
    url: str, *, trusted_origins=frozenset()
) -> EgressDecision:
    """Async variant of :func:`evaluate_url_policy` (event-loop DNS)."""
    trusted = _normalize_trusted(trusted_origins)
    pre = _pre_resolution(url, trusted)
    if isinstance(pre, EgressDecision):
        return pre
    try:
        ips = await _resolve_async(pre)
    except (OSError, socket.gaierror) as exc:
        return _blocked(url, "dns_failure", pre, str(exc))
    return _post_resolution(url, pre, ips, trusted)


def check_url_or_raise(url: str, *, trusted_origins=frozenset()) -> None:
    """Evaluate the egress policy for ``url`` (sync — blocking DNS) and raise if blocked.

    Args:
        url: The URL to check.
        trusted_origins: Hostnames the caller has already established as
            user-intended; private/internal IPs are allowed for these (cloud
            metadata endpoints are still blocked regardless).

    Raises:
        EgressBlockedError: If ``url`` is not allowed by the egress policy.
    """
    decision = evaluate_url_policy(url, trusted_origins=trusted_origins)
    if not decision.allowed:
        raise EgressBlockedError(url, decision.reason)


async def check_url_or_raise_async(url: str, *, trusted_origins=frozenset()) -> None:
    """Async variant of :func:`check_url_or_raise` (event-loop DNS).

    Args:
        url: The URL to check.
        trusted_origins: Hostnames the caller has already established as
            user-intended; private/internal IPs are allowed for these (cloud
            metadata endpoints are still blocked regardless).

    Raises:
        EgressBlockedError: If ``url`` is not allowed by the egress policy.
    """
    decision = await evaluate_url_policy_async(url, trusted_origins=trusted_origins)
    if not decision.allowed:
        raise EgressBlockedError(url, decision.reason)


# ---------------------------------------------------------------------------
# Guarded transport helpers
# ---------------------------------------------------------------------------
_STRIP_HEADERS = ("authorization", "cookie", "proxy-authorization")


class EgressFetchError(Exception):
    """Guarded-fetch transport failure: size cap, hop cap, missing Location."""

    def __init__(self, message: str, url: str = ""):
        self.url = url
        super().__init__(f"{message}" + (f" [{url}]" if url else ""))


@dataclass
class GuardedResponse:
    """Stack-neutral capped-fetch result. Never raises on construction."""

    status_code: int
    headers: object
    content: bytes
    final_url: str
    _response: object = field(default=None, repr=False)

    @property
    def text(self) -> str:
        ctype = ""
        try:
            ctype = self.headers.get("content-type", "") or ""
        except Exception:
            pass
        charset = "utf-8"
        for part in ctype.split(";"):
            part = part.strip()
            if part.lower().startswith("charset="):
                charset = part.split("=", 1)[1].strip().strip('"') or "utf-8"
        try:
            return self.content.decode(charset, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")

    def json(self):
        return _json.loads(self.text)

    def raise_for_status(self):
        if self._response is not None:
            return self._response.raise_for_status()
        if self.status_code >= 400:
            raise EgressFetchError(f"HTTP {self.status_code}", url=self.final_url)


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def host_of(url: str) -> str:
    """Lowercase hostname of ``url``; "" if unparseable (never raises)."""
    return _host_of(url)


def origin_set(url: str) -> frozenset:
    """Single-host trusted-origin set for a user-supplied ``url`` (empty if unparseable)."""
    h = _host_of(url)
    return frozenset({h}) if h else frozenset()


def _hop_headers(headers, same_origin: bool) -> dict:
    hop = {str(k): v for k, v in dict(headers or {}).items()}
    if not same_origin:
        for key in list(hop):
            if key.lower() in _STRIP_HEADERS:
                hop.pop(key)
    return hop


def guarded_fetch_httpx(
    url: str,
    *,
    client,
    max_bytes: int,
    trusted_origins: frozenset = frozenset(),
    headers: dict | None = None,
    params: dict | None = None,
) -> GuardedResponse:
    """Capped GET via httpx.Client with per-hop egress re-validation."""
    first_host = _host_of(url)
    current = url
    for hop in range(MAX_REDIRECT_HOPS + 1):
        check_url_or_raise(current, trusted_origins=trusted_origins)
        same_origin = _host_of(current) == first_host
        request = client.build_request(
            "GET",
            current,
            headers=_hop_headers(headers, same_origin),
            params=params if hop == 0 else None,
        )
        if not same_origin:
            # Strip credentials the client attaches at the transport-object
            # level (e.g. httpx.Client(headers={"Authorization": ...})) —
            # these are merged onto the built request by httpx and are
            # invisible to _hop_headers, which only sees the per-call
            # `headers` argument.
            for _h in _STRIP_HEADERS:
                request.headers.pop(_h, None)
        response = client.send(request, stream=True, follow_redirects=False)
        try:
            if response.is_redirect and response.status_code != 304:
                location = response.headers.get("location")
                if not location:
                    raise EgressFetchError("redirect without Location", url=current)
                current = urljoin(current, location)
                continue
            collected = bytearray()
            for chunk in response.iter_bytes():
                collected += chunk
                if len(collected) > max_bytes:
                    raise EgressFetchError(
                        f"response exceeds {max_bytes} bytes", url=current
                    )
            return GuardedResponse(
                status_code=response.status_code,
                headers=response.headers,
                content=bytes(collected),
                final_url=str(response.url),
                _response=response,
            )
        finally:
            response.close()
    raise EgressFetchError("too many redirects", url=url)


async def guarded_fetch_httpx_async(
    url: str,
    *,
    client,
    max_bytes: int,
    trusted_origins: frozenset = frozenset(),
    headers: dict | None = None,
    params: dict | None = None,
    auth=None,
) -> GuardedResponse:
    """Async capped GET via httpx.AsyncClient with per-hop re-validation.

    ``auth`` is applied on same-origin hops only (credential-stripping rule).
    Client-default sensitive headers (e.g. an ``httpx.AsyncClient(headers=...)``
    default ``Authorization``) are also stripped on cross-origin hops. A
    client-level ``auth=`` CALLABLE (set on the ``httpx.Client``/``AsyncClient``
    itself, as opposed to the ``auth`` parameter of this function) is NOT
    suppressed by this guard — no live caller uses that flow today; this is a
    documented residual risk.
    """
    first_host = _host_of(url)
    current = url
    for hop in range(MAX_REDIRECT_HOPS + 1):
        await check_url_or_raise_async(current, trusted_origins=trusted_origins)
        same_origin = _host_of(current) == first_host
        request = client.build_request(
            "GET",
            current,
            headers=_hop_headers(headers, same_origin),
            params=params if hop == 0 else None,
        )
        if not same_origin:
            # Strip credentials the client attaches at the transport-object
            # level — see guarded_fetch_httpx for the rationale.
            for _h in _STRIP_HEADERS:
                request.headers.pop(_h, None)
        send_kwargs = {"stream": True, "follow_redirects": False}
        if auth is not None and same_origin:
            send_kwargs["auth"] = auth
        response = await client.send(request, **send_kwargs)
        try:
            if response.is_redirect and response.status_code != 304:
                location = response.headers.get("location")
                if not location:
                    raise EgressFetchError("redirect without Location", url=current)
                current = urljoin(current, location)
                continue
            collected = bytearray()
            async for chunk in response.aiter_bytes():
                collected += chunk
                if len(collected) > max_bytes:
                    raise EgressFetchError(
                        f"response exceeds {max_bytes} bytes", url=current
                    )
            return GuardedResponse(
                status_code=response.status_code,
                headers=response.headers,
                content=bytes(collected),
                final_url=str(response.url),
                _response=response,
            )
        finally:
            await response.aclose()
    raise EgressFetchError("too many redirects", url=url)


def guarded_fetch_requests(
    url: str,
    *,
    session=None,
    max_bytes: int,
    trusted_origins: frozenset = frozenset(),
    timeout: float = 30.0,
    headers: dict | None = None,
    sink=None,
) -> "requests.Response":
    """Capped GET via requests with per-hop egress re-validation.

    Returns the final ``requests.Response`` with ``._content`` preloaded
    (unless ``sink`` is given, in which case bytes stream to ``sink`` and
    ``.content`` is empty). ``session.auth`` is suppressed on cross-origin
    hops (credential-stripping rule).
    """
    import requests

    sess = session or requests.Session()
    owns_session = session is None
    try:
        first_host = _host_of(url)
        current = url
        for _hop in range(MAX_REDIRECT_HOPS + 1):
            check_url_or_raise(current, trusted_origins=trusted_origins)
            same_origin = _host_of(current) == first_host
            prepared = sess.prepare_request(
                requests.Request(
                    "GET", current, headers=_hop_headers(headers, same_origin)
                )
            )
            if not same_origin:
                # prepare_request applies session.auth/cookies into headers;
                # a cross-origin hop must not carry them.
                for key in ("Authorization", "Cookie", "Proxy-Authorization"):
                    prepared.headers.pop(key, None)
            response = sess.send(
                prepared, stream=True, timeout=timeout, allow_redirects=False
            )
            if response.is_redirect:
                location = response.headers.get("location")
                response.close()
                if not location:
                    raise EgressFetchError("redirect without Location", url=current)
                current = urljoin(current, location)
                continue
            collected = bytearray() if sink is None else None
            received = 0
            try:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > max_bytes:
                        raise EgressFetchError(
                            f"response exceeds {max_bytes} bytes", url=current
                        )
                    if sink is None:
                        collected += chunk
                    else:
                        sink.write(chunk)
            finally:
                response.close()
            response._content = bytes(collected) if collected is not None else b""
            response._content_consumed = True
            return response
        raise EgressFetchError("too many redirects", url=url)
    finally:
        if owns_session:
            sess.close()


async def guarded_fetch_aiohttp(
    url: str,
    *,
    session,
    max_bytes: int,
    trusted_origins: frozenset = frozenset(),
    headers: dict | None = None,
    timeout=None,
) -> GuardedResponse:
    """Capped GET via aiohttp.ClientSession with per-hop re-validation."""
    from multidict import CIMultiDict

    first_host = _host_of(url)
    current = url
    for _hop in range(MAX_REDIRECT_HOPS + 1):
        await check_url_or_raise_async(current, trusted_origins=trusted_origins)
        same_origin = _host_of(current) == first_host
        kwargs = {
            "allow_redirects": False,
            "headers": _hop_headers(headers, same_origin),
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        async with session.get(current, **kwargs) as response:
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise EgressFetchError("redirect without Location", url=current)
                current = urljoin(current, location)
                continue
            collected = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                collected += chunk
                if len(collected) > max_bytes:
                    raise EgressFetchError(
                        f"response exceeds {max_bytes} bytes", url=current
                    )
            return GuardedResponse(
                status_code=response.status,
                headers=CIMultiDict(response.headers),
                content=bytes(collected),
                final_url=str(response.url),
                _response=response,
            )
    raise EgressFetchError("too many redirects", url=url)


def collect_navigation_chain(response) -> list:
    """Playwright ``Response`` -> every URL in its redirect chain, oldest first.

    Playwright route handlers intercept only the INITIAL request of a
    navigation; server redirect hops are followed by the browser without
    re-invoking the route. Post-navigation validation of this chain is
    therefore the enforcement point (the pre-``goto`` check blocks bad
    initial targets outright).
    """
    urls = []
    request = getattr(response, "request", None) if response is not None else None
    # Bounded walk: a real Playwright chain terminates at ``redirected_from is
    # None``, but cap iterations so a cyclic/pathological ``redirected_from``
    # graph (or a mock object whose attributes never resolve to None) can never
    # hang the scraper. A navigation cannot legitimately exceed the browser's
    # own redirect ceiling; this cap sits comfortably above it.
    for _ in range(MAX_REDIRECT_HOPS * 3 + 1):
        if request is None:
            break
        urls.append(request.url)
        request = getattr(request, "redirected_from", None)
    return list(reversed(urls))


def validate_navigation_chain(urls, *, trusted_origins=frozenset()) -> None:
    for u in urls:
        check_url_or_raise(u, trusted_origins=trusted_origins)


async def validate_navigation_chain_async(urls, *, trusted_origins=frozenset()) -> None:
    for u in urls:
        await check_url_or_raise_async(u, trusted_origins=trusted_origins)


_INSECURE_SSL_WARNED: set = set()


def warn_insecure_ssl(host: str) -> None:
    """Record a TLS-verification-disabled fetch (warn once per host)."""
    log_counter("web_insecure_ssl_fetch")
    h = (host or "").lower()
    if h not in _INSECURE_SSL_WARNED:
        _INSECURE_SSL_WARNED.add(h)
        logger.warning(
            f"TLS certificate verification DISABLED for fetches to {h} "
            "(subscription ssl_verify=0) — traffic to this host can be "
            "intercepted; only use for trusted self-signed intranet services."
        )
