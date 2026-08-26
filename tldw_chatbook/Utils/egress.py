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
from typing import Any, Iterable, List, Mapping, MutableMapping
from urllib.parse import urljoin, urlparse

import requests
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
        """Build the blocked-egress error with a credential-free message.

        ``self.url`` keeps the full URL for programmatic consumers (retry
        logic, callers that need the real target); the redaction boundary
        is ``str()``/``repr()`` -- what actually reaches logs and end
        users -- so the exception MESSAGE gets the credential-free origin
        label only.

        Args:
            url: The full request URL, retained verbatim on ``self.url``
                but rendered in the message via ``_log_origin``.
            reason: Short policy-reason slug (e.g. ``"private-ip"``).
            detail: Optional extra context appended to the message.
        """
        self.url = url
        self.reason = reason
        self.detail = detail
        super().__init__(
            f"Egress blocked ({reason}) for {_log_origin(url)}"
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
    """Classify one resolved IP: "metadata" | "private" | "public".

    ``ip.is_global`` is True for multicast (e.g. ``224.0.0.1``, or
    ``239.255.255.250`` -- a real SSDP/UPnP discovery address): the stdlib
    only excludes multicast from ``is_private``, not from ``is_global``, so
    without an explicit check here a multicast target would misclassify as
    "public" and sail through both this module's own ``evaluate_url_policy``
    (used by every ``guarded_fetch_*`` helper) and ``is_public_http_url``.
    Multicast is never a legitimate unicast HTTP/fetch target, so blocking
    it is strictly tightening for every consumer of this function -- caught
    in review of task-1356's pre-scrape SSRF guard (task-1356 CRITICAL 1).
    """
    ip = ipaddress.ip_address(ip_str)
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if ip in _METADATA_IPS:
        return "metadata"
    if ip.is_multicast:
        return "private"
    return "public" if ip.is_global else "private"


def is_public_http_url(url: str) -> bool:
    """Strict "is this genuinely a public internet address" check (task-1356).

    Unlike :func:`evaluate_url_policy`, this ignores ``trusted_origins`` and
    the ``[web_security]`` ``allowed_hosts``/``enabled`` config -- it answers
    the narrower question a pre-fetch guard needs when it must refuse
    REGARDLESS of user-configured trust (e.g. deep-search's relevance phase,
    which browses arbitrary search-result URLs with Playwright and must
    refuse a result pointing at ``http://169.254.169.254/`` even though the
    egress policy might otherwise be relaxed for this run). Reuses this
    module's own IP classification (``_classify_ip``, built on
    ``ipaddress.is_global`` -- which already excludes the CGNAT
    ``100.64.0.0/10`` and ``192.0.0.0/24`` ranges per the stdlib's
    documented exception, matching ``Tools/web_tool_impls._is_public_ip``'s
    classification without a second, hand-rolled network list) so a
    public/private/metadata verdict is computed in exactly one place.

    Args:
        url: The URL to classify.

    Returns:
        ``True`` iff ``url`` is http(s), has a host, that host resolves (or
        is already an IP literal), and every resolved IP classifies as
        public. ``False`` for anything else -- bad scheme, unparseable URL,
        DNS failure, or any private/loopback/link-local/multicast/reserved/
        unspecified/CGNAT/metadata IP. Never raises.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = parsed.hostname
    if parsed.scheme not in ("http", "https") or not host:
        return False
    try:
        _ = parsed.port
    except ValueError:
        return False
    h = host.lower()
    if h in METADATA_HOSTNAMES:
        return False
    try:
        ipaddress.ip_address(h)
        ips: List[str] = [h]
    except ValueError:
        try:
            ips = _resolve(h)
        except (OSError, socket.gaierror):
            return False
    if not ips:
        return False
    try:
        return all(_classify_ip(ip) == "public" for ip in ips)
    except ValueError:
        return False


def _log_origin(url: str) -> str:
    """Return a credential- and query-free URL label for transport logs."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return "<invalid-url>"
    if not host or parsed.scheme not in ("http", "https"):
        return "<invalid-url>"
    rendered_host = f"[{host}]" if ":" in host else host
    rendered_port = f":{port}" if port is not None else ""
    return f"{parsed.scheme}://{rendered_host}{rendered_port}"


def log_origin(url: str) -> str:
    """Public wrapper for :func:`_log_origin` (same ``_host_of``/``host_of``
    pattern used below): a credential- and query-free ``scheme://host[:port]``
    label, safe to interpolate into a log line outside this module.

    Any caller that would otherwise log a raw URL -- which may carry an
    embedded ``user:pass@`` credential or a query string with a token/API
    key -- should call this instead. It is the same redaction the
    ``EgressBlockedError``/``EgressFetchError`` messages already apply to
    themselves; this just makes it reusable so there is one spelling of
    "how we name a URL in a log", not a second copy of the logic.
    """
    return _log_origin(url)


def _blocked(url: str, reason: str, host: str, detail: str = "") -> EgressDecision:
    logger.warning(f"Egress blocked ({reason}): {_log_origin(url)} {detail}".rstrip())
    log_counter("egress_blocked", labels={"reason": reason})
    return EgressDecision(allowed=False, reason=reason, host=host)


def _pre_resolution(url: str, trusted_origins: frozenset):
    """Checks that need no DNS. Returns EgressDecision, or the host to resolve."""
    if not _config_enabled():
        logger.debug(f"Egress check disabled by [web_security] for {_log_origin(url)}")
        log_counter("egress_check_skipped")
        return EgressDecision(allowed=True, reason="disabled", host="")
    try:
        parsed = urlparse(url)
        host = parsed.hostname
    except ValueError:
        return _blocked(url, "scheme", "", "unparseable URL")
    if parsed.scheme not in ("http", "https") or not host:
        return _blocked(url, "scheme", host or "", "only http/https with a host")
    try:
        # urlparse defers port parsing until accessed; a malformed/out-of-range
        # port must fail HERE, at the policy boundary, not as a downstream
        # client InvalidURL — and it keeps this validator consistent with
        # origin_of(), which treats the same ValueError as unparseable.
        _ = parsed.port
    except ValueError:
        return _blocked(url, "scheme", host or "", "invalid port")
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
#: Header names stripped on any cross-origin redirect hop (see
#: ``_hop_headers`` / ``guarded_fetch_httpx``). ``x-goog-api-key`` is Gemini's
#: custom auth header (``Image_Generation/adapters/gemini_image_adapter.py``)
#: -- the first adapter in this app to authenticate via a header other than
#: ``Authorization``. Without listing it here, a redirect from the Gemini
#: base (or any user-configured base_url) to a different public host would
#: forward the real API key verbatim; every ``Authorization``-based adapter
#: is already protected by this tuple. There is no legitimate cross-origin
#: use of this header, so adding it is strictly tightening.
#:
#: NOTE (task-19733): this tuple is no longer the cross-origin RULE -- the rule
#: is :data:`CROSS_ORIGIN_SAFE_HEADERS` below, an allowlist. It survives as the
#: never-cross floor: both exemption sets below are constructed by SUBTRACTING
#: it, so a careless future edit that adds one of these names to an exemption
#: list cannot take effect.
_STRIP_HEADERS = ("authorization", "cookie", "proxy-authorization", "x-goog-api-key")

#: Request headers that MAY be forwarded across an origin boundary on a
#: redirect hop. Everything else the caller (or the client's own default
#: headers) supplied is dropped.
#:
#: This is an ALLOWLIST on purpose (task-19733). A denylist of credential
#: header names cannot be correct in this app, because the credential header
#: NAME is user-supplied: a subscription's ``auth_config`` chooses it
#: (``Subscriptions/monitoring_engine.py`` only *defaults* to ``X-API-Key``),
#: and so does a per-site config (``site_config_manager.SiteConfig.get_headers``,
#: same default). Growing ``_STRIP_HEADERS`` one incident at a time closes the
#: name someone happened to think of and forwards every other one verbatim to
#: whatever host the feed redirects to.
#:
#: Membership test: a header is listed here only when forwarding it to a
#: DIFFERENT origin is both (a) needed by a real caller and (b) incapable of
#: carrying a secret. That admits content negotiation, cache validators, and
#: range/partial-content headers -- a CDN redirect (feeds, model artifacts)
#: genuinely needs those or conditional GET and download resume break.
#:
#: Consequence to know about: a NON-credential custom header a user configured
#: for a feed (``custom_headers``) also stops being forwarded once that feed
#: redirects off-origin. That is deliberate -- nothing at this layer can tell
#: ``X-Feed-Token`` from ``X-Client-Version`` by name -- and it is the safe
#: direction to be wrong in.
CROSS_ORIGIN_SAFE_HEADERS = frozenset(
    {
        # Content negotiation
        "accept",
        "accept-charset",
        "accept-encoding",
        "accept-language",
        # Caching / conditional requests
        "cache-control",
        "pragma",
        "if-match",
        "if-modified-since",
        "if-none-match",
        "if-unmodified-since",
        # Partial content (Model_Artifacts resume; HF -> CDN is cross-origin)
        "if-range",
        "range",
        # Client identity (not a credential)
        "user-agent",
    }
) - frozenset(_STRIP_HEADERS)

#: Framing/connection headers owned by the HTTP client library itself, never a
#: caller credential. They are exempt from the allowlist so that stripping a
#: BUILT request cannot break the request it is protecting (dropping ``host``
#: would be fatal; dropping ``connection`` would silently change keep-alive).
#:
#: Every name here has a value drawn from a fixed vocabulary the client library
#: generates -- none of them can carry arbitrary caller text. That is the
#: membership test, and it is why ``content-type`` is NOT in this set (see
#: :data:`_BODY_DESCRIBING_HEADERS`).
_TRANSPORT_HEADERS = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "content-length",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-connection",
    }
) - frozenset(_STRIP_HEADERS)

#: Headers that describe a request BODY. They may cross an origin boundary
#: only on a hop that actually carries one.
#:
#: ``content-type`` sat in :data:`_TRANSPORT_HEADERS` until Qodo's review of
#: PR #1942 (task-19733) pointed out that it is the one exempted header whose
#: value is arbitrary caller-controlled text: ``multipart/form-data;
#: boundary=<anything>`` is a ready-made carrier, and a client-DEFAULT
#: ``Content-Type`` therefore crossed to the attacker origin on every bodyless
#: redirect hop. Deleting the exemption outright is also wrong -- dropping
#: ``Content-Type`` off a request that HAS a body corrupts it. The precise
#: rule is the conditional one: it describes a body, so it travels with a body.
_BODY_DESCRIBING_HEADERS = frozenset({"content-type"}) - frozenset(_STRIP_HEADERS)

#: Framing headers whose presence on a BUILT request proves it carries a body.
#: Verified against both clients this module drives: httpx and ``requests``
#: each emit ``Content-Length`` for a known-length body and
#: ``Transfer-Encoding: chunked`` for a streamed one, at BUILD time
#: (``Client.build_request`` / ``Session.prepare_request``), i.e. before this
#: module inspects the request. A bodyless GET gets neither; a bodyless POST
#: gets ``Content-Length: 0``, which is why the value is checked, not just the
#: name.
_BODY_FRAMING_HEADERS = frozenset({"content-length", "transfer-encoding"})


def _built_request_carries_body(request_headers: Mapping[str, str]) -> bool:
    """Whether a BUILT request actually has a body attached.

    Keys off the outgoing request's own framing headers rather than off the
    redirect status code. That is the durable form: 301/302/303 convert the
    method to GET and drop the body while 307/308 preserve both, but by the
    time this runs the client has already built the request for THIS hop, so
    its framing is the ground truth regardless of how it got here.

    Deny-by-default on anything unparseable: an unreadable ``Content-Length``
    is treated as "no body", so a header that describes a body it cannot prove
    exists does not cross an origin boundary. Neither client library emits
    such a value.

    Args:
        request_headers: Header mapping of a request the transport has
            already assembled (``httpx.Request.headers``,
            ``requests.PreparedRequest.headers``).

    Returns:
        ``True`` if the request carries a non-empty body.
    """
    for name, value in request_headers.items():
        lowered = str(name).lower()
        if lowered not in _BODY_FRAMING_HEADERS:
            continue
        text = str(value).strip()
        if lowered == "transfer-encoding":
            if text:
                return True
            continue
        try:
            if int(text) > 0:
                return True
        except ValueError:
            continue
    return False


def _may_cross_origin(name: str, *, has_body: bool) -> bool:
    """The single cross-origin rule, shared by both filtering layers.

    Args:
        name: Header name, any casing.
        has_body: Whether the hop this header would travel on carries a
            request body. Only :data:`_BODY_DESCRIBING_HEADERS` consult it.

    Returns:
        ``True`` if the header may be sent to a different origin.
    """
    lowered = name.lower()
    if lowered in CROSS_ORIGIN_SAFE_HEADERS or lowered in _TRANSPORT_HEADERS:
        return True
    if lowered in _BODY_DESCRIBING_HEADERS:
        return has_body
    return False


def filter_cross_origin_headers(
    headers: Mapping[str, str] | None,
    *,
    has_body: bool = False,
) -> dict[str, str]:
    """Caller-supplied headers reduced to the ones safe to send off-origin.

    Deny-by-default: a name absent from :data:`CROSS_ORIGIN_SAFE_HEADERS` is
    dropped, whatever it is called. Use this for the ``headers`` mapping a
    caller passed in, before it is handed to the transport.

    Args:
        headers: Any mapping of header name -> value (a plain ``dict``,
            ``httpx.Headers``, ``requests.structures.CaseInsensitiveDict`` and
            ``multidict.CIMultiDict`` all satisfy this). May be ``None``.
        has_body: Whether the request being described will carry a body. This
            is the same question :func:`strip_cross_origin_request_headers`
            answers by inspecting the built request; here it must be declared,
            because a ``headers`` mapping on its own says nothing about a
            body. Every guarded helper in this module issues a bodyless GET,
            hence the default. A body-carrying caller must pass ``True`` or
            its ``Content-Type`` will be dropped off-origin.

    Returns:
        A new plain ``dict`` containing only the forwardable entries,
        preserving the caller's casing.
    """
    return {
        str(k): v
        for k, v in dict(headers or {}).items()
        if _may_cross_origin(str(k), has_body=has_body)
    }


def strip_cross_origin_request_headers(
    request_headers: MutableMapping[str, str],
) -> None:
    """In place, drop every non-forwardable header from a BUILT request.

    The complement of :func:`filter_cross_origin_headers`: that one filters
    what the caller passed, this one filters what the transport actually
    assembled -- which additionally contains the client object's DEFAULT
    headers (``httpx.Client(headers=...)``, a ``requests.Session``'s
    ``headers``/``auth``/cookies). Those are invisible to the caller-side
    filter, and a credential set there leaks exactly the same way.

    Both layers apply the same rule (:func:`_may_cross_origin`); the only
    difference is where ``has_body`` comes from. Here it is INFERRED from the
    request's own framing headers, so the answer describes the actual outgoing
    hop rather than being taken on trust.

    Transport/framing headers (:data:`_TRANSPORT_HEADERS`) are left alone;
    ``Content-Type`` survives only if the request carries a body; everything
    else must be on :data:`CROSS_ORIGIN_SAFE_HEADERS`.

    Args:
        request_headers: A mutable, case-insensitive header mapping belonging
            to an already-built request (``httpx.Headers`` or
            ``requests.structures.CaseInsensitiveDict``). Mutated in place.
    """
    has_body = _built_request_carries_body(request_headers)
    for name in list(request_headers.keys()):
        if _may_cross_origin(str(name), has_body=has_body):
            continue
        request_headers.pop(name, None)


class EgressFetchError(Exception):
    """Guarded-fetch transport failure: size cap, hop cap, missing Location."""

    def __init__(self, message: str, url: str = ""):
        """Build the fetch-failure error with a credential-free message.

        Same redaction boundary as ``EgressBlockedError``: ``self.url``
        keeps the full URL, the message gets the origin label only.

        Args:
            message: Human-readable failure description.
            url: Optional full request URL, retained verbatim on
                ``self.url`` but rendered in the message via
                ``_log_origin``.
        """
        self.url = url
        super().__init__(f"{message}" + (f" [{_log_origin(url)}]" if url else ""))


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


def origin_of(url: str) -> tuple[str, str, int] | None:
    """Scheme+host+effective-port origin of ``url``, or ``None`` if undetermined.

    This is the STRICT origin used to gate credential forwarding
    (``Authorization``/``Cookie``/an httpx ``auth=``/a ``requests``
    ``session.auth``) across a redirect hop — a same-host HTTPS->HTTP
    downgrade or a same-host different-port redirect is NOT the same
    origin and must not carry credentials. It is deliberately NOT the
    same thing as the SSRF ``trusted_origins`` policy (see ``host_of``/
    ``origin_set``), which stays hostname-only by design: a user who
    trusts a host trusts it regardless of scheme/port.

    Ports are normalized to the scheme's default when omitted, so
    ``https://h/`` and ``https://h:443/`` produce the same origin.

    Args:
        url: The URL to derive an origin for.

    Returns:
        ``(scheme, host, port)`` with ``scheme``/``host`` lowercased, or
        ``None`` for a missing host, an unparseable URL, or a scheme other
        than http/https with no explicit port (nothing to default it to).
        Never raises.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is None:
        if scheme == "https":
            port = 443
        elif scheme == "http":
            port = 80
        else:
            return None
    return (scheme, host, port)


def same_origin(url_a: str, url_b: str) -> bool:
    """Whether ``url_a`` and ``url_b`` share scheme, host, and effective port.

    The credential-forwarding decision for a redirect hop: use this (not
    ``host_of`` equality) wherever a hop is judged "safe" to carry
    ``Authorization``/``Cookie``/auth across. Conservative on ambiguity —
    if either URL's origin can't be determined by :func:`origin_of`, this
    returns ``False`` (strip credentials rather than guess).

    Args:
        url_a: First URL.
        url_b: Second URL.

    Returns:
        ``True`` iff both resolve to the same ``(scheme, host, port)``.
    """
    origin_a = origin_of(url_a)
    return origin_a is not None and origin_a == origin_of(url_b)


def _hop_headers(
    headers: Mapping[str, str] | None, same_origin: bool
) -> dict[str, str]:
    """Caller headers for one hop: everything same-origin, allowlist otherwise.

    task-19733 inverted the cross-origin branch from a denylist of credential
    names to :func:`filter_cross_origin_headers`. Same-origin hops are
    untouched, so an authenticated feed that redirects within its own origin
    keeps working exactly as before.

    ``has_body`` is left at its default: every ``guarded_fetch_*`` helper in
    this module issues a bodyless GET, so a caller-supplied ``Content-Type``
    describes nothing and does not cross an origin boundary.

    Args:
        headers: The ``headers`` mapping the caller passed to a guarded fetch.
        same_origin: Whether this hop stays on the original request's origin.

    Returns:
        A new plain ``dict`` of the headers this hop may carry.
    """
    if same_origin:
        return {str(k): v for k, v in dict(headers or {}).items()}
    return filter_cross_origin_headers(headers)


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
    current = url
    for hop in range(MAX_REDIRECT_HOPS + 1):
        check_url_or_raise(current, trusted_origins=trusted_origins)
        is_same_origin = same_origin(url, current)
        request = client.build_request(
            "GET",
            current,
            headers=_hop_headers(headers, is_same_origin),
            params=params if hop == 0 else None,
        )
        if not is_same_origin:
            # Strip credentials the client attaches at the transport-object
            # level (e.g. httpx.Client(headers={"Authorization": ...})) —
            # these are merged onto the built request by httpx and are
            # invisible to _hop_headers, which only sees the per-call
            # `headers` argument. Allowlist, not denylist (task-19733): a
            # client-default header named by the user leaks identically to a
            # per-call one.
            strip_cross_origin_request_headers(request.headers)
        send_kwargs = {"stream": True, "follow_redirects": False}
        if not is_same_origin:
            # httpx applies a CLIENT-level ``auth=`` inside ``send()``, AFTER
            # build_request -- so header stripping above cannot see it and an
            # ``httpx.Client(auth=("u", "p"))`` re-attached ``Authorization``
            # on the very hop this guard exists to protect. Passing an
            # explicit ``auth=None`` (rather than the USE_CLIENT_DEFAULT
            # sentinel) suppresses it for this hop only.
            send_kwargs["auth"] = None
        response = client.send(request, **send_kwargs)
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
    A cross-origin hop carries :data:`CROSS_ORIGIN_SAFE_HEADERS` plus the
    client library's own framing (:data:`_TRANSPORT_HEADERS`) and nothing
    else — no matter whether the header came from the ``headers`` argument,
    from the client object's own default headers (e.g. an
    ``httpx.AsyncClient(headers=...)``), or from a client-level ``auth=`` (set
    on the ``httpx.Client``/``AsyncClient`` itself, as opposed to the ``auth``
    parameter of this function). That last one is applied by httpx inside
    ``send()``, so it is suppressed by passing an explicit ``auth=None`` on the
    cross-origin hop rather than by header stripping. ``Content-Type`` would
    be the one conditional exception (:data:`_BODY_DESCRIBING_HEADERS`), but
    this helper only ever issues a bodyless GET, so it never applies here.
    """
    current = url
    for hop in range(MAX_REDIRECT_HOPS + 1):
        await check_url_or_raise_async(current, trusted_origins=trusted_origins)
        is_same_origin = same_origin(url, current)
        request = client.build_request(
            "GET",
            current,
            headers=_hop_headers(headers, is_same_origin),
            params=params if hop == 0 else None,
        )
        if not is_same_origin:
            # Strip credentials the client attaches at the transport-object
            # level — see guarded_fetch_httpx for the rationale.
            strip_cross_origin_request_headers(request.headers)
        send_kwargs = {"stream": True, "follow_redirects": False}
        if not is_same_origin:
            # Explicit None, not "omit": omitting leaves httpx's
            # USE_CLIENT_DEFAULT sentinel in play, which re-applies a
            # CLIENT-level ``auth=`` inside send() -- after the header strip
            # above. See guarded_fetch_httpx for the full rationale.
            send_kwargs["auth"] = None
        elif auth is not None:
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


#: Config-driven default ``(connect, read)`` timeout for
#: ``create_default_session()`` -- see ``default_session_timeout()`` below.
#: The read half matches the house default this module already used for
#: ``guarded_fetch_requests``.
DEFAULT_SESSION_CONNECT_TIMEOUT = 10.0
DEFAULT_SESSION_READ_TIMEOUT = 30.0


def default_session_timeout() -> tuple[float, float]:
    """Config-driven ``(connect, read)`` timeout for ``create_default_session()``.

    Read from ``[web_security] request_connect_timeout_seconds`` /
    ``request_read_timeout_seconds``, the same ``get_cli_setting`` pattern
    ``_config_enabled``/``_config_allowed_hosts`` above already use, and
    consistent in spirit with the per-provider ``api_timeout`` settings
    ``LLM_Calls`` reads directly (task-19830).

    A tuple, not a single number, because ``requests`` applies the two
    halves differently: CONNECT bounds the TCP handshake/TLS setup once;
    READ is re-armed for every chunk of a streamed response (``stream=
    True``), so a slow-but-progressing stream is only killed by a stall on
    one chunk, never by total elapsed duration.
    """
    connect = get_cli_setting(
        "web_security",
        "request_connect_timeout_seconds",
        DEFAULT_SESSION_CONNECT_TIMEOUT,
    )
    read = get_cli_setting(
        "web_security", "request_read_timeout_seconds", DEFAULT_SESSION_READ_TIMEOUT
    )
    try:
        connect_seconds = float(connect)
    except (TypeError, ValueError):
        connect_seconds = DEFAULT_SESSION_CONNECT_TIMEOUT
    try:
        read_seconds = float(read)
    except (TypeError, ValueError):
        read_seconds = DEFAULT_SESSION_READ_TIMEOUT
    return (connect_seconds, read_seconds)


class DefaultTimeoutSession(requests.Session):
    """A ``requests.Session`` that fills in a default timeout when the
    caller omits one.

    ``get``/``post``/``put``/``delete``/``patch`` all funnel through
    ``Session.request()`` in the base ``requests.Session`` class, so
    overriding this one method covers every verb. An explicit ``timeout=``
    -- keyword, or the 9th positional argument the rare
    ``session.request(method, url, ...)`` call site passes -- always wins;
    this only fills the gap when the caller specified neither (task-19830).
    """

    def __init__(
        self,
        *args: Any,
        default_timeout: float | tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the session.

        Args:
            *args: Forwarded unchanged to ``requests.Session``.
            default_timeout: The timeout to fill in for calls that omit one
                -- a scalar (both halves) or a ``(connect, read)`` tuple.
                ``None`` means read it from config via
                ``default_session_timeout()`` at construction time.
            **kwargs: Forwarded unchanged to ``requests.Session``.
        """
        super().__init__(*args, **kwargs)
        self.default_timeout: float | tuple[float, float] = (
            default_session_timeout() if default_timeout is None else default_timeout
        )

    def request(  # type: ignore[override]
        self, method: str, url: str, *args: Any, **kwargs: Any
    ) -> requests.Response:
        """Dispatch the request, supplying ``default_timeout`` if needed.

        Args:
            method: HTTP method, as ``requests.Session.request`` takes it.
            url: The target URL.
            *args: Positional arguments forwarded to
                ``requests.Session.request``.
            **kwargs: Keyword arguments forwarded likewise. An explicit
                ``timeout`` here (or in the positional slot) is preserved.

        Returns:
            The ``requests.Response``, unchanged.

        Raises:
            requests.RequestException: Whatever the underlying request
                raises, including the ``Timeout`` this default exists to
                produce instead of blocking forever.
        """
        # Session.request's positional signature (after method, url) is
        # params, data, headers, cookies, files, auth, timeout -- 6
        # positionals before `timeout`, so a 7th positional argument IS an
        # explicit timeout and must not be overridden.
        if "timeout" not in kwargs and len(args) <= 6:
            kwargs["timeout"] = self.default_timeout
        return super().request(method, url, *args, **kwargs)


def create_default_session(
    *, timeout: float | tuple[float, float] | None = None
) -> "DefaultTimeoutSession":
    """Build a ``requests.Session`` whose default timeout is config-driven.

    Every call made through the returned session that omits ``timeout=``
    gets ``default_session_timeout()`` (or the ``timeout`` passed here, if
    given) instead of blocking forever on a half-open connection. A call
    that already passes its own ``timeout=`` is completely untouched --
    construction alone never changes behaviour for a deliberate per-call or
    per-provider timeout (task-19830).

    Args:
        timeout: Override for this session's default -- a scalar (applied
            to both halves) or a ``(connect, read)`` tuple. ``None`` reads
            the configured default.

    Returns:
        A ``DefaultTimeoutSession``. Usable anywhere a
        ``requests.Session`` is, including as a context manager.
    """
    return DefaultTimeoutSession(default_timeout=timeout)


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
        current = url
        for _hop in range(MAX_REDIRECT_HOPS + 1):
            check_url_or_raise(current, trusted_origins=trusted_origins)
            is_same_origin = same_origin(url, current)
            prepared = sess.prepare_request(
                requests.Request(
                    "GET", current, headers=_hop_headers(headers, is_same_origin)
                )
            )
            if not is_same_origin:
                # prepare_request applies session.auth/cookies AND the
                # session's default headers into the prepared request; a
                # cross-origin hop must not carry any of them unless they are
                # explicitly forwardable (task-19733).
                strip_cross_origin_request_headers(prepared.headers)
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
    """Capped GET via aiohttp.ClientSession with per-hop re-validation.

    Cross-origin hops carry only :data:`CROSS_ORIGIN_SAFE_HEADERS` out of the
    ``headers`` argument. Unlike the httpx/requests helpers there is no built
    request object to post-filter here, so a credential set as an
    ``aiohttp.ClientSession(headers=...)`` DEFAULT is not suppressed — a
    documented residual, unchanged by task-19733; no live caller does that.
    """
    from multidict import CIMultiDict

    current = url
    for _hop in range(MAX_REDIRECT_HOPS + 1):
        await check_url_or_raise_async(current, trusted_origins=trusted_origins)
        is_same_origin = same_origin(url, current)
        kwargs = {
            "allow_redirects": False,
            "headers": _hop_headers(headers, is_same_origin),
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
