"""Sync cores for web_* agent tools.

The SSRF guard below is written fresh for tldw_chatbook, using tldw_server's
tldw_Server_API/app/core/Web_Scraping/outbound_policy.py @ 5605b9d9906322c2e6b5342b48c391ae674d315e
(https://github.com/rmusser01/tldw_server, GPL-3.0-only) as the requirements
checklist — see re-plan spec §2.1 (2026-08-05).
"""

import ipaddress
import socket
from urllib.parse import urlsplit

from .local_tool_impls import LocalToolError

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DNS_CACHE_TTL_SECONDS = 300.0


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def validate_outbound_url(url: str) -> str:
    """Return ``url`` if it's safe to fetch; raise LocalToolError otherwise.

    Checks: scheme allowlist (http/https only), host resolves, and EVERY
    resolved IP is public (private/loopback/link-local/multicast/reserved/
    unspecified refused). Literal IPs are checked directly without DNS.

    Called for the initial URL AND every redirect hop. DNS-rebinding caveat:
    resolution happens again inside the HTTP client, so a hostile DNS server
    could return a public IP here and a private one at connection time;
    re-validating every redirect hop bounds that window. Pinning the
    connection to the validated IP is deliberately out of scope.

    Raises:
        LocalToolError: if the scheme is disallowed, the host is missing or
            does not resolve, or any resolved IP is not public.
    """
    parts = urlsplit(url.strip())
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
            infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80),
                                       proto=socket.IPPROTO_TCP)
        except (socket.gaierror, UnicodeError, OSError) as exc:
            raise LocalToolError(f"host does not resolve: {host!r}") from exc
        candidates = [info[4][0] for info in infos]
    if not candidates or not all(_is_public_ip(ip) for ip in candidates):
        raise LocalToolError(f"host resolves to a private/internal address: {host!r}")
    return url
