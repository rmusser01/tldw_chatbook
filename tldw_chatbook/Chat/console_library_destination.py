"""Credential-free Console provider destination and disclosure state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from ipaddress import IPv4Address, IPv6Address, ip_address, ip_network
from string import hexdigits
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution


_UNKNOWN_ENDPOINT_IDENTITY = "external/unknown"
_LOCAL_TRANSPORT_SCHEMES = frozenset({"unix", "http+unix"})
_HTTP_SCHEMES = frozenset({"http", "https"})
_PRIVATE_IPV4_NETWORKS = tuple(
    ip_network(cidr) for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_UNIQUE_LOCAL_IPV6 = ip_network("fc00::/7")
_CANONICAL_CLOUD_ORIGINS = frozenset(
    {
        ("https", "api.anthropic.com", None),
        ("https", "api.cohere.com", None),
        ("https", "api.deepseek.com", None),
        ("https", "generativelanguage.googleapis.com", None),
        ("https", "api.groq.com", None),
        ("https", "api-inference.huggingface.co", None),
        ("https", "router.huggingface.co", None),
        ("https", "api.mistral.ai", None),
        ("https", "api.moonshot.ai", None),
        ("https", "api.moonshot.cn", None),
        ("https", "api.openai.com", None),
        ("https", "openrouter.ai", None),
        ("https", "dashscope-intl.aliyuncs.com", None),
        ("https", "api.z.ai", None),
    }
)

ConsoleDestinationIdentity = tuple[
    str,
    str | None,
    str,
    ConsoleEgressClass,
]


@dataclass(frozen=True, slots=True)
class ConsoleLibraryEgressDisclosure:
    """One live-session disclosure raised by a destination transition."""

    previous_resolved_identity: ConsoleDestinationIdentity
    resolved_destination: ConsoleResolvedDestination


@dataclass(frozen=True, slots=True)
class ConsoleLibraryDestinationRuntimeState:
    """Non-durable destination state owned by one live Console session."""

    resolved_destination: ConsoleResolvedDestination | None = None
    last_resolved_identity: ConsoleDestinationIdentity | None = None
    disclosure: ConsoleLibraryEgressDisclosure | None = None


def resolve_console_destination(
    resolution: ConsoleProviderResolution,
) -> ConsoleResolvedDestination:
    """Classify a gateway-normalized endpoint without DNS or credential hints.

    Args:
        resolution: Provider readiness result after effective endpoint resolution.

    Returns:
        A credential-free destination containing only normalized origin identity.
    """
    provider = resolution.provider if isinstance(resolution.provider, str) else "unknown"
    model = resolution.model if isinstance(resolution.model, str) else None
    parsed = _parse_endpoint_origin(resolution.base_url)
    if parsed is None:
        return ConsoleResolvedDestination(
            provider=provider,
            model=model,
            endpoint_identity=_UNKNOWN_ENDPOINT_IDENTITY,
            egress_class=ConsoleEgressClass.UNKNOWN,
        )

    scheme, hostname, port, ip = parsed
    if scheme in _LOCAL_TRANSPORT_SCHEMES:
        endpoint_identity = f"{scheme}://local"
        egress_class = ConsoleEgressClass.ON_DEVICE
    else:
        endpoint_identity = _origin_identity(scheme, hostname, port, ip)
        egress_class = _classify_origin(scheme, hostname, port, ip)
    return ConsoleResolvedDestination(
        provider=provider,
        model=model,
        endpoint_identity=endpoint_identity,
        egress_class=egress_class,
    )


def update_console_library_destination_runtime(
    state: ConsoleLibraryDestinationRuntimeState,
    destination: ConsoleResolvedDestination,
    *,
    library_data_possible: bool,
) -> ConsoleLibraryDestinationRuntimeState:
    """Observe one resolved destination and update live disclosure state."""
    if not isinstance(state, ConsoleLibraryDestinationRuntimeState):
        raise TypeError("state must be ConsoleLibraryDestinationRuntimeState")
    if not isinstance(destination, ConsoleResolvedDestination):
        raise TypeError("destination must be ConsoleResolvedDestination")
    if type(library_data_possible) is not bool:
        raise TypeError("library_data_possible must be a bool")

    previous_identity = state.last_resolved_identity
    current_identity = destination.identity_key
    if previous_identity == current_identity:
        return replace(state, resolved_destination=destination)

    non_device = destination.egress_class is not ConsoleEgressClass.ON_DEVICE
    transitioned_from_device = (
        previous_identity is not None
        and previous_identity[3] is ConsoleEgressClass.ON_DEVICE
    )
    replace_existing = state.disclosure is not None
    disclosure = None
    if (
        library_data_possible
        and non_device
        and previous_identity is not None
        and (transitioned_from_device or replace_existing)
    ):
        disclosure = ConsoleLibraryEgressDisclosure(
            previous_resolved_identity=previous_identity,
            resolved_destination=destination,
        )
    return ConsoleLibraryDestinationRuntimeState(
        resolved_destination=destination,
        last_resolved_identity=current_identity,
        disclosure=disclosure,
    )


def settle_console_library_destination_runtime(
    state: ConsoleLibraryDestinationRuntimeState,
) -> ConsoleLibraryDestinationRuntimeState:
    """Clear a live disclosure while retaining the last resolved destination."""
    if not isinstance(state, ConsoleLibraryDestinationRuntimeState):
        raise TypeError("state must be ConsoleLibraryDestinationRuntimeState")
    if state.disclosure is None:
        return state
    return replace(state, disclosure=None)


def _parse_endpoint_origin(
    raw_endpoint: object,
) -> tuple[str, str, int | None, IPv4Address | IPv6Address | None] | None:
    if not isinstance(raw_endpoint, str):
        return None
    endpoint = raw_endpoint.strip()
    if (
        not endpoint
        or len(endpoint) > 2048
        or any(character.isspace() or ord(character) < 32 for character in endpoint)
    ):
        return None
    try:
        parsed = urlsplit(endpoint)
        scheme = parsed.scheme.lower()
        if scheme in _LOCAL_TRANSPORT_SCHEMES:
            return (scheme, "local", None, None)
        if (
            scheme not in _HTTP_SCHEMES
            or not parsed.hostname
            or not _userinfo_is_well_formed(parsed.netloc)
        ):
            return None
        hostname = parsed.hostname.lower().rstrip(".")
        if not hostname or len(hostname) > 253:
            return None
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (scheme, port) in {("http", 80), ("https", 443)}:
        port = None
    try:
        parsed_ip: IPv4Address | IPv6Address | None = ip_address(hostname)
    except ValueError:
        parsed_ip = None
    return (scheme, hostname, port, parsed_ip)


def _userinfo_is_well_formed(netloc: str) -> bool:
    separator_count = netloc.count("@")
    if separator_count == 0:
        return True
    if separator_count != 1:
        return False
    userinfo, hostinfo = netloc.rsplit("@", maxsplit=1)
    if not userinfo or not hostinfo:
        return False
    for index, character in enumerate(userinfo):
        if character == "%" and (
            index + 2 >= len(userinfo)
            or userinfo[index + 1] not in hexdigits
            or userinfo[index + 2] not in hexdigits
        ):
            return False
    return True


def _origin_identity(
    scheme: str,
    hostname: str,
    port: int | None,
    ip: IPv4Address | IPv6Address | None,
) -> str:
    normalized_host = str(ip) if ip is not None else hostname
    if isinstance(ip, IPv6Address):
        normalized_host = f"[{normalized_host}]"
    netloc = normalized_host if port is None else f"{normalized_host}:{port}"
    return urlunsplit((scheme, netloc, "", "", ""))


def _classify_origin(
    scheme: str,
    hostname: str,
    port: int | None,
    ip: IPv4Address | IPv6Address | None,
) -> ConsoleEgressClass:
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return ConsoleEgressClass.ON_DEVICE
    if ip is not None:
        if ip.is_loopback:
            return ConsoleEgressClass.ON_DEVICE
        if ip.is_link_local:
            return ConsoleEgressClass.PRIVATE_NETWORK
        if isinstance(ip, IPv4Address) and any(
            ip in private_network for private_network in _PRIVATE_IPV4_NETWORKS
        ):
            return ConsoleEgressClass.PRIVATE_NETWORK
        if isinstance(ip, IPv6Address) and ip in _UNIQUE_LOCAL_IPV6:
            return ConsoleEgressClass.PRIVATE_NETWORK
        if ip.is_global:
            return ConsoleEgressClass.PUBLIC_NETWORK
        return ConsoleEgressClass.UNKNOWN
    if (scheme, hostname, port) in _CANONICAL_CLOUD_ORIGINS:
        return ConsoleEgressClass.PUBLIC_NETWORK
    return ConsoleEgressClass.UNKNOWN
