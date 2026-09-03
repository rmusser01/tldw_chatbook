"""Pure provider endpoint interpretation for OpenAI-compatible APIs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from ipaddress import ip_address
from typing import Literal
from unicodedata import category
from urllib.parse import urlsplit, urlunsplit

EndpointForm = Literal[
    "origin",
    "api_base",
    "chat_url",
    "models_url",
    "legacy_local",
]

_LLAMA_PROVIDER_KEYS = frozenset({"llama_cpp", "local_llamacpp"})
#: Providers whose existing chat endpoint contract exposes a bounded models
#: route suitable for an explicit, non-generating connection check.
URL_BASED_PROVIDER_KEYS = frozenset(
    {
        "aphrodite",
        "custom",
        "custom_2",
        "koboldcpp",
        "llama_cpp",
        "local_llamacpp",
        "local_llamafile",
        "local_ollama",
        "local_vllm",
        "ollama",
        "oobabooga",
        "qwencloud",
        "tabbyapi",
        "vllm",
    }
)
_PROVIDER_ALIASES = {
    "Custom": "custom",
    "Custom OpenAI": "custom",
    "Custom OpenAI API": "custom",
    "custom-openai": "custom",
    "custom_openai": "custom",
    "custom-openai-api": "custom",
    "custom_openai_api": "custom",
    "Custom-2": "custom_2",
    "Custom 2": "custom_2",
    "custom-2": "custom_2",
    "Custom OpenAI 2": "custom_2",
    "custom-openai-2": "custom_2",
    "custom_openai_2": "custom_2",
    "Custom OpenAI API-2": "custom_2",
    "Custom OpenAI API 2": "custom_2",
    "custom-openai-api-2": "custom_2",
    "custom_openai_api_2": "custom_2",
    "llama.cpp": "llama_cpp",
    "local llama.cpp": "local_llamacpp",
    "OpenRouter": "openrouter",
    "local-llamacpp": "local_llamacpp",
}
_REMOTE_HTTP_WARNING = "Remote HTTP endpoints are not encrypted."
_PROVIDER_KEY = re.compile(r"[a-z0-9_]+")
_UNSAFE_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_UNRESERVED_ASCII = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_MAX_PROVIDER_LENGTH = 128
_MAX_ENDPOINT_LENGTH = 4096
_SUFFIXES: tuple[tuple[tuple[str, ...], EndpointForm], ...] = (
    (("v1", "chat", "completions"), "chat_url"),
    (("v1", "models"), "models_url"),
    (("completion",), "legacy_local"),
    (("v1",), "api_base"),
)


@dataclass(frozen=True, slots=True)
class ProviderEndpointResolution:
    """Normalized endpoint URLs and safe display values for one provider."""

    provider_key: str
    normalized_input: str
    persisted_endpoint: str | None
    chat_url: str | None
    models_url: str | None
    persisted_display: str
    chat_display: str
    models_display: str
    form: EndpointForm | None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class ConnectionProbeAvailability(StrEnum):
    """Whether the provider draft has a meaningful non-generating probe."""

    MODELS_ROUTE = "models_route"
    UNAVAILABLE = "unavailable"


def connection_probe_availability(
    provider: str,
    endpoint: str | None,
) -> ConnectionProbeAvailability:
    """Return probe availability without reading config or performing I/O."""

    provider_key = normalize_provider_key_for_contract(provider)
    if provider_key not in URL_BASED_PROVIDER_KEYS:
        return ConnectionProbeAvailability.UNAVAILABLE
    resolution = resolve_provider_endpoint(provider_key, endpoint)
    if resolution.errors or resolution.models_url is None:
        return ConnectionProbeAvailability.UNAVAILABLE
    return ConnectionProbeAvailability.MODELS_ROUTE


def resolve_provider_endpoint(
    provider: str, value: object
) -> ProviderEndpointResolution:
    """Interpret one endpoint value without reading config or performing I/O."""

    if not isinstance(provider, str) or len(provider) > _MAX_PROVIDER_LENGTH:
        return _invalid_resolution("", "Provider name is invalid or too long.")
    provider_key = normalize_provider_key_for_contract(provider)
    if not provider_key:
        return _invalid_resolution("", "Select a valid provider.")
    if not isinstance(value, str):
        return _invalid_resolution(provider_key, "Enter an endpoint URL.")
    if len(value) > _MAX_ENDPOINT_LENGTH:
        return _invalid_resolution(provider_key, "Endpoint URL is too long.")
    if _contains_unsafe_unicode(value):
        return _invalid_resolution(
            provider_key, "Endpoint URL contains invalid characters."
        )

    raw_value = value.strip()
    if not raw_value:
        return _invalid_resolution(provider_key, "Enter an endpoint URL.")
    if any(character.isspace() for character in raw_value):
        return _invalid_resolution(
            provider_key, "Endpoint URL contains invalid characters."
        )
    if "?" in raw_value:
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include a query."
        )
    if "#" in raw_value:
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include a fragment."
        )

    has_explicit_scheme = "://" in raw_value
    candidate = raw_value if has_explicit_scheme else f"http://{raw_value}"
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return _invalid_resolution(provider_key, "Endpoint URL is malformed.")

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return _invalid_resolution(provider_key, "Endpoint URL must use HTTP or HTTPS.")
    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
    ):
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include user information."
        )
    if parsed.netloc.endswith(":"):
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include an empty port."
        )
    if parsed.query:
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include a query."
        )
    if parsed.fragment:
        return _invalid_resolution(
            provider_key, "Endpoint URL must not include a fragment."
        )
    canonical_host = _canonical_host(hostname) if hostname else None
    if not canonical_host or port == 0:
        return _invalid_resolution(
            provider_key, "Endpoint URL must include a valid host and port."
        )
    if not has_explicit_scheme and not _is_exact_schemeless_host(parsed.netloc):
        return _invalid_resolution(
            provider_key,
            "Remote endpoint URLs must include an HTTP or HTTPS scheme.",
        )

    original_path = parsed.path
    if "//" in original_path or "\\" in original_path:
        return _invalid_resolution(provider_key, "Endpoint path is ambiguous.")
    canonical_path = _canonicalize_path(original_path)
    if canonical_path is None:
        return _invalid_resolution(
            provider_key, "Endpoint path contains invalid percent encoding."
        )

    path = canonical_path.removesuffix("/")
    segments = tuple(segment for segment in path.split("/") if segment)
    if any(segment in {".", ".."} for segment in segments):
        return _invalid_resolution(provider_key, "Endpoint path is ambiguous.")

    suffix = _terminal_suffix(
        segments, allow_legacy_local=provider_key in _LLAMA_PROVIDER_KEYS
    )
    if suffix is None:
        return _invalid_resolution(
            provider_key, "Endpoint path has ambiguous API suffixes."
        )
    prefix_segments, form = suffix

    netloc = _normalized_netloc(canonical_host, port)
    prefix_path = f"/{'/'.join(prefix_segments)}" if prefix_segments else ""
    root = urlunsplit((scheme, netloc, prefix_path, "", ""))
    normalized_path = f"/{'/'.join(segments)}" if segments else ""
    normalized_input = urlunsplit((scheme, netloc, normalized_path, "", ""))
    chat_url = f"{root}/v1/chat/completions"
    models_url = f"{root}/v1/models"
    persisted_endpoint = root if provider_key in _LLAMA_PROVIDER_KEYS else chat_url
    warnings = (
        (_REMOTE_HTTP_WARNING,)
        if scheme == "http" and not _is_loopback_host(canonical_host)
        else ()
    )

    return ProviderEndpointResolution(
        provider_key=provider_key,
        normalized_input=normalized_input,
        persisted_endpoint=persisted_endpoint,
        chat_url=chat_url,
        models_url=models_url,
        persisted_display=persisted_endpoint,
        chat_display=chat_url,
        models_display=models_url,
        form=form,
        warnings=warnings,
    )


def canonical_connection_identity(
    provider: str, value: object
) -> tuple[str, str] | None:
    """Return the provider and canonical persisted endpoint when valid."""

    resolution = resolve_provider_endpoint(provider, value)
    if resolution.persisted_endpoint is None:
        return None
    return (
        resolution.provider_key,
        _drop_default_port(resolution.persisted_endpoint),
    )


def normalize_provider_key_for_contract(provider: object) -> str:
    """Return the canonical provider key accepted by this endpoint contract.

    Args:
        provider: Provider config key or established display-name alias.

    Returns:
        The canonical provider key, or an empty string when unsupported.
    """
    if not isinstance(provider, str):
        return ""
    provider_key = provider.strip()
    alias = _PROVIDER_ALIASES.get(provider_key)
    if alias is not None:
        return alias
    if not _PROVIDER_KEY.fullmatch(provider_key):
        return ""
    return provider_key


def _invalid_resolution(
    provider_key: str, error: str
) -> ProviderEndpointResolution:
    return ProviderEndpointResolution(
        provider_key=provider_key,
        normalized_input="",
        persisted_endpoint=None,
        chat_url=None,
        models_url=None,
        persisted_display="",
        chat_display="",
        models_display="",
        form=None,
        errors=(error,),
    )


def _terminal_suffix(
    segments: tuple[str, ...],
    *,
    allow_legacy_local: bool,
) -> tuple[tuple[str, ...], EndpointForm] | None:
    occurrences: list[tuple[int, tuple[str, ...], EndpointForm]] = []
    for index in range(len(segments)):
        for suffix, form in _SUFFIXES:
            if form == "legacy_local" and not allow_legacy_local:
                continue
            if segments[index : index + len(suffix)] == suffix:
                occurrences.append((index, suffix, form))
                break

    if not occurrences:
        return (segments, "origin")
    if len(occurrences) != 1:
        return None

    index, suffix, form = occurrences[0]
    if index + len(suffix) != len(segments):
        return None
    return (segments[:index], form)


def _contains_unsafe_unicode(value: str) -> bool:
    return any(
        category(character) in _UNSAFE_UNICODE_CATEGORIES for character in value
    )


def _canonicalize_path(path: str) -> str | None:
    canonical: list[str] = []
    index = 0
    while index < len(path):
        character = path[index]
        if character != "%":
            if character.isascii():
                canonical.append(character)
            else:
                canonical.extend(
                    f"%{octet:02X}" for octet in character.encode("utf-8")
                )
            index += 1
            continue

        octets = bytearray()
        while index < len(path) and path[index] == "%":
            if (
                index + 2 >= len(path)
                or path[index + 1] not in _HEX_DIGITS
                or path[index + 2] not in _HEX_DIGITS
            ):
                return None
            octets.append(int(path[index + 1 : index + 3], 16))
            index += 3

        if _percent_octets_contain_unsafe_unicode(octets):
            return None
        for octet in octets:
            if octet in {0x2F, 0x5C} or octet < 0x20 or octet == 0x7F:
                return None
            decoded = chr(octet)
            canonical.append(
                decoded if decoded in _UNRESERVED_ASCII else f"%{octet:02X}"
            )
    return "".join(canonical)


def _percent_octets_contain_unsafe_unicode(octets: bytearray) -> bool:
    try:
        decoded = bytes(octets).decode("utf-8")
    except UnicodeDecodeError:
        return True
    return _contains_unsafe_unicode(decoded)


def _canonical_host(hostname: str) -> str | None:
    if not hostname.isascii() or "%" in hostname:
        return None
    try:
        return ip_address(hostname).compressed.lower()
    except ValueError:
        pass

    if hostname.endswith(".."):
        return None
    host = hostname.removesuffix(".")
    if not host or len(host) > 253:
        return None
    if all(character.isdigit() or character == "." for character in host):
        return None
    if not all(_valid_dns_label(label) for label in host.split(".")):
        return None
    return host.lower()


def _valid_dns_label(label: str) -> bool:
    return (
        1 <= len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
    )


def _normalized_netloc(hostname: str, port: int | None) -> str:
    host = hostname
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{port}" if port is not None else host


def _is_exact_schemeless_host(netloc: str) -> bool:
    return any(
        netloc == host or netloc.startswith(f"{host}:")
        for host in ("localhost", "127.0.0.1", "[::1]")
    )


def _drop_default_port(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    default_port = (parsed.scheme == "http" and parsed.port == 80) or (
        parsed.scheme == "https" and parsed.port == 443
    )
    if not default_port or parsed.hostname is None:
        return endpoint
    canonical_host = _canonical_host(parsed.hostname)
    if canonical_host is None:
        return endpoint
    netloc = _normalized_netloc(canonical_host, None)
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False
