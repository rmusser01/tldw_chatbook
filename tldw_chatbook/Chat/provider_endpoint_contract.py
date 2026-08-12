"""Pure provider endpoint interpretation for OpenAI-compatible APIs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

EndpointForm = Literal[
    "origin",
    "api_base",
    "chat_url",
    "models_url",
    "legacy_local",
]

_LLAMA_PROVIDER_KEYS = frozenset({"llama_cpp", "local_llamacpp"})
_CUSTOM_PROVIDER_ALIASES = frozenset(
    {"custom", "custom_openai", "custom_openai_api"}
)
_CUSTOM_2_PROVIDER_ALIASES = frozenset(
    {"custom_2", "custom_openai_2", "custom_openai_api_2"}
)
_REMOTE_HTTP_WARNING = "Remote HTTP endpoints are not encrypted."
_ENCODED_DELIMITER = re.compile(r"%(?:2f|5c)", re.IGNORECASE)
_PROVIDER_KEY = re.compile(r"[a-z0-9_]+")
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


def resolve_provider_endpoint(
    provider: str, value: object
) -> ProviderEndpointResolution:
    """Interpret one endpoint value without reading config or performing I/O."""

    provider_key = _normalize_provider_key(provider)
    if not provider_key:
        return _invalid_resolution("", "Select a valid provider.")
    if not isinstance(value, str):
        return _invalid_resolution(provider_key, "Enter an endpoint URL.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
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
    if not hostname or port == 0 or not _valid_host(hostname):
        return _invalid_resolution(
            provider_key, "Endpoint URL must include a valid host and port."
        )
    if not has_explicit_scheme and hostname.lower() not in {
        "localhost",
        "127.0.0.1",
    }:
        return _invalid_resolution(
            provider_key,
            "Remote endpoint URLs must include an HTTP or HTTPS scheme.",
        )
    if _ENCODED_DELIMITER.search(parsed.path):
        return _invalid_resolution(
            provider_key, "Endpoint path contains an encoded delimiter."
        )

    path = parsed.path.rstrip("/")
    segments = tuple(segment for segment in path.split("/") if segment)
    if (
        "//" in path
        or "\\" in path
        or any(segment in {".", ".."} for segment in segments)
    ):
        return _invalid_resolution(provider_key, "Endpoint path is ambiguous.")

    suffix = _terminal_suffix(
        segments, allow_legacy_local=provider_key in _LLAMA_PROVIDER_KEYS
    )
    if suffix is None:
        return _invalid_resolution(
            provider_key, "Endpoint path has ambiguous API suffixes."
        )
    prefix_segments, form = suffix

    netloc = _normalized_netloc(hostname, port)
    prefix_path = f"/{'/'.join(prefix_segments)}" if prefix_segments else ""
    root = urlunsplit((scheme, netloc, prefix_path, "", ""))
    normalized_path = f"/{'/'.join(segments)}" if segments else ""
    normalized_input = urlunsplit((scheme, netloc, normalized_path, "", ""))
    chat_url = f"{root}/v1/chat/completions"
    models_url = f"{root}/v1/models"
    persisted_endpoint = root if provider_key in _LLAMA_PROVIDER_KEYS else chat_url
    warnings = (
        (_REMOTE_HTTP_WARNING,)
        if scheme == "http" and not _is_loopback_host(hostname)
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


def _normalize_provider_key(provider: object) -> str:
    if not isinstance(provider, str):
        return ""
    provider_key = re.sub(r"[\s-]+", "_", provider.strip().lower())
    if not _PROVIDER_KEY.fullmatch(provider_key):
        return ""
    if provider_key in _CUSTOM_PROVIDER_ALIASES:
        return "custom"
    if provider_key in _CUSTOM_2_PROVIDER_ALIASES:
        return "custom_2"
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


def _valid_host(hostname: str) -> bool:
    if not hostname.isascii():
        return False
    try:
        ip_address(hostname)
        return True
    except ValueError:
        pass

    host = hostname.rstrip(".")
    if not host or len(host) > 253:
        return False
    return all(_valid_dns_label(label) for label in host.split("."))


def _valid_dns_label(label: str) -> bool:
    return (
        1 <= len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
    )


def _normalized_netloc(hostname: str, port: int | None) -> str:
    host = hostname.lower().rstrip(".")
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{port}" if port is not None else host


def _drop_default_port(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    default_port = (parsed.scheme == "http" and parsed.port == 80) or (
        parsed.scheme == "https" and parsed.port == 443
    )
    if not default_port or parsed.hostname is None:
        return endpoint
    netloc = _normalized_netloc(parsed.hostname, None)
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False
