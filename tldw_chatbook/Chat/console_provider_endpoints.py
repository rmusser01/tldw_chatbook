"""Shared Console provider endpoint display and comparison helpers."""

from __future__ import annotations

from collections.abc import Mapping
from ipaddress import ip_address
from urllib.parse import urlparse, urlunparse


UNSAVED_ENDPOINT_COPY = "Provider blocked: save the endpoint in Settings before using it from Console."
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
        "tabbyapi",
        "vllm",
    }
)
_ENDPOINT_SETTING_KEYS = (
    "api_base_url",
    "api_base",
    "base_url",
    "api_url",
    "api_endpoint",
    "endpoint",
)
_URL_PROVIDER_SETTING_KEYS = ("api_base_url", "api_base", "base_url", "api_url")
_INVALID_ENDPOINT_DISPLAY = "invalid endpoint"


def first_configured_endpoint(provider_settings: Mapping[str, object]) -> str | None:
    """Return the first configured provider endpoint from known config aliases.

    Args:
        provider_settings: Provider-specific configuration mapping.

    Returns:
        The first non-empty endpoint string, or ``None`` when no endpoint is
        configured.
    """
    for key in _ENDPOINT_SETTING_KEYS:
        value = provider_settings.get(key)
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def provider_uses_endpoint(provider_key: str, provider_settings: Mapping[str, object]) -> bool:
    """Return whether a provider should validate saved endpoint overrides.

    Args:
        provider_key: Normalized provider readiness key.
        provider_settings: Provider-specific configuration mapping.

    Returns:
        ``True`` when the provider is known to use endpoint URLs or has a saved
        base URL setting.
    """
    return provider_key in URL_BASED_PROVIDER_KEYS or any(
        key in provider_settings for key in _URL_PROVIDER_SETTING_KEYS
    )


def generic_endpoint_differs(base_url: str | None, provider_settings: Mapping[str, object]) -> bool:
    """Return whether a session endpoint differs from the persisted endpoint.

    Args:
        base_url: Session-selected provider endpoint.
        provider_settings: Provider-specific configuration mapping.

    Returns:
        ``True`` when both values normalize to different endpoint identities.
    """
    selected_base_url = normalize_generic_endpoint_for_compare(base_url)
    if not selected_base_url:
        return False
    configured_base_url = normalize_generic_endpoint_for_compare(
        first_configured_endpoint(provider_settings)
    )
    return selected_base_url != configured_base_url


def unsaved_endpoint_copy(base_url: str | None, provider_settings: Mapping[str, object]) -> str:
    """Return actionable recovery copy with safe endpoint details.

    Args:
        base_url: Session-selected provider endpoint.
        provider_settings: Provider-specific configuration mapping.

    Returns:
        User-visible recovery copy that omits credentials, query strings, and
        fragments from endpoint values.
    """
    selected = safe_endpoint_display(base_url) or "selected session endpoint"
    configured = safe_endpoint_display(first_configured_endpoint(provider_settings)) or "not saved"
    return f"{UNSAVED_ENDPOINT_COPY} Selected endpoint: {selected}. Saved endpoint: {configured}."


def safe_endpoint_display(url: str | None) -> str:
    """Return a credential-free endpoint label safe for user-visible UI.

    Args:
        url: Raw endpoint value from config or user input.

    Returns:
        A host/path endpoint label with user info, query strings, and fragments
        removed. Malformed endpoints return ``"invalid endpoint"`` instead of
        echoing raw input.
    """
    parsed_endpoint = _parse_http_endpoint(url)
    if parsed_endpoint is None:
        return "" if not str(url or "").strip() else _INVALID_ENDPOINT_DISPLAY
    return _format_endpoint(parsed_endpoint, drop_default_port=False)


def normalize_generic_endpoint_for_compare(url: str | None) -> str:
    """Normalize a generic provider endpoint for stable comparison.

    Args:
        url: Raw endpoint value from config or user input.

    Returns:
        Normalized endpoint identity. Empty input returns an empty string, while
        malformed input returns a non-secret invalid sentinel.
    """
    parsed_endpoint = _parse_http_endpoint(url)
    if parsed_endpoint is None:
        return "" if not str(url or "").strip() else _INVALID_ENDPOINT_DISPLAY
    return _format_endpoint(parsed_endpoint, drop_default_port=True)


def _parse_http_endpoint(url: str | None) -> tuple[bool, str, str, int | None, str] | None:
    raw_value = str(url or "")
    raw_url = raw_value.strip()
    if not raw_url:
        return None
    has_unsafe_character = any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in raw_value
    )
    if has_unsafe_character:
        return None
    has_scheme = "://" in raw_url
    candidate = raw_url if has_scheme else f"http://{raw_url}"
    try:
        parsed = urlparse(candidate)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if scheme not in {"http", "https"} or not hostname:
        return None
    hostname = hostname.lower()
    if not _is_allowed_endpoint_host(hostname, has_scheme=has_scheme, port=port):
        return None
    path = parsed.path.rstrip("/")
    return (has_scheme, scheme, hostname, port, path)


def _is_allowed_endpoint_host(hostname: str, *, has_scheme: bool, port: int | None) -> bool:
    if hostname == "localhost":
        return True
    try:
        ip_address(hostname)
        return True
    except ValueError:
        pass
    if _is_dotted_dns_name(hostname):
        return True
    return not has_scheme and port is not None


def _is_dotted_dns_name(hostname: str) -> bool:
    hostname = hostname.rstrip(".")
    if "." not in hostname or len(hostname) > 253:
        return False
    labels = hostname.split(".")
    return all(_is_dns_label(label) for label in labels)


def _is_dns_label(label: str) -> bool:
    return (
        1 <= len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isascii() and (character.isalnum() or character == "-") for character in label)
    )


def _format_endpoint(
    endpoint: tuple[bool, str, str, int | None, str],
    *,
    drop_default_port: bool,
) -> str:
    has_scheme, scheme, hostname, port, path = endpoint
    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if port is None or (drop_default_port and default_port) else f"{host}:{port}"
    if has_scheme:
        return urlunparse((scheme, netloc, path, "", "", "")).rstrip("/")
    return f"{netloc}{path}".rstrip("/")
