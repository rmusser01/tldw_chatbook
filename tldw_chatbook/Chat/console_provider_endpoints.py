"""Shared Console provider endpoint display and comparison helpers."""

from __future__ import annotations

from collections.abc import Mapping
from ipaddress import ip_address
from urllib.parse import urlparse, urlunparse

from tldw_chatbook.Chat.provider_endpoint_contract import (
    URL_BASED_PROVIDER_KEYS,
    canonical_connection_identity,
    resolve_provider_endpoint,
)

UNSAVED_ENDPOINT_COPY = (
    "Provider blocked: save the endpoint in Settings before using it from Console."
)
_ENDPOINT_SETTING_KEYS = (
    "api_base_url",
    "api_base",
    "base_url",
    "api_url",
    "api_endpoint",
    "endpoint",
    "router_base_url",
    "huggingface_router_base_url",
)
_URL_PROVIDER_SETTING_KEYS = (
    "api_base_url",
    "api_base",
    "base_url",
    "api_url",
    "router_base_url",
    "huggingface_router_base_url",
)
_INVALID_ENDPOINT_DISPLAY = "invalid endpoint"
_BUILTIN_PROVIDER_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1",
    "cohere": "https://api.cohere.com",
    "deepseek": "https://api.deepseek.com",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "groq": "https://api.groq.com/openai/v1",
    "huggingface": "https://api-inference.huggingface.co/v1",
    "mistral": "https://api.mistral.ai/v1",
    "mistralai": "https://api.mistral.ai/v1",
    "moonshot": "https://api.moonshot.ai/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "qwencloud": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "zai": "https://api.z.ai/api/paas/v4",
}


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


def effective_provider_endpoint(
    provider_key: str,
    selected_endpoint: str | None,
    provider_settings: Mapping[str, object],
) -> str | None:
    """Resolve the exact endpoint a provider call would use at this moment.

    Explicit Console selection wins, followed by the provider's configured
    endpoint aliases and finally the adapter's built-in cloud default. Custom
    OpenAI-compatible endpoint forms resolve to the chat URL actually used.

    Args:
        provider_key: Normalized provider readiness key.
        selected_endpoint: Optional Console-selected endpoint.
        provider_settings: Provider-specific configuration mapping.

    Returns:
        Effective endpoint to pin into the provider resolution, or ``None``
        when the provider has neither a configured nor built-in endpoint.
    """
    if isinstance(selected_endpoint, str) and selected_endpoint.strip():
        return _effective_contract_endpoint(provider_key, selected_endpoint)
    if provider_key == "huggingface" and _huggingface_router_mode(provider_settings):
        for key in ("router_base_url", "huggingface_router_base_url"):
            router_endpoint = provider_settings.get(key)
            if isinstance(router_endpoint, str) and router_endpoint.strip():
                return router_endpoint.strip()
        return builtin_provider_endpoint(provider_key, provider_settings)
    configured_endpoint = first_configured_endpoint(provider_settings)
    if configured_endpoint:
        return _effective_contract_endpoint(provider_key, configured_endpoint)
    return builtin_provider_endpoint(provider_key, provider_settings)


def effective_provider_discovery_endpoint(
    provider_key: str,
    selected_endpoint: str | None,
    provider_settings: Mapping[str, object],
) -> str | None:
    """Resolve the settings-aware OpenAI-compatible model-listing base."""

    endpoint = effective_provider_endpoint(
        provider_key,
        selected_endpoint,
        provider_settings,
    )
    if endpoint is None or provider_key != "huggingface":
        return endpoint
    if not _huggingface_router_mode(provider_settings):
        return endpoint
    try:
        parsed = urlparse(endpoint)
    except ValueError:
        return endpoint
    if (parsed.hostname or "").lower() != "router.huggingface.co":
        return endpoint
    return urlunparse((parsed.scheme, parsed.netloc, "/v1", "", "", ""))


def builtin_provider_endpoint(
    provider_key: str,
    provider_settings: Mapping[str, object] | None = None,
) -> str | None:
    """Return the canonical adapter fallback endpoint for a provider."""

    settings = provider_settings or {}
    if (
        provider_key == "moonshot"
        and str(settings.get("api_region", "")).lower() == "china"
    ):
        return "https://api.moonshot.cn/v1"
    if provider_key == "huggingface" and _huggingface_router_mode(settings):
        return "https://router.huggingface.co/hf-inference"
    return _BUILTIN_PROVIDER_ENDPOINTS.get(provider_key)


def _huggingface_router_mode(provider_settings: Mapping[str, object]) -> bool:
    return (
        str(
            provider_settings.get(
                "use_router_url_format",
                provider_settings.get("huggingface_use_router_url_format", "False"),
            )
        ).lower()
        == "true"
    )


def provider_uses_endpoint(
    provider_key: str, provider_settings: Mapping[str, object]
) -> bool:
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


def generic_endpoint_differs(
    base_url: str | None, provider_settings: Mapping[str, object]
) -> bool:
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


def unsaved_endpoint_copy(
    base_url: str | None, provider_settings: Mapping[str, object]
) -> str:
    """Return actionable recovery copy with safe endpoint details.

    Args:
        base_url: Session-selected provider endpoint.
        provider_settings: Provider-specific configuration mapping.

    Returns:
        User-visible recovery copy that omits credentials, query strings, and
        fragments from endpoint values.
    """
    selected = safe_endpoint_display(base_url) or "selected session endpoint"
    configured = (
        safe_endpoint_display(first_configured_endpoint(provider_settings))
        or "not saved"
    )
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
    resolution = resolve_provider_endpoint("custom", url)
    if resolution.persisted_endpoint is not None:
        return resolution.normalized_input
    legacy_display = _format_endpoint(parsed_endpoint, drop_default_port=False)
    has_scheme = parsed_endpoint[0]
    legacy_resolution = resolve_provider_endpoint("custom", legacy_display)
    if legacy_resolution.persisted_endpoint is not None:
        return legacy_resolution.normalized_input if has_scheme else legacy_display

    if not has_scheme:
        explicit_legacy_resolution = resolve_provider_endpoint(
            "custom",
            f"http://{legacy_display}",
        )
        if explicit_legacy_resolution.persisted_endpoint is not None:
            return legacy_display
    return _INVALID_ENDPOINT_DISPLAY


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
    identity = canonical_connection_identity("custom", url)
    if identity is not None:
        return identity[1]
    return _format_endpoint(parsed_endpoint, drop_default_port=True)


def _effective_contract_endpoint(provider_key: str, endpoint: str) -> str:
    """Resolve endpoint forms only for adapters governed by the new contract."""
    if provider_key not in {"custom", "custom_2", "llama_cpp", "local_llamacpp"}:
        return endpoint
    resolution = resolve_provider_endpoint(provider_key, endpoint)
    return resolution.persisted_endpoint or endpoint


def _parse_http_endpoint(
    url: str | None,
) -> tuple[bool, str, str, int | None, str] | None:
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


def _is_allowed_endpoint_host(
    hostname: str, *, has_scheme: bool, port: int | None
) -> bool:
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
        and all(
            character.isascii() and (character.isalnum() or character == "-")
            for character in label
        )
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
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    netloc = (
        host
        if port is None or (drop_default_port and default_port)
        else f"{host}:{port}"
    )
    if has_scheme:
        return urlunparse((scheme, netloc, path, "", "", "")).rstrip("/")
    return f"{netloc}{path}".rstrip("/")
