"""Shared Console provider endpoint display and comparison helpers."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlparse, urlunparse


UNSAVED_ENDPOINT_COPY = "Provider blocked: save the endpoint in Settings before using it from Console."


def first_configured_endpoint(provider_settings: Mapping[str, object]) -> str | None:
    """Return the first configured provider endpoint from known config aliases."""
    for key in ("api_url", "base_url", "api_base", "api_endpoint", "endpoint"):
        value = provider_settings.get(key)
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def generic_endpoint_differs(base_url: str | None, provider_settings: Mapping[str, object]) -> bool:
    """Return whether a session endpoint differs from the persisted provider endpoint."""
    selected_base_url = normalize_generic_endpoint_for_compare(base_url)
    if not selected_base_url:
        return False
    configured_base_url = normalize_generic_endpoint_for_compare(
        first_configured_endpoint(provider_settings)
    )
    return selected_base_url != configured_base_url


def unsaved_endpoint_copy(base_url: str | None, provider_settings: Mapping[str, object]) -> str:
    """Return actionable recovery copy with safe selected/saved endpoint details."""
    selected = safe_endpoint_display(base_url) or "selected session endpoint"
    configured = safe_endpoint_display(first_configured_endpoint(provider_settings)) or "not saved"
    return f"{UNSAVED_ENDPOINT_COPY} Selected endpoint: {selected}. Saved endpoint: {configured}."


def safe_endpoint_display(url: str | None) -> str:
    """Return a credential-free endpoint label safe for user-visible UI."""
    raw_url = str(url or "").strip()
    if not raw_url:
        return ""
    try:
        parsed = urlparse(raw_url)
    except ValueError:
        return raw_url.rstrip("/")
    if parsed.scheme and parsed.netloc:
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        if not hostname:
            return raw_url.rstrip("/")
        host = hostname.lower()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            return raw_url.rstrip("/")
        netloc = f"{host}:{port}" if port is not None else host
        return urlunparse((scheme, netloc, parsed.path.rstrip("/"), "", "", "")).rstrip("/")
    return raw_url.rstrip("/")


def normalize_generic_endpoint_for_compare(url: str | None) -> str:
    """Normalize a generic provider endpoint for stable comparison."""
    raw_url = str(url or "").strip()
    if not raw_url:
        return ""
    try:
        parsed = urlparse(raw_url)
    except ValueError:
        return raw_url.rstrip("/")
    if parsed.scheme and parsed.netloc:
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        if not hostname:
            return raw_url.rstrip("/")
        host = hostname.lower()
        try:
            port = parsed.port
        except ValueError:
            return raw_url.rstrip("/")
        default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host if default_port or port is None else f"{host}:{port}"
        return urlunparse((scheme, netloc, parsed.path.rstrip("/"), "", "", ""))
    return raw_url.rstrip("/")
