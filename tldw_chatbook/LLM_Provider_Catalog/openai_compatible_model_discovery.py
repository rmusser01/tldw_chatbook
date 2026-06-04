"""OpenAI-compatible endpoint model discovery helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import ParseResult, urlparse, urlunparse

import httpx

from tldw_chatbook.Chat.console_session_settings import normalize_llamacpp_base_url
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    DiscoveryErrorKind,
    ModelDiscoveryError,
    ModelDiscoveryResult,
)


_NATIVE_ENDPOINT_PATHS_BY_PROVIDER = {
    "koboldcpp": frozenset({"/api/v1/generate"}),
    "ollama": frozenset({"/api/tags"}),
}
_EXACT_SENSITIVE_METADATA_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "bearer",
        "client_secret",
        "key",
        "password",
        "secret",
        "token",
        "x_api_key",
        "x-api-key",
    }
)
_SENSITIVE_METADATA_KEY_SUBSTRINGS = frozenset(
    {
        "api_key",
        "access_token",
        "client_secret",
        "password",
        "secret",
        "x_api_key",
    }
)


def _normalized_provider_identity(provider_identity: str | None) -> str:
    """Return a stable provider identity string for endpoint policy checks."""
    return (provider_identity or "").strip().lower().replace("-", "_")


def _parse_endpoint(endpoint: str | None) -> ParseResult | None:
    """Parse a configured endpoint, accepting host-only local URLs."""
    raw_endpoint = str(endpoint or "").strip()
    if not raw_endpoint:
        return None
    candidate = raw_endpoint if "://" in raw_endpoint else f"http://{raw_endpoint}"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return parsed


def _normalized_path(parsed: ParseResult) -> str:
    """Return a lower-case endpoint path without a trailing slash."""
    path = (parsed.path or "").rstrip("/").lower()
    return path or "/"


def _safe_netloc(parsed: ParseResult) -> str:
    """Return host[:port] without any embedded credentials."""
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{host}:{parsed.port}" if parsed.port else host


def _models_path_for_endpoint_path(path: str) -> str | None:
    """Return the OpenAI-compatible models path for a supported endpoint path."""
    normalized_path = (path or "/").rstrip("/").lower() or "/"
    if normalized_path == "/":
        return "/v1/models"
    if normalized_path in {"/models", "/v1/models"}:
        return normalized_path
    if normalized_path == "/v1":
        return "/v1/models"
    if normalized_path in {"/completion", "/completions"}:
        return "/v1/models"
    if normalized_path.endswith("/v1/chat/completions"):
        return f"{normalized_path.removesuffix('/chat/completions')}/models"
    if normalized_path.endswith("/chat/completions"):
        return f"{normalized_path.removesuffix('/chat/completions')}/models"
    return None


def supports_openai_compatible_model_discovery(
    provider_identity: str,
    normalized_endpoint: str | None,
) -> bool:
    """Return whether an endpoint shape supports OpenAI-compatible discovery.

    Eligibility is based on explicit OpenAI-compatible URL paths. Native
    provider discovery URLs are rejected even when the provider can also expose
    an OpenAI-compatible API at another configured endpoint.
    """
    parsed = _parse_endpoint(normalized_endpoint)
    if parsed is None:
        return False

    provider_key = _normalized_provider_identity(provider_identity)
    path = _normalized_path(parsed)
    native_paths = _NATIVE_ENDPOINT_PATHS_BY_PROVIDER.get(provider_key, frozenset())
    if path in native_paths:
        return False

    return _models_path_for_endpoint_path(path) is not None


def build_models_url(endpoint: str, provider_identity: str) -> str:
    """Return the OpenAI-compatible models endpoint for a configured URL."""
    parsed = _parse_endpoint(endpoint)
    if parsed is None:
        return str(endpoint or "").strip()

    path = _normalized_path(parsed)
    if path in {"/completion", "/completions"}:
        base_url = normalize_llamacpp_base_url(endpoint)
        base_parsed = _parse_endpoint(base_url)
        if base_parsed is not None:
            return urlunparse(
                (base_parsed.scheme, _safe_netloc(base_parsed), "/v1/models", "", "", "")
            )

    models_path = _models_path_for_endpoint_path(path) or path

    return urlunparse((parsed.scheme, _safe_netloc(parsed), models_path, "", "", ""))


def fingerprint_endpoint(endpoint: str) -> str:
    """Return a safe endpoint fingerprint without secrets or query details."""
    parsed = _parse_endpoint(endpoint)
    if parsed is None:
        return str(endpoint or "").split("?", 1)[0].split("#", 1)[0].strip()

    path = (parsed.path or "").rstrip("/") or "/"
    return urlunparse((parsed.scheme, _safe_netloc(parsed), path, "", "", ""))


def _is_sensitive_metadata_key(key: object) -> bool:
    """Return whether a metadata key looks credential-bearing."""
    normalized_key = str(key).strip().lower().replace("-", "_")
    return normalized_key in _EXACT_SENSITIVE_METADATA_KEYS or any(
        sensitive_key in normalized_key
        for sensitive_key in _SENSITIVE_METADATA_KEY_SUBSTRINGS
    )


def _scrub_model_metadata_value(value: Any) -> Any:
    """Recursively remove sensitive-looking fields from endpoint metadata."""
    if isinstance(value, Mapping):
        return {
            str(key): _scrub_model_metadata_value(nested_value)
            for key, nested_value in value.items()
            if not _is_sensitive_metadata_key(key)
        }
    if isinstance(value, list):
        return [_scrub_model_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_model_metadata_value(item) for item in value)
    return value


def _safe_model_metadata(model_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy endpoint model metadata while dropping sensitive-looking fields."""
    return {
        str(key): _scrub_model_metadata_value(value)
        for key, value in model_payload.items()
        if not _is_sensitive_metadata_key(key)
    }


def normalize_models_response(
    payload: Mapping[str, Any],
    *,
    provider: str,
    provider_list_key: str,
    endpoint_fingerprint: str,
    now_iso: str,
) -> tuple[DiscoveredModel, ...]:
    """Normalize an OpenAI ``/models`` response into discovery contracts."""
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, list):
        raise ValueError("Invalid models response: expected data list")

    seen_model_ids: set[str] = set()
    models: list[DiscoveredModel] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError("Invalid models response: expected model objects")

        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Invalid models response: model id is required")

        model_id = model_id.strip()
        if model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)
        models.append(
            DiscoveredModel(
                provider=provider,
                provider_list_key=provider_list_key,
                model_id=model_id,
                display_name=model_id,
                source="runtime_discovered",
                endpoint_fingerprint=endpoint_fingerprint,
                discovered_at=now_iso,
                metadata_raw_safe=_safe_model_metadata(item),
            )
        )

    return tuple(models)


def _discovery_error(
    kind: DiscoveryErrorKind,
    message: str,
    recovery_hint: str,
) -> ModelDiscoveryError:
    """Build a typed safe discovery error."""
    return ModelDiscoveryError(
        kind=kind,
        message=message,
        recovery_hint=recovery_hint,
    )


async def discover_openai_compatible_models(
    *,
    provider: str,
    provider_list_key: str,
    endpoint: str,
    api_key: str | None,
    timeout_seconds: float = 10.0,
    client: httpx.AsyncClient | None = None,
) -> ModelDiscoveryResult:
    """Manually discover models from one configured OpenAI-compatible endpoint."""
    endpoint_fingerprint = fingerprint_endpoint(endpoint) if endpoint else None
    if not supports_openai_compatible_model_discovery(provider, endpoint):
        return ModelDiscoveryResult(
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint_fingerprint=endpoint_fingerprint,
            status="unsupported",
            error=_discovery_error(
                "unsupported_endpoint",
                "This endpoint is not an OpenAI-compatible models endpoint.",
                "Configure an explicit /v1 or /v1/models endpoint for discovery.",
            ),
        )

    models_url = build_models_url(endpoint, provider)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=timeout_seconds)
    try:
        try:
            response = await active_client.get(models_url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError:
            return ModelDiscoveryResult(
                provider=provider,
                provider_list_key=provider_list_key,
                endpoint_fingerprint=endpoint_fingerprint,
                status="error",
                error=_discovery_error(
                    "request_failed",
                    "Model discovery request failed.",
                    "Check the endpoint URL, server availability, and credentials.",
                ),
            )

        try:
            payload = response.json()
        except ValueError:
            return ModelDiscoveryResult(
                provider=provider,
                provider_list_key=provider_list_key,
                endpoint_fingerprint=endpoint_fingerprint,
                status="error",
                error=_discovery_error(
                    "invalid_response",
                    "The models endpoint did not return valid JSON.",
                    "Use an endpoint that returns a JSON object with a data array of model IDs.",
                ),
            )
    finally:
        if owns_client:
            await active_client.aclose()

    now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        models = normalize_models_response(
            payload,
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint_fingerprint=endpoint_fingerprint or "",
            now_iso=now_iso,
        )
    except ValueError:
        return ModelDiscoveryResult(
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint_fingerprint=endpoint_fingerprint,
            status="error",
            error=_discovery_error(
                "invalid_response",
                "The models endpoint did not return a valid OpenAI-compatible response.",
                "Use an endpoint that returns a JSON object with a data array of model IDs.",
            ),
        )

    return ModelDiscoveryResult(
        provider=provider,
        provider_list_key=provider_list_key,
        endpoint_fingerprint=endpoint_fingerprint,
        status="success",
        models=models,
    )
