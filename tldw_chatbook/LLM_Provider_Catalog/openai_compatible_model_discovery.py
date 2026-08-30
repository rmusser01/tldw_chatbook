"""OpenAI-compatible endpoint model discovery helpers."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from ipaddress import IPv6Address
from typing import Any
from urllib.parse import ParseResult, unquote, urlparse, urlunparse

import httpx

from tldw_chatbook.Chat.console_session_settings import normalize_llamacpp_base_url
from tldw_chatbook.Chat.local_server_discovery import (
    MODEL_ID_MAX_CHARS,
    MODEL_IDS_MAX_COUNT,
    MODEL_PROBE_RESPONSE_MAX_BYTES,
    UnsupportedModelResponseEncoding,
    read_bounded_model_response,
)
from tldw_chatbook.LLM_Calls.qwencloud_url import (
    QwenCloudBaseURLValidationError,
    normalize_qwencloud_base_url,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    DiscoveryErrorKind,
    ModelDiscoveryError,
    ModelDiscoveryResult,
)
from tldw_chatbook.Utils.input_validation import validate_url
from tldw_chatbook.Utils.tls_trust import build_httpx_async_client

_NATIVE_ENDPOINT_PATHS_BY_PROVIDER = {
    "koboldcpp": frozenset({"/api/v1/generate"}),
    "ollama": frozenset({"/api/tags"}),
    "local_ollama": frozenset({"/api/tags"}),
}
_QWENCLOUD_PROVIDER_KEY = "qwencloud"
_MAX_ENDPOINT_LENGTH = 2000
_MAX_GENERIC_PATH_DECODE_PASSES = 2
_GENERIC_ENDPOINT_TAILS = (("models",), ("responses",), ("chat", "completions"))
_GENERIC_REQUEST_ENDPOINT_TAILS = _GENERIC_ENDPOINT_TAILS[1:]
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_ENCODED_PATH_SEPARATOR_RE = re.compile(r"%(?:2[fF]|5[cC])")
_ZONE_ID_RE = re.compile(r"[A-Za-z0-9._~-]+")
_BASE_URL_INFERABLE_PROVIDER_KEYS = frozenset(
    {
        "aphrodite",
        "custom",
        "custom_2",
        "custom_openai_api",
        "custom_openai_api_2",
        "llama_cpp",
        "local_llamacpp",
        "local_llamafile",
        "local_llm",
        "local_vllm",
        "openai",
        "openrouter",
        "qwencloud",
        "tabbyapi",
        "vllm",
    }
)
_EXPLICIT_OPENAI_COMPATIBLE_ENDPOINT_PATHS = frozenset(
    {
        "/models",
        "/v1",
        "/v1/models",
        "/completion",
        "/completions",
        "/api/v1",
        "/api/paas/v4",
    }
)
_EXACT_SENSITIVE_METADATA_KEYS = frozenset(
    {
        "access_token",
        "auth_token",
        "authorization",
        "api_key",
        "apikey",
        "bearer",
        "client_secret",
        "credential",
        "credentials",
        "id_token",
        "key",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
        "x_api_key",
        "x-api-key",
    }
)
_SENSITIVE_METADATA_KEY_SUBSTRINGS = frozenset(
    {
        "access_token",
        "auth_token",
        "api_key",
        "client_secret",
        "credential",
        "id_token",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "x_api_key",
    }
)
_COMPACT_SENSITIVE_METADATA_KEYS = frozenset(
    sensitive_key.replace("_", "").replace("-", "")
    for sensitive_key in _EXACT_SENSITIVE_METADATA_KEYS
)
_COMPACT_SENSITIVE_METADATA_KEY_SUBSTRINGS = frozenset(
    sensitive_key.replace("_", "").replace("-", "")
    for sensitive_key in _SENSITIVE_METADATA_KEY_SUBSTRINGS
)
_COMPACT_SENSITIVE_METADATA_KEY_SUFFIXES = frozenset({"token"})

_ANTHROPIC_PROVIDER_KEY = "anthropic"
_ANTHROPIC_VERSION_HEADER = "2023-06-01"
MODEL_DISCOVERY_RESPONSE_MAX_BYTES = MODEL_PROBE_RESPONSE_MAX_BYTES
# Deliberately NOT MODEL_IDS_MAX_COUNT (100). That constant bounds how many ids
# the *probe* truncates to for a reachability sample; discovery instead fails
# closed on an over-bound catalog so no partial list is ever cached. Aliasing
# the two put the fail-closed bound below reality: api.openai.com returns 128
# models for an ordinary account, so live first-run discovery errored with
# "The models endpoint returned too many models" for every OpenAI user with a
# valid key. The fail-closed semantics are kept; only the calibration changes.
# MODEL_DISCOVERY_RESPONSE_MAX_BYTES (1 MiB) remains the real memory bound.
DISCOVERED_MODEL_MAX_COUNT = 512
DISCOVERED_MODEL_ID_MAX_CHARS = MODEL_ID_MAX_CHARS
MODEL_METADATA_MAX_DEPTH = 8
MODEL_METADATA_MAX_ITEMS = 256
MODEL_METADATA_MAX_SERIALIZED_BYTES = 16 * 1024
MODEL_METADATA_MAX_VALUE_CHARS = 4096
MODEL_METADATA_MAX_KEY_CHARS = 128

# Page size, not a total bound -- the two were the same constant only while
# DISCOVERED_MODEL_MAX_COUNT happened to equal 100. Anthropic's list-models
# `limit` is a per-request page size; the total stays bounded by
# DISCOVERED_MODEL_MAX_COUNT across _ANTHROPIC_MAX_MODEL_PAGES pages.
_ANTHROPIC_MODELS_PAGE_LIMIT = 100
_ANTHROPIC_MAX_MODEL_PAGES = 10


def build_discovery_auth_headers(
    provider_identity: str, api_key: str | None
) -> dict[str, str]:
    """Return provider-appropriate auth headers for a models request.

    Args:
        provider_identity: Provider name/identity used to pick the auth scheme
            (Anthropic gets x-api-key; everything else gets a Bearer token).
        api_key: The provider API key, or None for unauthenticated access.

    Returns:
        dict[str, str]: Headers for the request; empty when no key is given.
    """
    if not api_key:
        return {}
    if _normalized_provider_identity(provider_identity) == _ANTHROPIC_PROVIDER_KEY:
        return {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION_HEADER,
        }
    return {"Authorization": f"Bearer {api_key}"}


def _normalized_provider_identity(provider_identity: str | None) -> str:
    """Return a stable provider identity string for endpoint policy checks."""
    return (provider_identity or "").strip().lower().replace(" ", "_").replace("-", "_")


def _parse_endpoint(endpoint: str | None) -> ParseResult | None:
    """Parse a configured endpoint, accepting host-only local URLs."""
    raw_endpoint = str(endpoint or "").strip()
    if not raw_endpoint or len(raw_endpoint) > _MAX_ENDPOINT_LENGTH:
        return None
    candidate = raw_endpoint if "://" in raw_endpoint else f"http://{raw_endpoint}"
    if "\\" in candidate or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in candidate
    ):
        return None
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
    ):
        return None
    try:
        parsed.port
    except ValueError:
        return None
    return parsed if _is_structurally_safe_generic_endpoint(parsed) else None


def _is_structurally_safe_generic_endpoint(parsed: ParseResult) -> bool:
    """Validate generic discovery structure without provider suffix policy."""
    if not _is_safe_generic_authority(parsed.netloc):
        return False
    if not _is_safe_generic_path(parsed.path):
        return False
    safe_url = urlunparse(
        (parsed.scheme, _safe_netloc(parsed), parsed.path or "/", "", "", "")
    )
    return validate_url(safe_url)


def _is_safe_generic_authority(netloc: str) -> bool:
    """Allow percent only for an RFC 6874 bracketed IPv6 zone identifier."""
    if any(character in netloc for character in '\\|^{}<>"`') or netloc.endswith(":"):
        return False
    if "%" not in netloc:
        return True

    host_port = netloc.rsplit("@", 1)[-1]
    closing_bracket = host_port.find("]")
    if not host_port.startswith("[") or closing_bracket < 0:
        return False
    bracketed_host = host_port[1:closing_bracket]
    remainder = host_port[closing_bracket + 1 :]
    if bracketed_host.count("%25") != 1 or (
        remainder and not re.fullmatch(r":\d+", remainder)
    ):
        return False
    address, zone_id = bracketed_host.split("%25", 1)
    try:
        IPv6Address(address)
    except ValueError:
        return False
    return _ZONE_ID_RE.fullmatch(zone_id) is not None


def _is_safe_generic_path(path: str) -> bool:
    """Reject parser-ambiguous path structure without rewriting the URL."""
    if (
        "//" in path
        or re.search(r"%(?![0-9A-Fa-f]{2})", path) is not None
        or any(segment in {".", ".."} for segment in path.split("/"))
        or _has_unsafe_generic_endpoint_tail_structure(path)
    ):
        return False

    validation_path = path
    for _pass in range(_MAX_GENERIC_PATH_DECODE_PASSES):
        if _ENCODED_PATH_SEPARATOR_RE.search(validation_path):
            return False
        try:
            decoded_path = unquote(validation_path, errors="strict")
        except UnicodeDecodeError:
            return False
        if decoded_path == validation_path:
            break
        if (
            any(
                ord(character) < 32 or ord(character) == 127
                for character in decoded_path
            )
            or any(segment in {".", ".."} for segment in decoded_path.split("/"))
            or _has_unsafe_generic_endpoint_tail_structure(decoded_path)
        ):
            return False
        validation_path = decoded_path
    return _PERCENT_ESCAPE_RE.search(validation_path) is None


def _has_unsafe_generic_endpoint_tail_structure(path: str) -> bool:
    """Reject repeated or non-terminal generic request endpoint tails."""
    segments = tuple(segment.lower() for segment in path.strip("/").split("/"))
    request_tails = [
        (tail, index + len(tail))
        for tail in _GENERIC_REQUEST_ENDPOINT_TAILS
        for index in range(len(segments) - len(tail) + 1)
        if segments[index : index + len(tail)] == tail
    ]
    if len(request_tails) > 1 or any(
        end != len(segments) for _tail, end in request_tails
    ):
        return True
    return any(
        segments[-len(first + second) :] == first + second
        for first in _GENERIC_ENDPOINT_TAILS
        for second in _GENERIC_ENDPOINT_TAILS
    )


def _normalized_path(parsed: ParseResult) -> str:
    """Return a lower-case endpoint path without a trailing slash."""
    path = (parsed.path or "").rstrip("/").lower()
    return path or "/"


def _safe_netloc(parsed: ParseResult) -> str:
    """Return host[:port] without any embedded credentials."""
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    return f"{host}:{port}" if port else host


def _parse_endpoint_for_fingerprint(endpoint: str | None) -> ParseResult | None:
    """Parse any URL-like endpoint so safe display can strip credentials."""
    raw_endpoint = str(endpoint or "").strip()
    if not raw_endpoint or len(raw_endpoint) > _MAX_ENDPOINT_LENGTH:
        return None
    candidate = raw_endpoint if "://" in raw_endpoint else f"http://{raw_endpoint}"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    return parsed if parsed.scheme and parsed.netloc and parsed.hostname else None


def _models_path_for_endpoint_path(path: str) -> str | None:
    """Return the OpenAI-compatible models path for a supported endpoint path."""
    normalized_path = (path or "/").rstrip("/").lower() or "/"
    if normalized_path == "/":
        return "/v1/models"
    if normalized_path in {"/models", "/v1/models"}:
        return normalized_path
    if normalized_path == "/v1":
        return "/v1/models"
    if normalized_path in {"/api/v1", "/api/paas/v4"}:
        return f"{normalized_path}/models"
    if normalized_path in {"/completion", "/completions"}:
        return "/v1/models"
    if normalized_path.endswith("/v1/chat/completions"):
        return f"{normalized_path.removesuffix('/chat/completions')}/models"
    if normalized_path.endswith("/chat/completions"):
        return f"{normalized_path.removesuffix('/chat/completions')}/models"
    return None


def _models_path_preserving_encoding(path: str) -> str:
    """Return the models path without decoding or recasing its base prefix."""
    raw_path = (path or "/").rstrip("/") or "/"
    normalized_path = raw_path.lower()
    normalized_models_path = _models_path_for_endpoint_path(normalized_path)
    if normalized_models_path is None:
        return raw_path
    if normalized_path == "/" or normalized_path in {"/completion", "/completions"}:
        return normalized_models_path
    if normalized_path.endswith("/chat/completions"):
        return f"{raw_path[: -len('/chat/completions')]}/models"
    if normalized_path.endswith("/models"):
        return raw_path
    return f"{raw_path}/models"


def _is_base_url_path(path: str) -> bool:
    """Return whether a path requires provider identity to infer ``/v1/models``."""
    normalized_path = (path or "/").rstrip("/").lower() or "/"
    return normalized_path == "/"


def _is_explicit_openai_compatible_path(path: str) -> bool:
    """Return whether a path explicitly opts into OpenAI-compatible discovery."""
    normalized_path = (path or "/").rstrip("/").lower() or "/"
    return normalized_path in _EXPLICIT_OPENAI_COMPATIBLE_ENDPOINT_PATHS or (
        normalized_path.endswith("/chat/completions")
    )


def supports_openai_compatible_model_discovery(
    provider_identity: str,
    normalized_endpoint: str | None,
) -> bool:
    """Return whether an endpoint shape supports OpenAI-compatible discovery.

    Eligibility is based on explicit OpenAI-compatible URL paths. Native
    provider discovery URLs are rejected even when the provider can also expose
    an OpenAI-compatible API at another configured endpoint.
    """
    provider_key = _normalized_provider_identity(provider_identity)
    if provider_key == _QWENCLOUD_PROVIDER_KEY:
        try:
            normalize_qwencloud_base_url(normalized_endpoint)
        except QwenCloudBaseURLValidationError:
            return False
        return True

    parsed = _parse_endpoint(normalized_endpoint)
    if parsed is None:
        return False

    path = _normalized_path(parsed)
    native_paths = _NATIVE_ENDPOINT_PATHS_BY_PROVIDER.get(provider_key, frozenset())
    if path in native_paths:
        return False

    if _is_base_url_path(path):
        return provider_key in _BASE_URL_INFERABLE_PROVIDER_KEYS

    return _is_explicit_openai_compatible_path(path) and (
        _models_path_for_endpoint_path(path) is not None
    )


def build_models_url(endpoint: str, provider_identity: str) -> str:
    """Return the OpenAI-compatible models endpoint for a configured URL."""
    if _normalized_provider_identity(provider_identity) == _QWENCLOUD_PROVIDER_KEY:
        try:
            base_url = normalize_qwencloud_base_url(endpoint)
        except QwenCloudBaseURLValidationError:
            return fingerprint_endpoint(endpoint)
        return f"{base_url}/models"

    parsed = _parse_endpoint(endpoint)
    if parsed is None:
        return str(endpoint or "").strip()

    path = _normalized_path(parsed)
    if path in {"/completion", "/completions"}:
        base_url = normalize_llamacpp_base_url(endpoint)
        base_parsed = _parse_endpoint(base_url)
        if base_parsed is not None:
            return urlunparse(
                (
                    base_parsed.scheme,
                    _safe_netloc(base_parsed),
                    "/v1/models",
                    "",
                    "",
                    "",
                )
            )

    models_path = _models_path_preserving_encoding(parsed.path)

    return urlunparse((parsed.scheme, _safe_netloc(parsed), models_path, "", "", ""))


def fingerprint_endpoint(endpoint: str) -> str:
    """Return a safe endpoint fingerprint without secrets or query details."""
    parsed = _parse_endpoint(endpoint)
    if parsed is None:
        display_parsed = _parse_endpoint_for_fingerprint(endpoint)
        if display_parsed is not None:
            path = (display_parsed.path or "").rstrip("/") or "/"
            return urlunparse(
                (display_parsed.scheme, _safe_netloc(display_parsed), path, "", "", "")
            )
        raw_fingerprint = str(endpoint or "").split("?", 1)[0].split("#", 1)[0].strip()
        if "@" in raw_fingerprint:
            scheme, separator, _rest = raw_fingerprint.partition("://")
            return (
                f"{scheme}{separator}[invalid-endpoint]"
                if separator
                else "[invalid-endpoint]"
            )
        return raw_fingerprint

    path = (parsed.path or "").rstrip("/") or "/"
    return urlunparse((parsed.scheme, _safe_netloc(parsed), path, "", "", ""))


def _is_sensitive_metadata_key(key: object) -> bool:
    """Return whether a metadata key looks credential-bearing."""
    normalized_key = str(key).strip().lower().replace("-", "_")
    compact_key = normalized_key.replace("_", "")
    return (
        normalized_key in _EXACT_SENSITIVE_METADATA_KEYS
        or any(
            sensitive_key in normalized_key
            for sensitive_key in _SENSITIVE_METADATA_KEY_SUBSTRINGS
        )
        or compact_key in _COMPACT_SENSITIVE_METADATA_KEYS
        or any(
            sensitive_key in compact_key
            for sensitive_key in _COMPACT_SENSITIVE_METADATA_KEY_SUBSTRINGS
        )
        or any(
            compact_key.endswith(sensitive_suffix)
            for sensitive_suffix in _COMPACT_SENSITIVE_METADATA_KEY_SUFFIXES
        )
    )


def _scrub_model_metadata_value(
    value: Any,
    *,
    depth: int,
    budget: list[int],
) -> Any:
    """Return bounded JSON-like metadata with credential-looking fields removed."""

    if depth > MODEL_METADATA_MAX_DEPTH:
        raise ValueError("Invalid models response: metadata is too deep")
    budget[0] += 1
    if budget[0] > MODEL_METADATA_MAX_ITEMS:
        raise ValueError("Invalid models response: metadata has too many items")
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, nested_value in value.items():
            if type(key) is not str or len(key) > MODEL_METADATA_MAX_KEY_CHARS:
                raise ValueError("Invalid models response: metadata key is invalid")
            if _is_sensitive_metadata_key(key):
                continue
            result[key] = _scrub_model_metadata_value(
                nested_value,
                depth=depth + 1,
                budget=budget,
            )
        return result
    if type(value) is list:
        return [
            _scrub_model_metadata_value(item, depth=depth + 1, budget=budget)
            for item in value
        ]
    if type(value) is tuple:
        return tuple(
            _scrub_model_metadata_value(item, depth=depth + 1, budget=budget)
            for item in value
        )
    if type(value) is str:
        if len(value) > MODEL_METADATA_MAX_VALUE_CHARS:
            raise ValueError("Invalid models response: metadata value is too large")
        return value
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Invalid models response: metadata value is invalid")


def _safe_model_metadata(model_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy endpoint model metadata while dropping sensitive-looking fields."""
    if type(model_payload) is not dict:
        raise ValueError("Invalid models response: metadata is invalid")
    metadata = _scrub_model_metadata_value(model_payload, depth=0, budget=[0])
    if type(metadata) is not dict:
        raise ValueError("Invalid models response: metadata is invalid")
    serialized = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(serialized) > MODEL_METADATA_MAX_SERIALIZED_BYTES:
        raise ValueError("Invalid models response: metadata is too large")
    return metadata


def normalize_models_response(
    payload: Mapping[str, Any],
    *,
    provider: str,
    provider_list_key: str,
    endpoint_fingerprint: str,
    now_iso: str,
) -> tuple[DiscoveredModel, ...]:
    """Normalize an OpenAI ``/models`` response into discovery contracts."""
    data = payload.get("data") if type(payload) is dict else None
    if type(data) is not list:
        raise ValueError("Invalid models response: expected data list")
    if len(data) > DISCOVERED_MODEL_MAX_COUNT:
        raise ValueError("Invalid models response: too many models")

    seen_model_ids: set[str] = set()
    models: list[DiscoveredModel] = []
    for item in data:
        if type(item) is not dict:
            raise ValueError("Invalid models response: expected model objects")

        model_id = item.get("id")
        if type(model_id) is not str or not model_id.strip():
            raise ValueError("Invalid models response: model id is required")

        model_id = model_id.strip()
        if len(model_id) > DISCOVERED_MODEL_ID_MAX_CHARS or not model_id.isprintable():
            raise ValueError("Invalid models response: model id is invalid")
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
        if _parse_endpoint(endpoint) is None:
            # TASK-367: a malformed URL (invalid scheme / no host, e.g. a dropped
            # 'h' in 'ttp://…') is a DIFFERENT failure than a valid URL whose
            # path is not OpenAI-compatible. Report it as its own kind so the UI
            # stops misdiagnosing a scheme typo as a missing-/v1-path problem.
            error = _discovery_error(
                "malformed_endpoint",
                "This endpoint is not a valid URL.",
                "Enter a full http:// or https:// address, "
                "e.g. http://127.0.0.1:9099/v1.",
            )
        else:
            error = _discovery_error(
                "unsupported_endpoint",
                "This endpoint is not an OpenAI-compatible models endpoint.",
                "Configure an explicit /v1 or /v1/models endpoint for discovery.",
            )
        return ModelDiscoveryResult(
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint_fingerprint=endpoint_fingerprint,
            status="unsupported",
            error=error,
        )

    models_url = build_models_url(endpoint, provider)
    if not validate_url(models_url):
        return ModelDiscoveryResult(
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint_fingerprint=endpoint_fingerprint,
            status="unsupported",
            error=_discovery_error(
                "unsupported_endpoint",
                "This endpoint is not a valid OpenAI-compatible models URL.",
                "Configure an explicit http:// or https:// /v1 models endpoint.",
            ),
        )

    headers = {
        **build_discovery_auth_headers(provider, api_key),
        "Accept-Encoding": "identity",
    }
    paginate = _normalized_provider_identity(provider) == _ANTHROPIC_PROVIDER_KEY

    async def _request_payloads(
        active_client: httpx.AsyncClient,
    ) -> tuple[list[Mapping[str, Any]] | None, ModelDiscoveryResult | None]:
        payloads: list[Mapping[str, Any]] = []
        params: dict[str, Any] | None = (
            {"limit": _ANTHROPIC_MODELS_PAGE_LIMIT} if paginate else None
        )
        for _page in range(_ANTHROPIC_MAX_MODEL_PAGES if paginate else 1):
            try:
                async with active_client.stream(
                    "GET",
                    models_url,
                    headers=headers,
                    params=params,
                    follow_redirects=False,
                ) as response:
                    response.raise_for_status()
                    body = await read_bounded_model_response(response)
            except UnsupportedModelResponseEncoding:
                return None, ModelDiscoveryResult(
                    provider=provider,
                    provider_list_key=provider_list_key,
                    endpoint_fingerprint=endpoint_fingerprint,
                    status="error",
                    error=_discovery_error(
                        "invalid_response",
                        "Compressed models responses are not supported.",
                        "Use an endpoint that honors identity encoding.",
                    ),
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {401, 403}:
                    return None, ModelDiscoveryResult(
                        provider=provider,
                        provider_list_key=provider_list_key,
                        endpoint_fingerprint=endpoint_fingerprint,
                        status="error",
                        error=_discovery_error(
                            "missing_credentials",
                            "The models endpoint rejected the configured credentials.",
                            "Check the API key configured for this provider.",
                        ),
                    )
                if exc.response.status_code == 404:
                    return None, ModelDiscoveryResult(
                        provider=provider,
                        provider_list_key=provider_list_key,
                        endpoint_fingerprint=endpoint_fingerprint,
                        status="unsupported",
                        error=_discovery_error(
                            "unsupported_endpoint",
                            "The models endpoint is unavailable.",
                            "Enter the model ID used by this endpoint.",
                        ),
                    )
                return None, ModelDiscoveryResult(
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
            except httpx.HTTPError:
                return None, ModelDiscoveryResult(
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
            if body is None:
                return None, ModelDiscoveryResult(
                    provider=provider,
                    provider_list_key=provider_list_key,
                    endpoint_fingerprint=endpoint_fingerprint,
                    status="error",
                    error=_discovery_error(
                        "invalid_response",
                        "The models response is too large.",
                        "Use an endpoint with a bounded models response.",
                    ),
                )
            try:
                payload = await asyncio.to_thread(json.loads, body)
            except (RecursionError, UnicodeDecodeError, ValueError):
                return None, ModelDiscoveryResult(
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
            if type(payload) is not dict or type(payload.get("data")) is not list:
                return None, ModelDiscoveryResult(
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
            data = payload["data"]
            current_count = sum(len(item["data"]) for item in payloads)
            if len(data) > DISCOVERED_MODEL_MAX_COUNT - current_count:
                return None, ModelDiscoveryResult(
                    provider=provider,
                    provider_list_key=provider_list_key,
                    endpoint_fingerprint=endpoint_fingerprint,
                    status="error",
                    error=_discovery_error(
                        "invalid_response",
                        "The models endpoint returned too many models.",
                        "Use a narrower models endpoint or provider filter.",
                    ),
                )
            payloads.append(payload)
            if not paginate:
                break
            last_id = payload.get("last_id")
            if bool(payload.get("has_more")) and isinstance(last_id, str) and last_id:
                if current_count + len(data) >= DISCOVERED_MODEL_MAX_COUNT:
                    return None, ModelDiscoveryResult(
                        provider=provider,
                        provider_list_key=provider_list_key,
                        endpoint_fingerprint=endpoint_fingerprint,
                        status="error",
                        error=_discovery_error(
                            "invalid_response",
                            "The paginated models response exceeded the model limit.",
                            "Use a narrower provider-side model filter.",
                        ),
                    )
                params = {"limit": _ANTHROPIC_MODELS_PAGE_LIMIT, "after_id": last_id}
                continue
            break
        return payloads, None

    try:
        if client is not None:
            payloads, request_error = await _request_payloads(client)
        else:
            async with build_httpx_async_client(timeout=timeout_seconds) as active_client:
                payloads, request_error = await _request_payloads(active_client)
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
    if request_error is not None:
        return request_error
    if not payloads:
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

    combined_data: list[Any] = []
    for payload in payloads:
        combined_data.extend(payload["data"])

    now_iso = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    try:
        models = normalize_models_response(
            {"data": combined_data},
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
