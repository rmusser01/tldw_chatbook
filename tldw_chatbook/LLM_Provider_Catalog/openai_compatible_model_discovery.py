"""OpenAI-compatible endpoint model discovery helpers."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import ParseResult, unquote, urlparse, urlunparse

import httpx

from tldw_chatbook.Chat.console_session_settings import normalize_llamacpp_base_url
from tldw_chatbook.Chat.local_server_discovery import (
    MODEL_ID_MAX_CHARS,
    MODEL_PROBE_RESPONSE_MAX_BYTES,
    UnsupportedModelResponseEncoding,
    classify_transport_failure,
    read_bounded_model_response,
)
from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
    ConnectionProbeAvailability,
    ProviderEndpointResolution,
    canonical_endpoint_host,
    connection_probe_availability,
    is_exact_schemeless_host,
    read_schemeless_loopback,
    resolve_provider_endpoint,
)
from tldw_chatbook.LLM_Calls import recovery_review as _provider_recovery
from tldw_chatbook.LLM_Calls.qwencloud_url import (
    QwenCloudBaseURLValidationError,
    normalize_qwencloud_base_url,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    CAPABILITY_HINT_KEYS,
    DiscoveredModel,
    DiscoveryErrorKind,
    ModelDiscoveryError,
    ModelDiscoveryResult,
)
from tldw_chatbook.Utils.input_validation import validate_url
from tldw_chatbook.Utils.tls_trust import build_httpx_async_client
from tldw_chatbook.provider_registry import RECORDS_BY_KEY

_NATIVE_ENDPOINT_PATHS_BY_PROVIDER = {
    "ollama": frozenset({"/api/tags"}),
    "local_ollama": frozenset({"/api/tags"}),
}
_QWENCLOUD_PROVIDER_KEY = "qwencloud"
_KOBOLDCPP_PROVIDER_KEY = "koboldcpp"
#: The fix Settings offers with the contract's reason (TASK-36380). Only the
#: native route keeps sending: chat_with_kobold posts KoboldAI's native
#: payload to the saved endpoint, and Settings saves a root or a "/v1"
#: endpoint as <root>/v1/chat/completions.
KOBOLDCPP_ENDPOINT_HINT = (
    "Use the server's /api/v1/generate route, such as "
    "http://localhost:5001/api/v1/generate."
)
_MAX_ENDPOINT_LENGTH = 2000
_MAX_GENERIC_PATH_DECODE_PASSES = 2
_GENERIC_ENDPOINT_TAILS = (("models",), ("responses",), ("chat", "completions"))
_GENERIC_REQUEST_ENDPOINT_TAILS = _GENERIC_ENDPOINT_TAILS[1:]
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_ENCODED_PATH_SEPARATOR_RE = re.compile(r"%(?:2[fF]|5[cC])")
_BASE_URL_INFERABLE_PROVIDER_KEYS = frozenset(
    {
        "aphrodite",
        "custom",
        "custom_2",
        "custom_openai_api",
        "custom_openai_api_2",
        "llama_cpp",
        "strata",
        "polystrata",
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
        # Databricks AI Gateway base (the engine preset's /openai/v1 suffix,
        # ADR-179); also the standard Groq served-path prefix.
        "/openai/v1",
        # Fireworks served-path base (the engine preset's default URL,
        # ADR-179 Phase 2); models live at /inference/v1/models.
        "/inference/v1",
        # DeepInfra's OpenAI-compatible base (TASK-33201); models live at
        # /v1/openai/models.
        "/v1/openai",
        # Kilo Gateway base (TASK-33351); models live at /api/gateway/models.
        "/api/gateway",
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
# TASK-34100.2 (gap-03): Gemini lists models natively (models.list with an
# x-goog-api-key header, never a ?key= query); the same call verifies the key.
_GOOGLE_PROVIDER_KEY = "google"
_GOOGLE_API_PATHS = frozenset({"/v1beta", "/v1"})
_GOOGLE_PAGE_SIZE = 1000
#: A generateContent model whose id or display name names one of these is
#: not a chat model (speech, image, embedding, music, research agents) or is a
#: preview build. "banana" is Google's image-model brand ("Nano Banana 2.1",
#: ``gemini-nano-banana-2.1``: no image token in its id; review IV-3).
_GOOGLE_NON_CHAT_ID_TOKENS = (
    "tts",
    "image",
    "banana",
    "embed",
    "preview",
    "transcribe",
    "lyria",
    "robotics",
    "computer-use",
    "deep-research",
    "antigravity",
    "aqa",
    "imagen",
    "veo",
)
#: Gemini ids Google retired for new users: generateContent answered 404
#: ("no longer available to new users" / "not found for API version") with
#: the repo's key on 2026-10-03, and again for gemini-2.5-flash and
#: gemini-2.5-flash-lite on 2026-11-12. models.list still returns some of
#: them, so the listing sorts them last; it never hides them, since a key
#: Google still serves (an older or paid one) may answer (review v5r4,
#: G2C-R4-1). The shipped catalog and the code fallbacks name none of them.
RETIRED_GEMINI_IDS = frozenset(
    {
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-1.5-flash-latest",
    }
)
#: Gemini ids Google lists that did not serve the repo's key (TASK-34100.2
#: review v5r2, G2C-R3-1). On 2026-11-12, of 8 one-output-token calls each,
#: every one that got an HTTP answer was 503 UNAVAILABLE "This model is
#: currently experiencing high demand" (gemini-flash-latest 7, gemini-3.8-flash,
#: the model it resolves to, 6; the rest timed out at 60 s), with no 429,
#: while gemini-flash-lite-latest answered all 8 in about a second. Earlier
#: smoke runs from 2026-10-03 mixed answers, 503s and timeouts. Sorted last
#: like the retired ids, never hidden: another key may be served. Evidence:
#: .worktrees/setup-wizard-ux-qa/evidence/g2cv5-r3-catalog/, g2cv5-v3-catalog/.
NOT_ANSWERING_GEMINI_IDS = frozenset({"gemini-flash-latest", "gemini-3.8-flash"})
#: Longest provider reason kept for an error message (gap-01: the provider's
#: own words, bounded and scrubbed of the key that was sent).
_PROVIDER_REASON_MAX_CHARS = 160
MODEL_DISCOVERY_RESPONSE_MAX_BYTES = MODEL_PROBE_RESPONSE_MAX_BYTES
# Deliberately NOT MODEL_IDS_MAX_COUNT (100). That constant bounds how many ids
# the *probe* truncates to for a reachability sample; discovery instead fails
# closed on an over-bound catalog so no partial list is ever cached. Aliasing
# the two put the fail-closed bound below reality: api.openai.com returns 128
# models for an ordinary account, so live first-run discovery errored with
# "The models endpoint returned too many models" for every OpenAI user with a
# valid key. The fail-closed semantics are kept; only the calibration changes.
# MODEL_DISCOVERY_RESPONSE_MAX_BYTES (8 MiB) remains the real memory bound.
# TASK-32925: 512 sat just above OpenRouter's 456 live models (2026-09-23), and
# crossing it would list NOTHING for every OpenRouter user. Keep ~8x headroom;
# the disk and in-memory caches are pinned to hold a full-count list.
DISCOVERED_MODEL_MAX_COUNT = 4096
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
# Enough full pages to reach DISCOVERED_MODEL_MAX_COUNT (Qodo review on #2821:
# a fixed 10 silently truncated 1,001-4,096 model catalogs to 1,000). A last
# permitted page that still reports has_more fails closed; all pages together
# share one MODEL_DISCOVERY_RESPONSE_MAX_BYTES budget.
_ANTHROPIC_MAX_MODEL_PAGES = -(
    -DISCOVERED_MODEL_MAX_COUNT // _ANTHROPIC_MODELS_PAGE_LIMIT
)


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
    has_scheme = "://" in raw_endpoint
    candidate = raw_endpoint if has_scheme else f"http://{raw_endpoint}"
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
    if not has_scheme and not is_exact_schemeless_host(parsed.netloc):
        return None  # the save refuses a remote endpoint without its scheme
    if not _host_follows_contract(parsed):
        return None
    return parsed if _is_structurally_safe_generic_endpoint(parsed) else None


def _host_follows_contract(parsed: ParseResult) -> bool:
    """Whether the endpoint contract's host rule accepts the URL's host.

    TASK-38963: Discover models listed from hosts the save refuses, such as
    ``-kobold`` or ``1.2.3``. TASK-39041: a bracketed IPv6 zone id ("%25"),
    which PR #1562 had allowed, is refused too, as the save and TTS refuse it.
    """
    return canonical_endpoint_host(parsed.hostname or "") is not None


def _qwencloud_base_url(endpoint: str | None) -> str:
    """QwenCloud's base URL, reading a loopback endpoint typed without a
    scheme as ``http://``, as the save does (TASK-39041).

    Raises:
        QwenCloudBaseURLValidationError: If the URL is malformed or unsafe.
    """
    return normalize_qwencloud_base_url(
        endpoint if endpoint is None else read_schemeless_loopback(endpoint)
    )


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
    """Refuse parser-ambiguous authority characters, "%" included.

    TASK-39041: "%" was allowed for an RFC 6874 IPv6 zone id; the endpoint
    contract refuses zone ids, so discovery does too.
    """
    return not (
        any(character in netloc for character in '\\|^{}<>"`%')
        or netloc.endswith(":")
    )


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
    if normalized_path in {
        "/api/v1",
        "/api/paas/v4",
        "/openai/v1",
        "/inference/v1",
        "/v1/openai",
        "/api/gateway",
    }:
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


def _kobold_contract(
    provider_identity: str | None, endpoint: str | None
) -> ProviderEndpointResolution | None:
    """koboldcpp's endpoint-contract resolution; None for any other provider.

    koboldcpp's discovery follows the contract alone, so it lists where
    Switch model probes: the root's /v1/models for a bare root, a prefix root
    such as /kobold, or the native /api/v1/generate (TASK-35832, TASK-36380).
    """
    if _normalized_provider_identity(provider_identity) != _KOBOLDCPP_PROVIDER_KEY:
        return None
    return resolve_provider_endpoint(_KOBOLDCPP_PROVIDER_KEY, endpoint)


def contract_refusal_reason(
    provider_identity: str | None, endpoint: str | None
) -> str | None:
    """Return why the endpoint contract refuses an endpoint, else None.

    Settings shows it in place of its generic unsupported-endpoint copy, which
    asks for an http:// /v1 endpoint even when the URL is one: koboldcpp
    since TASK-36380 (with ``KOBOLDCPP_ENDPOINT_HINT``), every provider since
    TASK-38963's review (a refused host such as ``-kobold``).

    Args:
        provider_identity: Provider name/identity.
        endpoint: The configured or staged endpoint.

    Returns:
        The contract's first error for an endpoint it refuses, or None when
        it accepts the endpoint or the provider key is missing or malformed
        (any well-formed key is resolved, known or not).
    """
    resolution = resolve_provider_endpoint(
        _normalized_provider_identity(provider_identity), endpoint
    )
    if not resolution.provider_key or not resolution.errors:
        return None
    return resolution.errors[0]


def supports_openai_compatible_model_discovery(
    provider_identity: str,
    normalized_endpoint: str | None,
) -> bool:
    """Return whether an endpoint shape supports OpenAI-compatible discovery.

    Eligibility is based on explicit OpenAI-compatible URL paths. Native
    provider discovery URLs are rejected even when the provider can also expose
    an OpenAI-compatible API at another configured endpoint. The exception is
    koboldcpp: once its URL parses, it is eligible exactly when Switch
    model's probe is (the endpoint contract's
    ``connection_probe_availability``), and lists at the contract's models URL
    (TASK-35832, TASK-36380).
    """
    provider_key = _normalized_provider_identity(provider_identity)
    record = RECORDS_BY_KEY.get(provider_key)
    if record is not None and record.discovery_route is None:
        # Seeded-only preset (TASK-33201, MiniMax): no documented models
        # route, so never probe an unlisted URL.
        return False
    if provider_key == _QWENCLOUD_PROVIDER_KEY:
        try:
            base_url = _qwencloud_base_url(normalized_endpoint)
        except QwenCloudBaseURLValidationError:
            return False
        return _host_follows_contract(urlparse(base_url))

    parsed = _parse_endpoint(normalized_endpoint)
    if parsed is None:
        return False

    if provider_key == _KOBOLDCPP_PROVIDER_KEY:
        return connection_probe_availability(provider_key, normalized_endpoint) is (
            ConnectionProbeAvailability.MODELS_ROUTE
        )
    path = _normalized_path(parsed)
    if provider_key == _GOOGLE_PROVIDER_KEY:
        # TASK-34100.2: Google's own API base, listed natively.
        return parsed.scheme == "https" and path.rstrip("/") in _GOOGLE_API_PATHS
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
    if _normalized_provider_identity(provider_identity) == _GOOGLE_PROVIDER_KEY:
        parsed = _parse_endpoint(endpoint)
        if parsed is None:
            return str(endpoint or "").strip()
        base_path = parsed.path.rstrip("/")
        return urlunparse(
            (parsed.scheme, _safe_netloc(parsed), f"{base_path}/models", "", "", "")
        )
    if _normalized_provider_identity(provider_identity) == _QWENCLOUD_PROVIDER_KEY:
        try:
            base_url = _qwencloud_base_url(endpoint)
        except QwenCloudBaseURLValidationError:
            return fingerprint_endpoint(endpoint)
        return f"{base_url}/models"

    parsed = _parse_endpoint(endpoint)
    if parsed is None:
        return str(endpoint or "").strip()

    kobold = _kobold_contract(provider_identity, endpoint)
    if kobold is not None and kobold.models_url is not None:
        return kobold.models_url
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
    """Return a safe endpoint fingerprint without secrets or query details.

    TASK-38963 review r10e-rev3: an endpoint discovery and the contract both
    refuse may be a key pasted into the field (``sk-abc/def``, 64 hex), and
    its ``http://<key>/`` fingerprint reached the auto-refresh log. It gets
    the sentinel, keeping only a typed http(s)/ftp scheme.
    """
    parsed = _parse_endpoint(endpoint)
    if parsed is None and resolve_provider_endpoint("custom", endpoint).errors:
        scheme, separator, _rest = str(endpoint or "").strip().partition("://")
        typed = scheme.lower() if scheme.lower() in {"http", "https", "ftp"} else ""
        return f"{typed}://[invalid-endpoint]" if separator and typed else (
            "[invalid-endpoint]"
        )
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
            if type(nested_value) is str and len(nested_value) > MODEL_METADATA_MAX_VALUE_CHARS:
                # Drop the field, keep the model: Together ships chat templates
                # up to 16 KB in config.chat_template, and rejecting them
                # listed nothing for every Together user (TASK-34361).
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


def _bounded_model_metadata(model_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a model's safe metadata, dropping only what breaks a bound.

    One model's oversized details must not cost the user the whole list:
    Vercel's tiered pricing (270 items against a 256-item bound) listed none
    of its 407 models (TASK-34363). A top-level field that breaks a bound on
    its own is dropped first; if the rest together still exceed a bound, the
    largest fields go, biggest first, until they fit, and a field the
    capability inference reads goes only after every other. Nothing unbounded is
    ever kept. The work is bounded: a model with more top-level fields than
    the item bound cannot fit and gets no metadata, each field is measured
    once on its own, and the number to drop is found by bisection (about
    nine measurements of the record, however many fields it has).

    Args:
        model_payload: One model object from a ``/models`` response.

    Returns:
        The model's metadata within every bound; empty when no field fits.
    """
    try:
        return _safe_model_metadata(model_payload)
    except ValueError:
        pass
    if len(model_payload) > MODEL_METADATA_MAX_ITEMS:
        return {}
    kept: dict[str, Any] = {}
    for key, value in model_payload.items():
        try:
            _safe_model_metadata({key: value})
        except ValueError:
            continue
        kept[key] = value
    # Largest first, but the fields capability inference reads go last
    # (TASK-34402): shrinking a listing should not cost the model its hint.
    largest_first = sorted(
        kept,
        key=lambda name: (
            name not in CAPABILITY_HINT_KEYS,
            len(json.dumps(kept[name])),
        ),
        reverse=True,
    )
    # Every kept field passed on its own, so what remains can only break a
    # total (items or serialized bytes), and removing a field never raises
    # either: whether the record fits is monotone in how many fields are gone.
    # Bisect for the fewest evictions. Re-measuring after each one scrubbed a
    # record of 255 small fields 255 times, minutes of CPU for a large listing.
    measured: dict[int, dict[str, Any] | None] = {}

    def without(evicted: int) -> dict[str, Any] | None:
        if evicted not in measured:
            gone = set(largest_first[:evicted])
            try:
                measured[evicted] = _safe_model_metadata(
                    {name: value for name, value in kept.items() if name not in gone}
                )
            except ValueError:
                measured[evicted] = None
        return measured[evicted]

    fewest, most = 0, len(largest_first)
    while fewest < most:
        middle = (fewest + most) // 2
        if without(middle) is None:
            fewest = middle + 1
        else:
            most = middle
    return without(fewest) or {}


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
                metadata_raw_safe=_bounded_model_metadata(item),
            )
        )

    return tuple(models)


def _discovery_error(
    kind: DiscoveryErrorKind,
    message: str,
    recovery_hint: str,
    category: str | None = None,
) -> ModelDiscoveryError:
    """Build a typed safe discovery error."""
    return ModelDiscoveryError(
        kind=kind,
        message=message,
        recovery_hint=recovery_hint,
        category=category,
    )


async def _transport_category(error: httpx.HTTPError) -> str:
    """Name a failed request's bounded category (TASK-33005.4 AC#4).

    TASK-34100.6: certificate and plain-HTTP failures are named too, by the
    classifier the Settings check shares.
    """
    if isinstance(error, (httpx.TimeoutException, httpx.ConnectError)):
        return await classify_transport_failure(error)
    return "connection_error"


def _provider_reason(body: bytes | None, api_key: str | None) -> str:
    """The provider's own error message, bounded and scrubbed (gap-01).

    Reads only ``error.message`` (or a bare ``error``/``message`` string) from
    a JSON error body, collapses whitespace, drops unprintable characters,
    never echoes the key that was sent, and caps the length.

    Args:
        body: The raw (bounded) error response body.
        api_key: The key the request carried, redacted if echoed back.

    Returns:
        The reason, or ``""`` when the body carries none.
    """
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except (RecursionError, UnicodeDecodeError, ValueError):
        return ""
    message: object = None
    if type(payload) is dict:
        error = payload.get("error")
        if type(error) is dict:
            message = error.get("message")
        elif type(error) is str:
            message = error
        elif type(payload.get("message")) is str:
            message = payload["message"]
    if type(message) is not str:
        return ""
    text = " ".join(message.split())
    if api_key:
        text = text.replace(api_key, "[redacted]")
    text = "".join(character for character in text if character.isprintable())
    if len(text) > _PROVIDER_REASON_MAX_CHARS:
        text = text[: _PROVIDER_REASON_MAX_CHARS - 1].rstrip() + "…"
    return text


def _rejected_key_error(
    display_name: str, reason: str, status_code: int
) -> ModelDiscoveryError:
    """A rejected key, in the provider's words when it gave any."""
    return ModelDiscoveryError(
        kind="missing_credentials",
        message=f"{display_name} rejected this key: {reason}"
        if reason
        else f"{display_name} rejected this key.",
        recovery_hint="Check the API key configured for this provider.",
        category="forbidden" if status_code == 403 else "unauthorized",
        provider_reason=reason,
    )


def _is_google_chat_model(model: Mapping[str, Any]) -> bool:
    """A generateContent model that is a chat model and not a preview build."""
    methods = model.get("supportedGenerationMethods")
    if type(methods) is not list or "generateContent" not in methods:
        return False
    model_id = str(model.get("name") or "").removeprefix("models/").lower()
    display_name = model.get("displayName")
    names = (model_id, display_name.lower() if isinstance(display_name, str) else "")
    return bool(model_id) and not any(
        token in name for name in names for token in _GOOGLE_NON_CHAT_ID_TOKENS
    )


def _google_model_rank(model_id: str) -> int:
    """The shipped default first, then Google's order, then the pinned ids.

    Only the default is promoted (review v5r4, G2C-R4-1): promoting every
    ``-latest`` alias put ``gemini-flash-latest`` right under it. An id in
    ``RETIRED_GEMINI_IDS`` or ``NOT_ANSWERING_GEMINI_IDS`` sorts last; the
    sort is stable, so each group keeps the order Google listed it in.
    """
    from tldw_chatbook.config import DEFAULT_GOOGLE_CHAT_MODEL

    if model_id == DEFAULT_GOOGLE_CHAT_MODEL:
        return 0
    if model_id in RETIRED_GEMINI_IDS or model_id in NOT_ANSWERING_GEMINI_IDS:
        return 2
    return 1


async def _discover_google_models(
    *,
    provider: str,
    provider_list_key: str,
    endpoint: str,
    api_key: str | None,
    active_client: httpx.AsyncClient,
) -> ModelDiscoveryResult:
    """List Gemini chat models from Google's models.list (TASK-34100.2).

    The list is authenticated (``x-goog-api-key``; the key never rides in the
    URL), so a listing that answers also verifies the key, and Google's 400
    ``API_KEY_INVALID`` is a rejected key in Google's own words. Only
    generateContent chat models are kept -- no TTS, image, embedding or
    preview ids -- with the shipped default first (``_google_model_rank``).
    """
    endpoint_fingerprint = fingerprint_endpoint(endpoint)

    def result(
        status: Literal["success", "unsupported", "error"],
        error: ModelDiscoveryError | None = None,
        models: tuple[DiscoveredModel, ...] = (),
    ) -> ModelDiscoveryResult:
        return ModelDiscoveryResult(
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint_fingerprint=endpoint_fingerprint,
            status=status,
            models=models,
            error=error,
            key_verified=status == "success",
        )

    if not api_key:
        return result(
            "error",
            _discovery_error(
                "missing_credentials",
                "Google lists models only for a valid API key.",
                "Enter a Gemini API key from aistudio.google.com/apikey.",
            ),
        )
    models_url = build_models_url(endpoint, provider)
    if not validate_url(models_url):
        return result(
            "unsupported",
            _discovery_error(
                "unsupported_endpoint",
                "This endpoint is not a valid Google models URL.",
                "Use the Gemini API base https://generativelanguage.googleapis.com/v1beta.",
            ),
        )
    headers = {"x-goog-api-key": api_key, "Accept-Encoding": "identity"}
    listed: list[Mapping[str, Any]] = []
    bytes_read = 0
    page_token: str | None = None
    for _page in range(-(-DISCOVERED_MODEL_MAX_COUNT // _GOOGLE_PAGE_SIZE) + 1):
        params: dict[str, Any] = {"pageSize": _GOOGLE_PAGE_SIZE}
        if page_token:
            params["pageToken"] = page_token
        try:
            async with _provider_recovery.discovery_native_context(
                active_client.stream(
                    "GET",
                    models_url,
                    headers=headers,
                    params=params,
                    follow_redirects=False,
                ),
                response=True,
            ) as response:
                body = await read_bounded_model_response(
                    response, MODEL_DISCOVERY_RESPONSE_MAX_BYTES - bytes_read
                )
                status_code = response.status_code
        except UnsupportedModelResponseEncoding:
            return result(
                "error",
                _discovery_error(
                    "invalid_response",
                    "Compressed models responses are not supported.",
                    "Use an endpoint that honors identity encoding.",
                ),
            )
        except httpx.HTTPError as exc:
            return result(
                "error",
                _discovery_error(
                    "request_failed",
                    "Model discovery request failed.",
                    "Check your network connection.",
                    _transport_category(exc),
                ),
            )
        if status_code in {400, 401}:
            reason = _provider_reason(body, api_key)
            # A 400 rejects the key only when Google says so ('API key not
            # valid' / 'API key expired', reason API_KEY_INVALID).
            if status_code == 401 or "api key" in reason.lower():
                return result("error", _rejected_key_error("Google", reason, status_code))
        if status_code == 403:
            # Review round 2 (R2-8): only a 401 rejects a key (owner ruling,
            # PR #2958). A 403 -- the Gemini API disabled for the key's
            # project, a key restricted to other APIs -- keeps the shared 403
            # shape (``forbidden``, read by Settings as 'not permitted') and
            # Google's own reason, never 'rejected'.
            reason = _provider_reason(body, api_key)
            return result(
                "error",
                ModelDiscoveryError(
                    kind="missing_credentials",
                    message=f"Google refused this request: {reason}"
                    if reason
                    else "Google refused this request (HTTP 403).",
                    recovery_hint=(
                        "Enable the Gemini API for this key's project, or use a "
                        "key from aistudio.google.com/apikey."
                    ),
                    category="forbidden",
                    provider_reason=reason,
                ),
            )
        if status_code == 404:
            return result(
                "unsupported",
                _discovery_error(
                    "unsupported_endpoint",
                    "The models endpoint is unavailable.",
                    "Enter the model ID used by this endpoint.",
                    "http_status",
                ),
            )
        if status_code >= 400 or body is None:
            return result(
                "error",
                _discovery_error(
                    "request_failed" if status_code >= 400 else "invalid_response",
                    "Model discovery request failed."
                    if status_code >= 400
                    else "The models response is too large.",
                    "Check the endpoint URL, server availability, and credentials.",
                    "http_status" if status_code >= 400 else None,
                ),
            )
        bytes_read += len(body)
        try:
            payload = await asyncio.to_thread(json.loads, body)
        except (RecursionError, UnicodeDecodeError, ValueError):
            payload = None
        models = payload.get("models", []) if type(payload) is dict else None
        if type(models) is not list or any(type(item) is not dict for item in models):
            return result(
                "error",
                _discovery_error(
                    "invalid_response",
                    "Google did not return a valid model list.",
                    "Try again later.",
                ),
            )
        listed.extend(models)
        if len(listed) > DISCOVERED_MODEL_MAX_COUNT:
            return result(
                "error",
                _discovery_error(
                    "invalid_response",
                    "The models endpoint returned too many models.",
                    "Use a narrower models endpoint or provider filter.",
                ),
            )
        next_token = payload.get("nextPageToken")
        page_token = next_token if type(next_token) is str and next_token else None
        if page_token is None:
            break
    chat = [
        {**item, "id": str(item["name"]).removeprefix("models/")}
        for item in listed
        if _is_google_chat_model(item)
    ]
    chat.sort(key=lambda item: _google_model_rank(item["id"]))
    try:
        normalized = normalize_models_response(
            {"data": chat},
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint_fingerprint=endpoint_fingerprint or "",
            now_iso=datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        )
    except ValueError:
        return result(
            "error",
            _discovery_error(
                "invalid_response",
                "Google did not return a valid model list.",
                "Try again later.",
            ),
        )
    return result("success", models=normalized)


def _key_check_serves_endpoint(record: Any, endpoint: str) -> bool:
    """Whether the record's key check vouches for keys of this endpoint.

    TASK-34100.2 review F6b: the key check lives on the provider's own host
    (openrouter.ai, huggingface.co). A user who points the provider at a
    proxy or a dedicated endpoint holds that host's key, which must never be
    sent to the provider's key check (and would come back 'rejected').
    """
    configured = _parse_endpoint(endpoint)
    default = _parse_endpoint(getattr(record, "default_base_url", None))
    return (
        configured is not None
        and default is not None
        and (configured.hostname or "").lower() == (default.hostname or "").lower()
    )


async def _check_listing_key(
    active_client: httpx.AsyncClient,
    *,
    provider: str,
    provider_list_key: str,
    endpoint_fingerprint: str | None,
    endpoint: str,
    api_key: str,
) -> tuple[bool, ModelDiscoveryResult | None]:
    """Prove a public listing's key at the provider's key endpoint (gap-01).

    OpenRouter's and Hugging Face's model lists answer any key, so after a
    successful listing the key is checked where it can actually fail
    (``ProviderRecord.key_check_url``). Only a 401 rejects the key, with the
    provider's reason (owner ruling 2026-10-02, PR #2958). A check that gets
    no readable answer -- a 403, 429 or 5xx, a network failure, a body the
    client cannot decode -- verifies nothing and blocks nothing: the listing
    stands and the key stays unverified (review round 1, F6). A key for
    another host is never sent to the check at all.

    Returns:
        ``(key_verified, failure)``: ``failure`` is the rejected-key
        discovery, else ``None``.
    """
    record = RECORDS_BY_KEY.get(_normalized_provider_identity(provider))
    if (
        record is None
        or not record.key_check_url
        or not _key_check_serves_endpoint(record, endpoint)
    ):
        return False, None

    def failed(error: ModelDiscoveryError) -> ModelDiscoveryResult:
        return ModelDiscoveryResult(
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint_fingerprint=endpoint_fingerprint,
            status="error",
            error=error,
        )

    body: bytes | None = None
    status_code: int | None = None
    try:
        async with _provider_recovery.discovery_native_context(
            active_client.stream(
                "GET",
                record.key_check_url,
                headers={
                    **build_discovery_auth_headers(provider, api_key),
                    "Accept-Encoding": "identity",
                },
                follow_redirects=False,
            ),
            response=True,
        ) as response:
            status_code = response.status_code
            body = await read_bounded_model_response(response, 64 * 1024)
    except UnsupportedModelResponseEncoding:
        # The status stands; only the reason is unreadable. An unreadable
        # success proves nothing, so it never verifies the key.
        if status_code == 401:
            return False, failed(
                _rejected_key_error(record.display_name, "", status_code)
            )
        return False, None
    except httpx.HTTPError:
        return False, None
    if status_code == 401:
        return False, failed(
            _rejected_key_error(
                record.display_name, _provider_reason(body, api_key), status_code
            )
        )
    return status_code is not None and 200 <= status_code < 300, None


@_provider_recovery.discovery_call
async def discover_openai_compatible_models(
    *,
    provider: str,
    provider_list_key: str,
    endpoint: str,
    api_key: str | None,
    timeout_seconds: float = 10.0,
    client: httpx.AsyncClient | None = None,
) -> ModelDiscoveryResult:
    """Manually discover models from one configured OpenAI-compatible endpoint.

    TASK-34100.2: Gemini is listed natively (``_discover_google_models``),
    and a public listing with a key is followed by that provider's
    authenticated key check (``_check_listing_key``).
    """
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
    #: Set by the listing's authenticated key check (``_check_listing_key``).
    key_check = {"verified": False}

    async def _request_payloads(
        active_client: httpx.AsyncClient,
    ) -> tuple[list[Mapping[str, Any]] | None, ModelDiscoveryResult | None]:
        payloads: list[Mapping[str, Any]] = []
        bytes_read = 0
        params: dict[str, Any] | None = (
            {"limit": _ANTHROPIC_MODELS_PAGE_LIMIT} if paginate else None
        )
        for _page in range(_ANTHROPIC_MAX_MODEL_PAGES if paginate else 1):
            try:
                async with _provider_recovery.discovery_native_context(
                    active_client.stream(
                        "GET",
                        models_url,
                        headers=headers,
                        params=params,
                        follow_redirects=False,
                    ),
                    response=True,
                ) as response:
                    response.raise_for_status()
                    body = await read_bounded_model_response(
                        response, MODEL_DISCOVERY_RESPONSE_MAX_BYTES - bytes_read
                    )
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
                            "unauthorized"
                            if exc.response.status_code == 401
                            else "forbidden",
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
                            "http_status",
                        ),
                    )
                return None, ModelDiscoveryResult(
                    provider=provider,
                    provider_list_key=provider_list_key,
                    endpoint_fingerprint=endpoint_fingerprint,
                    status="error",
                    error=_discovery_error(
                        "request_failed",
                        # TASK-34100.6 review round 1 (live F7): the code says
                        # the server answered, so no 'is it running?' copy.
                        f"The models endpoint answered HTTP {exc.response.status_code}.",
                        "Check the endpoint URL, server availability, and credentials.",
                        "http_status",
                    ),
                )
            except httpx.HTTPError as exc:
                return None, ModelDiscoveryResult(
                    provider=provider,
                    provider_list_key=provider_list_key,
                    endpoint_fingerprint=endpoint_fingerprint,
                    status="error",
                    error=_discovery_error(
                        "request_failed",
                        "Model discovery request failed.",
                        "Check the endpoint URL, server availability, and credentials.",
                        await _transport_category(exc),
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
            bytes_read += len(body)
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
            if type(payload) is list:
                # Together answers /models with the bare array, no envelope
                # (TASK-34361, Tests/fixtures/cloud_live/together.json).
                payload = {"data": payload}
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
        else:
            # Every permitted page was used and the last still had more:
            # fail closed rather than cache a partial catalog.
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
        if api_key:
            verified, key_failure = await _check_listing_key(
                active_client,
                provider=provider,
                provider_list_key=provider_list_key,
                endpoint_fingerprint=endpoint_fingerprint,
                endpoint=endpoint,
                api_key=api_key,
            )
            if key_failure is not None:
                return None, key_failure
            key_check["verified"] = verified
        return payloads, None

    async def _run(
        active_client: httpx.AsyncClient,
    ) -> tuple[list[Mapping[str, Any]] | None, ModelDiscoveryResult | None]:
        if _normalized_provider_identity(provider) == _GOOGLE_PROVIDER_KEY:
            return None, await _discover_google_models(
                provider=provider,
                provider_list_key=provider_list_key,
                endpoint=endpoint,
                api_key=api_key,
                active_client=active_client,
            )
        return await _request_payloads(active_client)

    try:
        if client is not None:
            payloads, request_error = await _run(client)
        else:
            async with _provider_recovery.discovery_native_context(
                build_httpx_async_client(
                    timeout=timeout_seconds,
                    **_provider_recovery.discovery_client_options(),
                )
            ) as active_client:
                payloads, request_error = await _run(active_client)
    except httpx.HTTPError as exc:
        return ModelDiscoveryResult(
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint_fingerprint=endpoint_fingerprint,
            status="error",
            error=_discovery_error(
                "request_failed",
                "Model discovery request failed.",
                "Check the endpoint URL, server availability, and credentials.",
                await _transport_category(exc),
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
        # Off the event loop, like the JSON decode above: bounding a listing's
        # metadata is CPU work that grows with the response (a record whose
        # fields only overflow together is re-measured once per dropped field).
        models = await asyncio.to_thread(
            normalize_models_response,
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
        # The listing's key check ran and passed (``_check_listing_key``).
        key_verified=key_check["verified"],
    )
