"""Local LLM server discovery over localhost OpenAI-compatible endpoints.

Pure + async helpers used by the Console first-run setup card (task-188) and
the Console settings modal's "Discover models" affordance. Candidate probing
follows the ``console_provider_gateway`` precedent: short-timeout
``httpx.AsyncClient`` GETs against ``/v1/models`` (with an ``/api/tags``
fallback for Ollama-family candidates, which historically did not serve the
OpenAI-compatible route).

Automatic discovery (``discover_local_servers``) NEVER probes non-localhost
hosts: candidates are filtered strictly to ``127.0.0.1``/``localhost`` before
any request is made, so config typos or remote endpoints cannot trigger
background traffic. ``probe_models_endpoint`` is the explicit, user-initiated
single-URL probe used by the settings modal and carries no host filter.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse, urlunparse

import httpx

from tldw_chatbook.Chat.console_provider_endpoints import safe_endpoint_display


DISCOVERY_PROBE_TIMEOUT_SECONDS = 2.5
DEFAULT_LLAMACPP_DISCOVERY_URL = "http://127.0.0.1:8080"
DEFAULT_OLLAMA_DISCOVERY_URL = "http://127.0.0.1:11434"
#: api_settings sections whose configured endpoints are eligible candidates.
LOCAL_DISCOVERY_PROVIDER_KEYS = (
    "llama_cpp",
    "local_llamacpp",
    "ollama",
    "local_ollama",
    "vllm",
    "local_vllm",
    "koboldcpp",
    "oobabooga",
    "tabbyapi",
    "aphrodite",
)
_OLLAMA_PROVIDER_KEYS = frozenset({"ollama", "local_ollama"})
_LOCALHOST_HOSTNAMES = frozenset({"127.0.0.1", "localhost"})
_ENDPOINT_CONFIG_KEYS = (
    "api_url",
    "api_base_url",
    "api_base",
    "base_url",
    "api_endpoint",
    "endpoint",
)
#: Path suffixes stripped from configured endpoints so ``/v1/models`` appends
#: cleanly (mirrors ``normalize_llamacpp_base_url``'s endpoint-path handling).
_STRIPPED_PATH_SUFFIXES = ("/v1/models", "/v1", "/models")


@dataclass(frozen=True)
class LocalServerCandidate:
    """One localhost endpoint eligible for a discovery probe."""

    provider_key: str
    base_url: str


@dataclass(frozen=True)
class DiscoveredLocalServer:
    """A local server that answered a models probe."""

    provider_key: str
    base_url: str
    model_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalModelProbeResult:
    """Outcome of probing one endpoint for its model list.

    ``ok`` is ``True`` only when the endpoint answered a models route with a
    2xx response; ``model_ids`` may still be empty when the payload listed no
    models. ``detail`` carries short, honest user-facing failure copy.
    """

    ok: bool
    base_url: str
    model_ids: tuple[str, ...] = ()
    detail: str = ""


def endpoint_display(base_url: str) -> str:
    """Return a safe, credential-free display form of an endpoint.

    Args:
        base_url: Raw endpoint value from config or user input.

    Returns:
        A display label suitable for status copy; falls back to the raw
        stripped value only when the safe formatter yields nothing.
    """
    return safe_endpoint_display(base_url) or str(base_url or "").strip()


def normalize_probe_base_url(base_url: object) -> str | None:
    """Normalize a candidate endpoint to an http(s) origin-plus-path root.

    Args:
        base_url: Raw endpoint value (may lack a scheme, carry a trailing
            slash, or point directly at an API path such as ``/v1``).

    Returns:
        The normalized URL string, or ``None`` when the value is not a usable
        http(s) endpoint.
    """
    raw_url = str(base_url or "").strip()
    if not raw_url:
        return None
    candidate = raw_url if "://" in raw_url else f"http://{raw_url}"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    path = parsed.path.rstrip("/")
    lowered = path.lower()
    for suffix in _STRIPPED_PATH_SUFFIXES:
        if lowered.endswith(suffix):
            path = path[: len(path) - len(suffix)]
            break
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip("/")


def is_localhost_url(base_url: str) -> bool:
    """Return whether a URL strictly targets ``127.0.0.1`` or ``localhost``.

    Args:
        base_url: Normalized endpoint URL.

    Returns:
        ``True`` only for the two loopback spellings automatic discovery is
        allowed to probe; any other host (including other loopback aliases)
        is rejected.
    """
    try:
        hostname = urlparse(base_url).hostname
    except ValueError:
        return False
    return (hostname or "").lower() in _LOCALHOST_HOSTNAMES


def build_local_server_candidates(
    app_config: Mapping[str, object],
) -> tuple[LocalServerCandidate, ...]:
    """Build the ordered, localhost-only candidate list for discovery.

    Candidates are the llama.cpp and Ollama well-known localhost defaults plus
    any endpoint values configured under local-provider ``api_settings``
    sections. Non-localhost endpoints are dropped, and duplicates (same
    normalized URL) keep only the first occurrence.

    Args:
        app_config: Application configuration mapping.

    Returns:
        Deduplicated localhost candidates in probe order (defaults first).
    """
    ordered: list[LocalServerCandidate] = []
    seen_urls: set[str] = set()

    def _add(provider_key: str, raw_url: object) -> None:
        normalized = normalize_probe_base_url(raw_url)
        if not normalized or not is_localhost_url(normalized):
            return
        if normalized in seen_urls:
            return
        seen_urls.add(normalized)
        ordered.append(LocalServerCandidate(provider_key=provider_key, base_url=normalized))

    _add("llama_cpp", DEFAULT_LLAMACPP_DISCOVERY_URL)
    _add("ollama", DEFAULT_OLLAMA_DISCOVERY_URL)

    api_settings = app_config.get("api_settings", {}) if isinstance(app_config, Mapping) else {}
    if not isinstance(api_settings, Mapping):
        return tuple(ordered)
    for provider_key in LOCAL_DISCOVERY_PROVIDER_KEYS:
        section = api_settings.get(provider_key)
        if not isinstance(section, Mapping):
            continue
        for endpoint_key in _ENDPOINT_CONFIG_KEYS:
            _add(provider_key, section.get(endpoint_key))
    return tuple(ordered)


def _model_ids_from_payload(payload: object) -> tuple[str, ...]:
    """Extract model id strings from a models-endpoint JSON payload.

    Accepts the OpenAI ``{"data": [{"id": ...}]}`` shape, the Ollama
    ``{"models": [{"name": ...}]}`` shape, and bare lists of entries.

    Args:
        payload: Decoded JSON payload of any shape.

    Returns:
        Ordered unique model ids; empty when nothing recognizable is present.
    """
    entries: object = payload
    if isinstance(payload, Mapping):
        data = payload.get("data")
        entries = data if isinstance(data, list) else payload.get("models")
    if not isinstance(entries, list):
        return ()
    model_ids: list[str] = []
    for entry in entries:
        model_id: object = entry
        if isinstance(entry, Mapping):
            model_id = entry.get("id") or entry.get("name") or entry.get("model")
        if isinstance(model_id, str) and model_id.strip() and model_id.strip() not in model_ids:
            model_ids.append(model_id.strip())
    return tuple(model_ids)


async def _get_models_payload(
    http_client: httpx.AsyncClient,
    url: str,
    timeout: float,
    display: str,
) -> tuple[tuple[str, ...] | None, str]:
    """GET one models route and parse its payload.

    Args:
        http_client: Shared async HTTP client.
        url: Full route URL to fetch.
        timeout: Per-request timeout in seconds.
        display: Safe base-endpoint label used in failure copy.

    Returns:
        ``(model_ids, "")`` on a 2xx response (ids possibly empty), or
        ``(None, detail)`` with short failure copy otherwise.
    """
    try:
        response = await http_client.get(url, timeout=timeout)
    except httpx.TimeoutException:
        return None, f"Timed out contacting {display}."
    except httpx.HTTPError:
        return None, f"No models endpoint at {display}."
    except Exception:
        return None, f"No models endpoint at {display}."
    if response.status_code < 200 or response.status_code >= 300:
        return None, f"No models endpoint at {display} (HTTP {response.status_code})."
    try:
        payload = response.json()
    except Exception:
        return (), ""
    return _model_ids_from_payload(payload), ""


async def probe_models_endpoint(
    base_url: str,
    *,
    provider_key: str = "",
    http_client: httpx.AsyncClient | None = None,
    timeout: float = DISCOVERY_PROBE_TIMEOUT_SECONDS,
) -> LocalModelProbeResult:
    """Probe one endpoint for its model list.

    Tries ``<base_url>/v1/models`` first; Ollama-family providers fall back to
    ``<base_url>/api/tags`` when the OpenAI-compatible route fails.

    Args:
        base_url: Endpoint root to probe (scheme optional; API paths such as
            ``/v1`` are stripped).
        provider_key: Provider config key of the candidate origin; enables the
            Ollama ``/api/tags`` fallback for ``ollama``/``local_ollama``.
        http_client: Optional shared client; when omitted a short-lived one is
            created and closed.
        timeout: Per-request timeout in seconds.

    Returns:
        Probe result with ``ok``/``model_ids`` on success or honest short
        failure copy in ``detail``.
    """
    normalized = normalize_probe_base_url(base_url)
    if normalized is None:
        display = endpoint_display(base_url)
        return LocalModelProbeResult(
            ok=False,
            base_url=str(base_url or "").strip(),
            detail=f"No models endpoint at {display}." if display else "Enter a base URL first.",
        )
    display = endpoint_display(normalized)
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout)
    try:
        model_ids, detail = await _get_models_payload(
            client,
            f"{normalized}/v1/models",
            timeout,
            display,
        )
        if model_ids is None and provider_key in _OLLAMA_PROVIDER_KEYS:
            fallback_ids, _fallback_detail = await _get_models_payload(
                client,
                f"{normalized}/api/tags",
                timeout,
                display,
            )
            if fallback_ids is not None:
                model_ids, detail = fallback_ids, ""
    finally:
        if owns_client:
            await client.aclose()
    if model_ids is None:
        return LocalModelProbeResult(ok=False, base_url=normalized, detail=detail)
    return LocalModelProbeResult(ok=True, base_url=normalized, model_ids=model_ids)


async def discover_local_servers(
    app_config: Mapping[str, object],
    *,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = DISCOVERY_PROBE_TIMEOUT_SECONDS,
) -> tuple[DiscoveredLocalServer, ...]:
    """Probe localhost candidates and return every responding server.

    Args:
        app_config: Application configuration mapping (source of configured
            local-provider endpoints).
        http_client: Optional shared client; when omitted a short-lived one is
            created and closed.
        timeout: Per-request timeout in seconds.

    Returns:
        Responding servers in candidate order (well-known defaults first);
        empty when nothing answered. Never raises for network failures.
    """
    candidates = build_local_server_candidates(app_config)
    if not candidates:
        return ()
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout)
    try:
        results = await asyncio.gather(
            *(
                probe_models_endpoint(
                    candidate.base_url,
                    provider_key=candidate.provider_key,
                    http_client=client,
                    timeout=timeout,
                )
                for candidate in candidates
            ),
            return_exceptions=True,
        )
    finally:
        if owns_client:
            await client.aclose()
    discovered: list[DiscoveredLocalServer] = []
    for candidate, result in zip(candidates, results):
        if not isinstance(result, LocalModelProbeResult) or not result.ok:
            continue
        discovered.append(
            DiscoveredLocalServer(
                provider_key=candidate.provider_key,
                base_url=candidate.base_url,
                model_ids=result.model_ids,
            )
        )
    return tuple(discovered)
