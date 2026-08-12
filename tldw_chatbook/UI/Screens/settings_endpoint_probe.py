"""Short live connectivity probe for the Settings provider Test action.

After the local readiness check passes for a URL-based/local provider, the Test
action derives and probes that provider's models route. A successful model
listing reports only models-route reachability; it does not claim that a chat
request succeeded.

URL normalization is shared with ``Chat.local_server_discovery`` (the Console
setup card's discovery module). The HTTP call itself is intentionally local to
this helper: the Settings toast must distinguish transport-failure categories
(connection refused vs timeout vs HTTP status) that
``LocalModelProbeResult.detail`` only exposes as prose copy.

Outcome summaries never include endpoint URLs, exception text, or secrets;
callers may embed them directly in user-visible toasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import httpx

from tldw_chatbook.Chat.local_server_discovery import model_ids_from_payload
from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint

SETTINGS_ENDPOINT_PROBE_TIMEOUT_SECONDS = 2.5
"""Per-request timeout keeping the Test action short even when unreachable."""


EndpointProbeState = Literal[
    "reachable",
    "unreachable",
    "model_listing_unavailable",
]
EndpointProbeCategory = Literal[
    "timeout",
    "connection_refused",
    "unauthorized",
    "forbidden",
    "http_status",
    "invalid_payload",
    "connection_error",
]


@dataclass(frozen=True, slots=True, init=False)
class SettingsEndpointProbeOutcome:
    """Result of one live ``/v1/models`` probe for the Settings Test toast.

    Attributes:
        state: Structured models-route result state.
        summary: Short toast fragment such as ``"reachable (3 models)"`` or
            ``"unreachable: connection refused"``. Contains no URLs/secrets.
        category: Bounded failure category, or ``None`` for valid outcomes and
            invalid local input.
        model_ids: Bounded, sanitized model identifiers from the response.
    """

    state: EndpointProbeState
    summary: str
    category: EndpointProbeCategory | None
    model_ids: tuple[str, ...]
    _legacy_model_count: int | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        state: EndpointProbeState | None = None,
        summary: str = "",
        category: EndpointProbeCategory | None = None,
        model_ids: tuple[str, ...] = (),
        *,
        reachable: bool | None = None,
        model_count: int | None = None,
    ) -> None:
        """Build a structured outcome while accepting the legacy keywords."""
        resolved_state: EndpointProbeState
        if state is None:
            resolved_state = "reachable" if reachable else "unreachable"
        else:
            resolved_state = state
            if reachable is not None and reachable != (state == "reachable"):
                raise ValueError("reachable conflicts with state")
        if resolved_state not in {
            "reachable",
            "unreachable",
            "model_listing_unavailable",
        }:
            raise ValueError("invalid endpoint probe state")
        object.__setattr__(self, "state", resolved_state)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "model_ids", tuple(model_ids))
        object.__setattr__(self, "_legacy_model_count", model_count)

    @property
    def reachable(self) -> bool:
        """Compatibility view of whether the models route returned a listing."""
        return self.state == "reachable"

    @property
    def model_count(self) -> int | None:
        """Compatibility count for old callers that did not retain model IDs."""
        if self.model_ids:
            return len(self.model_ids)
        if self._legacy_model_count is not None:
            return self._legacy_model_count
        return 0 if self.state == "reachable" else None


def _reachable_outcome(response: httpx.Response) -> SettingsEndpointProbeOutcome:
    try:
        payload = response.json()
    except ValueError:
        return SettingsEndpointProbeOutcome(
            state="unreachable",
            category="invalid_payload",
            summary="unreachable: invalid models response",
        )
    model_ids = model_ids_from_payload(payload)
    if model_ids is None:
        return SettingsEndpointProbeOutcome(
            state="unreachable",
            category="invalid_payload",
            summary="unreachable: invalid models response",
        )
    count = len(model_ids)
    noun = "model" if count == 1 else "models"
    return SettingsEndpointProbeOutcome(
        state="reachable",
        summary=f"reachable ({count} {noun})",
        model_ids=model_ids,
    )


def _failure(
    category: EndpointProbeCategory,
    summary: str,
) -> SettingsEndpointProbeOutcome:
    return SettingsEndpointProbeOutcome(
        state="unreachable",
        category=category,
        summary=summary,
    )


async def _request_models(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
) -> SettingsEndpointProbeOutcome:
    try:
        response = await client.get(
            url,
            timeout=timeout,
            follow_redirects=False,
        )
    except httpx.TimeoutException:
        return _failure("timeout", "unreachable: timeout")
    except httpx.ConnectError:
        return _failure("connection_refused", "unreachable: connection refused")
    except httpx.HTTPError:
        return _failure("connection_error", "unreachable: connection error")
    except Exception:  # noqa: BLE001 - this UI boundary must never leak or raise.
        return _failure("connection_error", "unreachable: connection error")

    if response.status_code == 404:
        return SettingsEndpointProbeOutcome(
            state="model_listing_unavailable",
            category="http_status",
            summary="Model listing unavailable; chat endpoint not tested",
        )
    if response.status_code == 401:
        return _failure("unauthorized", "unreachable: unauthorized")
    if response.status_code == 403:
        return _failure("forbidden", "unreachable: forbidden")
    if response.status_code < 200 or response.status_code >= 300:
        return _failure(
            "http_status",
            f"unreachable: HTTP {response.status_code}",
        )
    return _reachable_outcome(response)


async def probe_settings_endpoint(
    base_url: str,
    *,
    provider: str = "custom",
    timeout: float = SETTINGS_ENDPOINT_PROBE_TIMEOUT_SECONDS,
    http_client: httpx.AsyncClient | None = None,
) -> SettingsEndpointProbeOutcome:
    """Probe the contract-derived models route once with a short timeout.

    Never raises: every transport or protocol failure is folded into an
    ``unreachable: <category>`` summary safe for user-visible toasts.

    Args:
        base_url: Configured provider endpoint in origin, API-base, full chat,
            or full models form.
        provider: Canonical provider key controlling endpoint interpretation
            and the Ollama fallback. Defaults to ``custom`` for legacy callers.
        timeout: Per-request timeout in seconds.
        http_client: Optional client override (tests pass a
            ``httpx.MockTransport``-backed client); when omitted a short-lived
            client is created and closed.

    Returns:
        The probe outcome with a toast-ready ``summary`` fragment.
    """
    resolution = resolve_provider_endpoint(provider, base_url)
    if resolution.models_url is None:
        return SettingsEndpointProbeOutcome(
            state="unreachable",
            summary="unreachable: invalid endpoint URL",
        )
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout)
    try:
        outcome = await _request_models(
            client,
            resolution.models_url,
            timeout,
        )
        if outcome.state == "model_listing_unavailable" and provider in {
            "ollama",
            "local_ollama",
        }:
            root = resolution.models_url.removesuffix("/v1/models")
            return await _request_models(client, f"{root}/api/tags", timeout)
        return outcome
    finally:
        if owns_client:
            await client.aclose()
