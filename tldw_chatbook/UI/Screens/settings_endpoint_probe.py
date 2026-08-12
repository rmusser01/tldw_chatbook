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

import json
from dataclasses import dataclass, field
from typing import Literal

import httpx

from tldw_chatbook.Chat.local_server_discovery import (
    connect_error_is_refused,
    model_ids_from_payload,
    normalize_probe_provider_key,
    read_bounded_model_response,
)
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
_ENDPOINT_PROBE_CATEGORIES = frozenset(
    {
        "timeout",
        "connection_refused",
        "unauthorized",
        "forbidden",
        "http_status",
        "invalid_payload",
        "connection_error",
    }
)
_MODEL_IDS_UNSET = object()


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
    )

    def __init__(
        self,
        state: EndpointProbeState | None = None,
        summary: str = "",
        category: EndpointProbeCategory | None = None,
        model_ids: tuple[str, ...] | object = _MODEL_IDS_UNSET,
        *,
        reachable: bool | None = None,
        model_count: int | None = None,
    ) -> None:
        """Build a structured outcome while accepting the legacy keywords."""
        if state is None and reachable is None:
            raise ValueError("Provide endpoint probe state or reachable.")
        if reachable is not None and not isinstance(reachable, bool):
            raise ValueError("Reachable must be a boolean.")
        if not isinstance(summary, str):
            # Public compatibility contract requires bounded ValueError messages.
            raise ValueError(  # noqa: TRY004
                "Endpoint probe summary must be text."
            )
        if state is not None and not isinstance(state, str):
            raise ValueError("Endpoint probe state is invalid.")

        model_ids_provided = model_ids is not _MODEL_IDS_UNSET
        if model_ids_provided and isinstance(model_ids, (str, bytes)):
            raise ValueError("Model IDs are invalid.")
        try:
            resolved_model_ids = (
                tuple(model_ids) if model_ids_provided else ()
            )
        except TypeError:
            raise ValueError("Model IDs are invalid.") from None
        if any(not isinstance(model_id, str) for model_id in resolved_model_ids):
            raise ValueError("Model IDs are invalid.")
        if model_count is not None and (
            isinstance(model_count, bool)
            or not isinstance(model_count, int)
            or model_count < 0
        ):
            raise ValueError("Model count must be a non-negative integer.")
        if category is not None and (
            not isinstance(category, str)
            or category not in _ENDPOINT_PROBE_CATEGORIES
        ):
            raise ValueError("Endpoint probe category is invalid.")

        resolved_state: EndpointProbeState
        if state is None:
            resolved_state = "reachable" if reachable else "unreachable"
        else:
            resolved_state = state
            if reachable is not None and reachable != (state == "reachable"):
                raise ValueError("Endpoint probe state conflicts with reachable.")
        if resolved_state not in {
            "reachable",
            "unreachable",
            "model_listing_unavailable",
        }:
            raise ValueError("Endpoint probe state is invalid.")

        resolved_model_count: int | None = None
        if resolved_state == "reachable":
            if category is not None:
                raise ValueError(
                    "Reachable probe cannot include a failure category."
                )
            if (
                model_ids_provided
                and model_count is not None
                and model_count != len(resolved_model_ids)
            ):
                raise ValueError("Reachable probe model data is inconsistent.")
            resolved_model_count = (
                len(resolved_model_ids) if model_ids_provided else model_count
            )
        elif resolved_state == "unreachable":
            if resolved_model_ids or model_count is not None:
                raise ValueError("Unreachable probe data must be empty.")
        else:
            if resolved_model_ids or model_count is not None:
                raise ValueError("Model listing probe data must be empty.")
            if category not in {None, "http_status"}:
                raise ValueError("Model listing probe category is invalid.")

        object.__setattr__(self, "state", resolved_state)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "model_ids", resolved_model_ids)
        object.__setattr__(self, "_legacy_model_count", resolved_model_count)

    @property
    def reachable(self) -> bool:
        """Compatibility view of whether the models route returned a listing."""
        return self.state == "reachable"

    @property
    def model_count(self) -> int | None:
        """Compatibility count for old callers that did not retain model IDs."""
        return self._legacy_model_count


def _reachable_outcome(body: bytes) -> SettingsEndpointProbeOutcome:
    try:
        payload = json.loads(body)
    except (RecursionError, UnicodeDecodeError, ValueError):
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
        async with client.stream(
            "GET",
            url,
            timeout=timeout,
            follow_redirects=False,
        ) as response:
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
            body = await read_bounded_model_response(response)
    except httpx.TimeoutException:
        return _failure("timeout", "unreachable: timeout")
    except httpx.ConnectError as error:
        if connect_error_is_refused(error):
            return _failure("connection_refused", "unreachable: connection refused")
        return _failure("connection_error", "unreachable: connection error")
    except httpx.HTTPError:
        return _failure("connection_error", "unreachable: connection error")
    except Exception:  # noqa: BLE001 - this UI boundary must never leak or raise.
        return _failure("connection_error", "unreachable: connection error")

    if body is None:
        return _failure(
            "invalid_payload",
            "unreachable: models response too large",
        )
    return _reachable_outcome(body)


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
    provider_key = normalize_probe_provider_key(provider)
    resolution = resolve_provider_endpoint(provider_key, base_url)
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
        if outcome.state == "model_listing_unavailable" and provider_key in {
            "ollama",
            "local_ollama",
        }:
            root = resolution.models_url.removesuffix("/v1/models")
            return await _request_models(client, f"{root}/api/tags", timeout)
        return outcome
    finally:
        if owns_client:
            await client.aclose()
