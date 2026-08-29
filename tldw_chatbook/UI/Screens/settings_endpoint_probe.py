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
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import httpx

from tldw_chatbook.Chat.local_server_discovery import (
    connect_error_is_refused,
    model_ids_from_payload,
    normalize_probe_provider_key,
    read_bounded_model_response,
)
from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint
from tldw_chatbook.TTS.openai_compatible_config import (
    normalize_openai_compatible_endpoint,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConnectionState,
    SpeechTTSTestOperation,
)
from tldw_chatbook.Utils.egress import (
    check_url_or_raise_async,
    origin_of,
    origin_set,
)

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


class SettingsEndpointProbePurpose(StrEnum):
    """Closed operation context for the shared Settings endpoint probe."""

    CHAT_CATALOG = "chat_catalog"
    TTS_CATALOG = "tts_catalog"


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

    state: EndpointProbeState | SpeechTTSConnectionState
    operation: SpeechTTSTestOperation
    summary: str
    category: EndpointProbeCategory | None
    model_ids: tuple[str, ...]
    _legacy_model_count: int | None = field(
        default=None,
        repr=False,
    )

    def __init__(
        self,
        state: EndpointProbeState | SpeechTTSConnectionState | None = None,
        summary: str = "",
        category: EndpointProbeCategory | None = None,
        model_ids: tuple[str, ...] | object = _MODEL_IDS_UNSET,
        *,
        operation: SpeechTTSTestOperation | str = SpeechTTSTestOperation.CATALOG,
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
        try:
            resolved_operation = SpeechTTSTestOperation(operation)
        except (TypeError, ValueError):
            raise ValueError("Endpoint probe operation is invalid.") from None

        model_ids_provided = model_ids is not _MODEL_IDS_UNSET
        if model_ids_provided and isinstance(model_ids, (str, bytes)):
            raise ValueError("Model IDs are invalid.")
        try:
            resolved_model_ids = tuple(model_ids) if model_ids_provided else ()
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
            not isinstance(category, str) or category not in _ENDPOINT_PROBE_CATEGORIES
        ):
            raise ValueError("Endpoint probe category is invalid.")

        resolved_state: EndpointProbeState | SpeechTTSConnectionState
        if state is None:
            resolved_state = "reachable" if reachable else "unreachable"
        else:
            resolved_state = state
            if reachable is not None and reachable != (state == "reachable"):
                raise ValueError("Endpoint probe state conflicts with reachable.")
        if str(resolved_state) not in {
            "reachable",
            "unreachable",
            "model_listing_unavailable",
            "not_tested",
            "unsupported",
        }:
            raise ValueError("Endpoint probe state is invalid.")

        resolved_model_count: int | None = None
        if resolved_state == SpeechTTSConnectionState.REACHABLE:
            if category is not None:
                raise ValueError("Reachable probe cannot include a failure category.")
            if (
                model_ids_provided
                and model_count is not None
                and model_count != len(resolved_model_ids)
            ):
                raise ValueError("Reachable probe model data is inconsistent.")
            resolved_model_count = (
                len(resolved_model_ids) if model_ids_provided else model_count
            )
        elif resolved_state == SpeechTTSConnectionState.UNREACHABLE:
            if resolved_model_ids or model_count is not None:
                raise ValueError("Unreachable probe data must be empty.")
        elif str(resolved_state) in {
            "model_listing_unavailable",
            SpeechTTSConnectionState.UNSUPPORTED.value,
        }:
            if resolved_model_ids or model_count is not None:
                raise ValueError("Model listing probe data must be empty.")
            if category not in {None, "http_status"}:
                raise ValueError("Model listing probe category is invalid.")
        else:
            if resolved_model_ids or model_count is not None or category is not None:
                raise ValueError("Untested probe data must be empty.")

        object.__setattr__(self, "state", resolved_state)
        object.__setattr__(self, "operation", resolved_operation)
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
            headers={"Accept-Encoding": "identity"},
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


def _tts_outcome(
    state: SpeechTTSConnectionState,
    summary: str,
    *,
    category: EndpointProbeCategory | None = None,
    model_ids: tuple[str, ...] = (),
) -> SettingsEndpointProbeOutcome:
    return SettingsEndpointProbeOutcome(
        state=state,
        operation=SpeechTTSTestOperation.CATALOG,
        summary=summary,
        category=category,
        model_ids=model_ids,
    )


def _tts_reachable_outcome(body: bytes) -> SettingsEndpointProbeOutcome:
    try:
        payload = json.loads(body)
    except (RecursionError, UnicodeDecodeError, ValueError):
        return _tts_outcome(
            SpeechTTSConnectionState.UNREACHABLE,
            "unreachable: invalid models response",
            category="invalid_payload",
        )
    model_ids = model_ids_from_payload(_without_chat_capability_labels(payload))
    if model_ids is None:
        return _tts_outcome(
            SpeechTTSConnectionState.UNREACHABLE,
            "unreachable: invalid models response",
            category="invalid_payload",
        )
    count = len(model_ids)
    noun = "model" if count == 1 else "models"
    return _tts_outcome(
        SpeechTTSConnectionState.REACHABLE,
        f"reachable ({count} {noun})",
        model_ids=model_ids,
    )


def _without_chat_capability_labels(payload: object) -> object:
    """Project a models payload without chat-only task filtering metadata."""

    entries: object = payload
    container: str | None = None
    if isinstance(payload, Mapping):
        if isinstance(payload.get("data"), list):
            container = "data"
            entries = payload[container]
        elif isinstance(payload.get("models"), list):
            container = "models"
            entries = payload[container]
    if not isinstance(entries, list):
        return payload
    projected_entries = [
        {
            key: value
            for key, value in entry.items()
            if key not in {"task", "task_type", "model_type"}
        }
        if isinstance(entry, Mapping)
        else entry
        for entry in entries
    ]
    if container is None:
        return projected_entries
    projected_payload = dict(payload)
    projected_payload[container] = projected_entries
    return projected_payload


async def _request_tts_catalog(
    client: httpx.AsyncClient,
    *,
    catalog_url: str,
    expected_origin: str,
    timeout: float,
) -> SettingsEndpointProbeOutcome:
    try:
        async with client.stream(
            "GET",
            catalog_url,
            headers={"Accept-Encoding": "identity"},
            timeout=timeout,
            follow_redirects=False,
        ) as response:
            if origin_of(str(response.request.url)) != origin_of(expected_origin):
                return _tts_outcome(
                    SpeechTTSConnectionState.UNREACHABLE,
                    "unreachable: connection error",
                    category="connection_error",
                )
            if response.status_code in {404, 405, 501}:
                return _tts_outcome(
                    SpeechTTSConnectionState.UNSUPPORTED,
                    "catalog unsupported; speech endpoint not tested",
                    category="http_status",
                )
            if response.status_code == 401:
                return _tts_outcome(
                    SpeechTTSConnectionState.UNREACHABLE,
                    "unreachable: unauthorized",
                    category="unauthorized",
                )
            if response.status_code == 403:
                return _tts_outcome(
                    SpeechTTSConnectionState.UNREACHABLE,
                    "unreachable: forbidden",
                    category="forbidden",
                )
            if response.status_code < 200 or response.status_code >= 300:
                return _tts_outcome(
                    SpeechTTSConnectionState.UNREACHABLE,
                    f"unreachable: HTTP {response.status_code}",
                    category="http_status",
                )
            body = await read_bounded_model_response(response)
    except httpx.TimeoutException:
        return _tts_outcome(
            SpeechTTSConnectionState.UNREACHABLE,
            "unreachable: timeout",
            category="timeout",
        )
    except httpx.ConnectError as error:
        category: EndpointProbeCategory = (
            "connection_refused"
            if connect_error_is_refused(error)
            else "connection_error"
        )
        summary = (
            "unreachable: connection refused"
            if category == "connection_refused"
            else "unreachable: connection error"
        )
        return _tts_outcome(
            SpeechTTSConnectionState.UNREACHABLE,
            summary,
            category=category,
        )
    except Exception:  # noqa: BLE001 - never expose transport or response details.
        return _tts_outcome(
            SpeechTTSConnectionState.UNREACHABLE,
            "unreachable: connection error",
            category="connection_error",
        )

    if body is None:
        return _tts_outcome(
            SpeechTTSConnectionState.UNREACHABLE,
            "unreachable: models response too large",
            category="invalid_payload",
        )
    return _tts_reachable_outcome(body)


async def _probe_openai_tts_catalog(
    base_url: str,
    *,
    timeout: float,
    http_client: httpx.AsyncClient | None,
) -> SettingsEndpointProbeOutcome:
    try:
        endpoint = normalize_openai_compatible_endpoint(base_url)
    except (TypeError, ValueError):
        return _tts_outcome(
            SpeechTTSConnectionState.NOT_TESTED,
            "not tested: invalid endpoint configuration",
        )
    if endpoint.catalog_url is None:
        return _tts_outcome(
            SpeechTTSConnectionState.NOT_TESTED,
            "not tested: catalog operation is not declared",
        )
    try:
        await check_url_or_raise_async(
            endpoint.catalog_url,
            trusted_origins=origin_set(endpoint.origin),
        )
    except Exception:  # noqa: BLE001 - policy details stay out of UI summaries.
        return _tts_outcome(
            SpeechTTSConnectionState.NOT_TESTED,
            "not tested: endpoint blocked by network policy",
        )

    owns_client = http_client is None
    try:
        client = http_client or httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
        )
    except Exception:  # noqa: BLE001 - construction failures are bounded.
        return _tts_outcome(
            SpeechTTSConnectionState.UNREACHABLE,
            "unreachable: connection error",
            category="connection_error",
        )
    close_failed = False
    try:
        outcome = await _request_tts_catalog(
            client,
            catalog_url=endpoint.catalog_url,
            expected_origin=endpoint.origin,
            timeout=timeout,
        )
    finally:
        if owns_client:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - cleanup failure cannot leak details.
                close_failed = True
    if close_failed:
        return _tts_outcome(
            SpeechTTSConnectionState.UNREACHABLE,
            "unreachable: connection error",
            category="connection_error",
        )
    return outcome


async def probe_settings_endpoint(
    base_url: str,
    *,
    provider: str = "custom",
    purpose: SettingsEndpointProbePurpose | str = (
        SettingsEndpointProbePurpose.CHAT_CATALOG
    ),
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
        purpose: Explicit chat or TTS catalog contract. Defaults to chat to
            preserve the shared helper's existing Settings behavior.
        timeout: Per-request timeout in seconds.
        http_client: Optional client override (tests pass a
            ``httpx.MockTransport``-backed client); when omitted a short-lived
            client is created and closed.

    Returns:
        The probe outcome with a toast-ready ``summary`` fragment.
    """
    try:
        resolved_purpose = SettingsEndpointProbePurpose(purpose)
    except (TypeError, ValueError):
        raise ValueError("Endpoint probe purpose is invalid.") from None
    provider_key = normalize_probe_provider_key(provider)
    if resolved_purpose is SettingsEndpointProbePurpose.TTS_CATALOG:
        if provider_key != "openai":
            return _tts_outcome(
                SpeechTTSConnectionState.NOT_TESTED,
                "not tested: TTS catalog probe is unsupported for this provider",
            )
        return await _probe_openai_tts_catalog(
            base_url,
            timeout=timeout,
            http_client=http_client,
        )
    resolution = resolve_provider_endpoint(provider_key, base_url)
    if resolution.models_url is None:
        return SettingsEndpointProbeOutcome(
            state="unreachable",
            summary="unreachable: invalid endpoint URL",
        )
    owns_client = http_client is None
    try:
        client = http_client or httpx.AsyncClient(timeout=timeout)
    except Exception:  # noqa: BLE001 - keep UI failures bounded and secret-free.
        return _failure("connection_error", "unreachable: connection error")
    outcome = _failure("connection_error", "unreachable: connection error")
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
            outcome = await _request_models(client, f"{root}/api/tags", timeout)
    except Exception:  # noqa: BLE001 - helper contract is explicitly no-raise.
        outcome = _failure("connection_error", "unreachable: connection error")
    finally:
        if owns_client:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - shutdown errors are not actionable UI.
                outcome = _failure(
                    "connection_error",
                    "unreachable: connection error",
                )
    return outcome
