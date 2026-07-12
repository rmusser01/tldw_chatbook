"""Short live connectivity probe for the Settings provider Test action.

task-191 (Settings half): after the local readiness check passes for a
URL-based/local provider, the Test action runs one short ``GET
<base_url>/v1/models`` and folds the outcome into the toast ("endpoint
reachable (3 models)" / "endpoint unreachable: connection refused").

URL normalization is shared with ``Chat.local_server_discovery`` (the Console
setup card's discovery module). The HTTP call itself is intentionally local to
this helper: the Settings toast must distinguish transport-failure categories
(connection refused vs timeout vs HTTP status) that
``LocalModelProbeResult.detail`` only exposes as prose copy.

Outcome summaries never include endpoint URLs, exception text, or secrets;
callers may embed them directly in user-visible toasts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from tldw_chatbook.Chat.local_server_discovery import normalize_probe_base_url
from tldw_chatbook.Utils.input_validation import validate_url


SETTINGS_ENDPOINT_PROBE_TIMEOUT_SECONDS = 2.5
"""Per-request timeout keeping the Test action short even when unreachable."""


@dataclass(frozen=True)
class SettingsEndpointProbeOutcome:
    """Result of one live ``/v1/models`` probe for the Settings Test toast.

    Attributes:
        reachable: Whether the endpoint answered the models route with 2xx.
        summary: Short toast fragment such as ``"reachable (3 models)"`` or
            ``"unreachable: connection refused"``. Contains no URLs/secrets.
        model_count: Number of models listed by the endpoint, when countable.
    """

    reachable: bool
    summary: str
    model_count: int | None = None


def _model_count_from_payload(payload: object) -> int | None:
    """Count model entries in an OpenAI/Ollama-style models payload.

    Args:
        payload: Decoded JSON payload of any shape.

    Returns:
        Number of listed models, or ``None`` when the shape is unrecognized.
    """
    entries: object = payload
    if isinstance(payload, Mapping):
        data = payload.get("data")
        entries = data if isinstance(data, list) else payload.get("models")
    if isinstance(entries, list):
        return len(entries)
    return None


def _reachable_outcome(response: httpx.Response) -> SettingsEndpointProbeOutcome:
    try:
        payload = response.json()
    except Exception:
        payload = None
    count = _model_count_from_payload(payload)
    if count is None:
        return SettingsEndpointProbeOutcome(reachable=True, summary="reachable")
    noun = "model" if count == 1 else "models"
    return SettingsEndpointProbeOutcome(
        reachable=True,
        summary=f"reachable ({count} {noun})",
        model_count=count,
    )


async def probe_settings_endpoint(
    base_url: str,
    *,
    timeout: float = SETTINGS_ENDPOINT_PROBE_TIMEOUT_SECONDS,
    http_client: httpx.AsyncClient | None = None,
) -> SettingsEndpointProbeOutcome:
    """Probe ``<base_url>/v1/models`` once with a short timeout.

    Never raises: every transport or protocol failure is folded into an
    ``unreachable: <category>`` summary safe for user-visible toasts.

    Args:
        base_url: Configured provider endpoint (scheme optional; API paths
            such as ``/v1`` are stripped before appending ``/v1/models``).
        timeout: Per-request timeout in seconds.
        http_client: Optional client override (tests pass a
            ``httpx.MockTransport``-backed client); when omitted a short-lived
            client is created and closed.

    Returns:
        The probe outcome with a toast-ready ``summary`` fragment.
    """
    normalized = normalize_probe_base_url(base_url)
    # PR #608 review: user-entered endpoints must pass the shared
    # input_validation boundary before any network use.
    if normalized is None or not validate_url(normalized):
        return SettingsEndpointProbeOutcome(
            reachable=False,
            summary="unreachable: invalid endpoint URL",
        )
    url = f"{normalized}/v1/models"
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await client.get(url, timeout=timeout)
    except httpx.TimeoutException:
        return SettingsEndpointProbeOutcome(reachable=False, summary="unreachable: timeout")
    except httpx.ConnectError:
        return SettingsEndpointProbeOutcome(
            reachable=False,
            summary="unreachable: connection refused",
        )
    except Exception:
        # Any other transport/protocol error: never surface raw exception
        # text (it can echo URLs, hosts, or userinfo).
        return SettingsEndpointProbeOutcome(
            reachable=False,
            summary="unreachable: connection error",
        )
    finally:
        if owns_client:
            await client.aclose()
    if response.status_code < 200 or response.status_code >= 300:
        return SettingsEndpointProbeOutcome(
            reachable=False,
            summary=f"unreachable: HTTP {response.status_code}",
        )
    return _reachable_outcome(response)
