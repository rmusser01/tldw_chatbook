"""Pure first-run setup-card state contracts for the native Console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from tldw_chatbook.Chat.console_provider_endpoints import safe_endpoint_display
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsReadiness
from tldw_chatbook.Chat.provider_catalog import provider_display_name

CONSOLE_SETUP_CARD_TITLE = "Get started"
CONSOLE_READY_EMPTY_COPY = "Ready — type a message to begin."
CONSOLE_QUIET_EMPTY_COPY = "No messages yet."
CONSOLE_SETUP_STEP_GLYPHS = {"done": "✓", "active": "●", "pending": "○"}

_STEP_ONE_LABELS = {
    "Missing key": "Connect a provider (API key or local server)",
    "Invalid URL": "Save the provider endpoint",
    "Endpoint not saved": "Save the provider endpoint",
    "Unknown": "Choose a supported provider",
    "Pending": "Choose a send-capable provider",
}
CONSOLE_SETUP_STEP_THREE_DETAIL = "Composer unlocks after setup"
_TRUE_STRINGS = {"true", "yes", "1", "on"}


@dataclass(frozen=True)
class ConsoleSetupStep:
    """One numbered step in the Console first-run setup card."""

    state: str
    label: str
    detail: str = ""

    @property
    def glyph(self) -> str:
        return CONSOLE_SETUP_STEP_GLYPHS.get(self.state, "○")


@dataclass(frozen=True)
class ConsoleSetupCardState:
    """Display state for the Console empty-transcript onboarding surface."""

    mode: str
    steps: tuple[ConsoleSetupStep, ...] = ()
    body_copy: str = ""


@dataclass(frozen=True)
class ConsoleDetectedServerAction:
    """Secondary setup-card affordance for an auto-detected local server."""

    label: str
    tooltip: str
    provider_key: str
    base_url: str
    model_id: str | None = None


def _detected_server_host_display(base_url: str) -> str:
    """Return a compact credential-free ``host:port`` label for card copy.

    Args:
        base_url: Detected server base URL.

    Returns:
        The endpoint display with any http(s) scheme prefix removed.
    """
    display = safe_endpoint_display(base_url)
    for prefix in ("http://", "https://"):
        if display.startswith(prefix):
            return display[len(prefix):]
    return display


def build_console_detected_server_action(
    server: Any,
    *,
    card_mode: str,
) -> ConsoleDetectedServerAction | None:
    """Build the detected-local-server card affordance, if one applies.

    Args:
        server: Discovery result with ``provider_key``, ``base_url``, and
            ``model_ids`` attributes (``DiscoveredLocalServer``), or ``None``.
        card_mode: Current ``ConsoleSetupCardState.mode``; the affordance only
            exists while the blocking ``card`` is showing.

    Returns:
        The action (label, tooltip, and the values to persist) or ``None``
        when no detected server should be offered.
    """
    if server is None or card_mode != "card":
        return None
    provider_key = str(getattr(server, "provider_key", "") or "").strip()
    base_url = str(getattr(server, "base_url", "") or "").strip()
    if not provider_key or not base_url:
        return None
    try:
        hostname = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return None
    if hostname not in {"127.0.0.1", "localhost"}:
        # The card only ever offers loopback servers; anything else means the
        # discovery filter was bypassed and the offer is dropped, not shown.
        return None
    host_display = _detected_server_host_display(base_url)
    if not host_display or host_display == "invalid endpoint":
        return None
    display_name = provider_display_name(provider_key)
    model_ids = tuple(
        model_id
        for model_id in (getattr(server, "model_ids", ()) or ())
        if isinstance(model_id, str) and model_id.strip()
    )
    model_id = model_ids[0].strip() if model_ids else None
    if model_id:
        tooltip = f"Sets provider to {display_name} at {host_display} and model to {model_id}."
    else:
        tooltip = f"Sets provider to {display_name} at {host_display}. Pick a model next."
    return ConsoleDetectedServerAction(
        label=f"Use detected {display_name} ({host_display})",
        tooltip=tooltip,
        provider_key=provider_key,
        base_url=base_url,
        model_id=model_id,
    )


def coerce_console_first_send_completed(raw: Any) -> bool:
    """Normalize the persisted first-send flag from arbitrary config input.

    Config values may arrive as a native bool, a legacy int/string form, or
    anything else if the config was hand-edited or corrupted. Anything
    unrecognized is treated as not-yet-completed rather than raising.

    Args:
        raw: The raw ``first_send_completed`` value read from config (or
            ``None`` if absent).

    Returns:
        ``True`` if ``raw`` unambiguously represents completion (``True``,
        a nonzero int, or a truthy string like ``"true"``/``"yes"``/``"1"``/
        ``"on"``); ``False`` for everything else, including ``None`` and
        unrecognized types.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw != 0
    if isinstance(raw, str):
        return raw.strip().lower() in _TRUE_STRINGS
    return False


def build_console_setup_card_state(
    *,
    readiness: ConsoleSettingsReadiness,
    provider_label: str,
    has_model: bool,
    first_send_completed: bool,
    has_messages: bool,
    guidance_dismissed: bool,
) -> ConsoleSetupCardState:
    """Derive the onboarding surface state from the readiness single source.

    Args:
        readiness: Current settings readiness from
            ``build_console_settings_readiness``.
        provider_label: User-facing provider name for the done-step detail.
        has_model: Whether a model is selected for the session.
        first_send_completed: Persisted global flag; once True the card never
            returns, including new tabs and workspaces.
        has_messages: Whether the active transcript has any messages.
        guidance_dismissed: In-session dismissal (user started composing).

    Returns:
        Card state: full ``card`` while setup is incomplete, one ``ready_line``
        when setup is complete but nothing was ever sent, ``quiet`` otherwise.
    """
    if has_messages or first_send_completed:
        return ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)

    provider_done = bool(readiness.native_send_supported)
    if provider_done and has_model:
        if guidance_dismissed:
            return ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)
        return ConsoleSetupCardState(mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY)

    provider_name = str(provider_label or "Provider").strip() or "Provider"
    step_one = ConsoleSetupStep(
        state="done" if provider_done else "active",
        label=(
            "Provider ready"
            if provider_done
            else _STEP_ONE_LABELS.get(readiness.label, "Finish provider setup")
        ),
        detail=f"{provider_name} ready" if provider_done else "",
    )
    # Step two only counts as done once the provider is actually ready AND a
    # model is set: template defaults (e.g. gpt-4o) must not pre-check the
    # step on a virgin profile where the provider is still blocked.
    step_two = ConsoleSetupStep(
        state=(
            "done"
            if provider_done and has_model
            else ("active" if provider_done else "pending")
        ),
        label="Pick a model",
    )
    step_three = ConsoleSetupStep(
        state="pending",
        label="Send your first message",
        detail=CONSOLE_SETUP_STEP_THREE_DETAIL,
    )
    return ConsoleSetupCardState(
        mode="card",
        steps=(step_one, step_two, step_three),
    )
