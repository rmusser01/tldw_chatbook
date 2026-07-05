"""Pure first-run setup-card state contracts for the native Console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsReadiness

CONSOLE_SETUP_CARD_TITLE = "Get started"
CONSOLE_READY_EMPTY_COPY = "Ready — type a message to begin."
CONSOLE_QUIET_EMPTY_COPY = "No messages yet."
CONSOLE_SETUP_STEP_GLYPHS = {"done": "✓", "active": "●", "pending": "○"}

_STEP_ONE_LABELS = {
    "Missing key": "Add an API key",
    "Invalid URL": "Save the provider endpoint",
    "Endpoint not saved": "Save the provider endpoint",
    "Unknown": "Choose a supported provider",
    "Pending": "Choose a send-capable provider",
}
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
    step_two = ConsoleSetupStep(
        state="done" if has_model else ("active" if provider_done else "pending"),
        label="Pick a model",
    )
    step_three = ConsoleSetupStep(
        state="pending",
        label="Send your first message",
        detail="Type below, Enter to send",
    )
    return ConsoleSetupCardState(
        mode="card",
        steps=(step_one, step_two, step_three),
    )
