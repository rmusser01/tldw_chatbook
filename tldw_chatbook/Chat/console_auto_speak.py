"""Pure eligibility and consent policy for automatic Console reply speech."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_speech_preferences import (
    ConsoleSpeechPreferences,
    is_console_speech_destination,
)


class AutoSpeakDisposition(str, Enum):
    """Reason automatic reply speech should proceed or remain silent."""

    SPEAK = "speak"
    DISABLED = "disabled"
    PAUSED = "paused"
    NEEDS_CONSENT = "needs_consent"
    HANDSFREE_OWNS = "handsfree_owns"
    BACKGROUND = "background"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True, slots=True)
class AutoSpeakContext:
    """Current conversation, destination, and speech-ownership state."""

    preferences: ConsoleSpeechPreferences
    destination_fingerprint: str
    active_session_id: str
    hands_free_active: bool


def decide_auto_speak(
    message: ConsoleChatMessage,
    *,
    session_id: str,
    context: AutoSpeakContext,
) -> AutoSpeakDisposition:
    """Return the fail-closed disposition for one completed reply."""
    if not _eligible_reply(message):
        return AutoSpeakDisposition.INELIGIBLE
    if not isinstance(context, AutoSpeakContext):
        return AutoSpeakDisposition.DISABLED

    preferences = _validated_preferences(context.preferences)
    if preferences is None or not preferences.auto_speak:
        return AutoSpeakDisposition.DISABLED
    if preferences.paused:
        return AutoSpeakDisposition.PAUSED

    if (
        type(session_id) is not str
        or not session_id
        or type(context.active_session_id) is not str
        or not context.active_session_id
        or context.active_session_id != session_id
    ):
        return AutoSpeakDisposition.BACKGROUND

    if context.hands_free_active is not False:
        return AutoSpeakDisposition.HANDSFREE_OWNS

    destination = context.destination_fingerprint
    if (
        not is_console_speech_destination(destination)
        or preferences.consent_destination != destination
    ):
        return AutoSpeakDisposition.NEEDS_CONSENT
    return AutoSpeakDisposition.SPEAK


def _eligible_reply(message: object) -> bool:
    return bool(
        isinstance(message, ConsoleChatMessage)
        and message.role is ConsoleMessageRole.ASSISTANT
        and message.status == "complete"
        and type(message.content) is str
        and message.content.strip()
    )


def _validated_preferences(value: object) -> ConsoleSpeechPreferences | None:
    if not isinstance(value, ConsoleSpeechPreferences):
        return None
    try:
        return ConsoleSpeechPreferences(
            auto_speak=value.auto_speak,
            paused=value.paused,
            consent_destination=value.consent_destination,
            consent_version=value.consent_version,
        )
    except (AttributeError, TypeError, ValueError):
        return None
