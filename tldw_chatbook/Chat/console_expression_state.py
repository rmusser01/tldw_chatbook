"""Pure Console expression precedence for live and historical avatars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

EXPRESSION_STATES = ("idle", "thinking", "speaking", "error")
EXPRESSION_IMAGE_STATES = ("thinking", "speaking", "error")

ExpressionSource = Literal["idle", "operational", "explicit", "historical"]

_STATUS_TO_STATE = {
    "pending": "thinking",
    "streaming": "speaking",
    "complete": "idle",
    "stopped": "idle",
    "failed": "error",
}


@dataclass(frozen=True, slots=True)
class CharacterEmoteHistoryIdentity:
    """Bounded immutable identity required to restore one final expression."""

    actor_id: int | None
    pack_id: int | None
    pack_version_id: int | None
    expression_key: str | None
    expression_id: int | None
    asset_id: int | None


@dataclass(frozen=True, slots=True)
class ConsoleExpressionSelection:
    """One precedence-resolved avatar request without display overrides."""

    state: str
    source: ExpressionSource
    message_id: str | None = None
    history_identity: CharacterEmoteHistoryIdentity | None = None


def resolve_console_expression_selection(
    store,
    active_session_id,
    *,
    react_enabled: bool,
    explicit_message_id: str | None = None,
    explicit_state: str | None = None,
) -> ConsoleExpressionSelection:
    """Return the operational, live-explicit, or final historical selection.

    Args:
        store: Console chat store that owns the session messages.
        active_session_id: Session whose latest assistant message controls state.
        react_enabled: Whether automatic character reactions are enabled.
        explicit_message_id: Streaming message associated with an explicit event.
        explicit_state: Most recent normalized explicit state for that message.

    Returns:
        The precedence-resolved selection before manual display overrides.
    """

    idle = ConsoleExpressionSelection("idle", "idle")
    if not react_enabled or active_session_id is None or store is None:
        return idle
    try:
        messages = store.messages_for_session(active_session_id)
    except Exception:
        return idle
    for message in reversed(messages):
        if getattr(message, "role", None) is not ConsoleMessageRole.ASSISTANT:
            continue
        message_id = getattr(message, "id", None)
        status = getattr(message, "status", "complete")
        if status == "streaming" and (
            explicit_state is not None and explicit_message_id == message_id
        ):
            return ConsoleExpressionSelection(
                explicit_state,
                "explicit",
                message_id=message_id,
            )
        if status == "complete":
            emote = getattr(getattr(message, "metadata", None), "character_emote", None)
            mood_label = getattr(emote, "mood_label", None)
            if isinstance(mood_label, str) and mood_label:
                return ConsoleExpressionSelection(
                    mood_label,
                    "historical",
                    message_id=message_id,
                    history_identity=CharacterEmoteHistoryIdentity(
                        actor_id=getattr(emote, "actor_id", None),
                        pack_id=getattr(emote, "pack_id", None),
                        pack_version_id=getattr(emote, "pack_version_id", None),
                        expression_key=getattr(emote, "expression_key", None),
                        expression_id=getattr(emote, "expression_id", None),
                        asset_id=getattr(emote, "asset_id", None),
                    ),
                )
        state = _STATUS_TO_STATE.get(status, "idle")
        return ConsoleExpressionSelection(
            state,
            "idle" if state == "idle" else "operational",
            message_id=message_id,
        )
    return idle


def resolve_console_expression_state(
    store,
    active_session_id,
    *,
    react_enabled: bool,
) -> str:
    """Return the legacy state string for callers without live-event context."""

    return resolve_console_expression_selection(
        store,
        active_session_id,
        react_enabled=react_enabled,
    ).state
