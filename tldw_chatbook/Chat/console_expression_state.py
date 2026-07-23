"""Pure, DB-free derivation of the Console 'expression state' for the reactive
character avatar (P3d). Reads only the store's in-memory message statuses -- safe
to call on the 0.2s transcript poll tick.

State machine (from the active session's last assistant message status):
  pending   -> "thinking"   (created, awaiting first token)
  streaming -> "speaking"   (tokens flowing)
  complete  -> "idle"
  stopped   -> "idle"       (user stop is not an error)
  failed    -> "error"
  (no assistant message / no session / react disabled) -> "idle"
"""
from __future__ import annotations

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

EXPRESSION_STATES = ("idle", "thinking", "speaking", "error")
EXPRESSION_IMAGE_STATES = ("thinking", "speaking", "error")

_STATUS_TO_STATE = {
    "pending": "thinking",
    "streaming": "speaking",
    "complete": "idle",
    "stopped": "idle",
    "failed": "error",
}


def resolve_console_expression_state(store, active_session_id, *, react_enabled: bool) -> str:
    """Return the current expression state for the active Console session.

    Args:
        store: the ConsoleChatStore (exposes ``messages_for_session``).
        active_session_id: the live session id, or None.
        react_enabled: whether reactive swapping is enabled; when False, always idle.

    Returns:
        One of EXPRESSION_STATES. Never raises (any lookup failure -> "idle").
    """
    if not react_enabled or active_session_id is None or store is None:
        return "idle"
    try:
        messages = store.messages_for_session(active_session_id)
    except Exception:
        return "idle"
    # Scan from the end and stop at the first assistant turn: the active
    # turn is at/near the tail, so this is O(1) amortized rather than a full
    # forward walk of the transcript on every 0.2s Console tick. (The
    # ``messages_for_session`` snapshot above is itself O(n), but that cost is
    # pre-existing and shared with the controller's own streaming accessor,
    # and only paid on active-streaming ticks.)
    for message in reversed(messages):
        if getattr(message, "role", None) is ConsoleMessageRole.ASSISTANT:
            return _STATUS_TO_STATE.get(getattr(message, "status", "complete"), "idle")
    return "idle"
