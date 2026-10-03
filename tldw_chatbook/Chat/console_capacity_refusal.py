"""Plain-words copy for a send refused before dispatch because it cannot fit.

TASK-34100.5 AC#3 (entry-exit-handoff-03). The context policy reports why a
request cannot fit in internal terms ("Response reservation and safety margin
leave no model input capacity", "mandatory context"), and the old refusal
added "Repair the model limit…" whatever the reason. A first-time user on a
model whose window was only a fallback guess could not tell what was wrong or
what to do. This module turns the limiting reason into copy that names the
model, says the context size isn't known ONLY when the window is a guess, and
names the fix: Switch model (Alt+M) always; Set context size when the window
is unknown; a lower reply limit when the reply reservation is what fills it.
"""

from __future__ import annotations

from collections.abc import Sequence

#: The policy's own reason strings (console_context_policy.resolve_context_policy).
RESERVATION_REASON = (
    "Response reservation and safety margin leave no model input capacity."
)
MANDATORY_REASON = "Mandatory request material leaves no conversation capacity."

_SWITCH = "Switch model (Alt+M) to one with a larger, known window"
_SET_SIZE = (
    "or Set context size: add a context_window for {model} under "
    "model_capabilities.models in config.toml"
)


def capacity_refusal_copy(
    *,
    model: str | None,
    window_tokens: int | None,
    window_verified: bool,
    response_tokens: int | None,
    reasons: Sequence[str] = (),
    no_older_turns: bool = False,
) -> str:
    """Return the refusal shown when a request cannot fit the selected model.

    Args:
        model: The selected model id.
        window_tokens: The context window the request was sized against.
        window_verified: Whether that window came from the catalog or the
            server rather than a fallback guess.
        response_tokens: Tokens reserved for the reply.
        reasons: The context policy's validation errors.
        no_older_turns: True when compaction had nothing older to summarize.

    Returns:
        One or two plain sentences naming the model, the cause and the fix.
    """
    name = (model or "").strip() or "The selected model"
    window = f"{window_tokens:,}" if isinstance(window_tokens, int) else ""
    reply = f"{response_tokens:,}" if isinstance(response_tokens, int) else ""
    reserved_all = RESERVATION_REASON in reasons
    if window_tokens is None or not window_verified:
        assumed = f", so chatbook assumed {window} tokens" if window else ""
        reserve = (
            f" and reserved {reply} of them for the reply" if reserved_all and reply else ""
        )
        return (
            f"{name}'s context size isn't known{assumed}{reserve}, and this "
            f"message can't fit. {_SWITCH}, {_SET_SIZE.format(model=name)}."
        )
    if reserved_all:
        return (
            f"{name}'s {window}-token window is used up by the {reply}-token "
            "reply limit. Lower the reply limit (max tokens) in Console "
            f"settings, or {_SWITCH}."
        )
    older = ", and there are no older turns to summarize" if no_older_turns else ""
    return (
        f"This message, the instructions and the tools need more room than "
        f"{name}'s {window}-token window allows{older}. Shorten the message, "
        f"or {_SWITCH}."
    )


def is_repeat_of_last_row(store: object, session_id: str, copy: str) -> bool:
    """Whether the session already ends with this exact refusal row.

    A Retry refused again for the same reason must not append a second copy
    of the row it is already showing.
    """
    try:
        messages = store.messages_for_session(session_id)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 -- an unknown session appends as before
        return False
    for message in reversed(tuple(messages)):
        role = getattr(getattr(message, "role", None), "value", None)
        if role == "assistant" and getattr(message, "status", "") == "failed":
            continue  # the refused turn's own empty assistant row
        return role == "system" and getattr(message, "content", None) == copy
    return False
