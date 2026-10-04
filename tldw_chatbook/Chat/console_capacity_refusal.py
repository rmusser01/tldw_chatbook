"""Keep a repeated pre-dispatch refusal from adding a second identical row.

TASK-34100.5 AC#3 (entry-exit-handoff-03): each refused Retry of a turn the
context preflight blocked appended another copy of the same system row. The
refusal's copy itself now comes from TASK-34350's context-overflow alert
(``console_context_budget_copy``), which names the model, what fills the
window and the setting that changes it; this module only keeps the row single.
"""

from __future__ import annotations


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
