"""User-facing copy for a Console send that is over its context budget.

TASK-34350, owner ruling 2026-10-03: a send over budget prompts to compact,
auto-compacts when that is enabled, or alerts when compacting cannot make
room. Before this, Ask mode blocked with "Review and approve compaction before
sending again" and offered nothing to approve, and a send that compaction
could not help was refused with copy that treated an estimated context window
as fact (the 4,096-token fallback fixed in TASK-33940.5 refused every send).

Pure functions over token counts: no Console widget, store, or controller
imports.
"""

from __future__ import annotations

from enum import Enum

#: Where the response reservation is set (``MODEL_FIELD_LABELS["max_tokens"]``).
MAX_TOKENS_SETTING = "Max tokens in Conversation settings > Model and generation"
#: Where a model's real context window is set; the same destination the
#: Context and memory view names for an estimated window.
MODEL_WINDOW_SETTING = "F4 Settings > Providers & Models"


class ContextOverflowCause(str, Enum):
    """Why compacting older turns cannot make a send fit."""

    #: Max tokens plus the safety margin already fill the whole window.
    NO_INPUT_CAPACITY = "no_input_capacity"
    #: The system prompt, tools and attached context alone exceed what is left.
    MANDATORY_EXCEEDS = "mandatory_exceeds"
    #: The request is over the limit and no older complete turn can be summarized.
    NOTHING_TO_COMPACT = "nothing_to_compact"


def _tokens(value: int) -> str:
    return f"{value:,}"


def _window_phrase(window_tokens: int | None, *, estimated: bool) -> str:
    if window_tokens is None:
        return "context window"
    phrase = f"{_tokens(window_tokens)}-token context window"
    return f"{phrase} (an estimate)" if estimated else phrase


def _estimate_hint(estimated: bool) -> str:
    if not estimated:
        return ""
    return (
        " This model's context window is an estimate; if it is larger, set "
        f"the real value in {MODEL_WINDOW_SETTING}."
    )


def context_overflow_alert_copy(
    cause: ContextOverflowCause,
    *,
    model: str,
    window_tokens: int | None,
    window_estimated: bool,
    response_tokens: int,
    input_ceiling_tokens: int | None,
) -> str:
    """Compose the alert for a send that compacting cannot make fit.

    Args:
        cause: What leaves no room.
        model: The selected model's name.
        window_tokens: The context window the limit was computed from.
        window_estimated: True when that window is a fallback estimate rather
            than detected or set by the user.
        response_tokens: The response reservation (Max tokens).
        input_ceiling_tokens: Input tokens the window leaves for the request.

    Returns:
        Copy that says the message was not sent, what fills the window, the
        setting that changes it, and, for an estimate, where to set the real
        window.
    """

    window = _window_phrase(window_tokens, estimated=window_estimated)
    hint = _estimate_hint(window_estimated)
    if cause is ContextOverflowCause.NO_INPUT_CAPACITY:
        return (
            f"Your message was not sent: {model}'s {window} is used up by the "
            f"response reservation, Max tokens ({_tokens(response_tokens)}), and "
            "the safety margin, so compacting older turns cannot make room. "
            f"Lower {MAX_TOKENS_SETTING}.{hint}"
        )
    if cause is ContextOverflowCause.MANDATORY_EXCEEDS:
        room = (
            f"the {_tokens(input_ceiling_tokens)} tokens"
            if input_ceiling_tokens is not None
            else "the room"
        )
        return (
            "Your message was not sent: the system prompt, tools and attached "
            f"context need more than {room} {model}'s {window} leaves for a "
            "request, so compacting older turns cannot make room. Remove "
            f"attached sources or tools, or lower {MAX_TOKENS_SETTING}.{hint}"
        )
    return (
        f"Your message was not sent: it does not fit {model}'s {window}, and "
        "there are no older complete turns to compact. Shorten the message or "
        f"its attachments, lower {MAX_TOKENS_SETTING}, or start a new chat."
        f"{hint}"
    )


def _estimate_window_sentence(estimated: bool) -> str:
    if not estimated:
        return ""
    return (
        " The budget comes from an estimated context window; set the real "
        f"value in {MODEL_WINDOW_SETTING}."
    )


def compaction_prompt_copy(
    *,
    used_tokens: int,
    budget_tokens: int,
    estimated: bool,
) -> str:
    """Compose the one-line status for a send held at the threshold.

    Args:
        used_tokens: Conversation tokens this send would carry.
        budget_tokens: The conversation budget the threshold is a share of.
        estimated: True when the budget comes from an estimated window.

    Returns:
        Copy stating the hold, the numbers and the three choices.
    """

    return (
        "Your message is held: this chat reached its compaction threshold "
        f"({_tokens(used_tokens)} of {_tokens(budget_tokens)} tokens). Choose "
        "Compact and send, Send without compacting, or Cancel."
        f"{_estimate_window_sentence(estimated)}"
    )


def compaction_hold_detail(
    *,
    used_tokens: int,
    budget_tokens: int,
    estimated: bool,
) -> tuple[str, str]:
    """Compose the hold card's two detail rows.

    Args:
        used_tokens: Conversation tokens this send would carry.
        budget_tokens: The conversation budget the threshold is a share of.
        estimated: True when the budget comes from an estimated window.

    Returns:
        ``(usage, choices)``: the numbers, then what each action does.
    """

    usage = (
        f"Context: {_tokens(used_tokens)} of {_tokens(budget_tokens)} tokens; "
        "this chat reached its compaction threshold."
        f"{_estimate_window_sentence(estimated)}"
    )
    choices = (
        "Nothing was sent. Compact and send summarizes older turns first "
        "(one extra model call). Send without compacting sends it as is. "
        "Cancel puts it back in the composer."
    )
    return usage, choices
