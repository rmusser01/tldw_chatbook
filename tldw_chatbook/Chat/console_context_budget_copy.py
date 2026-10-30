"""User-facing copy for a Console send that is over its context budget.

TASK-34350, owner ruling 2026-10-03: a send over budget prompts to compact,
auto-compacts when that is enabled, or alerts when compacting cannot make
room. Before this, Ask mode blocked with "Review and approve compaction before
sending again" and offered nothing to approve, and a send that compaction
could not help was refused with copy that treated an assumed context window
as fact (the 4,096-token fallback fixed in TASK-33940.5 refused every send).

One fact, one word (TASK-33007 #12): a window no source knows is "assumed",
as Chat settings and Settings say it ("unknown, 32,000 assumed"). The
``estimated`` arguments keep their name; only the words changed.

Pure functions over token counts: no Console widget, store, or controller
imports.
"""

from __future__ import annotations

from enum import Enum

#: Where the response reservation is set (``MODEL_FIELD_LABELS["max_tokens"]``).
#: The surface is Chat settings (Ctrl+O), spelled as Settings is below.
MAX_TOKENS_SETTING = "Max tokens in Chat settings ▸ Model and generation (Ctrl+O)"
#: Where a model's real context window is set. Chat settings' Context and
#: memory view names it with this constant, in the words its Model view
#: uses (TASK-33007 capture finding 17).
MODEL_WINDOW_SETTING = "Settings ▸ Providers & Models (F4)"


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
    return f"{phrase} (assumed)" if estimated else phrase


def _estimate_hint(estimated: bool) -> str:
    if not estimated:
        return ""
    return (
        " This model's context window is assumed, not known; if it is larger, "
        f"set the real value in {MODEL_WINDOW_SETTING}."
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
        window_estimated: True when that window is assumed (a fallback)
            rather than detected or set by the user.
        response_tokens: The response reservation (Max tokens).
        input_ceiling_tokens: Input tokens the window leaves for the request.

    Returns:
        Copy that says the message was not sent, what fills the window, the
        setting that changes it, and, for an assumed one, where to set the real
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
        " The budget comes from an assumed context window; set the real "
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
        estimated: True when the budget comes from an assumed window.

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
    trigger_tokens: int | None = None,
) -> tuple[str, str, str]:
    """Compose the hold card's three detail rows.

    Args:
        used_tokens: Conversation tokens this send would carry.
        budget_tokens: The conversation budget the threshold is a share of.
        estimated: True when the budget comes from an assumed window.
        trigger_tokens: Where Ask compaction starts (TASK-33621.4 AC#1);
            the clause is left out when unknown.

    Returns:
        ``(usage, choices, preview)``: the numbers, what each action does,
        and where to preview a summary first (owner decision 2026-10-27:
        the card points to /rewind rather than embedding its preview).
    """

    threshold = "this chat reached its compaction threshold"
    if trigger_tokens is not None and budget_tokens > 0:
        share = round(trigger_tokens * 100 / budget_tokens)
        threshold = f"compaction trigger {_tokens(trigger_tokens)} tokens ({share}%)"
    usage = (
        f"Context: {_tokens(used_tokens)} of {_tokens(budget_tokens)} tokens; "
        f"{threshold}.{_estimate_window_sentence(estimated)}"
    )
    choices = (
        "Nothing was sent. Compact and send summarizes older turns first "
        "(one extra model call). Send without compacting sends it as is. "
        "Change policy opens Chat settings ▸ Context and memory. "
        "Cancel puts it back in the composer."
    )
    preview = (
        "To preview what would be summarized, use /rewind ▸ Summarize up to here."
    )
    return usage, choices, preview
