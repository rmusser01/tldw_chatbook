"""User-facing copy for a Console compaction that did not complete.

TASK-33621.3. Every compaction failure used to surface as one fixed line --
"Conversation compaction did not complete; the provider request was not
sent." -- whatever the cause, after a summary call that had already been
billed. The copy here names the cause, discloses what the failed summary call
spent (and its cost when the model is priced), says plainly whether the
user's message was sent, and gives a next step: omit older turns, raise the
budget, turn compaction off, or start a new chat.

Pure functions over content-free inputs (a reason code and a usage record):
no Console widget, store, or controller imports.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from tldw_chatbook.Chat.cost_display import format_cost_amount
from tldw_chatbook.Chat.provider_usage import ProviderUsage

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_context_compaction import (
        CompactionTransactionResult,
    )

#: Reason code -> the clause completing "... because <clause>".
_REASON_CLAUSES: dict[str, str] = {
    "auxiliary_timed_out": "the summary call timed out",
    "auxiliary_provider_failed": "the summary call failed at the provider",
    "invalid_summary_output": (
        "the model returned an empty, oversized or malformed summary"
    ),
    "summary_projection_failed": "the summary could not be fitted into the request",
    "summary_did_not_make_progress": (
        "the summary plus the recent turns kept with it would still be over "
        "the target size"
    ),
    "memory_commit_failed": "the summary could not be saved to this chat",
    "admission_changed": "the conversation changed while it was being summarized",
    "branch_memory_changed_before_commit": (
        "the conversation changed while it was being summarized"
    ),
    "compaction_already_running": (
        "another compaction is already running for this chat"
    ),
    "invalid_automatic_admission": (
        "the conversation changed before compaction could start"
    ),
    "invalid_manual_admission": (
        "the conversation changed before compaction could start"
    ),
    # Planner outcomes: no summary call was made for any of these.
    "plan_unreachable": (
        "one summary call cannot bring this chat under its target size"
    ),
    # A catch-all: the kept recent turns alone fill the target, or no summary
    # would be smaller than the turns it replaces, or the older turns do not
    # fit one summary call.
    "no_positive_useful_summary_allowance": (
        "no summary of the older turns would both fit under the chat's target "
        "size and make it smaller"
    ),
    "no_complete_durable_units": (
        "this chat has no older complete turns to summarize"
    ),
    "unknown_or_empty_budget": "this chat's conversation budget is unknown",
    "automatic_visual_input_limit_exceeded": (
        "the older turns carry more images than one summary call accepts"
    ),
    "automatic_visual_input_unsupported": (
        "the older turns carry images this model cannot read"
    ),
}
_DEFAULT_CLAUSE = "compaction did not complete"
#: Integrity faults in a saved memory range (invalid_effective_range_*,
#: range_projection_*): not a size problem, so never worded as one.
_RANGE_FAULT_CLAUSE = "this chat's saved memory range does not match its messages"

#: The planner outcome that means "nothing to summarize yet"; a manual
#: Compact now says so plainly, with no failure and no next step.
_NOTHING_TO_COMPACT = "no_complete_durable_units"

#: Failures a plain retry can fix; everything else needs a settings change.
_TRANSIENT_REASONS = frozenset(
    {
        "auxiliary_timed_out",
        "auxiliary_provider_failed",
        "admission_changed",
        "branch_memory_changed_before_commit",
        "compaction_already_running",
        "invalid_automatic_admission",
        "invalid_manual_admission",
    }
)

#: Reasons that mean the conversation moved under the attempt; the send can
#: simply be retried, so automatic compaction is not paused for them.
_STALE_REASONS = frozenset({"admission_changed", "branch_memory_changed_before_commit"})

NEXT_STEP_SETTINGS = (
    "Next: in Conversation settings > Context and memory, raise Conversation "
    "max tokens, set If compaction fails to Omit older context to send without "
    "compacting, or set When limit nears to Off; or start a new chat."
)
#: The pause lives in memory (ConsoleCompactionService), so say it ends on
#: restart rather than promise it outlives the app session.
_PAUSED = (
    "Automatic compaction stays paused for this chat until you change these "
    "settings or its earlier messages, or restart the app."
)


def compaction_failure_reason_clause(reason: str | None) -> str:
    """Return the plain-language cause for one compaction reason code.

    Args:
        reason: A ``CompactionTransactionResult.reason`` code, or None.

    Returns:
        A lowercase clause suitable after "because".
    """

    if (
        reason is not None
        and reason not in _REASON_CLAUSES
        and (reason.startswith(("invalid_effective_range", "range_projection")))
    ):
        return _RANGE_FAULT_CLAUSE
    return _REASON_CLAUSES.get(reason or "", _DEFAULT_CLAUSE)


def _usage_cost(usage: ProviderUsage) -> Decimal | float | None:
    try:
        from tldw_chatbook.LLM_Calls.pricing_catalog import get_pricing_catalog

        breakdown = get_pricing_catalog().cost_for_usage(usage)
    except Exception:  # noqa: BLE001 -- pricing is advisory; never block copy
        return None
    return None if breakdown is None else breakdown.total


def compaction_spend_copy(usage: ProviderUsage | None, *, attempted: bool) -> str:
    """Disclose what a failed summary call spent.

    Args:
        usage: Provider-reported usage for the failed call, or None.
        attempted: Whether a summary call was actually made.

    Returns:
        One sentence, or "" when no summary call was made.
    """

    if not attempted:
        return ""
    if usage is None:
        return (
            "The summary call reported no token usage; the provider may still bill it."
        )
    input_tokens = usage.uncached_input + usage.cache_read + usage.cache_write
    text = (
        f"The failed summary call spent {input_tokens:,} input + "
        f"{usage.output:,} output tokens"
    )
    cost = _usage_cost(usage)
    if cost is not None:
        text += f" (about ${format_cost_amount(cost)})"
    return text + "."


def compaction_failure_copy(
    reason: str | None,
    *,
    manual: bool,
    usage: ProviderUsage | None = None,
    attempted: bool = False,
    suppressed: bool = False,
    omitted: bool = False,
) -> str:
    """Compose the copy shown when a compaction does not complete.

    Args:
        reason: The transaction's reason code.
        manual: True for Compact now, where no message was being sent.
        usage: What the failed summary call reported spending.
        attempted: Whether a summary call was made (and may be billed).
        suppressed: True when an earlier failure stopped this automatic
            attempt before any call.
        omitted: True when If compaction fails is Omit older context, so the
            message goes out uncompacted instead of being held.

    Returns:
        Reason-specific copy with the spend, whether the message was sent,
        and a next step. Never claims "the provider request was not sent".
    """

    if omitted:
        return _omitted_copy(reason, usage=usage)
    clause = compaction_failure_reason_clause(reason)
    retry = (
        "Try Compact now in Conversation settings > Context and memory later. "
        if reason in _TRANSIENT_REASONS
        else ""
    )
    if suppressed:
        return (
            "Your message was not sent: automatic compaction is paused for this "
            f"chat because its last attempt failed ({clause}). No new summary "
            f"call was made. {retry}{NEXT_STEP_SETTINGS}"
        )
    spend = compaction_spend_copy(usage, attempted=attempted)
    if manual and reason == _NOTHING_TO_COMPACT:
        return f"Nothing to compact yet: {clause}."
    if manual:
        head = f"Compaction failed and nothing changed: {clause}."
    else:
        head = (
            "Your message was not sent: this chat could not be compacted "
            f"because {clause}."
        )
    parts = [head]
    if spend:
        parts.append(spend)
    if not manual and reason in _STALE_REASONS:
        parts.append("Send again to retry.")
        return " ".join(parts)
    parts.append(f"{retry}{NEXT_STEP_SETTINGS}")
    if not manual and attempted:
        parts.append(_PAUSED)
    return " ".join(parts)


def transaction_failure_copy(
    result: CompactionTransactionResult,
    *,
    manual: bool,
    omitted: bool = False,
) -> str:
    """Compose failure copy straight from a compaction transaction result.

    Args:
        result: The FAILED or STALE transaction result.
        manual: True for Compact now.
        omitted: True when the send goes out uncompacted (Omit older context).

    Returns:
        The same copy as :func:`compaction_failure_copy`.
    """

    return compaction_failure_copy(
        result.reason,
        manual=manual,
        usage=result.usage,
        attempted=result.attempted,
        suppressed=result.suppressed,
        omitted=omitted,
    )


def _omitted_copy(reason: str | None, *, usage: ProviderUsage | None) -> str:
    """Disclose a billed failure that the send survived uncompacted.

    With If compaction fails set to Omit older context the message still
    goes out, so nothing blocks -- but the failed summary call was made and
    may be billed, and automatic compaction pauses. The controller shows it
    once, on the attempt that was billed; the paused sends that follow stay
    quiet.
    """

    parts = [
        "Your message was sent without compacting (If compaction fails is set "
        "to Omit older context): this chat could not be compacted because "
        f"{compaction_failure_reason_clause(reason)}.",
        compaction_spend_copy(usage, attempted=True),
    ]
    if reason not in _STALE_REASONS:
        parts.append(
            "Automatic compaction is paused for this chat until you change its "
            "compaction settings or earlier messages, or restart the app."
        )
    return " ".join(parts)
