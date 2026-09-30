"""TASK-33621.3: compaction failure copy, reason ledger and retry latch.

The end-to-end contract (real store, real durable send) lives in
``test_console_compaction_live_session.py``. This file pins the two pieces it
is built from: the copy a user reads, and the service rule that a FAILED
automatic attempt is never billed again on an unchanged conversation.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from Tests.Chat.test_console_context_compaction import (
    _Gateway,
    _prepare,
    _Repository,
    _resolution,
    _transaction_inputs,
)
from tldw_chatbook.Chat.console_compaction_failure import (
    compaction_failure_copy,
    compaction_spend_copy,
)
from tldw_chatbook.Chat.console_context_compaction import (
    CompactionRetryFence,
    CompactionTerminal,
    ConsoleCompactionService,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage

_PRICED = ProviderUsage(
    uncached_input=2_000, output=50, provider="openai", model="gpt-4.1-mini"
)
_UNPRICED = replace(_PRICED, model="no-such-model-anywhere")

_FORBIDDEN = "provider request was not sent"


def test_automatic_failure_copy_names_cause_spend_and_next_step() -> None:
    copy = compaction_failure_copy(
        "summary_did_not_make_progress",
        manual=False,
        usage=_PRICED,
        attempted=True,
    )

    assert copy.startswith("Your message was not sent: ")
    assert "still be over the target size" in copy
    assert "2,000 input + 50 output tokens (about $" in copy
    assert "raise Conversation max tokens" in copy
    assert "Omit older context" in copy
    assert "When limit nears to Off" in copy
    assert "start a new chat" in copy
    assert "stays paused" in copy
    assert _FORBIDDEN not in copy


def test_manual_failure_copy_says_nothing_changed() -> None:
    copy = compaction_failure_copy(
        "memory_commit_failed", manual=True, usage=_PRICED, attempted=True
    )

    assert copy.startswith("Compaction failed and nothing changed: ")
    assert "could not be saved" in copy
    assert "not sent" not in copy
    assert "paused" not in copy


def test_suppressed_copy_says_no_new_call_was_made() -> None:
    copy = compaction_failure_copy(
        "invalid_summary_output", manual=False, suppressed=True
    )

    assert "automatic compaction is paused" in copy
    assert "No new summary call was made." in copy
    assert "spent" not in copy
    assert _FORBIDDEN not in copy


def test_stale_copy_invites_a_resend_instead_of_a_settings_change() -> None:
    copy = compaction_failure_copy(
        "admission_changed", manual=False, usage=_PRICED, attempted=True
    )

    assert "changed while it was being summarized" in copy
    assert copy.endswith("Send again to retry.")
    assert "paused" not in copy


@pytest.mark.parametrize(
    ("usage", "attempted", "expected"),
    [
        (None, False, ""),
        (None, True, "reported no token usage"),
        (_UNPRICED, True, "2,000 input + 50 output tokens."),
    ],
)
def test_spend_copy_never_fabricates_a_price(usage, attempted, expected) -> None:
    copy = compaction_spend_copy(usage, attempted=attempted)

    if expected:
        assert expected in copy
    else:
        assert copy == ""
    assert "$" not in copy


def test_manual_compact_now_with_nothing_to_summarize_says_so_plainly() -> None:
    copy = compaction_failure_copy("no_positive_useful_summary_allowance", manual=True)

    assert copy == (
        "Compaction failed and nothing changed: summarizing this chat's older "
        "turns would not make it smaller."
    )


def test_unmapped_range_planner_reasons_read_as_unreachable_target() -> None:
    copy = compaction_failure_copy("invalid_effective_range_memory", manual=False)

    assert "cannot bring this chat under its target size" in copy
    assert "raise Conversation max tokens" in copy


def test_every_known_reason_has_specific_copy() -> None:
    generic = compaction_failure_copy("not-a-real-reason", manual=True)
    for reason in (
        "auxiliary_timed_out",
        "auxiliary_provider_failed",
        "invalid_summary_output",
        "summary_projection_failed",
        "summary_did_not_make_progress",
        "memory_commit_failed",
        "admission_changed",
        "compaction_already_running",
        "invalid_automatic_admission",
        "plan_unreachable",
        "no_positive_useful_summary_allowance",
        "no_complete_durable_units",
        "unknown_or_empty_budget",
        "automatic_visual_input_limit_exceeded",
        "automatic_visual_input_unsupported",
    ):
        copy = compaction_failure_copy(reason, manual=True)
        assert copy != generic, reason


async def _compact(service, inputs, fence, *, honor=True):
    plan, prompt, prefix, admission, branch_commit = inputs
    return await service.compact(
        admission=admission,
        branch_commit=branch_commit,
        plan=plan,
        resolution=_resolution(),
        prompt=prompt,
        current_admission=lambda: admission,
        prepare_main=_prepare,
        prefix_messages=prefix,
        retry_fence=fence,
        honor_failure_latch=honor,
    )


@pytest.mark.asyncio
async def test_failed_attempt_suppresses_rebilling_until_the_fence_changes() -> None:
    repository = _Repository()
    gateway = _Gateway(text="")
    service = ConsoleCompactionService(repository, gateway)
    inputs = _transaction_inputs()
    prefix = inputs[2]
    fence = CompactionRetryFence("conversation-1", "settings-a", prefix[:4])

    first = await _compact(service, inputs, fence)
    assert first.terminal is CompactionTerminal.FAILED
    assert first.reason == "invalid_summary_output"
    assert first.attempted is True
    assert repository.finishes[-1][1]["failure_reason"] == "invalid_summary_output"
    assert gateway.calls == 1

    # Same settings, turns only APPENDED after the failed history: no call,
    # no ledger row.
    appended = CompactionRetryFence("conversation-1", "settings-a", prefix)
    again = await _compact(service, inputs, appended)
    assert again.suppressed is True
    assert again.attempted is False
    assert gateway.calls == 1
    assert len(repository.starts) == 1

    # Compact now is an explicit action: it may always try (and bill) again.
    manual = await _compact(service, inputs, fence, honor=False)
    assert manual.attempted is True
    assert gateway.calls == 2

    # A policy change lifts the block ...
    policy_changed = CompactionRetryFence("conversation-1", "settings-b", prefix)
    assert (await _compact(service, inputs, policy_changed)).attempted is True
    assert gateway.calls == 3

    # ... and so does an edit to an earlier message.
    edited = CompactionRetryFence(
        "conversation-1",
        "settings-b",
        (replace(prefix[0], version=(prefix[0].version or 0) + 1),) + prefix[1:],
    )
    assert (await _compact(service, inputs, edited)).attempted is True
    assert gateway.calls == 4


@pytest.mark.asyncio
async def test_success_clears_the_latch() -> None:
    repository = _Repository()
    gateway = _Gateway(text="")
    service = ConsoleCompactionService(repository, gateway)
    inputs = _transaction_inputs()
    fence = CompactionRetryFence("conversation-1", "settings-a", inputs[2][:4])
    await _compact(service, inputs, fence)

    gateway.text = "Compact facts."
    succeeded = await _compact(service, inputs, fence, honor=False)
    assert succeeded.terminal is CompactionTerminal.SUCCEEDED

    gateway.text = ""
    retried = await _compact(service, inputs, fence)
    assert retried.suppressed is False
    assert retried.attempted is True
