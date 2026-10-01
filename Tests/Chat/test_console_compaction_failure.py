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
    _message,
    _prepare,
    _Repository,
    _resolution,
    _resolved,
    _transaction_inputs,
)
from tldw_chatbook.Chat.console_compaction_failure import (
    compaction_failure_copy,
    compaction_spend_copy,
)
from tldw_chatbook.Chat.console_context_compaction import (
    CompactionPromptSnapshot,
    CompactionRetryFence,
    CompactionTerminal,
    ConsoleCompactionService,
    DurableMessageSnapshot,
    EffectiveMemoryKind,
    EffectiveMemoryResult,
    compaction_retry_fence,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage

_PRICED = ProviderUsage(
    uncached_input=2_000, output=50, provider="openai", model="gpt-4.1-mini"
)
_UNPRICED = replace(_PRICED, model="no-such-model-anywhere")

_FORBIDDEN = "provider request was not sent"


# The cost comes from the real pricing catalog, which reads the guarded config
# loader; under the per-test sandbox that admission fails closed and the copy
# (correctly) drops the price. Keep the bootstrap profile for this one node.
@pytest.mark.bootstrap_profile
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
    # The pause is in memory only; the copy must not promise more.
    assert copy.endswith("or restart the app.")
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
    copy = compaction_failure_copy("no_complete_durable_units", manual=True)

    assert copy == (
        "Nothing to compact yet: this chat has no older complete turns to "
        "summarize."
    )


def test_manual_compact_now_with_no_useful_summary_gives_a_next_step() -> None:
    """The planner's catch-all is a size problem, not 'nothing to do'."""

    copy = compaction_failure_copy("no_positive_useful_summary_allowance", manual=True)

    assert copy.startswith("Compaction failed and nothing changed: ")
    assert "fit under the chat's target size and make it smaller" in copy
    assert "raise Conversation max tokens" in copy
    assert "spent" not in copy  # the planner made no summary call


def test_saved_range_faults_are_not_worded_as_a_size_problem() -> None:
    for reason in (
        "invalid_effective_range_memory",
        "invalid_effective_range_anchors",
        "range_projection_units_mismatch",
    ):
        copy = compaction_failure_copy(reason, manual=False)

        assert "saved memory range does not match its messages" in copy, reason
        assert "target size" not in copy, reason


@pytest.mark.parametrize(
    ("reason", "paused"),
    [("invalid_summary_output", True), ("branch_memory_changed_before_commit", False)],
)
def test_omitted_copy_discloses_a_billed_failure_the_send_survived(
    reason: str, paused: bool
) -> None:
    copy = compaction_failure_copy(
        reason, manual=False, usage=_UNPRICED, attempted=True, omitted=True
    )

    assert copy.startswith("Your message was sent without compacting")
    assert "2,000 input + 50 output tokens." in copy
    assert ("Automatic compaction is paused" in copy) is paused
    assert "not sent" not in copy


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
async def test_each_route_keeps_its_own_pause() -> None:
    """A failure on another model must not replace the sends' pause."""

    repository = _Repository()
    gateway = _Gateway(text="")
    service = ConsoleCompactionService(repository, gateway)
    inputs = _transaction_inputs()
    prefix = inputs[2]
    sends = CompactionRetryFence("conversation-1", "main", prefix[:4], "openai/main")
    auxiliary = CompactionRetryFence("conversation-1", "aux", prefix[:4], "openai/aux")

    await _compact(service, inputs, sends)
    await _compact(service, inputs, auxiliary, honor=False)
    assert gateway.calls == 2

    assert (await _compact(service, inputs, sends)).suppressed is True
    assert (await _compact(service, inputs, auxiliary)).suppressed is True
    assert gateway.calls == 2


@pytest.mark.asyncio
async def test_a_commit_that_can_never_land_fails_before_the_billed_call() -> None:
    """The live-session P0 shape (no parent chain) must cost nothing."""

    repository = _Repository()
    gateway = _Gateway()
    service = ConsoleCompactionService(repository, gateway)
    plan, prompt, prefix, admission, branch_commit = _transaction_inputs()
    orphaned = replace(
        branch_commit,
        durable_lineage=tuple(
            replace(row, parent_message_id=None) for row in branch_commit.durable_lineage
        ),
    )

    result = await _compact(
        service, (plan, prompt, prefix, admission, orphaned), None
    )

    assert result.terminal is CompactionTerminal.FAILED
    assert result.reason == "memory_commit_failed"
    assert result.attempted is False
    assert gateway.calls == 0
    assert repository.starts == []


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


def _lineage_fence(
    lineage: tuple[DurableMessageSnapshot, ...], *, active_request: bool = True
) -> CompactionRetryFence:
    """The fence the controller builds for ``lineage`` (the durable path)."""

    return compaction_retry_fence(
        "conversation-1",
        _resolution(),
        CompactionPromptSnapshot("Preserve decisions."),
        _resolved(),
        EffectiveMemoryResult(EffectiveMemoryKind.RAW),
        lineage,
        active_request=active_request,
    )


def _reply(message_id: str, status: str, content: str = "") -> DurableMessageSnapshot:
    return replace(_message(message_id, "assistant", content), status=status)


# The paused lineage is u0 a0 u1 a1 u2 a2 (``_transaction_inputs``' prefix).
# Each shape is the durable path a request sees when it is built.
_RESUME_SHAPES = {
    # Continue persists its new empty reply before the preflight runs.
    "continue": lambda p: p + (_reply("a-continue", "pending"),),
    # Regenerate forks an unsaved sibling: the durable path ends at u2.
    "regenerate": lambda p: p[:-1],
    # Retry of a regenerate whose sibling failed: the sibling is the request.
    "retry-failed-regenerate-sibling": lambda p: p[:-1]
    + (_reply("a2-sibling", "failed", "partial"),),
    # A send appends after the whole paused lineage.
    "send": lambda p: p + (_message("u3", "user", "next"),),
}
_LIFTING_SHAPES = {
    # The User Guide: deleting the latest exchange, then sending, lifts it.
    "send-after-deleting-the-latest-exchange": lambda p: p[:-2]
    + (_message("u3", "user", "next"),),
    "regenerate-after-editing-the-question": lambda p: p[:-2]
    + (replace(p[-2], version=2, content="edited question"),),
    "continue-after-editing-the-reply": lambda p: p[:-1]
    + (
        replace(p[-1], version=2, content="edited reply"),
        _reply("a-continue", "pending"),
    ),
    "regenerate-an-earlier-reply": lambda p: p[:3],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shape", "paused"),
    [(name, True) for name in _RESUME_SHAPES]
    + [(name, False) for name in _LIFTING_SHAPES],
)
async def test_a_compact_now_pause_covers_requests_that_resume_its_latest_exchange(
    shape: str, paused: bool
) -> None:
    """AC#5: Compact now fences the whole lineage, latest exchange included.

    A Continue, a regenerate, or a Retry of a failed regenerate sibling
    starts its request at that exchange's user turn, so its own history is
    shorter than the pause's. The pause still covers it while every paused
    row the request carries is unchanged; an edit, a delete or another
    request lifts it.
    """

    repository = _Repository()
    gateway = _Gateway(text="")
    service = ConsoleCompactionService(repository, gateway)
    inputs = _transaction_inputs()
    prefix = inputs[2]
    compact_now = _lineage_fence(prefix, active_request=False)
    assert compact_now.history == prefix
    await _compact(service, inputs, compact_now, honor=False)
    assert gateway.calls == 1

    build = {**_RESUME_SHAPES, **_LIFTING_SHAPES}[shape]
    result = await _compact(service, inputs, _lineage_fence(build(prefix)))

    assert result.suppressed is paused
    assert gateway.calls == (1 if paused else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["continue", "regenerate"])
async def test_a_send_pause_covers_the_previous_reply_after_a_discard(
    shape: str,
) -> None:
    """AC#5: Discard, then Continue or regenerate the reply before it.

    A failed send pauses on the history before its own question. Discarding
    that question and resuming the reply before it is no change to that
    history, exactly like Discard plus a fresh send, so it stays paused.
    """

    repository = _Repository()
    gateway = _Gateway(text="")
    service = ConsoleCompactionService(repository, gateway)
    inputs = _transaction_inputs()
    prefix = inputs[2]
    send = _lineage_fence(prefix + (_message("u3", "user", "next"),))
    assert send.history == prefix
    await _compact(service, inputs, send)
    assert gateway.calls == 1

    result = await _compact(
        service, inputs, _lineage_fence(_RESUME_SHAPES[shape](prefix))
    )

    assert result.suppressed is True
    assert gateway.calls == 1
