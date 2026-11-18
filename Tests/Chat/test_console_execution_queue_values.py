"""Original queue states survive the closed selected-display value boundary."""

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleControllerActivity
from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError
from tldw_chatbook.Chat.console_execution_queue_values import (
    ConsoleQueueDisplay,
    queue_display,
    queue_presentation,
)
from tldw_chatbook.Chat.console_prompt_queue import ConsolePromptQueueRegistry


def _facts(*, accepted=False):
    registry = ConsolePromptQueueRegistry()
    snapshot = registry.snapshot("session")
    activity = ConsoleControllerActivity(
        "session", accepted, False, accepted, False, 0, False, False
    )
    return registry, ConsoleQueueDisplay(
        snapshot,
        activity,
        False,
        False,
        None,
        "",
        None,
        False,
        False,
        False,
        accepted,
        False,
    )


def test_selected_queue_retains_original_send_preparing_queue_full_and_start_precedence():
    registry, facts = _facts()

    def shown(value, *, sending=False, collapsed=False):
        detached = queue_display(queue_display(value, False), True)
        assert detached == value and detached is not value
        return queue_presentation(detached, sending=sending, collapsed=collapsed)

    assert shown(facts).send_label == "Send"
    assert shown(facts, sending=True).send_enabled is False
    preparing = replace(
        facts,
        activity=replace(
            facts.activity, occupies_slot=True, preparing_before_acceptance=True
        ),
    )
    assert shown(preparing).send_label == "Preparing..."
    assert not shown(preparing).send_enabled
    registry.begin_chain(
        "session", context_epoch=1, expected_revision=facts.snapshot.revision
    )
    queued = replace(
        facts,
        snapshot=registry.snapshot("session"),
        activity=replace(facts.activity, occupies_slot=True, accepted_live_turn=True),
    )
    assert shown(queued).send_label == "Queue" and shown(queued).send_enabled
    for number in range(10):
        snapshot = registry.admit(
            "session",
            text=f"Actual queue entry {number}",
            expected_revision=queued.snapshot.revision,
        ).snapshot
        queued = replace(
            queued,
            snapshot=snapshot,
            activity=replace(queued.activity, queued_count=snapshot.total_count),
        )
    full = shown(queued)
    assert full.count == 10 and full.next_preview == "Actual queue entry 0"
    assert (
        full.send_label == "Queue full" and not full.send_enabled and full.shelf_visible
    )
    assert not shown(queued, collapsed=True).shelf_visible
    assert shown(replace(queued, chat_start_accepted=True)).send_label == "Running"
    prepared = replace(queued, chat_start_prepared=True, chat_start_accepted=True)
    assert shown(prepared).send_enabled and shown(prepared).send_label == "Send"
    received = replace(prepared, received_preparing=True)
    assert (
        shown(received).send_label == "Preparing..."
        and not shown(received).send_enabled
    )


def test_queue_codec_refuses_foreign_session_and_non_values():
    _, facts = _facts()
    payload = queue_display(facts, False)
    for changed in (
        dict(payload, controller=object()),
        dict(payload, run_stop_allowed=1),
        dict(payload, activity=dict(payload["activity"], session_id="other")),
        dict(payload, snapshot=dict(payload["snapshot"], mode="unknown")),
    ):
        with pytest.raises(ConsoleProtocolError):
            queue_display(changed, True)


def test_recovery_and_acknowledgement_keep_original_queue_priority():
    from tldw_chatbook.Chat.console_prompt_queue import (
        PromptQueueMode,
        PromptQueuePauseReason,
    )

    _, facts = _facts()
    failed = replace(
        facts,
        snapshot=replace(
            facts.snapshot,
            mode=PromptQueueMode.PAUSED,
            pause_reason=PromptQueuePauseReason.FAILED,
        ),
        failed_turn_preview="original prompt",
        dispatch_recovery_blocks_queue=True,
    )
    display = queue_presentation(failed, sending=False, collapsed=False)
    assert (
        display.state_label == "Paused for response recovery"
        and not display.pause_enabled
    )
    unsent = replace(
        failed, turn_recovery_id="turn", turn_recovery_reason="Provider refused"
    )
    display = queue_presentation(unsent, sending=True, collapsed=False)
    assert (
        display.turn_recovery_id == "turn" and display.primary_action == "turn-recovery"
    )
    assert display.shelf_visible and not display.send_enabled
