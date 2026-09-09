"""Only bound conversation/workspace activity animates the selected Buddy."""

import pytest

from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController


def scoped_adapter():
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)
    assert hasattr(adapter, "set_scope"), "Buddy lifecycle scope is missing"
    adapter.set_scope(
        session_ids=frozenset({"chat-a"}), conversation_ids=frozenset({"durable-a"})
    )
    return controller, adapter


def test_unrelated_running_or_approval_events_do_not_animate_buddy():
    controller, adapter = scoped_adapter()
    adapter.run_state("chat-b", "streaming")
    adapter.approval_round("chat-b", "question-b", pending=True)
    assert controller.snapshot().state == "idle"
    adapter.run_state("chat-a", "streaming")
    assert controller.snapshot().state == "speaking"
    adapter.approval_round("chat-a", "question-a", pending=True)
    assert controller.snapshot().state == "approval_needed"


def test_scope_change_releases_previous_leases_and_allows_new_scope():
    controller, adapter = scoped_adapter()
    adapter.run_state("chat-a", "streaming")
    adapter.set_scope(session_ids=frozenset({"chat-b"}), conversation_ids=frozenset())
    assert controller.snapshot().state == "idle"
    adapter.approval_round("chat-a", "old-question", pending=True)
    assert controller.snapshot().state == "idle"
    adapter.run_state("chat-b", "validating")
    assert controller.snapshot().state == "thinking"


def test_scoped_tool_events_require_their_explicit_session_and_wakes_match_durable_id():
    controller, adapter = scoped_adapter()
    adapter.tool_step("run-b", 1, "tool_call", session_id="chat-b")
    adapter.tool_step("unknown-run", 1, "tool_call")
    adapter.wake("durable-b", "wake-b", active=True)
    assert controller.snapshot().state == "idle"
    adapter.tool_step("run-a", 1, "tool_call", session_id="chat-a")
    assert controller.snapshot().state == "tool_running"
    adapter.release_run("run-a")
    adapter.wake("durable-a", "wake-a", active=True)
    assert controller.snapshot().state == "wake_armed"


def test_unscoped_legacy_adapter_retains_existing_behavior():
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)
    adapter.tool_step("legacy", 1, "tool_call")
    assert controller.snapshot().state == "tool_running"


@pytest.mark.parametrize("status", ["validating", "streaming"])
def test_rebinding_away_and_back_keeps_the_accepted_run_owner_until_completion(status):
    controller, adapter = scoped_adapter()
    accepted = adapter.run_state("chat-a", "validating")
    if status == "streaming":
        adapter.run_state("chat-a", status, run_owner=accepted)
    adapter.set_scope(session_ids=frozenset({"chat-b"}), conversation_ids=frozenset())
    adapter.set_scope(session_ids=frozenset({"chat-a"}), conversation_ids=frozenset())
    assert adapter.run_state("chat-a", status, replay=True) == accepted
    adapter.run_state("chat-a", "completed", run_owner=accepted)
    assert controller.snapshot().state == "idle"


def test_new_run_started_outside_scope_rejects_late_prior_completion_after_rebind():
    controller, adapter = scoped_adapter()
    old = adapter.run_state("chat-a", "validating")
    adapter.set_scope(session_ids=frozenset({"chat-b"}), conversation_ids=frozenset())
    current = adapter.run_state("chat-a", "validating")
    assert current is not None and current != old
    assert controller.snapshot().state == "idle"
    adapter.set_scope(session_ids=frozenset({"chat-a"}), conversation_ids=frozenset())
    adapter.run_state("chat-a", "streaming")
    adapter.run_state("chat-a", "completed", run_owner=old)
    assert controller.snapshot().state == "speaking"
    adapter.run_state("chat-a", "completed", run_owner=current)
    assert controller.snapshot().state == "idle"


def test_adding_workspace_member_keeps_retained_tool_voice_and_wake_leases():
    for source, expected in (
        ("tool", "tool_running"),
        ("voice", "listening"),
        ("wake", "wake_armed"),
    ):
        controller, adapter = scoped_adapter()
        if source == "tool":
            adapter.tool_step("tool-a", 1, "tool_call", session_id="chat-a")
        elif source == "voice":
            adapter.voice_state("chat-a", 1, "listening")
        else:
            adapter.wake("durable-a", "wake-a", active=True)
        adapter.set_scope(
            session_ids=frozenset({"chat-a", "chat-b"}),
            conversation_ids=frozenset({"durable-a"}),
        )
        assert controller.snapshot().state == expected
