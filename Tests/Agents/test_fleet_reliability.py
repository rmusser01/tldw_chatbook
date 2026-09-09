"""Executable regressions for the September fleet reliability review."""

from __future__ import annotations

import pytest

from Tests.Agents.test_fleet_runtime import FLEET_CFG, make_fleet_service
from Tests.Agents.test_fleet_steering_mailbox import STEER_CFG, fence, make_deps
from tldw_chatbook.Agents.agent_models import ModelTurn
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def test_mailbox_refuses_entry_33_without_losing_fifo_and_drain_frees_room():
    fleet = FleetCoordinator(3, lambda: 0.0)
    handle = fleet.reserve("child", None)
    expected = [("user", str(i)) for i in range(32)]
    for source, message in expected:
        assert fleet.post_steering(handle.handle_id, source, message)
    assert not fleet.post_steering(handle.handle_id, "user", "overflow")
    assert fleet.drain_steering(handle.handle_id) == expected
    assert fleet.post_steering(handle.handle_id, "user", "room again")


def test_mailbox_refuses_aggregate_character_overflow():
    fleet = FleetCoordinator(3, lambda: 0.0)
    handle = fleet.reserve("child", None)
    for _ in range(16):
        assert fleet.post_steering(handle.handle_id, "user", "x" * 4000)
    assert not fleet.post_steering(handle.handle_id, "user", "x")
    assert len(fleet.drain_steering(handle.handle_id)) == 16


def test_retention_size_includes_unread_steering():
    fleet = FleetCoordinator(3, lambda: 0.0, retained_transcript_max_chars=200)
    handle = fleet.reserve("child", None)
    assert fleet.post_steering(handle.handle_id, "user", "x" * 180)
    fleet.finish(
        handle.handle_id, "done", transcript=[{"role": "user", "content": "go"}]
    )
    assert fleet.get_retained(handle.handle_id) is None
    assert fleet.get(handle.handle_id).queued_steering == 1


def test_uncopyable_transcript_refuses_retention_without_claiming_steering():
    class UncopyableContent:
        def __deepcopy__(self, memo):
            raise TypeError("content cannot be copied")

        def __str__(self):
            return "provider content"

    fleet = FleetCoordinator(3, lambda: 0.0)
    handle = fleet.reserve("child", None)
    assert fleet.post_steering(handle.handle_id, "user", "late correction")
    fleet.finish(
        handle.handle_id,
        "done",
        transcript=[{"role": "assistant", "content": UncopyableContent()}],
    )
    assert fleet.get_retained(handle.handle_id) is None
    terminal = fleet.get(handle.handle_id)
    assert terminal.queued_steering == terminal.undelivered_steering == 1
    assert not terminal.can_resume


def test_service_reports_a_full_live_mailbox_without_calling_it_finished(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="review")
    holder = {}

    def send():
        return fence("send_to_agent", {"id": holder["id"], "message": "overflow"})

    service, _chat, fleet = make_fleet_service(db, [send, "done"])
    handle = fleet.reserve("survivor", None)
    holder["id"] = handle.handle_id
    for i in range(32):
        assert fleet.post_steering(handle.handle_id, "user", str(i))
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        results = [step.result for step in outcome.steps if step.kind == "tool_result"]
        assert any("queue is full" in text for text in results), results
        assert fleet.get(handle.handle_id).status == "running"
        assert fleet.get(handle.handle_id).queued_steering == 32
    finally:
        fleet.finish(handle.handle_id, "cancelled")


def test_pruning_removes_obsolete_events_but_preserves_live_survivor_events():
    fleet = FleetCoordinator(3, lambda: 0.0)
    survivor = fleet.reserve("survivor", None)
    for i in range(100):
        handle = fleet.reserve(str(i), None)
        fleet.finish(handle.handle_id, "done")
        fleet.prune_terminal()
    events = fleet.drain_events()
    assert [(event.kind, event.handle_id) for event in events] == [
        ("fleet_started", survivor.handle_id)
    ]
    assert fleet.drain_events() == []


@pytest.mark.parametrize("mutate_input", [True, False])
def test_retained_native_calls_are_deeply_isolated(mutate_input):
    fleet = FleetCoordinator(3, lambda: 0.0)
    handle = fleet.reserve("child", None)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "calculator", "arguments": "{}"}}
            ],
        }
    ]
    fleet.finish(handle.handle_id, "done", transcript=messages)
    exposed = (
        messages if mutate_input else fleet.get_retained(handle.handle_id).messages
    )
    exposed[0]["tool_calls"][0]["id"] = "corrupted"
    assert (
        fleet.get_retained(handle.handle_id).messages[0]["tool_calls"][0]["id"]
        == "call-1"
    )


@pytest.mark.parametrize("retain", [True, False])
def test_final_call_steering_is_reported_unread_without_an_automatic_restart(retain):
    fleet = FleetCoordinator(3, lambda: 0.0, retained_transcripts=5 if retain else 0)
    handle = fleet.reserve("child", None)
    seen = []

    def model(messages, _tools):
        seen.append(list(messages))
        assert fleet.post_steering(handle.handle_id, "user", "also check tests")
        return ModelTurn(text="finished")

    outcome = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "go"}],
        [],
        make_deps(model, drain=lambda: fleet.drain_steering(handle.handle_id)),
    )
    fleet.finish(handle.handle_id, outcome.status, transcript=outcome.final_messages)
    terminal = fleet.get(handle.handle_id)
    assert len(seen) == 1
    assert terminal.undelivered_steering == 1
    assert terminal.can_resume is retain
