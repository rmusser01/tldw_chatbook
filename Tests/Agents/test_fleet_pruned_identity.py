"""Steering refusals retain bounded identity, never payloads or authority."""

from dataclasses import asdict

import pytest

from Tests.Agents.test_fleet_continuation import _run
from Tests.Agents.test_fleet_runtime import _tool_results, make_fleet_service
from Tests.Agents.test_fleet_steering_mailbox import fence
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.mark.parametrize("address", ["handle", "run"])
@pytest.mark.parametrize("status", ["cancelled", "superseded", "error"])
def test_pruned_child_receives_honest_terminal_refusal(tmp_path, address, status):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="identity")
    target = {}

    def steer():
        return fence("send_to_agent", {"id": target["id"], "message": "follow up"})

    service, _chat, fleet = make_fleet_service(db, [steer, "answer"])
    handle = fleet.reserve("old task", None)
    child_id = db.create_run(conversation_id="c", agent_kind="subagent")
    fleet.attach_run(handle.handle_id, child_id)
    db.set_status(child_id, status)
    fleet.finish(handle.handle_id, status)
    fleet.prune_terminal()
    target["id"] = handle.handle_id if address == "handle" else child_id
    run_id, outcome = _run(service)
    assert outcome.status == "done"
    result = _tool_results(db.get_run(run_id), "send_to_agent")[0]
    assert status in result
    assert "no retained transcript" in result
    assert "cannot be resumed" in result
    assert "no sub-agent matches" not in result
    assert "earlier session" not in result
    assert db.count_subagent_runs("c") == 1


def test_db_only_terminal_run_does_not_invent_an_earlier_session(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="identity")
    child_id = db.create_run(conversation_id="c", agent_kind="subagent")
    db.set_status(child_id, "cancelled")
    service, _chat, _fleet = make_fleet_service(
        db,
        [fence("send_to_agent", {"id": child_id, "message": "follow up"}), "answer"],
    )
    run_id, _outcome = _run(service)
    result = _tool_results(db.get_run(run_id), "send_to_agent")[0]
    assert "cancelled" in result
    assert "earlier session" not in result
    assert "transcript" in result and "fresh sub-agent" in result


def test_foreign_conversation_run_stays_unknown(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="identity")
    child_id = db.create_run(conversation_id="foreign", agent_kind="subagent")
    db.set_status(child_id, "cancelled", result="private result")
    service, _chat, _fleet = make_fleet_service(
        db,
        [fence("send_to_agent", {"id": child_id, "message": "follow up"}), "answer"],
    )
    run_id, _outcome = _run(service)
    result = _tool_results(db.get_run(run_id), "send_to_agent")[0]
    assert "no sub-agent matches" in result
    assert "private result" not in result
    assert "cancelled" not in result


def test_pruned_identity_is_bounded_and_contains_no_run_payload():
    fleet = FleetCoordinator(3, lambda: 0)
    ids = []
    for i in range(257):
        handle = fleet.reserve("private task", "private agent")
        ids.append(handle.handle_id)
        fleet.attach_run(handle.handle_id, f"run-{i}")
        fleet.finish(handle.handle_id, "cancelled", result="private result")
        fleet.prune_terminal()
    assert fleet.get_pruned_identity(ids[0]) is None
    assert fleet.get_pruned_identity("run-0") is None
    assert fleet.get_pruned_identity(ids[1]) is not None
    assert asdict(fleet.get_pruned_identity("run-256")) == {
        "handle_id": ids[-1],
        "run_id": "run-256",
        "status": "cancelled",
    }
    assert fleet.snapshot() == []
    assert fleet.drain_events() == []


def test_pruned_handle_identity_wins_over_a_colliding_run_id():
    fleet = FleetCoordinator(3, lambda: 0)
    first = fleet.reserve("first", None)
    second = fleet.reserve("second", None)
    fleet.attach_run(first.handle_id, second.handle_id)
    fleet.finish(first.handle_id, "cancelled")
    fleet.finish(second.handle_id, "error")
    fleet.prune_terminal()
    assert fleet.get_pruned_identity(second.handle_id).status == "error"


def test_live_run_id_still_wins_over_pruned_identity(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="identity")
    target = {}

    def steer():
        return fence("send_to_agent", {"id": target["id"], "message": "correction"})

    service, _chat, fleet = make_fleet_service(db, [steer, "answer"])
    old = fleet.reserve("old", None)
    fleet.finish(old.handle_id, "cancelled")
    fleet.prune_terminal()
    survivor = fleet.reserve("live", None)
    fleet.attach_run(survivor.handle_id, old.handle_id)
    target["id"] = old.handle_id
    try:
        run_id, _outcome = _run(service)
        result = _tool_results(db.get_run(run_id), "send_to_agent")[0]
        assert "queued" in result
        assert fleet.drain_steering(survivor.handle_id) == [
            ("supervisor", "correction")
        ]
    finally:
        fleet.finish(survivor.handle_id, "cancelled")
