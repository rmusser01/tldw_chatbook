"""App-lifetime Console activity receipt coordination."""

from __future__ import annotations

import sqlite3
import threading

from tldw_chatbook.Chat.console_activity_receipts import (
    ConsoleActivityReceiptService,
)
from tldw_chatbook.Chat.console_agent_bridge import FleetDrained, SettledChild
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


class RecordingMarks:
    FLEET_UNSEEN = "fleet_unseen"

    def __init__(self) -> None:
        self.marked: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    def set_mark(self, conversation_id: str, mark: str) -> None:
        assert mark == self.FLEET_UNSEEN
        self.calls.append(("set", conversation_id))
        self.marked.add(conversation_id)

    def clear_mark(self, conversation_id: str, mark: str) -> None:
        assert mark == self.FLEET_UNSEEN
        self.calls.append(("clear", conversation_id))
        self.marked.discard(conversation_id)

    def list_marked_conversation_ids(self, mark: str) -> tuple[str, ...]:
        assert mark == self.FLEET_UNSEEN
        return tuple(sorted(self.marked))


def _child(
    status: str,
    *,
    run_id: str | None,
    session_id: str = "session-1",
    after_turn: bool = True,
) -> SettledChild:
    return SettledChild(
        run_id=run_id,
        status=status,
        session_id=session_id,
        assistant_message_id="assistant-1",
        settled_after_turn=after_turn,
    )


def test_fleet_drain_maps_statuses_and_is_idempotent(tmp_path):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    service = ConsoleActivityReceiptService(database, marks)
    event = FleetDrained(
        conversation_id="conversation-1",
        children=(
            _child("done", run_id="run-done"),
            _child("error", run_id="run-failed"),
            _child("cancelled", run_id="run-cancelled"),
            _child("done", run_id="within-turn", after_turn=False),
        ),
        drain_id="drain-explicit",
    )

    first = service.publish_fleet_drain(event)
    second = service.publish_fleet_drain(event)

    assert second == first
    assert len(first) == 3
    assert {receipt.status for receipt in service.unseen_snapshot()} == {
        "done",
        "failed",
        "cancelled",
    }
    assert marks.marked == {"conversation-1"}
    assert service.hydration_state() == "ready"


def test_null_run_children_use_drain_identity_without_cross_drain_collision(tmp_path):
    database = AgentRunsDB(tmp_path / "runs.db")
    service = ConsoleActivityReceiptService(database, RecordingMarks())

    first = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("error", run_id=None),),
            drain_id="drain-one",
        )
    )
    duplicate = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("error", run_id=None),),
            drain_id="drain-one",
        )
    )
    second = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("error", run_id=None),),
            drain_id="drain-two",
        )
    )

    assert duplicate == first
    assert second != first
    assert len(service.unseen_snapshot()) == 2


def test_unknown_fleet_status_fails_closed_and_degrades(tmp_path):
    service = ConsoleActivityReceiptService(
        AgentRunsDB(tmp_path / "runs.db"), RecordingMarks()
    )

    published = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("mystery", run_id="run-mystery"),),
            drain_id="drain-mystery",
        )
    )

    assert published == ()
    assert service.unseen_snapshot() == ()
    assert service.hydration_state() == "degraded"


def test_cold_hydration_rebuilds_snapshot_and_reconciles_marks(tmp_path):
    path = tmp_path / "runs.db"
    database = AgentRunsDB(path)
    activity_id, _ = database.publish_console_activity(
        origin="fleet_survivor",
        logical_outcome_id="fleet-run:hydrated",
        status="done",
        session_id=None,
        conversation_id="conversation-hydrated",
        run_id="hydrated",
    )
    marks = RecordingMarks()
    marks.marked.add("stale-conversation")
    service = ConsoleActivityReceiptService(database, marks)

    assert service.hydration_state() == "cold"
    assert service.unseen_snapshot() == ()
    assert service.hydrate_from_storage() == 1

    assert [row.activity_id for row in service.unseen_snapshot()] == [activity_id]
    assert service.hydration_state() == "ready"
    assert marks.marked == {"conversation-hydrated"}


def test_cold_or_degraded_clear_request_preserves_coarse_mark(tmp_path, monkeypatch):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    marks.marked.add("conversation-1")
    service = ConsoleActivityReceiptService(database, marks)

    assert service.clear_fleet_mark_if_seen("conversation-1") is False
    assert marks.marked == {"conversation-1"}

    monkeypatch.setattr(
        database,
        "list_unseen_console_activity",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("private")),
    )
    assert service.hydrate_from_storage() == 0
    assert service.hydration_state() == "degraded"
    assert service.clear_fleet_mark_if_seen("conversation-1") is False
    assert marks.marked == {"conversation-1"}


def test_ready_clear_request_respects_receipts_and_viewless_set(tmp_path):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    service = ConsoleActivityReceiptService(database, marks)
    receipts = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("done", run_id="run-one"),),
            drain_id="drain-one",
        )
    )

    assert service.clear_fleet_mark_if_seen("conversation-1") is False
    assert marks.marked == {"conversation-1"}
    assert service.acknowledge(receipts) == 1
    assert marks.marked == set()

    assert service.ensure_fleet_mark("conversation-1") is True
    assert marks.marked == {"conversation-1"}
    assert service.clear_fleet_mark_if_seen("conversation-1") is True
    assert marks.marked == set()


def test_acknowledge_reconciles_badge_without_clearing_newer_receipt(tmp_path):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    service = ConsoleActivityReceiptService(database, marks)
    first = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("error", run_id="run-one"),),
            drain_id="drain-one",
        )
    )
    second = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("done", run_id="run-two"),),
            drain_id="drain-two",
        )
    )

    assert service.acknowledge(first) == 1
    assert marks.marked == {"conversation-1"}
    assert [row.activity_id for row in service.unseen_snapshot()] == list(second)

    assert service.acknowledge(second) == 1
    assert marks.marked == set()


def test_publication_and_acknowledgement_share_one_lock(tmp_path, monkeypatch):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    service = ConsoleActivityReceiptService(database, marks)
    original = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("done", run_id="run-old"),),
            drain_id="drain-old",
        )
    )
    acknowledge_entered = threading.Event()
    release_acknowledge = threading.Event()
    real_acknowledge = database.acknowledge_console_activity

    def blocked_acknowledge(activity_ids):
        acknowledge_entered.set()
        assert release_acknowledge.wait(5)
        return real_acknowledge(activity_ids)

    monkeypatch.setattr(database, "acknowledge_console_activity", blocked_acknowledge)
    acknowledge_thread = threading.Thread(target=service.acknowledge, args=(original,))
    acknowledge_thread.start()
    assert acknowledge_entered.wait(5)

    published: list[tuple[str, ...]] = []
    publish_thread = threading.Thread(
        target=lambda: published.append(
            service.publish_fleet_drain(
                FleetDrained(
                    conversation_id="conversation-1",
                    children=(_child("done", run_id="run-new"),),
                    drain_id="drain-new",
                )
            )
        )
    )
    publish_thread.start()
    release_acknowledge.set()
    acknowledge_thread.join(5)
    publish_thread.join(5)

    assert published and marks.marked == {"conversation-1"}
    assert [row.activity_id for row in service.unseen_snapshot()] == list(published[0])


def test_storage_failure_is_content_free_degradation_with_retry(tmp_path, monkeypatch):
    database = AgentRunsDB(tmp_path / "runs.db")
    service = ConsoleActivityReceiptService(database, RecordingMarks())
    real_list = database.list_unseen_console_activity
    calls = {"count": 0}

    def flaky_list():
        calls["count"] += 1
        if calls["count"] == 1:
            raise sqlite3.OperationalError("secret database path and content")
        return real_list()

    monkeypatch.setattr(database, "list_unseen_console_activity", flaky_list)

    assert service.hydrate_from_storage() == 0
    assert service.hydration_state() == "degraded"
    assert service.unseen_snapshot() == ()
    assert service.hydrate_from_storage() == 0
    assert service.hydration_state() == "ready"
