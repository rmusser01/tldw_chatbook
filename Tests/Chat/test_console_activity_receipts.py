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
    FLEET_RECEIPT_FALLBACK = "fleet_receipt_fallback"

    def __init__(self) -> None:
        self._marked_by_type: dict[str, set[str]] = {
            self.FLEET_UNSEEN: set(),
            self.FLEET_RECEIPT_FALLBACK: set(),
        }
        self.calls: list[tuple[str, str]] = []

    @property
    def marked(self) -> set[str]:
        return self._marked_by_type[self.FLEET_UNSEEN]

    def set_mark(self, conversation_id: str, mark: str) -> None:
        self.calls.append(("set", conversation_id))
        self._marked_by_type[mark].add(conversation_id)

    def clear_mark(self, conversation_id: str, mark: str) -> None:
        self.calls.append(("clear", conversation_id))
        self._marked_by_type[mark].discard(conversation_id)

    def list_marked_conversation_ids(self, mark: str) -> tuple[str, ...]:
        return tuple(sorted(self._marked_by_type[mark]))


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
    assert first.complete is True
    assert len(first.activity_ids) == 3
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

    assert published.activity_ids == ()
    assert published.complete is False
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
    assert service.acknowledge(receipts.activity_ids) == 1
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

    assert service.acknowledge(first.activity_ids) == 1
    assert marks.marked == {"conversation-1"}
    assert [row.activity_id for row in service.unseen_snapshot()] == list(
        second.activity_ids
    )

    assert service.acknowledge(second.activity_ids) == 1
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
    acknowledge_thread = threading.Thread(
        target=service.acknowledge, args=(original.activity_ids,)
    )
    acknowledge_thread.start()
    assert acknowledge_entered.wait(5)

    published = []
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
    assert [row.activity_id for row in service.unseen_snapshot()] == list(
        published[0].activity_ids
    )


def test_all_failed_fleet_publication_keeps_coarse_mark_and_degraded_state(
    tmp_path, monkeypatch
):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    service = ConsoleActivityReceiptService(database, marks)
    monkeypatch.setattr(
        database,
        "publish_console_activity",
        lambda **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("private")),
    )

    result = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("done", run_id="run-one"),),
            drain_id="drain-one",
        )
    )

    assert result.activity_ids == ()
    assert result.complete is False
    assert service.hydration_state() == "degraded"
    assert marks.marked == {"conversation-1"}


def test_partial_fleet_publication_keeps_coarse_mark_and_reports_incomplete(
    tmp_path, monkeypatch
):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    service = ConsoleActivityReceiptService(database, marks)
    real_publish = database.publish_console_activity
    calls = {"count": 0}

    def publish_once(**kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise sqlite3.OperationalError("private")
        return real_publish(**kwargs)

    monkeypatch.setattr(database, "publish_console_activity", publish_once)

    result = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(
                _child("done", run_id="run-one"),
                _child("error", run_id="run-two"),
            ),
            drain_id="drain-one",
        )
    )

    assert len(result.activity_ids) == 1
    assert result.complete is False
    assert service.hydration_state() == "degraded"
    assert marks.marked == {"conversation-1"}


def test_fleet_reload_failure_keeps_coarse_mark_and_reports_incomplete(
    tmp_path, monkeypatch
):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    service = ConsoleActivityReceiptService(database, marks)
    monkeypatch.setattr(
        database,
        "list_unseen_console_activity",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("private")),
    )

    result = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("done", run_id="run-one"),),
            drain_id="drain-one",
        )
    )

    assert len(result.activity_ids) == 1
    assert result.complete is False
    assert service.hydration_state() == "degraded"
    assert marks.marked == {"conversation-1"}


def test_failed_fleet_publication_debt_survives_recovery_until_explicit_visit(
    tmp_path, monkeypatch
):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    service = ConsoleActivityReceiptService(database, marks)
    real_publish = database.publish_console_activity
    monkeypatch.setattr(
        database,
        "publish_console_activity",
        lambda **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("private")),
    )

    result = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("done", run_id="run-one"),),
            drain_id="drain-one",
        )
    )
    assert result.complete is False
    assert marks.marked == {"conversation-1"}
    assert marks.list_marked_conversation_ids(
        marks.FLEET_RECEIPT_FALLBACK
    ) == ("conversation-1",)

    monkeypatch.setattr(database, "publish_console_activity", real_publish)
    recovered = ConsoleActivityReceiptService(database, marks)
    assert recovered.hydrate_from_storage() == 0
    recovered.publish_ordinary(
        logical_outcome_id="ordinary-one",
        status="done",
        session_id="session-ordinary",
        conversation_id="conversation-ordinary",
    )

    assert recovered.hydration_state() == "ready"
    assert "conversation-1" in marks.marked
    assert recovered.clear_fleet_mark_if_seen("conversation-1") is True
    assert "conversation-1" not in marks.marked
    assert marks.list_marked_conversation_ids(
        marks.FLEET_RECEIPT_FALLBACK
    ) == ()


def test_acknowledge_retry_confirms_ids_absent_after_post_write_reload_failure(
    tmp_path, monkeypatch
):
    database = AgentRunsDB(tmp_path / "runs.db")
    service = ConsoleActivityReceiptService(database, RecordingMarks())
    publication = service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("done", run_id="run-one"),),
            drain_id="drain-one",
        )
    )
    real_list = database.list_unseen_console_activity
    calls = {"count": 0}

    def fail_first_reload():
        calls["count"] += 1
        if calls["count"] == 1:
            raise sqlite3.OperationalError("private")
        return real_list()

    monkeypatch.setattr(database, "list_unseen_console_activity", fail_first_reload)

    assert service.acknowledge(publication.activity_ids) == 1
    assert service.hydration_state() == "degraded"
    assert service.acknowledge(publication.activity_ids) == 1
    assert service.hydration_state() == "ready"
    assert service.unseen_snapshot() == ()


def test_complete_fleet_replay_replaces_fallback_debt_with_exact_receipt(
    tmp_path, monkeypatch
):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    service = ConsoleActivityReceiptService(database, marks)
    event = FleetDrained(
        conversation_id="conversation-1",
        children=(_child("done", run_id="run-one"),),
        drain_id="drain-one",
    )
    real_publish = database.publish_console_activity
    monkeypatch.setattr(
        database,
        "publish_console_activity",
        lambda **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("private")),
    )
    assert service.publish_fleet_drain(event).complete is False
    monkeypatch.setattr(database, "publish_console_activity", real_publish)

    replay = service.publish_fleet_drain(event)

    assert replay.complete is True
    assert marks.list_marked_conversation_ids(
        marks.FLEET_RECEIPT_FALLBACK
    ) == ()
    assert marks.marked == {"conversation-1"}
    assert [receipt.activity_id for receipt in service.unseen_snapshot()] == list(
        replay.activity_ids
    )


def test_transient_fallback_mark_failure_is_repaired_before_ready_and_restart(
    tmp_path, monkeypatch
):
    database = AgentRunsDB(tmp_path / "runs.db")
    marks = RecordingMarks()
    service = ConsoleActivityReceiptService(database, marks)
    real_set_mark = marks.set_mark
    fallback_attempts = {"count": 0}

    def fail_first_fallback(conversation_id: str, mark: str) -> None:
        if mark == marks.FLEET_RECEIPT_FALLBACK:
            fallback_attempts["count"] += 1
            if fallback_attempts["count"] == 1:
                raise sqlite3.OperationalError("private")
        real_set_mark(conversation_id, mark)

    monkeypatch.setattr(marks, "set_mark", fail_first_fallback)
    monkeypatch.setattr(
        database,
        "publish_console_activity",
        lambda **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("private")),
    )

    assert service.publish_fleet_drain(
        FleetDrained(
            conversation_id="conversation-1",
            children=(_child("done", run_id="run-one"),),
            drain_id="drain-one",
        )
    ).complete is False
    assert service.hydration_state() == "degraded"
    assert marks.marked == {"conversation-1"}
    assert marks.list_marked_conversation_ids(
        marks.FLEET_RECEIPT_FALLBACK
    ) == ()

    assert service.hydrate_from_storage() == 0
    assert service.hydration_state() == "ready"
    assert marks.list_marked_conversation_ids(
        marks.FLEET_RECEIPT_FALLBACK
    ) == ("conversation-1",)

    restarted = ConsoleActivityReceiptService(database, marks)
    assert restarted.hydrate_from_storage() == 0
    assert restarted.hydration_state() == "ready"
    assert marks.marked == {"conversation-1"}


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
