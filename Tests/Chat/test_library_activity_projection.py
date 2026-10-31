"""Pure selected-turn projection and sidecar-only trajectory behavior."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from tldw_chatbook.Chat.library_activity import (
    LibraryActivityEvent,
    encode_library_activity_event,
    project_library_activity,
)
from tldw_chatbook.Chat.trajectory import derive_trajectory


def _event(event_id: str = "event-1") -> LibraryActivityEvent:
    return LibraryActivityEvent(
        version=1,
        event_id=event_id,
        attempt_id="attempt-1",
        run_id="run-child",
        actor_kind="subagent",
        parent_run_id="run-parent",
        library_provider="rag",
        operation="search_library_rag",
        status="succeeded",
        result_count=2,
        query_preview="bounded query",
        source_refs=(),
        error_code=None,
        error_summary=None,
    )


def _row(
    turn_id: str,
    event: LibraryActivityEvent,
    *,
    seq: int,
    message_id: str | None = None,
    payload_json: str | None = None,
):
    return SimpleNamespace(
        message_id=message_id or turn_id,
        conversation_id="conversation-1",
        turn_id=turn_id,
        seq=seq,
        event_kind="library_activity",
        step_started_at=100.0 + seq,
        first_token_at=None,
        completed_at=None,
        model=None,
        provider=None,
        payload_json=payload_json or encode_library_activity_event(event),
    )


def test_projection_filters_active_lineage_and_selected_turn() -> None:
    rows = [
        _row("turn-1", _event("event-1"), seq=2),
        _row("turn-2", _event("event-2"), seq=4),
        _row("off-branch", _event("event-3"), seq=3),
    ]

    view = project_library_activity(rows, ("turn-1", "turn-2"), "turn-2")

    assert view.selected_turn_id == "turn-2"
    assert [action.event.event_id for action in view.actions] == ["event-2"]
    assert view.actions[0].event.actor_kind == "subagent"
    assert view.actions[0].event.parent_run_id == "run-parent"
    assert view.actions[0].occurred_at == 104.0
    assert view.corrupt_row_count == 0


def test_projection_reports_bounded_corrupt_status_without_partial_event() -> None:
    rows = [
        _row("turn-1", _event(), seq=1, payload_json='{"version":99}'),
        _row("turn-1", replace(_event(), event_id="event-2"), seq=2),
    ]

    view = project_library_activity(rows, ("turn-1",), "turn-1")

    assert [action.event.event_id for action in view.actions] == ["event-2"]
    assert view.corrupt_row_count == 1
    assert view.status == "corrupt"


def test_library_activity_is_excluded_from_generic_trajectory() -> None:
    messages = [
        {
            "id": "turn-1",
            "sender": "user",
            "content": "question",
            "timestamp": 1.0,
            "parent_message_id": None,
            "deleted": False,
        },
        {
            "id": "assistant-1",
            "sender": "assistant",
            "content": "answer",
            "timestamp": 2.0,
            "parent_message_id": "turn-1",
            "deleted": False,
        },
    ]
    rows = [
        SimpleNamespace(
            message_id="turn-1",
            conversation_id="conversation-1",
            turn_id="turn-1",
            seq=1,
            event_kind="user",
            step_started_at=None,
            first_token_at=None,
            completed_at=None,
            model=None,
            provider=None,
            payload_json=None,
        ),
        _row("turn-1", _event(), seq=2),
        SimpleNamespace(
            message_id="assistant-1",
            conversation_id="conversation-1",
            turn_id="turn-1",
            seq=3,
            event_kind="assistant",
            step_started_at=None,
            first_token_at=None,
            completed_at=None,
            model=None,
            provider=None,
            payload_json=None,
        ),
    ]

    snapshot = derive_trajectory(
        messages, {}, rows, (), (), active_leaf_message_id="assistant-1"
    )
    kinds = [record.kind for turn in snapshot.turns for record in turn.records]

    assert kinds == ["user", "assistant"]


def test_counts_by_turn_match_projecting_every_turn() -> None:
    """TASK-33628.5.2: one pass over the rows gives the per-turn answer."""
    from tldw_chatbook.Chat import library_activity
    from tldw_chatbook.Chat.library_activity import count_library_activity_by_turn

    rows = [
        _row("turn-1", _event("event-1"), seq=2),
        _row("turn-1", _event("event-1"), seq=3),  # duplicate event id
        _row("turn-1", _event("event-2"), seq=0),  # bad sequence
        _row("turn-3", _event("event-3"), seq=5, payload_json='{"version":99}'),
        _row("turn-4", _event("event-4"), seq=6, message_id="other"),
        _row("off-branch", _event("event-5"), seq=7),
        {
            "message_id": "turn-2",
            "turn_id": "turn-2",
            "seq": 8,
            "event_kind": "library_activity",
            "step_started_at": 1.0,
            "payload_json": encode_library_activity_event(_event("event-6")),
        },
        {"turn_id": "turn-2", "event_kind": "tool_call", "seq": 9},
    ]
    turns = ("turn-1", "turn-2", "turn-3", "turn-4", "turn-5")
    expected = {
        turn: len(project_library_activity(rows, turns, turn).actions)
        for turn in turns
    }

    counts = count_library_activity_by_turn(rows, turns)

    assert counts == {turn: n for turn, n in expected.items() if n}
    assert counts == {"turn-1": 1, "turn-2": 1}
    # Linear in rows, not quadratic in turns: 1,500 turns with activity on
    # two of them project two turns, not 1,500.
    many = tuple(f"t{index}" for index in range(1500))
    projected: list[str | None] = []
    real = library_activity.project_library_activity

    def spy(rows_, active, selected):
        projected.append(selected)
        return real(rows_, active, selected)

    library_activity.project_library_activity = spy
    try:
        sparse = [_row("t7", _event("e7"), seq=1), _row("t900", _event("e9"), seq=2)]
        assert count_library_activity_by_turn(sparse, many) == {"t7": 1, "t900": 1}
    finally:
        library_activity.project_library_activity = real
    assert sorted(projected) == ["t7", "t900"]
