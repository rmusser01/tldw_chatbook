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
