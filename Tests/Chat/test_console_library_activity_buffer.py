"""Store-owned pending persistence for minimized Library activity."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_library_activity_buffer import (
    LIBRARY_ACTIVITY_NOT_SAVED_COPY,
    ConsoleLibraryActivityBuffer,
)
from tldw_chatbook.Chat.library_activity import LibraryActivityEvent
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _event(index: int) -> LibraryActivityEvent:
    return LibraryActivityEvent(
        version=1,
        event_id=f"event-{index}",
        attempt_id="attempt-1",
        run_id="run-1",
        actor_kind="primary",
        parent_run_id=None,
        library_provider="direct",
        operation="library_search_notes",
        status="succeeded",
        result_count=1,
        query_preview=f"query {index}",
        source_refs=(),
        error_code=None,
        error_summary=None,
    )


def test_concurrent_admission_is_unique_and_keeps_one_stable_order() -> None:
    buffer = ConsoleLibraryActivityBuffer(lambda _session_id, _batch: None)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda index: buffer.admit("session-1", "turn-1", _event(index)),
                range(32),
            )
        )

    first = buffer.pending_events("session-1")
    second = buffer.pending_events("session-1")
    assert first == second
    assert len(first) == 32
    assert {item.event.event_id for item in first} == {
        f"event-{index}" for index in range(32)
    }


def test_failed_flush_retains_exact_batch_and_retry_confirms_once() -> None:
    seen: list[tuple[str, ...]] = []

    def persist(_session_id, contribution) -> None:
        event_ids = tuple(item.event.event_id for item in contribution.items)
        seen.append(event_ids)
        if len(seen) == 1:
            raise RuntimeError("private storage detail")

    buffer = ConsoleLibraryActivityBuffer(persist, max_attempts=2)
    buffer.admit("session-1", "turn-1", _event(1))
    buffer.admit("session-1", "turn-1", _event(2))

    first = buffer.flush("session-1")
    assert first.status == "pending"
    assert first.saved_count == 0
    assert first.pending_count == 2
    assert buffer.pending_events("session-1")

    retried = buffer.retry("session-1")
    assert retried.status == "saved"
    assert retried.saved_count == 2
    assert retried.pending_count == 0
    assert seen == [("event-1", "event-2"), ("event-1", "event-2")]
    assert buffer.pending_events("session-1") == ()


def test_retry_exhaustion_exposes_bounded_not_saved_state() -> None:
    calls = 0

    def fail(_session_id, _contribution) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("must never surface")

    buffer = ConsoleLibraryActivityBuffer(fail, max_attempts=2)
    buffer.admit("session-1", "turn-1", _event(1))

    assert buffer.flush("session-1").status == "pending"
    exhausted = buffer.retry("session-1")

    assert exhausted.status == "failed"
    assert exhausted.error_code == "retry_exhausted"
    assert exhausted.warning == LIBRARY_ACTIVITY_NOT_SAVED_COPY
    assert calls == 2
    assert len(buffer.pending_events("session-1")) == 1


def test_final_flush_is_one_bounded_attempt() -> None:
    calls = 0

    def fail(_session_id, _contribution) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("unavailable")

    buffer = ConsoleLibraryActivityBuffer(fail)
    buffer.admit("session-1", "turn-1", _event(1))

    first = buffer.final_flush("session-1")
    second = buffer.final_flush("session-1")

    assert first.status == second.status == "failed"
    assert first.pending_count == second.pending_count == 1
    assert buffer.state("session-1") == first
    assert calls == 1


class _FailingContribution:
    def write(self, *, writer, conversation_id, message_ids) -> None:
        raise RuntimeError("injected promotion failure")


def test_promotion_failure_rolls_back_rows_and_retains_activity_for_retry(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "activity-promotion.db", "activity-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(ephemeral=True)
    user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="question"
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    store.admit_library_activity(session.id, user.id, _event(1))
    before = store.pending_library_activity(session.id)
    revision_before_promotion = store.library_activity_revision(session.id)

    with pytest.raises(RuntimeError, match="injected promotion failure"):
        store.promote_ephemeral_session(
            session.id, contributions=(_FailingContribution(),)
        )

    assert store.pending_library_activity(session.id) == before
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM message_trajectory_metadata"
    ).fetchone()[0] == 0

    conversation_id = store.promote_ephemeral_session(session.id)
    rows = db.get_trajectory_rows(conversation_id)
    activity = [row for row in rows if row.event_kind == "library_activity"]
    durable_user_id = store._nodes_by_session[session.id][user.id].persisted_message_id
    assert len(activity) == 1
    assert activity[0].message_id == durable_user_id
    assert activity[0].turn_id == durable_user_id
    assert store.pending_library_activity(session.id) == ()
    assert store.library_activity_revision(session.id) > revision_before_promotion


def test_close_performs_one_final_flush_for_a_durable_session(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "activity-close.db", "activity-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(ephemeral=True)
    user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="question"
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    conversation_id = store.promote_ephemeral_session(session.id)
    store.admit_library_activity(session.id, user.id, _event(1))

    store.close_session(session.id)

    activity = [
        row
        for row in db.get_trajectory_rows(conversation_id)
        if row.event_kind == "library_activity"
    ]
    assert len(activity) == 1


def test_store_projects_pending_and_durable_activity_to_native_assistant(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "activity-projection.db", "activity-test")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(ephemeral=True)
    user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="question"
    )
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    store.promote_ephemeral_session(session.id)
    store.admit_library_activity(session.id, user.id, _event(1))

    pending_view, pending_counts, pending_state = store.library_activity_snapshot(
        session.id, assistant.id
    )

    assert [item.event.event_id for item in pending_view.actions] == ["event-1"]
    assert pending_counts == ((assistant.id, 1),)
    assert pending_state.status == "pending"
    assert store.current_library_activity_turn_id(session.id) == user.id

    assert store.flush_library_activity(session.id).status == "saved"
    durable_view, durable_counts, durable_state = store.library_activity_snapshot(
        session.id, assistant.id
    )

    assert [item.event.event_id for item in durable_view.actions] == ["event-1"]
    assert durable_counts == ((assistant.id, 1),)
    assert durable_state.status == "saved"


def test_store_projection_follows_active_branch_without_losing_other_activity() -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    first_user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="first question"
    )
    first_assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="first answer"
    )
    store.admit_library_activity(session.id, first_user.id, _event(1))

    second_user = store.create_sibling(
        first_user.id,
        role=ConsoleMessageRole.USER,
        content="second question",
    )
    second_assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="second answer"
    )
    store.admit_library_activity(session.id, second_user.id, _event(2))

    second_view, second_counts, _state = store.library_activity_snapshot(
        session.id, second_assistant.id
    )
    assert [item.event.event_id for item in second_view.actions] == ["event-2"]
    assert second_counts == ((second_assistant.id, 1),)

    store.set_active_leaf(session.id, first_assistant.id)
    first_view, first_counts, _state = store.library_activity_snapshot(
        session.id, first_assistant.id
    )
    assert [item.event.event_id for item in first_view.actions] == ["event-1"]
    assert first_counts == ((first_assistant.id, 1),)
    assert len(store.pending_library_activity(session.id)) == 2
