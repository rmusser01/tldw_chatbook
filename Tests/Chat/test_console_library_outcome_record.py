"""Each Automatic Library send persists what its search did (TASK-33621.51).

Before this task the controller wrote a ``library_preparation`` record on the
USER message only for a zero-match search and for "Send once without
Library", and the bypass record dropped why the search had paused. A grounded
send wrote nothing, so the transcript had nothing to project for it.

Real controller, store and SQLite database; only the provider and the Library
search service are doubles.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_automatic_library_preparation import (
    ConsoleChatStore,
    _HeldRagService,
    _PolicyCoordinator,
    _RagService,
    _StreamingFence,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState
from tldw_chatbook.Chat.library_preparation import (
    LIBRARY_PREPARATION_EVENT_KIND,
    decode_library_preparation_event,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow

pytestmark = pytest.mark.bootstrap_profile

SESSION = "session-1"


def _rows(count: int) -> list[LibraryRagResultRow]:
    return [
        LibraryRagResultRow.from_result(
            {
                "source_id": f"note-{index}",
                "chunk_id": f"chunk-{index}",
                "title": f"Note {index}",
                "content": f"evidence {index}",
                "score": 0.9,
                "runtime_backend": "local",
                "source_type": "notes",
            }
        )
        for index in range(1, count + 1)
    ]


def _automatic_controller(tmp_path, service, *, timeout: float = 5.0):
    db = CharactersRAGDB(tmp_path / "outcome.sqlite", client_id="task-33621-51")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    store.library_policy_coordinator = _PolicyCoordinator(
        ConsoleAutoRetrieve.AUTOMATIC
    )
    store.create_session(session_id=SESSION)
    store.active_session_id = SESSION
    gateway = _StreamingFence()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)
    controller._library_preparation_timeout = timeout
    return db, store, controller, gateway


def _records(db, store):
    """Decode every persisted preparation record of the session's chat."""
    session = next(item for item in store.sessions() if item.id == SESSION)
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    return [
        (row.message_id, decode_library_preparation_event(row.payload_json))
        for row in db.get_trajectory_rows(conversation_id)
        if row.event_kind == LIBRARY_PREPARATION_EVENT_KIND
    ]


def _user_ids(db, store) -> list[str]:
    session = next(item for item in store.sessions() if item.id == SESSION)
    return [
        str(row["id"])
        for row in db.get_messages_for_conversation(
            session.persisted_conversation_id, limit=50, include_image_data=False
        )
        if row["sender"] == "user"
    ]


@pytest.mark.asyncio
async def test_a_send_that_found_sources_records_how_many_it_sent(
    tmp_path, owned_console_databases
):
    service = _RagService({"runtime_backend": "local", "results": _rows(3)})
    db, store, controller, gateway = _automatic_controller(tmp_path, service)
    owned_console_databases(db, controller)

    result = await controller.submit_draft("what do my notes say", session_id=SESSION)

    assert result.accepted, result.visible_copy
    assert gateway.provider_calls == 1
    assert "evidence 3" in repr(gateway.messages)  # the sources really went out
    records = _records(db, store)
    assert [(event.outcome, event.result_count) for _, event in records] == [
        ("evidence_sent", 3)
    ]
    # Owned by the USER message the search ran for.
    assert [owner for owner, _ in records] == _user_ids(db, store)


@pytest.mark.asyncio
async def test_a_zero_match_send_records_zero_matches(
    tmp_path, owned_console_databases
):
    """Control: this record already existed before the task."""
    service = _RagService({"runtime_backend": "local", "results": []})
    db, store, controller, _gateway = _automatic_controller(tmp_path, service)
    owned_console_databases(db, controller)

    result = await controller.submit_draft("anything", session_id=SESSION)

    assert result.accepted, result.visible_copy
    assert [(event.outcome, event.result_count) for _, event in _records(db, store)] == [
        ("zero_matches", 0)
    ]


async def _pause_then_bypass(controller, store, cause, service):
    """Pause the send on ``cause``, then press Send once without Library."""
    if cause == "stopped":
        send = asyncio.create_task(
            controller.submit_draft("stopped search", session_id=SESSION)
        )
        await asyncio.wait_for(service.entered.wait(), 5)
        assert controller.stop_active_run() is True
        await asyncio.wait_for(send, 5)
    else:
        await controller.submit_draft(f"{cause} search", session_id=SESSION)
    paused = store.preparation_for_session(SESSION)
    assert paused is not None and paused.state is ConsoleTurnPreparationState.PAUSED
    return await controller.bypass_library_preparation(paused.preparation_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cause", "reason"),
    [("failed", "failed"), ("timeout", "timed_out"), ("stopped", "stopped")],
)
async def test_a_send_once_without_library_records_why_the_search_paused(
    tmp_path, owned_console_databases, cause, reason
):
    if cause == "failed":
        service = _RagService(error=RuntimeError("secret search failure"))
        timeout = 5.0
    elif cause == "timeout":
        service = _RagService(delay=0.5)
        timeout = 0.01
    else:
        service = _HeldRagService()
        timeout = 60.0
    db, store, controller, gateway = _automatic_controller(
        tmp_path, service, timeout=timeout
    )
    owned_console_databases(db, controller)

    result = await _pause_then_bypass(controller, store, cause, service)

    assert result.accepted, result.visible_copy
    assert gateway.provider_calls == 1
    records = _records(db, store)
    assert [
        (event.outcome, event.bypass_reason, event.result_count)
        for _, event in records
    ] == [("bypassed", reason, 0)]
    assert "secret" not in repr(records)
