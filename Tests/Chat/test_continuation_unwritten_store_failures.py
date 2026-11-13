"""TASK-34100.3 review: an "unwritten" store failure really wrote nothing.

When the optional reasoning checkpoint of a tool-free first final turn cannot
be stored, the runtime keeps the delivered reply as an ordinary reply, but
only for the store codes in ``UNWRITTEN_STORE_FAILURES``: failures raised
before the store wrote the continuation or settled the dispatch. That premise
used to rest on a comment and on runtime tests whose persist callback raised
a fake sentence. These tests drive the real ``ConsoleChatStore`` and a
file-backed ChaChaNotes database to the reachable codes and check that:

* the store's sentence maps to an unwritten code;
* no continuation JSON, version bump or settlement was written; and
* the ordinary completion path then saves the reply, with no recovery left.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.Chat.test_console_dispatch_continuation_handoff import (
    _checkpoint_count,
    _event,
    _moonshot_acceptance,
)
from Tests.Chat.test_continuation_destination_identity import (
    _kimi_final,
    _kimi_target,
    _started_store,
    _with_real_destination,
)
from tldw_chatbook.Agents.agent_models import (
    ContinuationEventContext,
    FinalContinuation,
    ToolBatchReady,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
from tldw_chatbook.Chat.provider_continuation_diagnostics import (
    UNWRITTEN_STORE_FAILURES,
    store_failure_code,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.bootstrap_profile]

REPLY = "answer"


def _context(owner_id: str) -> ContinuationEventContext:
    return ContinuationEventContext(
        owner_message_id=owner_id,
        run_id="run-primary",
        agent_kind="primary",
        durability="persistent",
    )


def _unwritten_code(error: pytest.ExceptionInfo[RuntimeError]) -> str:
    code = store_failure_code(error.value)
    assert code in UNWRITTEN_STORE_FAILURES, code
    return code


def _local_store(tmp_path: Path):
    """A saved chat with no durable dispatch row: the store's create path."""
    database = CharactersRAGDB(tmp_path / "chat.sqlite", "task-34100-3-review")
    store = ConsoleChatStore(persistence=ChatPersistenceService(database))
    session = store.create_session(title="Unwritten continuation")
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="Say hello", persist=True
    )
    owner = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    assert owner.persisted_message_id is None
    return database, store, session, owner


def _assistant_rows(database, session) -> list[dict]:
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    return [
        row
        for row in database.get_messages_for_conversation(conversation_id, limit=100)
        if row["sender"] == "assistant"
    ]


def _streaming_dispatch(tmp_path):
    db, store, session_id, _ = _started_store(
        tmp_path,
        lambda conversation_id: _with_real_destination(
            _moonshot_acceptance(conversation_id),
            "moonshot",
            "kimi-k3",
            "https://api.moonshot.ai/v1",
        ),
    )
    before = db.get_message_by_id("assistant-1")
    assistant = store._message_or_raise("assistant-1")
    assistant.status = "streaming"
    assistant.assistant_generation_state = "streaming"
    assistant.content = REPLY
    return db, store, session_id, before


def _assert_ordinary_completion_settles(db, store, session_id, before) -> None:
    after = db.get_message_by_id("assistant-1")
    assert after["provider_continuation_json"] is None
    assert after["version"] == before["version"]
    assert _checkpoint_count(db) == 1
    recovery = store.dispatch_recovery_for_session(session_id)
    assert recovery is not None
    assert recovery.in_flight and not recovery.recovery_needed

    completed = store.mark_message_complete("assistant-1")

    assert completed.status == "complete"
    row = db.get_message_by_id("assistant-1")
    assert (
        row["content"],
        row["assistant_generation_state"],
        row["provider_continuation_json"],
    ) == (REPLY, "complete", None)
    assert _checkpoint_count(db) == 0
    assert store.dispatch_recovery_for_session(session_id) is None


def test_destination_mismatch_leaves_the_dispatch_to_ordinary_completion(
    tmp_path,
) -> None:
    """destination_mismatch: the frozen destination refuses before any write.

    Since dev's TASK-35582 a run whose target moved to another origin is
    refused by the store's binding check, before settlement is attempted.
    """
    db, store, session_id, before = _streaming_dispatch(tmp_path)

    with pytest.raises(RuntimeError) as error:
        store.persist_provider_continuation_event(
            FinalContinuation(
                _event().context,
                _kimi_final("https://api.moonshot.cn/v1"),
                None,
                REPLY,
            ),
            continuation_target=_kimi_target("https://api.moonshot.cn/v1"),
        )

    assert _unwritten_code(error) == "destination_mismatch"
    _assert_ordinary_completion_settles(db, store, session_id, before)
    db.close()


def test_settlement_conflict_leaves_the_dispatch_to_ordinary_completion(
    tmp_path, monkeypatch
) -> None:
    """terminal_settlement_failed: the atomic settlement rolled back whole.

    The repository refuses the continuation settlement once (a lost CAS race);
    nothing is written, and the ordinary completion then settles the turn.
    """
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleDispatchResultStatus,
        ConsoleDispatchWriteResult,
    )

    db, store, session_id, before = _streaming_dispatch(tmp_path)
    repository = store.persistence.console_dispatch_repository
    settle = repository.settle_with_assistant

    def refuse_the_continuation_settlement(settlement):
        if settlement.provider_continuation_json is not None:
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.CONFLICT, None, None, None
            )
        return settle(settlement)

    monkeypatch.setattr(
        repository, "settle_with_assistant", refuse_the_continuation_settlement
    )

    with pytest.raises(RuntimeError) as error:
        store.persist_provider_continuation_event(
            FinalContinuation(_event().context, _kimi_final(), None, REPLY),
            continuation_target=_kimi_target(),
        )

    assert _unwritten_code(error) == "terminal_settlement_failed"
    _assert_ordinary_completion_settles(db, store, session_id, before)
    db.close()


def test_missing_create_operation_writes_no_owner_row(tmp_path, monkeypatch) -> None:
    """create_unavailable: raised before the owner row is created."""
    database, store, session, owner = _local_store(tmp_path)
    store.append_stream_chunk(owner.id, REPLY)
    # A persistence backend without the schema-v37 create operation.
    monkeypatch.setattr(database, "create_assistant_with_continuation", None)
    try:
        with pytest.raises(RuntimeError) as error:
            store.persist_provider_continuation_event(
                FinalContinuation(
                    _context(owner.id),
                    _kimi_final(),
                    None,
                    REPLY,
                ),
                continuation_target=_kimi_target(),
            )

        assert _unwritten_code(error) == "create_unavailable"
        # Saving the chat (persist_session_if_needed) may already have saved
        # the streamed reply as an ORDINARY message; no continuation was.
        assert store.get_message(owner.id).provider_continuation is None
        assert all(
            row["provider_continuation_json"] is None
            for row in _assistant_rows(database, session)
        )

        store.mark_message_complete(owner.id)

        assert [
            (
                row["content"],
                row["assistant_generation_state"],
                row["provider_continuation_json"],
            )
            for row in _assistant_rows(database, session)
        ] == [(REPLY, "complete", None)]
    finally:
        database.close_connection()


def _active_batch() -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=3,
        selected_provider="moonshot",
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="active",
        rounds=(
            ContinuationRound(
                "",
                ("tool reasoning",),
                (
                    ContinuationCall(
                        "call-1", "calculator", '{"expression":"2+2"}', "pending"
                    ),
                ),
            ),
        ),
    )


def test_version_conflict_is_raised_before_the_update(tmp_path) -> None:
    """version_conflict: the stored row is left exactly as the other writer left it.

    A first final checkpoint has no recorded continuation version, so this
    code cannot reach the keep-the-reply fallback; it is still classed as
    unwritten, which this test proves against the real update path.
    """
    database, store, session, owner = _local_store(tmp_path)
    try:
        store.persist_provider_continuation_event(
            ToolBatchReady(
                context=_context(owner.id),
                checkpoint=_active_batch(),
                expected_checkpoint_revision=None,
            ),
            continuation_target=_kimi_target(),
        )
        stored = database.get_message_by_id(owner.id)
        # Another writer moves the row on behind this store's back.
        database.update_message(
            owner.id,
            {"content": "edited elsewhere"},
            stored["version"],
            preserve_provider_continuation=True,
        )
        moved = database.get_message_by_id(owner.id)
        assert moved["version"] == stored["version"] + 1

        with pytest.raises(RuntimeError) as error:
            store.persist_provider_continuation_event(
                FinalContinuation(_context(owner.id), _kimi_final(), 1, REPLY),
                continuation_target=_kimi_target(),
            )

        assert _unwritten_code(error) == "version_conflict"
        after = database.get_message_by_id(owner.id)
        assert (
            after["version"],
            after["content"],
            after["provider_continuation_json"],
            after["assistant_generation_state"],
        ) == (
            moved["version"],
            moved["content"],
            moved["provider_continuation_json"],
            moved["assistant_generation_state"],
        )
    finally:
        database.close_connection()
