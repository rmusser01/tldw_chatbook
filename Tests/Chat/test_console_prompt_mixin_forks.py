"""Native live-fork receipt custody and transactional promotion."""

import json
from dataclasses import replace

import pytest

from Tests.Chat.test_console_prompt_mixin_retry import capture_mode, select
from Tests.Chat.test_console_trace_current_turn_transforms import (
    trace_harness as trace_harness,  # noqa: PLC0414 - shared fixture
)
from Tests.Chat.test_console_trace_runtime import (
    make_database as make_database,  # noqa: PLC0414 - shared fixture
)
from Tests.DB.test_console_prompt_mixins import attempt
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_prompt_mixin_service import ConsolePromptMixinService
from tldw_chatbook.Chat.console_prompt_mixins import MixinSnapshotUnavailableError
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def fork_runtime(make_database):
    db = make_database(":memory:", "fork-mixin")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    try:
        yield db, store
    finally:
        for session in store.sessions():
            store.close_session(session.id)
        store.end_app_runtime()
        db.close()


@pytest.fixture
async def fork_trace_harness(trace_harness):
    try:
        yield trace_harness
    finally:
        for session in trace_harness.store.sessions():
            trace_harness.store.close_session(session.id)
        trace_harness.store.end_app_runtime()


def fork(store, assistant_id, target, *, durable=False):
    return store.stage_fork_snapshot(
        store.issue_fork_fence(assistant_id),
        title="Fork",
        fork_session_id=target,
        fork_conversation_id=target + "-conversation" if durable else None,
        destination_durable=durable,
    )


def commit(store, snapshot):
    from Tests.Chat.test_console_chat_fork_persistence import _commit

    return _commit(store.persistence, snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("detached_predecessor", [False, True])
async def test_temporary_nested_fork_owns_retry_chain_after_close_and_promotion(
    fork_runtime, empty, detached_predecessor
):
    db, store = fork_runtime
    service = ConsolePromptMixinService(
        store=store, persistence=store.persistence, prompt_scope_service=None
    )
    source = store.create_session(
        ephemeral=True, settings=ConsoleSessionSettings(provider="openai", model="test")
    )
    user = store.append_message(
        source.id, role=ConsoleMessageRole.USER, content="Question"
    )
    reply = store.append_message(
        source.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(reply.id, "Failed")
    reply = store.mark_message_failed(reply.id)
    batch = replace(
        attempt(empty=empty), user_message_id=user.id, assistant_message_id=reply.id
    )
    store.publish_prompt_mixin_attempt(source.id, batch)
    if detached_predecessor:
        reply = store.create_sibling(
            reply.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=False
        )
        store.append_stream_chunk(reply.id, "Failed")
        reply = store.mark_message_failed(reply.id)
    for index in range(2):
        batch = replace(
            batch,
            attempt_id=f"retry-{index}",
            retry_of_attempt_id=batch.attempt_id,
            assistant_message_id=reply.id,
            replaces_attempt_id=None,
        )
        store.publish_prompt_mixin_attempt(source.id, batch)
    first_snapshot = fork(store, reply.id, "fork-one")
    first = store.register_fork_snapshot(first_snapshot)
    store.close_session(source.id)
    copied = store.messages_for_session(first.id)[-1]
    assert copied.metadata.mixin_attempt_id == "retry-1"
    assert copied.mixin_usage_count == (0 if empty else 1)
    assert (
        await service.retry_snapshot(first.id, assistant_message_id=copied.id)
        == batch.snapshot
    )
    second = store.register_fork_snapshot(fork(store, copied.id, "fork-two"))
    store.close_session(first.id)
    copied = store.messages_for_session(second.id)[-1]
    assert (
        await service.retry_snapshot(second.id, assistant_message_id=copied.id)
        == batch.snapshot
    )
    assert service.preferences(second.id).entries == ()
    retry_batch = replace(
        batch,
        attempt_id="fork-retry",
        retry_of_attempt_id=batch.attempt_id,
        user_message_id=store.messages_for_session(second.id)[0].id,
        assistant_message_id=copied.id,
    )
    await service.accept_attempt(second.id, retry_batch)
    assert store.get_message(copied.id).metadata.mixin_attempt_id == "fork-retry"
    cid = store.promote_ephemeral_session(second.id)
    assert cid
    copied = store.get_message(copied.id)
    result = store.persistence.get_conversation_mixin_attempt(
        cid, assistant_message_id=copied.persisted_message_id
    )
    assert result.status == "recorded" and result.snapshot == batch.snapshot
    assert copied.metadata.mixin_attempt_id == "fork-retry"
    raw_marker = json.loads(
        db.get_connection()
        .execute(
            "SELECT metadata_json FROM messages WHERE id=?",
            (copied.persisted_message_id,),
        )
        .fetchone()[0]
    )
    assert raw_marker["mixin_attempt_id"] == "fork-retry"
    with db.transaction() as cursor:
        rows = cursor.execute(
            "SELECT attempt_id,assistant_message_id,replaces_attempt_id,result_revision_id,trace_call_id FROM console_prompt_mixin_attempts WHERE conversation_id=?",
            (cid,),
        ).fetchall()
        assert len(rows) == 4
        assert all(tuple(row)[2:] == (None, None, None) for row in rows)
        original = next(row for row in rows if row[0] == "attempt")
        assert (original[1] is None) is detached_predecessor
        assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
    store.close_session(second.id)
    store.end_app_runtime()


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_durable_fork_immediate_publication_and_original_retry(
    fork_trace_harness, monkeypatch, durable
):
    h = fork_trace_harness
    capture_mode(monkeypatch, False)
    _, scope = select(h, monkeypatch)
    h.adapter_error = True
    first = await h.controller.submit_draft("Question", session_id="session-1")
    await h.controller.retry_message(first.assistant_message_id)
    source = h.store._sessions["session-1"].persisted_conversation_id
    h.store.update_message_content(first.assistant_message_id, "partial answer")
    snapshot = fork(
        h.store, first.assistant_message_id, "durable-fork", durable=durable
    )
    if durable:
        commit(h.store, snapshot)
    destination = h.store.register_fork_snapshot(snapshot)
    copied = h.store.messages_for_session(destination.id)[-1]
    assert (
        copied.metadata.mixin_attempt_id
        == h.store.get_message(first.assistant_message_id).metadata.mixin_attempt_id
    )
    assert copied.mixin_usage_count == 1
    assert destination.prompt_mixin_preferences.preferences.entries == ()
    h.store.close_session("session-1")
    from tldw_chatbook.Canvas.repository import CanvasRepository

    version = h.db.get_conversation_by_id(source)["version"]
    h.db.soft_delete_conversation(source, expected_version=version)
    CanvasRepository(h.db).hard_purge_conversation(source)
    scope.records.clear()
    h.adapter_error = False
    result = await h.controller.retry_message(copied.id)
    assert result.accepted, result
    assert h.entries[-1][-1]["content"] == "body A\n\nQuestion"
    copied = h.store.get_message(copied.id)
    assert (
        copied.metadata.mixin_attempt_id
        != json.loads(dict(snapshot.prompt_mixin_history.markers)[copied.id])[
            "mixin_attempt_id"
        ]
    )
    if not durable:
        cid = h.store.promote_ephemeral_session(destination.id)
        copied = h.store.get_message(copied.id)
        replay = h.store.persistence.get_conversation_mixin_attempt(
            cid,
            assistant_message_id=copied.persisted_message_id,
            attempt_id=copied.metadata.mixin_attempt_id,
        )
        assert (
            replay.status == "recorded"
            and replay.snapshot.items[0].definition.body == "body A"
        )
    h.store.close_session(destination.id)
    h.store.end_app_runtime()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "marker",
    [
        {},
        {"mixin_attempt_id": "lost", "mixin_snapshot_schema_version": 1},
        {"mixin_attempt_id": "partial"},
        {"mixin_snapshot_schema_version": 1},
        {"mixin_attempt_id": "future", "mixin_snapshot_schema_version": 2},
    ],
)
async def test_durable_to_temporary_fork_preserves_legacy_and_unavailable_markers(
    fork_runtime, marker
):
    db, store = fork_runtime
    service = ConsolePromptMixinService(
        store=store, persistence=store.persistence, prompt_scope_service=None
    )
    source = store.create_session(
        ephemeral=True, settings=ConsoleSessionSettings(provider="openai", model="test")
    )
    store.append_message(source.id, role=ConsoleMessageRole.USER, content="Question")
    assistant = store.append_message(
        source.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(assistant.id, "Failed")
    assistant = store.mark_message_failed(assistant.id)
    store.promote_ephemeral_session(source.id)
    assistant = store.get_message(assistant.id)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE messages SET metadata_json=? WHERE id=?",
            (json.dumps(marker), assistant.persisted_message_id),
        )
    snapshot = fork(store, assistant.id, "temp-marker")
    destination = store.register_fork_snapshot(snapshot)
    store.close_session(source.id)
    copied = store.messages_for_session(destination.id)[-1]
    assert copied.mixin_usage_count == (None if marker else 0)
    assert copied.mixin_replay_status == ("unavailable" if marker else "legacy")
    from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService

    actions = ConsoleMessageActionService().available_actions(copied)
    assert any(
        row.action_id == ("regenerate" if marker else "retry") for row in actions
    )
    assert not any(
        row.action_id == ("retry" if marker else "regenerate") for row in actions
    )
    if marker:
        with pytest.raises(MixinSnapshotUnavailableError):
            await service.retry_snapshot(destination.id, assistant_message_id=copied.id)
    else:
        assert (
            await service.retry_snapshot(destination.id, assistant_message_id=copied.id)
        ).items == ()
    promoted = store.promote_ephemeral_session(destination.id)
    copied = store.get_message(copied.id)
    result = store.persistence.get_conversation_mixin_attempt(
        promoted, assistant_message_id=copied.persisted_message_id
    )
    assert result.status == ("unavailable" if marker else "legacy")
    raw = json.loads(
        db.get_connection()
        .execute(
            "SELECT metadata_json FROM messages WHERE id=?",
            (copied.persisted_message_id,),
        )
        .fetchone()[0]
    )
    assert {
        key: value for key, value in raw.items() if key.startswith("mixin_")
    } == marker
    store.close_session(destination.id)
    store.end_app_runtime()


def test_historical_import_rejects_foreign_anchor_and_rolls_back(make_database):
    from Tests.DB.test_console_prompt_mixins import saved_turn
    from tldw_chatbook.DB.console_prompt_mixins import (
        read_fork_state,
        write_attempt,
        write_fork_state,
    )

    db = make_database(":memory:", "fork-owner")
    source, source_ids = saved_turn(db)
    target, target_ids = saved_turn(db)
    with db.transaction() as cursor:
        write_attempt(cursor, source, batch=attempt(), native_message_ids=source_ids)
        state = read_fork_state(
            cursor, source, {key: key for key in source_ids.values()}
        )
    with pytest.raises(MixinSnapshotUnavailableError), db.transaction() as cursor:
        write_fork_state(
            cursor,
            target,
            state=state,
            message_ids={key: key for key in source_ids.values()},
        )
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT count(*) FROM console_prompt_mixin_attempts WHERE conversation_id=?",
                (target,),
            ).fetchone()[0]
            == 0
        )
    mapped = {source_ids[key]: target_ids[key] for key in source_ids}
    with pytest.raises(RuntimeError), db.transaction() as cursor:
        write_fork_state(cursor, target, state=state, message_ids=mapped)
        raise RuntimeError("parent rollback")
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT count(*) FROM console_prompt_mixin_attempts WHERE conversation_id=?",
                (target,),
            ).fetchone()[0]
            == 0
        )


def test_historical_state_rejects_mutable_markers_and_broken_lineage():
    from tldw_chatbook.Chat.console_prompt_mixins import (
        MixinForkAttempt,
        MixinForkState,
    )

    with pytest.raises(ValueError, match="mixin_invalid_fork_marker"):
        MixinForkState(markers=(["assistant", "{}"],))
    batch = attempt()
    with pytest.raises(ValueError, match="mixin_invalid_fork_lineage"):
        MixinForkState(
            (MixinForkAttempt("retry", None, None, "missing", batch.snapshot),)
        )


@pytest.mark.asyncio
async def test_durable_fork_receipt_change_refuses_parent_commit(
    fork_trace_harness, monkeypatch
):
    h = fork_trace_harness
    capture_mode(monkeypatch, False)
    select(h, monkeypatch)
    first = await h.controller.submit_draft("Question", session_id="session-1")
    snapshot = fork(h.store, first.assistant_message_id, "stale-receipts", durable=True)
    source = h.store._sessions["session-1"].persisted_conversation_id
    with h.db.transaction() as cursor:
        cursor.execute(
            "DELETE FROM console_prompt_mixin_attempt_items WHERE conversation_id=?",
            (source,),
        )
    with pytest.raises(MixinSnapshotUnavailableError, match="mixin_copy_unavailable"):
        commit(h.store, snapshot)
    assert h.db.get_conversation_by_id(snapshot.fork_conversation_id) is None
    assert all(row.id != snapshot.fork_session_id for row in h.store.sessions())
