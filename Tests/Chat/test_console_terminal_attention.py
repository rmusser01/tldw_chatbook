"""Exact durable Console terminal-attention receipts (TASK-22514 Task 6)."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tldw_chatbook.Chat import message_metadata as metadata_module
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_conversation_hydration import (
    console_messages_from_conversation_tree,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    GenerationVariantMeta,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleDispatchSettlementError,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata


RECEIPT_A = "11111111-1111-4111-8111-111111111111"
RECEIPT_B = "22222222-2222-4222-8222-222222222222"


def _video_metadata(**overrides: object) -> VideoGenerationMetadata:
    values: dict[str, object] = {
        "name": "terminal-video",
        "prompt": "a quiet terminal result",
        "backend": "test-backend",
    }
    values.update(overrides)
    return VideoGenerationMetadata(**values)  # type: ignore[arg-type]


def test_message_metadata_round_trips_a_bounded_local_terminal_receipt() -> None:
    metadata = MessageMetadata(
        engine="stream",
        provider="local",
        terminal_receipt_id=RECEIPT_A,
    )

    restored = MessageMetadata.from_json(metadata.to_json())

    assert restored == metadata
    assert restored is not None
    assert restored.terminal_receipt_id == RECEIPT_A
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="visible result",
        metadata=metadata,
    )
    assert metadata_module.terminal_receipt_id_for_message(message) == RECEIPT_A


@pytest.mark.parametrize("receipt_id", ["", "visible-result", "x" * 200])
def test_message_metadata_rejects_nonopaque_terminal_receipts(receipt_id: str) -> None:
    if receipt_id == "":
        assert MessageMetadata(terminal_receipt_id=receipt_id).terminal_receipt_id == ""
        return
    with pytest.raises(ValueError, match="terminal receipt"):
        MessageMetadata(terminal_receipt_id=receipt_id)


def test_durable_metadata_degrades_only_an_invalid_terminal_receipt() -> None:
    restored = MessageMetadata.from_json(
        json.dumps(
            {
                "engine": "stream",
                "interrupted": True,
                "terminal_receipt_id": "body-or-secret-not-opaque",
            }
        )
    )

    assert restored is not None
    assert restored.engine == "stream"
    assert restored.interrupted is True
    assert restored.terminal_receipt_id == ""


def test_video_metadata_preserves_full_payload_and_terminal_receipt_together() -> None:
    metadata = _video_metadata(
        negative_prompt="no blur",
        model="model-a",
        seed=7,
        duration_seconds=6.0,
        fps=24.0,
        width=1280,
        height=720,
        ratio="16:9",
        source_image_message_id="source-message",
        container="webm",
        terminal_receipt_id=RECEIPT_B,
    )

    restored = VideoGenerationMetadata.from_json(metadata.to_json())

    assert restored == metadata
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[video] terminal-video",
        video_metadata=metadata,
    )
    assert metadata_module.terminal_receipt_id_for_message(message) == RECEIPT_B


def _real_store(tmp_path):
    db = CharactersRAGDB(tmp_path / "terminal.sqlite", client_id="terminal-test")
    persistence = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Terminal attention")
    store.active_session_id = session.id
    return db, store, session


@pytest.mark.parametrize("terminal", ["complete", "failed"])
def test_store_terminalizes_ordinary_rows_with_one_stable_receipt(
    tmp_path, terminal
) -> None:
    db, store, session = _real_store(tmp_path)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(message.id, "terminal body")

    first_receipt = store._ensure_terminal_receipt(message.id)
    second_receipt = store._ensure_terminal_receipt(message.id)
    terminal_message = (
        store.mark_message_complete(message.id)
        if terminal == "complete"
        else store.mark_message_failed(message.id)
    )

    assert first_receipt == second_receipt
    assert metadata_module.terminal_receipt_id_for_message(terminal_message) == (
        first_receipt
    )
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == (
        (conversation_id, first_receipt),
    )
    durable = db.get_message_by_id(terminal_message.persisted_message_id)
    assert durable is not None
    assert durable["assistant_generation_state"] == terminal


def test_explicit_stop_terminalizes_without_a_receipt(tmp_path) -> None:
    db, store, session = _real_store(tmp_path)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(message.id, "partial")

    stopped = store.mark_message_stopped(message.id)

    assert metadata_module.terminal_receipt_id_for_message(stopped) == ""
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == ()


def test_temporary_completion_omits_receipt_before_and_after_promotion(
    tmp_path,
) -> None:
    """A non-durable terminal row cannot publish a speculative receipt."""

    db = CharactersRAGDB(
        tmp_path / "temporary-terminal.sqlite",
        client_id="terminal-test",
    )
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(title="Temporary terminal", ephemeral=True)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(message.id, "temporary completed body")

    completed = store.mark_message_complete(message.id)

    assert completed.status == "complete"
    assert completed.persisted_message_id is None
    assert metadata_module.terminal_receipt_id_for_message(completed) == ""
    assert message.id not in store._pending_terminal_receipts
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == ()

    conversation_id = store.promote_ephemeral_session(session.id)

    assert conversation_id is not None
    promoted = store.get_message(message.id)
    assert promoted.persisted_message_id is not None
    row = db.get_message_by_id(promoted.persisted_message_id)
    assert row is not None
    metadata = MessageMetadata.from_json(row.get("metadata_json"))
    assert metadata is None or metadata.terminal_receipt_id == ""
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == ()


def test_pending_terminal_write_reports_success_only_after_row_and_mark_exist(
    tmp_path,
    monkeypatch,
) -> None:
    """A pending create cannot claim success from a no-op persistence attempt."""

    db, store, session = _real_store(tmp_path)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    real_persist = store._persist_new_message
    monkeypatch.setattr(store, "_persist_new_message", lambda **_kwargs: None)
    store.append_stream_chunk(message.id, "eventually durable")
    live_message = store._nodes_by_session[session.id][message.id]
    store._materialize_stream_buffer(live_message)
    receipt_id = store._ensure_terminal_receipt(message.id)

    first = store._persist_existing_message(
        live_message,
        terminal_receipt_id=receipt_id,
        terminal_outcome="complete",
    )

    assert first is False
    assert live_message.persisted_message_id is None
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == ()
    assert live_message.id in store._pending_persistence_message_ids
    assert live_message.id not in store._terminal_persistence_deferred_ids
    assert live_message.content == "eventually durable"

    monkeypatch.setattr(store, "_persist_new_message", real_persist)
    second = store._persist_existing_message(
        live_message,
        terminal_receipt_id=receipt_id,
        terminal_outcome="complete",
    )

    assert second is True
    assert live_message.persisted_message_id is not None
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == (
        (session.persisted_conversation_id, receipt_id),
    )
    durable = db.get_message_by_id(live_message.persisted_message_id)
    assert durable is not None
    assert durable["assistant_generation_state"] == "complete"


@pytest.mark.parametrize("reason", ["user_stop", "session_close", "app_shutdown"])
def test_explicit_cancellation_reasons_use_stopped_without_a_receipt(
    tmp_path, reason
) -> None:
    db, store, session = _real_store(tmp_path)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(message.id, f"partial before {reason}")

    terminal = store.mark_message_stopped(message.id)

    assert terminal.status == "stopped"
    assert metadata_module.terminal_receipt_id_for_message(terminal) == ""
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == ()


def test_unexpected_cancellation_terminalizes_as_failure_with_a_receipt(
    tmp_path,
) -> None:
    db, store, session = _real_store(tmp_path)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(message.id, "partial before unexpected cancellation")

    terminal = store.mark_message_failed(message.id)

    receipt_id = metadata_module.terminal_receipt_id_for_message(terminal)
    assert terminal.status == "failed"
    assert receipt_id
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == (
        (session.persisted_conversation_id, receipt_id),
    )
    durable = db.get_message_by_id(terminal.persisted_message_id)
    assert durable is not None
    assert durable["assistant_generation_state"] == "failed"


def test_new_retry_attempt_rotates_receipt_and_exact_ack_keeps_new_attention(
    tmp_path,
) -> None:
    db, store, session = _real_store(tmp_path)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(message.id, "first failed attempt")
    first = store.mark_message_failed(message.id)
    receipt_a = metadata_module.terminal_receipt_id_for_message(first)

    store.prepare_message_retry(message.id)
    store.append_stream_chunk(message.id, "later successful attempt")
    second = store.mark_message_complete(message.id)
    receipt_b = metadata_module.terminal_receipt_id_for_message(second)

    assert receipt_a and receipt_b and receipt_a != receipt_b
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    marks = ConversationLocalMarksService(db)
    assert marks.console_terminal_outcome(conversation_id, receipt_a) == "failed"
    assert marks.console_terminal_outcome(conversation_id, receipt_b) == "complete"
    assert marks.acknowledge_console_unseen(conversation_id, receipt_a) is True
    assert marks.console_terminal_outcome(conversation_id, receipt_a) is None
    assert marks.console_terminal_outcome(conversation_id, receipt_b) == "complete"
    assert marks.list_console_unseen_marks() == ((conversation_id, receipt_b),)


@pytest.mark.parametrize("outcome", ["complete", "failed"])
def test_variant_attempt_rotates_receipt_for_success_or_failure(
    tmp_path, outcome
) -> None:
    db, store, session = _real_store(tmp_path)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(message.id, "base answer")
    base = store.mark_message_complete(message.id)
    receipt_a = metadata_module.terminal_receipt_id_for_message(base)

    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "replacement answer")
    terminal = (
        store.finalize_variant_stream(message.id)
        if outcome == "complete"
        else store.mark_message_failed(message.id)
    )
    receipt_b = metadata_module.terminal_receipt_id_for_message(terminal)

    assert receipt_a and receipt_b and receipt_a != receipt_b
    if outcome == "failed":
        assert terminal.content == "base answer"
        assert terminal.status == "complete"
        assert terminal.assistant_generation_state == "complete"
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    marks = ConversationLocalMarksService(db)
    assert marks.console_terminal_outcome(conversation_id, receipt_a) == "complete"
    assert marks.console_terminal_outcome(conversation_id, receipt_b) == outcome
    durable = db.get_message_by_id(terminal.persisted_message_id)
    assert durable is not None
    assert durable["assistant_generation_state"] == "complete"
    assert marks.acknowledge_console_unseen(conversation_id, receipt_a) is True
    assert marks.list_console_unseen_marks() == ((conversation_id, receipt_b),)


@pytest.mark.parametrize("terminal", ["complete", "failed"])
def test_ordinary_terminal_mark_failure_restores_live_state_and_retries_same_receipt(
    tmp_path, terminal
) -> None:
    db, store, session = _real_store(tmp_path)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(message.id, "terminal body")
    live = store._nodes_by_session[session.id][message.id]
    before = deepcopy(live)
    connection = db.get_connection()
    connection.execute(
        "CREATE TRIGGER fail_ordinary_terminal_mark BEFORE INSERT ON "
        "conversation_local_marks BEGIN SELECT RAISE(ABORT, 'mark failure'); END"
    )
    connection.commit()

    settle = (
        store.mark_message_complete
        if terminal == "complete"
        else store.mark_message_failed
    )
    with pytest.raises(Exception, match="mark failure"):
        settle(message.id)

    assert vars(live) == vars(before)
    pending_receipt = store._pending_terminal_receipts[message.id]
    assert metadata_module.terminal_receipt_id_for_message(live) == ""
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == ()

    connection.execute("DROP TRIGGER fail_ordinary_terminal_mark")
    connection.commit()
    retried = settle(message.id)
    assert metadata_module.terminal_receipt_id_for_message(retried) == pending_receipt
    assert message.id not in store._pending_terminal_receipts


@pytest.mark.parametrize("terminal", ["complete", "failed"])
def test_variant_terminal_mark_failure_restores_owner_and_retry_gates(
    tmp_path, terminal
) -> None:
    db, store, session = _real_store(tmp_path)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(message.id, "base answer")
    store.mark_message_complete(message.id)
    store.begin_variant_stream(message.id)
    store.append_stream_chunk(message.id, "replacement answer")
    live = store._nodes_by_session[session.id][message.id]
    before = deepcopy(live)
    base_before = store._variant_stream_bases[message.id]
    connection = db.get_connection()
    connection.execute(
        "CREATE TRIGGER fail_variant_terminal_mark BEFORE INSERT ON "
        "conversation_local_marks BEGIN SELECT RAISE(ABORT, 'mark failure'); END"
    )
    connection.commit()

    settle = (
        store.finalize_variant_stream
        if terminal == "complete"
        else store.mark_message_failed
    )
    with pytest.raises(Exception, match="mark failure"):
        settle(message.id)

    assert vars(live) == vars(before)
    assert store._variant_stream_bases[message.id] == base_before
    assert message.id not in store._variant_restored_message_ids
    pending_receipt = store._pending_terminal_receipts[message.id]

    connection.execute("DROP TRIGGER fail_variant_terminal_mark")
    connection.commit()
    retried = settle(message.id)
    assert metadata_module.terminal_receipt_id_for_message(retried) == pending_receipt
    if terminal == "failed":
        assert retried.status == "complete"
        assert retried.assistant_generation_state == "complete"
        durable = db.get_message_by_id(retried.persisted_message_id)
        assert durable is not None
        assert durable["assistant_generation_state"] == "complete"
        assert ConversationLocalMarksService(db).console_terminal_outcome(
            session.persisted_conversation_id, pending_receipt
        ) == "failed"


@pytest.mark.parametrize("media_kind", ["image", "video"])
def test_immediate_media_mark_failure_rolls_back_live_append_and_can_retry(
    tmp_path, media_kind
) -> None:
    db, store, session = _real_store(tmp_path)
    connection = db.get_connection()
    connection.execute(
        "CREATE TRIGGER fail_media_terminal_mark BEFORE INSERT ON "
        "conversation_local_marks BEGIN SELECT RAISE(ABORT, 'mark failure'); END"
    )
    connection.commit()

    def append_media():
        if media_kind == "image":
            return store.append_generation_message(
                session.id,
                content="[image] generated result",
                variants=[(b"png-bytes", "image/png", _generation_meta())],
                persist=True,
            )
        return store.append_video_message(
            session.id,
            video_metadata=_video_metadata(),
            persist=True,
        )

    with pytest.raises(Exception, match="mark failure"):
        append_media()

    assert store.messages_for_session(session.id) == []
    assert store._pending_terminal_receipts == {}
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == ()
    connection.execute("DROP TRIGGER fail_media_terminal_mark")
    connection.commit()

    retried = append_media()
    receipt_id = metadata_module.terminal_receipt_id_for_message(retried)
    assert receipt_id
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == (
        (session.persisted_conversation_id, receipt_id),
    )
    durable = db.get_message_by_id(retried.persisted_message_id)
    assert durable is not None
    assert durable["assistant_generation_state"] == "complete"


def test_merge_persisted_generation_message_retains_image_terminal_receipt(
    tmp_path,
) -> None:
    db, source, source_session = _real_store(tmp_path)
    persisted = source.append_generation_message(
        source_session.id,
        content="[image] generated result",
        variants=[(b"png-bytes", "image/png", _generation_meta())],
        persist=True,
    )
    receipt_id = metadata_module.terminal_receipt_id_for_message(persisted)
    conversation_id = source_session.persisted_conversation_id
    assert receipt_id and conversation_id is not None
    fresh = ConsoleChatStore(persistence=ChatPersistenceService(db))
    fresh_session = fresh.restore_persisted_session(
        title="Merge image",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=[],
        active_leaf_persisted_id=None,
    )

    merged = fresh.merge_persisted_generation_message(
        fresh_session.id,
        persisted.persisted_message_id,
    )

    assert merged is not None
    assert metadata_module.terminal_receipt_id_for_message(merged) == receipt_id


def _generation_meta() -> GenerationVariantMeta:
    return GenerationVariantMeta(
        prompt="a generated result",
        negative_prompt="",
        backend="test",
        model="model-a",
        seed=7,
        style=None,
        params={"steps": 1},
    )


def test_immediate_image_create_publishes_receipt_with_full_media_metadata(
    tmp_path,
) -> None:
    db, store, session = _real_store(tmp_path)

    message = store.append_generation_message(
        session.id,
        content="[image] generated result",
        variants=[(b"png-bytes", "image/png", _generation_meta())],
        persist=True,
    )

    receipt_id = metadata_module.terminal_receipt_id_for_message(message)
    assert receipt_id
    assert message.generation_metadata == (_generation_meta(),)
    assert message.attachments[0].data == b"png-bytes"
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == (
        (session.persisted_conversation_id, receipt_id),
    )
    durable = db.get_message_by_id(message.persisted_message_id)
    assert durable is not None
    assert durable["assistant_generation_state"] == "complete"


def test_immediate_video_create_preserves_payload_and_publishes_receipt(tmp_path) -> None:
    db, store, session = _real_store(tmp_path)
    original = _video_metadata(
        negative_prompt="no blur",
        model="model-a",
        seed=7,
        duration_seconds=6.0,
        fps=24.0,
        width=1280,
        height=720,
        ratio="16:9",
        source_image_message_id="source-message",
        container="webm",
    )

    message = store.append_video_message(
        session.id,
        video_metadata=original,
        persist=True,
    )

    receipt_id = metadata_module.terminal_receipt_id_for_message(message)
    assert receipt_id
    assert message.video_metadata == _video_metadata(
        negative_prompt="no blur",
        model="model-a",
        seed=7,
        duration_seconds=6.0,
        fps=24.0,
        width=1280,
        height=720,
        ratio="16:9",
        source_image_message_id="source-message",
        container="webm",
        terminal_receipt_id=receipt_id,
    )
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == (
        (session.persisted_conversation_id, receipt_id),
    )
    durable = db.get_message_by_id(message.persisted_message_id)
    assert durable is not None
    assert durable["assistant_generation_state"] == "complete"


@pytest.mark.parametrize("media_kind", ["image", "video"])
@pytest.mark.parametrize("ephemeral", [False, True])
def test_media_expected_durability_distinguishes_temporary_from_missing_row(
    tmp_path, monkeypatch, media_kind, ephemeral
) -> None:
    db = CharactersRAGDB(tmp_path / "media.sqlite", client_id="terminal-test")
    persistence = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Media", ephemeral=ephemeral)
    connection = db.get_connection()
    changes_before = connection.total_changes
    original = _video_metadata(
        negative_prompt="no blur",
        model="model-a",
        seed=7,
        duration_seconds=6.0,
        fps=24.0,
        width=1280,
        height=720,
        ratio="16:9",
        container="webm",
    )
    if ephemeral:
        monkeypatch.setattr(
            persistence,
            "create_conversation",
            lambda **_: pytest.fail("temporary conversation write"),
        )
        monkeypatch.setattr(
            persistence,
            "create_message",
            lambda **_: pytest.fail("temporary message write"),
        )
    else:
        monkeypatch.setattr(store, "_persist_new_message", lambda **_: None)

    def append_media():
        if media_kind == "image":
            return store.append_generation_message(
                session.id,
                content="[image] generated result",
                variants=[(b"png-bytes", "image/png", _generation_meta())],
                persist=True,
            )
        return store.append_video_message(
            session.id, video_metadata=original, persist=True
        )

    try:
        if ephemeral:
            message = append_media()
            assert store.get_message(message.id) == message
            assert store.active_leaf(session.id) == message.id
            assert message.status == "complete"
            assert message.assistant_generation_state == "complete"
            assert message.persisted_message_id is None
            assert session.persisted_conversation_id is None
            assert metadata_module.terminal_receipt_id_for_message(message) == ""
            if media_kind == "image":
                assert message.generation_metadata == (_generation_meta(),)
                assert [
                    (item.data, item.mime_type) for item in message.attachments
                ] == [(b"png-bytes", "image/png")]
            else:
                assert message.video_metadata == original
            assert connection.total_changes == changes_before
            assert (
                connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
            )
        else:
            with pytest.raises(
                ConsoleDispatchSettlementError,
                match=f"Terminal {media_kind} persistence failed",
            ):
                append_media()
            assert store.messages_for_session(session.id) == []
            assert store.active_leaf(session.id) is None
        assert store._pending_terminal_receipts == {}
        assert store._pending_persistence_message_ids == set()
        assert ConversationLocalMarksService(db).list_console_unseen_marks() == ()
    finally:
        db.close_connection()


@pytest.mark.parametrize("media_kind", ["image", "video"])
def test_unpersisted_media_does_not_publish_terminal_attention(
    tmp_path, media_kind
) -> None:
    db, store, session = _real_store(tmp_path)

    if media_kind == "image":
        message = store.append_generation_message(
            session.id,
            content="[image] generated result",
            variants=[(b"png-bytes", "image/png", _generation_meta())],
            persist=False,
        )
    else:
        message = store.append_video_message(
            session.id,
            video_metadata=_video_metadata(),
            persist=False,
        )

    assert metadata_module.terminal_receipt_id_for_message(message) == ""
    assert ConversationLocalMarksService(db).list_console_unseen_marks() == ()


def test_two_results_keep_distinct_receipts_and_acknowledge_only_the_exact_row(
    tmp_path,
) -> None:
    db, store, session = _real_store(tmp_path)
    receipts: list[str] = []
    for body in ("first terminal result", "second terminal result"):
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        store.append_stream_chunk(message.id, body)
        terminal = store.mark_message_complete(message.id)
        receipts.append(metadata_module.terminal_receipt_id_for_message(terminal))

    assert receipts[0] and receipts[1] and receipts[0] != receipts[1]
    marks = ConversationLocalMarksService(db)
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    assert set(marks.list_console_unseen_marks()) == {
        (conversation_id, receipts[0]),
        (conversation_id, receipts[1]),
    }
    assert marks.acknowledge_console_unseen(conversation_id, receipts[0]) is True
    assert marks.list_console_unseen_marks() == ((conversation_id, receipts[1]),)


def test_terminal_receipt_never_enters_provider_message_content() -> None:
    store = ConsoleChatStore()
    session = store.create_session(title="Provider privacy", ephemeral=True)
    store.active_session_id = session.id
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="visible result",
        metadata=MessageMetadata(terminal_receipt_id=RECEIPT_A),
        persist=False,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=object(),
        agent_runtime_enabled=False,
    )

    payload = controller._provider_messages_for_session(session.id)

    assert any("visible result" in str(row.get("content")) for row in payload)
    assert RECEIPT_A not in json.dumps(payload)


@pytest.mark.parametrize("message_kind", ["ordinary", "image", "video"])
def test_terminal_attention_survives_restart_hydrates_and_acks_exactly(
    tmp_path, message_kind
) -> None:
    database_path = tmp_path / "terminal-restart.sqlite"
    db = CharactersRAGDB(database_path, client_id="terminal-first-process")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session(title="Terminal restart")
    store.active_session_id = session.id
    if message_kind == "ordinary":
        live = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        store.append_stream_chunk(live.id, "ordinary terminal result")
        live = store.mark_message_complete(live.id)
    elif message_kind == "image":
        live = store.append_generation_message(
            session.id,
            content="[image] generated result",
            variants=[(b"png-bytes", "image/png", _generation_meta())],
            persist=True,
        )
    else:
        live = store.append_video_message(
            session.id,
            video_metadata=_video_metadata(
                negative_prompt="no blur",
                model="model-a",
                seed=7,
                duration_seconds=6.0,
                fps=24.0,
                width=1280,
                height=720,
                ratio="16:9",
                container="webm",
            ),
            persist=True,
        )
    receipt_id = metadata_module.terminal_receipt_id_for_message(live)
    conversation_id = session.persisted_conversation_id
    assert receipt_id
    assert conversation_id is not None
    db.close_connection()

    restarted_db = CharactersRAGDB(
        database_path,
        client_id="terminal-second-process",
    )
    tree = ChatConversationService(restarted_db).get_conversation_tree(
        conversation_id,
        depth_cap=10_000,
        root_limit=10_000,
    )
    restored_nodes = console_messages_from_conversation_tree(tree, db=restarted_db)
    restarted_store = ConsoleChatStore(
        persistence=ChatPersistenceService(restarted_db)
    )
    restored_session = restarted_store.restore_persisted_session(
        title="Terminal restart",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=restored_nodes,
        active_leaf_persisted_id=restarted_db.get_conversation_active_leaf(
            conversation_id
        ),
    )
    restored = restarted_store.messages_for_session(restored_session.id)[0]

    assert metadata_module.terminal_receipt_id_for_message(restored) == receipt_id
    assert restored.assistant_generation_state == "complete"
    if message_kind == "image":
        assert restored.attachments[0].data == b"png-bytes"
        assert restored.generation_metadata == (_generation_meta(),)
    elif message_kind == "video":
        assert restored.video_metadata == _video_metadata(
            negative_prompt="no blur",
            model="model-a",
            seed=7,
            duration_seconds=6.0,
            fps=24.0,
            width=1280,
            height=720,
            ratio="16:9",
            container="webm",
            terminal_receipt_id=receipt_id,
        )
    marks = ConversationLocalMarksService(restarted_db)
    assert marks.list_console_unseen_marks() == ((conversation_id, receipt_id),)
    assert marks.acknowledge_console_unseen(conversation_id, receipt_id) is True
    restarted_db.close_connection()

    final_db = CharactersRAGDB(database_path, client_id="terminal-third-process")
    assert ConversationLocalMarksService(final_db).list_console_unseen_marks() == ()
