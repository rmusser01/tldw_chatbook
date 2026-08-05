"""Trusted Console speech snapshot contracts."""

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_speech import (
    ConsoleSpeechSnapshotRejected,
    ConsoleSpeechSnapshotRejectionCode,
    TTSMessageSpeechSnapshot,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.TTS.profile_types import CharacterRef


def _snapshot(
    *,
    raw_content: str = "private response",
    character_ref: CharacterRef | None = None,
) -> TTSMessageSpeechSnapshot:
    return TTSMessageSpeechSnapshot(
        session_id="session-1",
        message_id="message-1",
        persisted_conversation_id=None,
        persisted_message_id=None,
        raw_content=raw_content,
        selected_variant_id="message-1",
        speech_revision=0,
        persisted_message_version=None,
        role=ConsoleMessageRole.ASSISTANT,
        status="complete",
        assistant_kind="character" if character_ref is not None else "generic",
        character_ref=character_ref,
    )


def test_snapshot_is_frozen_and_redacts_content_and_authority_from_repr():
    character_ref = CharacterRef(
        source="local",
        authority_id="authority-secret",
        character_id="17",
    )
    snapshot = _snapshot(
        raw_content="do not log this response",
        character_ref=character_ref,
    )

    rendered = repr(snapshot)

    assert "do not log this response" not in rendered
    assert "authority-secret" not in rendered
    assert snapshot.raw_content == "do not log this response"
    assert snapshot.character_ref == character_ref
    with pytest.raises(FrozenInstanceError):
        snapshot.raw_content = "changed"  # type: ignore[misc]


def test_snapshot_rejects_invalid_structural_values():
    with pytest.raises(ValueError, match="speech_revision"):
        TTSMessageSpeechSnapshot(
            session_id="session-1",
            message_id="message-1",
            persisted_conversation_id=None,
            persisted_message_id=None,
            raw_content="response",
            selected_variant_id="message-1",
            speech_revision=-1,
            persisted_message_version=None,
            role=ConsoleMessageRole.ASSISTANT,
            status="complete",
            assistant_kind="generic",
            character_ref=None,
        )


def test_snapshot_rejection_exposes_only_bounded_code_and_safe_retry_copy():
    error = ConsoleSpeechSnapshotRejected(
        ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
    )

    assert error.code is ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
    assert str(error) == "Message changed before speech started; select Speak again."
    assert "private response" not in repr(error)

    with pytest.raises(ValueError, match="rejection code"):
        ConsoleSpeechSnapshotRejected("unbounded-detail")  # type: ignore[arg-type]


def _assert_rejected(
    store: ConsoleChatStore,
    snapshot: TTSMessageSpeechSnapshot,
    code: ConsoleSpeechSnapshotRejectionCode,
) -> None:
    with pytest.raises(ConsoleSpeechSnapshotRejected) as caught:
        store.validate_tts_message_speech_snapshot(snapshot)
    assert caught.value.code is code


def _revision(store: ConsoleChatStore, message_id: str) -> int:
    return store._message_speech_revisions[message_id]


def test_store_issues_and_validates_exact_scoped_character_snapshot():
    store = ConsoleChatStore()
    session = store.create_session(
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="local-authority",
        character_id=7,
    )
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="  Exact visible response.\n",
    )

    snapshot = store.issue_tts_message_speech_snapshot(message.id)

    assert snapshot.session_id == session.id
    assert snapshot.message_id == message.id
    assert snapshot.persisted_conversation_id is None
    assert snapshot.persisted_message_id is None
    assert snapshot.raw_content == "  Exact visible response.\n"
    assert snapshot.selected_variant_id == message.id
    assert snapshot.speech_revision == 0
    assert snapshot.persisted_message_version is None
    assert snapshot.role is ConsoleMessageRole.ASSISTANT
    assert snapshot.status == "complete"
    assert snapshot.assistant_kind == "character"
    assert snapshot.character_ref == CharacterRef(
        source="local",
        authority_id="local-authority",
        character_id="7",
    )
    assert (
        store.validate_tts_message_speech_snapshot(snapshot)
        == "  Exact visible response.\n"
    )


@pytest.mark.parametrize(
    "session_kwargs, expected_kind",
    [
        ({}, "generic"),
        (
            {
                "runtime_backend": "local",
                "assistant_kind": "persona",
                "assistant_id": "persona-1",
            },
            "persona",
        ),
        (
            {
                "runtime_backend": "server",
                "assistant_kind": "character",
                "assistant_id": "character-1",
                "assistant_authority_id": None,
            },
            "character",
        ),
    ],
    ids=["generic", "persona", "authority-null-character"],
)
def test_store_snapshot_does_not_invent_character_authority(
    session_kwargs,
    expected_kind,
):
    store = ConsoleChatStore()
    session = store.create_session(**session_kwargs)
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="response",
    )

    snapshot = store.issue_tts_message_speech_snapshot(message.id)

    assert snapshot.assistant_kind == expected_kind
    assert snapshot.character_ref is None
    assert store.validate_tts_message_speech_snapshot(snapshot) == "response"


@pytest.mark.parametrize(
    "scenario",
    [
        "user",
        "system",
        "tool",
        "blank",
        "pending",
        "streaming",
        "stopped",
        "failed",
    ],
)
def test_store_refuses_non_assistant_incomplete_or_blank_messages(scenario):
    store = ConsoleChatStore()
    session = store.create_session()
    if scenario in {"user", "system", "tool"}:
        role = {
            "user": ConsoleMessageRole.USER,
            "system": ConsoleMessageRole.SYSTEM,
            "tool": ConsoleMessageRole.TOOL,
        }[scenario]
        message = store.append_message(session.id, role=role, content="text")
    elif scenario == "blank":
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=" \n ",
        )
    else:
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
        )
        if scenario == "streaming":
            store.append_stream_chunk(message.id, "partial")
        elif scenario == "stopped":
            store.mark_message_stopped(message.id)
        elif scenario == "failed":
            store.mark_message_failed(message.id)

    with pytest.raises(ConsoleSpeechSnapshotRejected):
        store.issue_tts_message_speech_snapshot(message.id)


def test_edit_then_restore_identical_content_still_invalidates_snapshot():
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="original",
    )
    snapshot = store.issue_tts_message_speech_snapshot(message.id)

    store.update_message_content(message.id, "edited")
    store.update_message_content(message.id, "original")

    assert store.get_message(message.id).content == snapshot.raw_content
    _assert_rejected(
        store,
        snapshot,
        ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED,
    )


def test_session_switch_rejects_snapshot_without_mutating_message():
    store = ConsoleChatStore()
    first = store.create_session()
    message = store.append_message(
        first.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="response",
    )
    snapshot = store.issue_tts_message_speech_snapshot(message.id)

    store.create_session(title="Second")

    _assert_rejected(
        store,
        snapshot,
        ConsoleSpeechSnapshotRejectionCode.SESSION_CHANGED,
    )
    assert store.get_message(message.id).content == "response"


def test_message_moved_off_active_path_rejects_snapshot():
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="first branch",
    )
    snapshot = store.issue_tts_message_speech_snapshot(message.id)

    store.create_sibling(
        message.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="second branch",
    )

    _assert_rejected(
        store,
        snapshot,
        ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED,
    )


def test_delete_rejects_snapshot_and_releases_process_local_revision():
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="response",
    )
    snapshot = store.issue_tts_message_speech_snapshot(message.id)

    store.delete_message(message.id)

    _assert_rejected(
        store,
        snapshot,
        ConsoleSpeechSnapshotRejectionCode.MISSING_MESSAGE,
    )
    assert message.id not in store._message_speech_revisions


def test_close_session_releases_all_process_local_revisions():
    store = ConsoleChatStore()
    session = store.create_session()
    first = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    second = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="response",
    )

    store.close_session(session.id)

    assert first.id not in store._message_speech_revisions
    assert second.id not in store._message_speech_revisions


def test_authorship_change_rejects_snapshot():
    store = ConsoleChatStore()
    session = store.create_session(
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="local-authority",
        character_id=7,
    )
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="response",
    )
    snapshot = store.issue_tts_message_speech_snapshot(message.id)

    session.assistant_authority_id = "different-authority"

    _assert_rejected(
        store,
        snapshot,
        ConsoleSpeechSnapshotRejectionCode.AUTHORSHIP_CHANGED,
    )


def test_selected_text_variant_identity_and_content_are_revalidated():
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
    )
    linear_snapshot = store.issue_tts_message_speech_snapshot(message.id)

    with_variant = store.add_variant(message.id, "second")
    second_snapshot = store.issue_tts_message_speech_snapshot(message.id)

    assert with_variant.variants is not None
    assert second_snapshot.selected_variant_id == with_variant.variants.current.id
    assert second_snapshot.raw_content == "second"
    _assert_rejected(
        store,
        linear_snapshot,
        ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED,
    )

    store.select_variant(message.id, 0)
    first_variant_snapshot = store.issue_tts_message_speech_snapshot(message.id)

    assert first_variant_snapshot.selected_variant_id == (
        with_variant.variants.variants[0].id
    )
    assert first_variant_snapshot.raw_content == "first"
    _assert_rejected(
        store,
        second_snapshot,
        ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED,
    )

    store.update_message_content(message.id, "changed first")
    _assert_rejected(
        store,
        first_variant_snapshot,
        ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED,
    )


def test_variant_stream_lifecycle_invalidates_pre_stream_snapshot():
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="base",
    )
    snapshot = store.issue_tts_message_speech_snapshot(message.id)

    before_begin = _revision(store, message.id)
    store.begin_variant_stream(message.id)
    assert _revision(store, message.id) > before_begin
    _assert_rejected(
        store,
        snapshot,
        ConsoleSpeechSnapshotRejectionCode.MESSAGE_NOT_SPEAKABLE,
    )

    before_chunk = _revision(store, message.id)
    store.append_stream_chunk(message.id, "new")
    assert _revision(store, message.id) > before_chunk

    before_finalize = _revision(store, message.id)
    store.finalize_variant_stream(message.id)
    assert _revision(store, message.id) > before_finalize
    _assert_rejected(
        store,
        snapshot,
        ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED,
    )


def test_speech_revision_bumps_for_content_status_and_variant_mutations():
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    assert _revision(store, message.id) == 0

    before = _revision(store, message.id)
    store.append_stream_chunk(message.id, "partial")
    assert _revision(store, message.id) > before

    before = _revision(store, message.id)
    store.get_message(message.id)
    assert _revision(store, message.id) > before

    before = _revision(store, message.id)
    store.reset_stream_content(message.id)
    assert _revision(store, message.id) > before

    store.append_stream_chunk(message.id, "answer")
    before = _revision(store, message.id)
    store.mark_message_complete(message.id)
    assert _revision(store, message.id) > before

    before = _revision(store, message.id)
    store.update_message_content(message.id, "edited")
    assert _revision(store, message.id) > before

    before = _revision(store, message.id)
    store.add_variant(message.id, "variant")
    assert _revision(store, message.id) > before

    before = _revision(store, message.id)
    store.select_variant(message.id, 0)
    assert _revision(store, message.id) > before

    before = _revision(store, message.id)
    store.update_message_content(message.id, "edited variant")
    assert _revision(store, message.id) > before


def test_speech_revision_bumps_for_provisional_retry_and_terminal_mutations():
    store = ConsoleChatStore()
    session = store.create_session()
    provisional = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        defer_terminal_persistence=True,
    )

    before = _revision(store, provisional.id)
    store.replace_deferred_terminal_body(provisional.id, "selected")
    assert _revision(store, provisional.id) > before

    before = _revision(store, provisional.id)
    store.mark_message_complete(provisional.id)
    assert _revision(store, provisional.id) > before

    failed = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    before = _revision(store, failed.id)
    store.mark_message_failed(failed.id)
    assert _revision(store, failed.id) > before

    before = _revision(store, failed.id)
    store.prepare_message_retry(failed.id)
    assert _revision(store, failed.id) > before

    stopped = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    before = _revision(store, stopped.id)
    store.mark_message_stopped(stopped.id)
    assert _revision(store, stopped.id) > before

    blocked = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="not sent",
    )
    before = _revision(store, blocked.id)
    store.mark_message_send_blocked(blocked.id)
    assert _revision(store, blocked.id) > before


def test_persistence_service_reads_only_current_positive_message_version(tmp_path):
    database = CharactersRAGDB(tmp_path / "speech-version.sqlite", "speech-test")
    try:
        service = ChatPersistenceService(database)
        conversation_id = service.create_conversation()
        message_id = service.create_message(
            conversation_id=conversation_id,
            sender="assistant",
            content="response",
            image_data=None,
            image_mime_type=None,
        )

        assert service.get_message_version(message_id) == 1
        assert service.get_message_version("missing") is None

        database.soft_delete_message(message_id, expected_version=1)

        assert service.get_message_version(message_id) is None
    finally:
        database.close_connection()


def test_persisted_snapshot_rejects_external_edit_then_restore(tmp_path):
    database = CharactersRAGDB(tmp_path / "speech-stale.sqlite", "speech-test")
    try:
        persistence = ChatPersistenceService(database)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="original",
            persist=True,
        )
        snapshot = store.issue_tts_message_speech_snapshot(message.id)
        assert snapshot.persisted_conversation_id is not None
        assert snapshot.persisted_message_id is not None
        assert snapshot.persisted_message_version == 1

        for content in ("external edit", "original"):
            assert persistence.update_message_content(
                message_id=snapshot.persisted_message_id,
                content=content,
                image_data=None,
                image_mime_type=None,
            )

        assert store.get_message(message.id).content == snapshot.raw_content
        _assert_rejected(
            store,
            snapshot,
            ConsoleSpeechSnapshotRejectionCode.PERSISTED_VERSION_CHANGED,
        )
    finally:
        database.close_connection()


def test_persisted_message_without_version_reader_fails_closed():
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="response",
    )
    session.persisted_conversation_id = "conversation-1"
    store._nodes_by_session[session.id][
        message.id
    ].persisted_message_id = "persisted-message-1"
    store.persistence = object()  # type: ignore[assignment]

    with pytest.raises(ConsoleSpeechSnapshotRejected) as caught:
        store.issue_tts_message_speech_snapshot(message.id)

    assert (
        caught.value.code
        is ConsoleSpeechSnapshotRejectionCode.PERSISTED_VERSION_UNAVAILABLE
    )
