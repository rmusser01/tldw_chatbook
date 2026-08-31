"""Console store character-emote sanitization and terminal contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import pytest

from tldw_chatbook.Character_Chat.emote_directives import (
    CharacterEmoteAssetReference,
    CharacterEmoteRunSnapshot,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def _snapshot() -> CharacterEmoteRunSnapshot:
    return CharacterEmoteRunSnapshot(
        actor_id=7,
        pack_id=11,
        pack_version_id=13,
        states=("happy", "sad", "smug"),
        assets=(
            CharacterEmoteAssetReference("happy", "happy", 17, 19),
            CharacterEmoteAssetReference("sad", "sad", 23, 29),
            CharacterEmoteAssetReference("smug", "custom:smug", 31, 37),
        ),
    )


def _armed_message(store: ConsoleChatStore):
    session = store.create_session(
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Please tell me the result.",
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.begin_character_emote_capture(assistant.id, _snapshot())
    return session, assistant


def test_armed_stream_strips_controls_and_publishes_every_event() -> None:
    store = ConsoleChatStore()
    session, assistant = _armed_message(store)

    store.append_stream_chunk(assistant.id, "Emo")
    assert store.get_message(assistant.id).content == ""
    store.append_stream_chunk(
        assistant.id,
        "te: smug\nHello\nEmote: sad\nDone",
    )

    assert store.get_message(assistant.id).content == "Hello\nDone"
    events = store.character_emote_events_after(session.id, 0)
    assert [(event.sequence, event.message_id, event.state) for event in events] == [
        (1, assistant.id, "smug"),
        (2, assistant.id, "sad"),
    ]

    completed = store.mark_message_complete(assistant.id)
    emote = completed.metadata.character_emote
    assert completed.content == "Hello\nDone"
    assert emote is not None
    assert emote.mood_label == "sad"
    assert [(event.state, event.at_char) for event in emote.emote_events] == [
        ("smug", 0),
        ("sad", 6),
    ]
    assert emote.actor_id == 7
    assert emote.pack_version_id == 13
    assert emote.expression_key == "sad"
    assert emote.asset_id == 23


def test_unarmed_generic_stream_remains_byte_for_byte_unchanged() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )

    store.append_stream_chunk(assistant.id, "Emote: sad\nVisible")
    completed = store.mark_message_complete(assistant.id)

    assert completed.content == "Emote: sad\nVisible"
    assert completed.metadata is None


def test_success_without_explicit_event_uses_pinned_heuristic() -> None:
    store = ConsoleChatStore()
    _session, assistant = _armed_message(store)

    store.append_stream_chunk(assistant.id, "I am happy and glad!")
    completed = store.mark_message_complete(assistant.id)

    emote = completed.metadata.character_emote
    assert emote is not None
    assert emote.emote_events == ()
    assert emote.mood_label == "happy"
    assert emote.mood_confidence is not None
    assert emote.expression_key == "happy"
    assert emote.asset_id == 17


@pytest.mark.parametrize(
    ("terminal", "reason"), [("stopped", "stopped"), ("failed", "failed")]
)
def test_unsuccessful_terminal_discards_candidate_and_never_uses_heuristic(
    terminal: str,
    reason: str,
) -> None:
    store = ConsoleChatStore()
    _session, assistant = _armed_message(store)
    store.append_stream_chunk(assistant.id, "Visible\nEmote: happy")

    completed = getattr(store, f"mark_message_{terminal}")(assistant.id)

    assert completed.content == "Visible\n"
    emote = completed.metadata.character_emote
    assert emote is not None
    assert emote.mood_label is None
    assert emote.emote_events == ()
    assert emote.fallback_reason == reason


def test_citation_body_replacement_is_sanitized_and_replaces_durable_events() -> None:
    store = ConsoleChatStore()
    session = store.create_session(
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        defer_terminal_persistence=True,
    )
    store.begin_character_emote_capture(assistant.id, _snapshot())
    store.append_stream_chunk(assistant.id, "Emote: happy\nProvisional")

    replaced = store.replace_deferred_terminal_body(
        assistant.id,
        "Emote: ../../bad\nSelected\nEmote: smug",
    )
    completed = store.mark_message_complete(assistant.id)

    assert replaced.content == "Selected\n"
    assert completed.content == "Selected\n"
    emote = completed.metadata.character_emote
    assert emote is not None
    assert [(event.state, event.at_char) for event in emote.emote_events] == [
        ("smug", 9)
    ]
    assert emote.mood_label == "smug"
    assert [
        event.state for event in store.character_emote_events_after(session.id, 0)
    ] == [
        "happy",
        "smug",
    ]


def test_parser_fault_reprocesses_whole_chunk_with_eventless_fail_closed_sanitizer() -> (
    None
):
    store = ConsoleChatStore()
    _session, assistant = _armed_message(store)

    class FaultingParser:
        def safe_copy(self):
            from tldw_chatbook.Character_Chat.emote_directives import (
                CharacterEmoteStreamParser,
            )

            return CharacterEmoteStreamParser()

        def push(self, _chunk):
            raise RuntimeError("injected parser fault")

    store._character_emote_captures[assistant.id].parser = FaultingParser()

    store.append_stream_chunk(assistant.id, "Safe\nEmote: happy\nAfter")
    store.append_stream_chunk(assistant.id, "\nEmote: sad\nTail")
    completed = store.mark_message_complete(assistant.id)

    assert completed.content == "Safe\nAfter\nTail"
    emote = completed.metadata.character_emote
    assert emote is not None
    assert emote.emote_events == ()
    assert emote.fallback_reason == "parser_error"


def test_failed_variant_commit_restores_terminal_emote_capture_and_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConsoleChatStore()
    session, assistant = _armed_message(store)
    store.begin_variant_stream(assistant.id)
    store.append_stream_chunk(assistant.id, "Replacement\nEmote: smug")
    feed_before = store.character_emote_events_after(session.id, 0)
    sequence_before = store._character_emote_sequence
    original_persist = store._persist_generation_variant

    def fail_commit(*_args, **_kwargs):
        raise RuntimeError("injected generation failure")

    monkeypatch.setattr(store, "_persist_generation_variant", fail_commit)
    with pytest.raises(RuntimeError, match="injected generation failure"):
        store.finalize_variant_stream(assistant.id)

    assert assistant.id in store._character_emote_captures
    assert store.character_emote_events_after(session.id, 0) == feed_before
    assert store._character_emote_sequence == sequence_before

    monkeypatch.setattr(store, "_persist_generation_variant", original_persist)
    completed = store.finalize_variant_stream(assistant.id)
    assert completed.content == "Replacement\n"
    assert completed.metadata is not None
    assert completed.metadata.character_emote is not None
    assert completed.metadata.character_emote.mood_label == "smug"


def test_real_sqlite_persists_only_sanitized_text_and_bounded_metadata() -> None:
    db = CharactersRAGDB(":memory:", "emote-test")
    try:
        character_id = int(db.add_character_card({"name": "Emote test"}))
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(
            assistant_kind="character",
            assistant_id=str(character_id),
            character_id=character_id,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Question",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
            defer_terminal_persistence=True,
        )
        store.begin_character_emote_capture(
            assistant.id,
            replace(_snapshot(), actor_id=character_id),
        )

        store.append_stream_chunk(
            assistant.id,
            "Emote: smug\nSafe answer\nEmote: ../../bad",
        )
        completed = store.mark_message_complete(assistant.id)
        row = db.get_message_by_id(completed.persisted_message_id)

        assert row["content"] == "Safe answer\n"
        assert "Emote:" not in row["content"]
        assert row["metadata_json"] is not None
        assert "Safe answer" not in row["metadata_json"]
        assert "smug" in row["metadata_json"]
        assert db.search_conversations_by_content("Safe answer")
        assert db.search_conversations_by_content("smug") == []
        export_source = db.get_messages_for_conversation(row["conversation_id"])
        assert [message["content"] for message in export_source] == [
            "Question",
            "Safe answer\n",
        ]
        assert all("Emote:" not in message["content"] for message in export_source)
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "terminal_name",
    ["mark_message_complete", "mark_message_stopped", "mark_message_failed"],
)
def test_real_sqlite_terminal_update_failure_restores_emote_variant_and_fences(
    terminal_name: str, monkeypatch
) -> None:
    db = CharactersRAGDB(":memory:", f"terminal-rollback-{terminal_name}")
    try:
        character_id = int(db.add_character_card({"name": "Rollback actor"}))
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session(
            assistant_kind="character",
            assistant_id=str(character_id),
            character_id=character_id,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Question",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Original",
            persist=True,
        )
        store.begin_variant_stream(assistant.id)
        token = store.begin_generation_attempt(assistant.id)
        store.begin_character_emote_capture(
            assistant.id, replace(_snapshot(), actor_id=character_id)
        )
        store.append_stream_chunk(assistant.id, "Replacement\nEmote: smug")

        before = store.get_message(assistant.id)
        feed_before = store.character_emote_events_after(session.id, 0)
        payload_before = store.payload_revision(session.id)
        speech_before = store._message_speech_revisions.get(assistant.id)
        epoch_before = store.conversation_context_epoch(session.id)
        base_before = store._variant_stream_bases[assistant.id]
        capture_before = store._character_emote_captures[assistant.id]
        persisted_before = db.get_message_by_id(assistant.persisted_message_id)
        original_writer = persistence.replace_assistant_generation_projection
        monkeypatch.setattr(
            persistence,
            "replace_assistant_generation_projection",
            lambda **_kwargs: 0,
        )

        with pytest.raises(
            RuntimeError, match="Selected generation persistence did not commit"
        ):
            getattr(store, terminal_name)(assistant.id)

        assert store.get_message(assistant.id) == before
        assert store.character_emote_events_after(session.id, 0) == feed_before
        assert store.payload_revision(session.id) == payload_before
        assert store._message_speech_revisions.get(assistant.id) == speech_before
        assert store.conversation_context_epoch(session.id) == epoch_before
        assert store._variant_stream_bases[assistant.id] == base_before
        assert assistant.id in store._character_emote_captures
        assert store._character_emote_captures[assistant.id] is not capture_before
        assert store._generation_attempt_tokens[assistant.id] == token
        assert db.get_message_by_id(assistant.persisted_message_id) == persisted_before

        monkeypatch.setattr(
            persistence, "replace_assistant_generation_projection", original_writer
        )
        getattr(store, terminal_name)(assistant.id)
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "terminal_name",
    ["mark_message_complete", "mark_message_stopped", "mark_message_failed"],
)
def test_real_sqlite_unpersisted_terminal_falsy_create_restores_exact_state(
    terminal_name: str, monkeypatch
) -> None:
    db = CharactersRAGDB(":memory:", f"terminal-create-rollback-{terminal_name}")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session()
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
            defer_terminal_persistence=True,
        )
        token = store.begin_generation_attempt(assistant.id)
        store.append_stream_chunk(assistant.id, "Not durable")
        before = store.get_message(assistant.id)
        payload_before = store.payload_revision(session.id)
        speech_before = store._message_speech_revisions.get(assistant.id)
        epoch_before = store.conversation_context_epoch(session.id)
        monkeypatch.setattr(store, "_persist_new_message", lambda **_kwargs: False)

        with pytest.raises(
            RuntimeError,
            match="Terminal (?:generation|message) persistence did not commit",
        ):
            getattr(store, terminal_name)(assistant.id)

        assert store.get_message(assistant.id) == before
        assert store.payload_revision(session.id) == payload_before
        assert store._message_speech_revisions.get(assistant.id) == speech_before
        assert store.conversation_context_epoch(session.id) == epoch_before
        assert store._generation_attempt_tokens[assistant.id] == token
        assert assistant.persisted_message_id is None
        assert (
            db.get_messages_for_conversation(
                session.persisted_conversation_id or "missing"
            )
            == []
        )
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "terminal_name, terminal_state",
    [
        ("mark_message_complete", "complete"),
        ("mark_message_stopped", "stopped"),
        ("mark_message_failed", "failed"),
    ],
)
def test_real_sqlite_post_commit_metadata_failure_reconciles_terminal_owner(
    terminal_name: str, terminal_state: str, monkeypatch
) -> None:
    db = CharactersRAGDB(":memory:", f"terminal-post-commit-{terminal_name}")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session()
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Original",
            persist=True,
        )
        live = store._message_or_raise(assistant.id)
        live.content = "Replacement"
        live.status = "streaming"
        live.assistant_generation_state = "streaming"
        live.metadata = MessageMetadata(engine="post-commit")
        monkeypatch.setattr(
            persistence,
            "update_message_metadata",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("sidecar failed")),
        )

        with pytest.raises(RuntimeError, match="sidecar failed"):
            getattr(store, terminal_name)(assistant.id)

        row = db.get_message_by_id(assistant.persisted_message_id)
        current = store.get_message(assistant.id)
        assert row["content"] == current.content == "Replacement"
        assert row["assistant_generation_state"] == terminal_state
        assert current.assistant_generation_state == terminal_state
        assert current.status == terminal_state
        assert current.provider_continuation_message_version == row["version"] == 2
    finally:
        db.close_connection()


def test_real_sqlite_post_commit_citation_failure_keeps_committed_body(
    monkeypatch,
) -> None:
    db = CharactersRAGDB(":memory:", "terminal-citation-post-commit")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session()
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Original",
            persist=True,
        )
        live = store._message_or_raise(assistant.id)
        live.content = "Replacement"
        live.status = "streaming"
        live.assistant_generation_state = "streaming"
        store._terminal_citation_finalizers[assistant.id] = lambda _body: object()
        store._terminal_persistence_deferred_ids.add(assistant.id)
        monkeypatch.setattr(
            store,
            "_persist_new_message",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("citation failed")),
        )

        with pytest.raises(RuntimeError, match="citation failed"):
            store.mark_message_complete(assistant.id)

        row = db.get_message_by_id(assistant.persisted_message_id)
        current = store.get_message(assistant.id)
        assert row["content"] == current.content == "Replacement"
        assert row["assistant_generation_state"] == "complete"
        assert current.status == current.assistant_generation_state == "complete"
        assert current.provider_continuation_message_version == row["version"] == 2
    finally:
        db.close_connection()


def test_failed_terminal_restore_does_not_clobber_other_message_success(
    tmp_path: Path, monkeypatch
) -> None:
    db = CharactersRAGDB(
        tmp_path / "terminal-concurrency.sqlite", "terminal-concurrency"
    )
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session()
        first = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="first", persist=True
        )
        second = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="second",
            persist=True,
        )
        for message in (first, second):
            live = store._message_or_raise(message.id)
            live.status = "streaming"
            live.assistant_generation_state = "streaming"
        entered = Event()
        release = Event()
        original = persistence.replace_assistant_generation_projection

        def controlled_writer(**kwargs):
            if kwargs["message_id"] == first.persisted_message_id:
                entered.set()
                assert release.wait(5)
                return 0
            return original(**kwargs)

        monkeypatch.setattr(
            persistence, "replace_assistant_generation_projection", controlled_writer
        )
        failure: list[BaseException] = []
        before_payload = store.payload_revision(session.id)

        def fail_first() -> None:
            try:
                store.mark_message_complete(first.id)
            except BaseException as exc:
                failure.append(exc)

        thread = Thread(target=fail_first)
        thread.start()
        assert entered.wait(5)
        completed = store.mark_message_complete(second.id)
        second_completion_generation = store.message_completion_generation(second.id)
        second_epoch = store._message_completion_epoch
        release.set()
        thread.join(5)

        assert failure and "did not commit" in str(failure[0])
        assert store.get_message(second.id) == completed
        assert store.payload_revision(session.id) == before_payload + 1
        assert (
            store.message_completion_generation(second.id)
            == second_completion_generation
        )
        assert store._message_completion_epoch >= second_epoch
        row = db.get_message_by_id(second.persisted_message_id)
        assert row["assistant_generation_state"] == "complete"
        assert row["version"] == 2
    finally:
        db.close_connection()


def test_failed_terminal_restore_preserves_one_concurrent_failed_retry_context_bump(
    tmp_path: Path, monkeypatch
) -> None:
    """Rollback removes A's counters, not B's single provider-context change."""

    db = CharactersRAGDB(
        tmp_path / "terminal-context-concurrency.sqlite", "terminal-context"
    )
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session()
        first = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="first", persist=True
        )
        second = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="failed once",
            persist=True,
        )
        for message in (first, second):
            live = store._message_or_raise(message.id)
            live.status = "streaming"
            live.assistant_generation_state = "streaming"
        store.mark_message_failed(second.id)
        store.prepare_message_retry(second.id)
        store.append_stream_chunk(second.id, "retry succeeded")
        before_context = store.conversation_context_epoch(session.id)
        before_payload = store.payload_revision(session.id)
        entered = Event()
        release = Event()
        original = persistence.replace_assistant_generation_projection

        def controlled_writer(**kwargs):
            if kwargs["message_id"] == first.persisted_message_id:
                entered.set()
                assert release.wait(5)
                return 0
            return original(**kwargs)

        monkeypatch.setattr(
            persistence, "replace_assistant_generation_projection", controlled_writer
        )
        failure: list[BaseException] = []

        def fail_first() -> None:
            try:
                store.mark_message_complete(first.id)
            except BaseException as exc:
                failure.append(exc)

        thread = Thread(target=fail_first)
        thread.start()
        assert entered.wait(5)
        completed = store.mark_message_complete(second.id)
        completion_after_second = store.message_completion_generation(second.id)
        release.set()
        thread.join(5)

        assert failure and "did not commit" in str(failure[0])
        assert store.conversation_context_epoch(session.id) == before_context + 1
        assert store.payload_revision(session.id) == before_payload + 1
        assert store.message_completion_generation(second.id) == completion_after_second
        assert store.get_message(second.id) == completed
        row = db.get_message_by_id(second.persisted_message_id)
        assert row["content"] == "retry succeeded"
        assert row["assistant_generation_state"] == "complete"
    finally:
        db.close_connection()
