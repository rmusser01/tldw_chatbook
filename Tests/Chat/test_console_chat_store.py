import pytest
from datetime import datetime

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID, LocalWorkspaceRegistryService


def test_store_creates_session_and_appends_messages():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1", workspace_id="global")

    user_message = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    assistant_message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")

    assert store.active_session_id == session.id
    assert user_message.content == "hello"
    assert assistant_message.status == "pending"
    assert [message.role for message in store.messages_for_session(session.id)] == [
        ConsoleMessageRole.USER,
        ConsoleMessageRole.ASSISTANT,
    ]


def test_store_records_message_feedback():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="answer")

    updated = store.set_message_feedback(message.id, "up")

    assert updated.feedback == "up"
    assert store.get_message(message.id).feedback == "up"


def test_store_deletes_message_from_transcript():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="answer")

    deleted = store.delete_message(message.id)

    assert deleted.id == message.id
    assert store.messages_for_session(session.id) == []
    with pytest.raises(KeyError):
        store.get_message(message.id)


def test_delete_message_clears_orphaned_variant_stream_base():
    """A message deleted mid-regenerate (stopped, then deleted) must not
    leave a stale entry in _variant_stream_bases -- Plan-B Task 1 Minor
    finding (console_chat_store.py delete_message previously popped the
    stream-chunk maps but not this one)."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="answer")
    store.begin_variant_stream(message.id)
    store.mark_message_stopped(message.id)

    assert message.id in store._variant_stream_bases

    store.delete_message(message.id)

    assert message.id not in store._variant_stream_bases


def test_store_updates_message_content():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="answer")

    updated = store.update_message_content(message.id, "edited answer")

    assert updated.content == "edited answer"
    assert store.get_message(message.id).content == "edited answer"


def test_store_updates_current_variant_content():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="first")
    store.add_variant(message.id, "second")

    updated = store.update_message_content(message.id, "edited second")

    assert updated.content == "edited second"
    assert updated.variants is not None
    assert updated.variants.selected_index == 1
    assert updated.variants.current.content == "edited second"
    assert updated.variants.variants[0].content == "first"


def test_store_updates_streaming_message_and_marks_stopped():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")

    store.append_stream_chunk(message.id, "hel")
    store.append_stream_chunk(message.id, "lo")
    store.mark_message_stopped(message.id)

    updated = store.get_message(message.id)
    assert updated.content == "hello"
    assert updated.status == "stopped"


def test_store_buffers_stream_chunks_until_messages_are_materialized():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")

    chunk_result = store.append_stream_chunk(message.id, "hel")

    assert chunk_result.content == ""
    materialized = store.messages_for_session(session.id)[0]
    assert materialized.content == "hel"
    assert materialized.status == "streaming"


def test_store_tracks_active_workspace_context():
    context = ConsoleWorkspaceContext(active_workspace_id="workspace-a")
    store = ConsoleChatStore(workspace_context=context)

    assert store.workspace_context.active_workspace_id == "workspace-a"

    store.set_workspace_context(ConsoleWorkspaceContext(active_workspace_id="workspace-b"))

    assert store.workspace_context.active_workspace_id == "workspace-b"


def test_store_creates_and_switches_sessions():
    store = ConsoleChatStore()
    first = store.ensure_session(title="Chat 1")
    store.append_message(first.id, role=ConsoleMessageRole.USER, content="first")
    second = store.create_session(title="Chat 2")

    assert store.active_session_id == second.id

    store.switch_session(first.id)

    assert store.active_session_id == first.id
    assert store.messages_for_session(first.id)[0].content == "first"


def test_store_restore_state_replaces_sessions_and_rebuilds_message_indexes():
    store = ConsoleChatStore()
    stale_session = store.ensure_session(title="Stale")
    stale_message = store.append_message(
        stale_session.id,
        role=ConsoleMessageRole.USER,
        content="stale",
    )
    restored_session = ConsoleChatSession(id="session-a", title="Restored")
    restored_message = ConsoleChatMessage(
        id="message-a",
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
    )

    store.restore_state(
        sessions=[restored_session],
        messages_by_session={"session-a": [restored_message]},
        active_session_id="session-a",
    )

    assert [session.id for session in store.sessions()] == ["session-a"]
    assert store.active_session_id == "session-a"
    assert store.messages_for_session("session-a")[0].content == "answer"
    assert store.session_id_for_message("message-a") == "session-a"
    with pytest.raises(KeyError):
        store.get_message(stale_message.id)


def test_store_renames_session_with_trimmed_title():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")

    renamed = store.rename_session(session.id, "  Planning tab  ")

    assert renamed is session
    assert store.sessions()[0].title == "Planning tab"


def test_store_rejects_blank_session_title_without_mutating_existing_title():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")

    with pytest.raises(ValueError):
        store.rename_session(session.id, "   ")

    assert store.sessions()[0].title == "Chat 1"


def test_console_sessions_store_independent_settings_snapshots() -> None:
    store = ConsoleChatStore()
    first_settings = ConsoleSessionSettings(provider="llama_cpp", model="a", temperature=0.1)
    second_settings = ConsoleSessionSettings(provider="openai", model="b", temperature=0.9)

    first = store.create_session(title="A", settings=first_settings)
    second = store.create_session(title="B", settings=second_settings)

    assert store.session_settings(first.id).model == "a"
    assert store.session_settings(second.id).model == "b"


def test_replacing_session_settings_does_not_mutate_other_sessions() -> None:
    store = ConsoleChatStore()
    first = store.create_session(settings=ConsoleSessionSettings(provider="llama_cpp", model="a"))
    second = store.create_session(settings=ConsoleSessionSettings(provider="llama_cpp", model="b"))

    store.replace_session_settings(
        first.id,
        ConsoleSessionSettings(provider="llama_cpp", model="changed"),
    )

    assert store.session_settings(first.id).model == "changed"
    assert store.session_settings(second.id).model == "b"


def test_replace_session_settings_returns_stored_session_instance() -> None:
    store = ConsoleChatStore()
    session = store.create_session(settings=ConsoleSessionSettings(provider="llama_cpp", model="a"))

    returned = store.replace_session_settings(
        session.id,
        ConsoleSessionSettings(provider="llama_cpp", model="changed"),
    )

    assert returned is store.switch_session(session.id)
    assert returned.settings.model == "changed"


def test_ensure_session_applies_settings_only_when_creating_session() -> None:
    store = ConsoleChatStore()
    settings = ConsoleSessionSettings(provider="llama_cpp", model="new")

    session = store.ensure_session(settings=settings)

    assert store.session_settings(session.id) == settings


def test_ensure_session_settings_do_not_mutate_existing_active_session() -> None:
    store = ConsoleChatStore()
    original_settings = ConsoleSessionSettings(provider="llama_cpp", model="original")
    session = store.ensure_session(settings=original_settings)

    ensured = store.ensure_session(
        settings=ConsoleSessionSettings(provider="openai", model="ignored"),
    )

    assert ensured.id == session.id
    assert store.session_settings(session.id) == original_settings


def test_session_settings_returns_none_when_session_has_no_settings() -> None:
    store = ConsoleChatStore()
    session = store.create_session()

    assert store.session_settings(session.id) is None


def test_store_closes_session_and_activates_neighbor():
    store = ConsoleChatStore()
    first = store.ensure_session(title="Chat 1")
    second = store.create_session(title="Chat 2")
    store.append_message(second.id, role=ConsoleMessageRole.USER, content="second")

    activated = store.close_session(second.id)

    assert activated == first
    assert store.active_session_id == first.id
    assert [session.id for session in store.sessions()] == [first.id]
    with pytest.raises(KeyError):
        store.messages_for_session(second.id)


def test_store_closes_last_session_returns_none():
    store = ConsoleChatStore()
    only = store.ensure_session(title="Solo")
    store.append_message(only.id, role=ConsoleMessageRole.USER, content="msg")

    activated = store.close_session(only.id)

    assert activated is None
    assert store.active_session_id is None
    assert store.sessions() == []


def test_store_adds_regenerated_variant_and_selects_it():
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
    )

    store.add_variant(message.id, "second")

    updated = store.get_message(message.id)
    assert updated.variants.current.content == "second"
    assert updated.variants.can_go_previous is True


class FakePersistence:
    def __init__(self):
        self.created_conversations = []
        self.created_messages = []
        self.updated_messages = []
        self.updated_system_prompts = []

    def create_conversation(self, **kwargs):
        self.created_conversations.append(kwargs)
        return "conv-1"

    def update_conversation_system_prompt(self, *, conversation_id, system_prompt):
        self.updated_system_prompts.append(
            {"conversation_id": conversation_id, "system_prompt": system_prompt}
        )
        return True

    def create_message(
        self,
        *,
        conversation_id,
        sender,
        content,
        image_data,
        image_mime_type,
        message_id=None,
        parent_message_id=None,
        feedback=None,
    ):
        kwargs = {
            "conversation_id": conversation_id,
            "sender": sender,
            "content": content,
            "image_data": image_data,
            "image_mime_type": image_mime_type,
            "message_id": message_id,
            "parent_message_id": parent_message_id,
            "feedback": feedback,
        }
        self.created_messages.append(kwargs)
        return f"msg-{len(self.created_messages)}"

    def update_message_content(
        self,
        *,
        message_id,
        content,
        image_data,
        image_mime_type,
        parent_message_id=None,
        feedback=None,
        update_parent=False,
        update_feedback=False,
    ):
        self.updated_messages.append(
            {
                "message_id": message_id,
                "content": content,
                "image_data": image_data,
                "image_mime_type": image_mime_type,
                "parent_message_id": parent_message_id,
                "feedback": feedback,
                "update_parent": update_parent,
                "update_feedback": update_feedback,
            }
        )
        return True


class FakeChatSyncProducer:
    def __init__(self):
        self.enqueued = []

    def enqueue_chat_message(self, **kwargs):
        self.enqueued.append(kwargs)
        return {
            "status": "enqueued",
            "outbox_entry": {
                "outbox_id": len(self.enqueued),
                "envelope": {
                    "payload_hash": f"hash:{kwargs['role']}:{kwargs['content']}",
                },
            },
        }


class FailingChatSyncProducer:
    def enqueue_chat_message(self, **kwargs):
        raise RuntimeError("sync unavailable")


def test_store_can_persist_user_and_assistant_messages_through_adapter():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")

    store.persist_session_if_needed(session.id)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello", persist=True)

    assert persistence.created_conversations[0]["conversation_title"] == "Chat 1"
    assert persistence.created_messages[0]["conversation_id"] == "conv-1"
    assert persistence.created_messages[0]["sender"] == "user"
    assert persistence.created_messages[0]["content"] == "hello"
    assert persistence.created_messages[0]["image_data"] is None
    assert persistence.created_messages[0]["image_mime_type"] is None


def test_persist_session_if_needed_passes_system_prompt_from_settings():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp", system_prompt="Be terse."),
    )

    store.persist_session_if_needed(session.id)

    assert persistence.created_conversations[0]["system_prompt"] == "Be terse."


def test_persist_session_if_needed_passes_none_system_prompt_without_settings():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")

    store.persist_session_if_needed(session.id)

    assert persistence.created_conversations[0]["system_prompt"] is None


def test_set_session_system_prompt_updates_settings_without_persisting_when_unsaved():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp"),
    )

    updated, persisted = store.set_session_system_prompt(session.id, "New system prompt")

    assert updated.settings.system_prompt == "New system prompt"
    assert persisted is True
    assert persistence.updated_system_prompts == []
    assert persistence.created_conversations == []


def test_set_session_system_prompt_persists_change_when_conversation_already_saved():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp"),
    )
    store.persist_session_if_needed(session.id)

    updated, persisted = store.set_session_system_prompt(session.id, "Answer in French.")

    assert updated.settings.system_prompt == "Answer in French."
    assert persisted is True
    assert persistence.updated_system_prompts == [
        {"conversation_id": "conv-1", "system_prompt": "Answer in French."}
    ]


def test_set_session_system_prompt_preserves_formatting_verbatim():
    """Only blank/whitespace-only text is treated as "no system prompt";
    leading whitespace and internal formatting (e.g. a blank line between
    paragraphs) must survive into storage unchanged rather than being
    stripped."""
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp"),
    )
    store.persist_session_if_needed(session.id)
    formatted_prompt = "  line1\n\n  line2  "

    updated, persisted = store.set_session_system_prompt(session.id, formatted_prompt)

    assert updated.settings.system_prompt == formatted_prompt
    assert persisted is True
    assert persistence.updated_system_prompts == [
        {"conversation_id": "conv-1", "system_prompt": formatted_prompt}
    ]


def test_set_session_system_prompt_normalizes_blank_to_none():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp", system_prompt="Old prompt"),
    )
    store.persist_session_if_needed(session.id)

    updated, persisted = store.set_session_system_prompt(session.id, "   ")

    assert updated.settings.system_prompt is None
    assert persisted is True
    assert persistence.updated_system_prompts == [
        {"conversation_id": "conv-1", "system_prompt": None}
    ]


def test_set_session_system_prompt_survives_persistence_failure():
    """A persistence error (e.g. the conversation was deleted, or a DB
    conflict) must not escape `set_session_system_prompt`, and the
    in-memory session keeps the applied value (this store's existing
    convention: mutations are not rolled back when the durable write that
    follows them fails); the caller gets `persisted=False` back so it can
    surface the failure honestly instead of assuming the change was saved.
    """

    class RaisingPersistence(FakePersistence):
        def update_conversation_system_prompt(self, *, conversation_id, system_prompt):
            raise RuntimeError("conversation vanished")

    persistence = RaisingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(
        title="Chat 1",
        settings=ConsoleSessionSettings(provider="llama_cpp"),
    )
    store.persist_session_if_needed(session.id)

    updated, persisted = store.set_session_system_prompt(session.id, "New prompt")

    assert persisted is False
    assert updated.settings.system_prompt == "New prompt"
    assert store.session_settings(session.id).system_prompt == "New prompt"


def test_store_enqueues_chat_sync_after_user_message_is_durable():
    persistence = FakePersistence()
    sync_producer = FakeChatSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync_producer,
        sync_v2_server_profile_id="server-a",
        sync_v2_authenticated_principal_id="user-a",
    )
    session = store.ensure_session(title="Chat 1")

    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
        persist=True,
    )

    assert message.persisted_message_id == "msg-1"
    assert sync_producer.enqueued == [
        {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": None,
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "role": "user",
            "content": "hello",
            "parent_message_id": None,
            "sequence": 1,
            "variant_turn_id": None,
            "variant_index": None,
            "variant_count": None,
            "selected_variant_id": None,
            "base_version": None,
            "entity_version": None,
        }
    ]


def test_store_enqueues_streaming_assistant_only_after_completion():
    persistence = FakePersistence()
    sync_producer = FakeChatSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync_producer,
        sync_v2_server_profile_id="server-a",
    )
    session = store.ensure_session(title="Chat 1")
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello?",
        persist=True,
    )
    sync_producer.enqueued.clear()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )

    store.append_stream_chunk(assistant.id, "hel")
    store.append_stream_chunk(assistant.id, "lo")

    assert sync_producer.enqueued == []

    completed = store.mark_message_complete(assistant.id)

    assert completed.persisted_message_id == "msg-2"
    assert sync_producer.enqueued[-1]["message_id"] == "msg-2"
    assert sync_producer.enqueued[-1]["role"] == "assistant"
    assert sync_producer.enqueued[-1]["content"] == "hello"
    assert sync_producer.enqueued[-1]["parent_message_id"] == "msg-1"
    assert sync_producer.enqueued[-1]["sequence"] == 2


def test_store_does_not_enqueue_failed_assistant_final_content():
    persistence = FakePersistence()
    sync_producer = FakeChatSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync_producer,
        sync_v2_server_profile_id="server-a",
    )
    session = store.ensure_session(title="Chat 1")
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )

    store.append_stream_chunk(assistant.id, "partial")
    store.mark_message_failed(assistant.id)

    assert sync_producer.enqueued == []


def test_mark_message_failed_restores_prior_status_when_variant_base_present():
    """Plan-B Task 1 finding: a zero-chunk (empty-stream) regenerate of a
    previously-complete message must restore that prior status, not flip to
    "failed" -- every send path builds provider context with skip_failed=True
    (see console_chat_controller._provider_messages_for_session), so a wrong
    "failed" status here would silently drop an otherwise-good turn from the
    model's context for the rest of the session. Pre-refactor, a failed
    regenerate was a pure no-op on the existing message."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="original"
    )
    assert message.status == "complete"

    store.begin_variant_stream(message.id)
    # Zero-chunk stream: no append_stream_chunk calls before failure.
    failed = store.mark_message_failed(message.id)

    assert failed.status == "complete"
    assert failed.content == "original"
    assert message.id not in store._variant_stream_bases


def test_mark_message_failed_without_variant_base_still_marks_failed():
    """A normal (non-regenerate) send failure keeps today's "failed" status;
    only the variant-regenerate path has a known-good prior state to
    restore."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(assistant.id, "partial")

    failed = store.mark_message_failed(assistant.id)

    assert failed.status == "failed"
    assert failed.content == "partial"


def test_store_persists_chat_when_sync_enqueue_fails():
    persistence = FakePersistence()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=FailingChatSyncProducer(),
        sync_v2_server_profile_id="server-a",
    )
    session = store.ensure_session(title="Chat 1")

    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
        persist=True,
    )

    assert message.persisted_message_id == "msg-1"
    assert persistence.created_messages[0]["content"] == "hello"


def test_store_enqueues_selected_variant_with_restore_metadata():
    persistence = FakePersistence()
    sync_producer = FakeChatSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync_producer,
        sync_v2_server_profile_id="server-a",
    )
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
        persist=True,
    )
    sync_producer.enqueued.clear()

    updated = store.add_variant(message.id, "second")

    assert updated.variants is not None
    assert sync_producer.enqueued[-1]["message_id"] == "msg-1"
    assert sync_producer.enqueued[-1]["content"] == "second"
    assert sync_producer.enqueued[-1]["base_version"] == "hash:assistant:first"
    assert sync_producer.enqueued[-1]["sequence"] == 1
    assert sync_producer.enqueued[-1]["variant_turn_id"] == updated.variants.turn_id
    assert sync_producer.enqueued[-1]["variant_index"] == 1
    assert sync_producer.enqueued[-1]["variant_count"] == 2
    assert sync_producer.enqueued[-1]["selected_variant_id"] == updated.variants.current.id


def test_store_sequences_only_sync_eligible_messages():
    persistence = FakePersistence()
    sync_producer = FakeChatSyncProducer()
    store = ConsoleChatStore(
        persistence=persistence,
        sync_v2_chat_producer=sync_producer,
        sync_v2_server_profile_id="server-a",
    )
    session = store.ensure_session(title="Chat 1")
    store.append_message(
        session.id,
        role=ConsoleMessageRole.SYSTEM,
        content="visible only",
        persist=False,
    )
    failed = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )
    store.append_stream_chunk(failed.id, "partial")
    store.mark_message_failed(failed.id)

    first_synced = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
        persist=True,
    )
    second_synced = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="again",
        persist=True,
    )

    assert first_synced.persisted_message_id == "msg-2"
    assert second_synced.persisted_message_id == "msg-3"
    assert [entry["sequence"] for entry in sync_producer.enqueued] == [1, 2]


def test_store_updates_persisted_streaming_assistant_content_and_status():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session(title="Chat 1")
    store.persist_session_if_needed(session.id)
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=True,
    )

    store.append_stream_chunk(assistant.id, "hel")
    store.append_stream_chunk(assistant.id, "lo")
    store.mark_message_complete(assistant.id)

    completed = store.get_message(assistant.id)
    assert persistence.updated_messages[-1]["message_id"] == completed.persisted_message_id
    assert persistence.updated_messages[-1]["content"] == "hello"
    assert persistence.updated_messages[-1]["image_data"] is None
    assert persistence.updated_messages[-1]["image_mime_type"] is None


def test_store_persists_workspace_session_with_real_chat_persistence_service(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="test_client")
        )
        registry.create_workspace(workspace_id="workspace-a", name="Workspace A")
        store = ConsoleChatStore(
            persistence=ChatPersistenceService(db, workspace_registry=registry)
        )
        session = store.ensure_session(title="Chat 1", workspace_id="workspace-a")

        conversation_id = store.persist_session_if_needed(session.id)
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="hello",
            persist=True,
        )

        conversation = db.get_conversation_by_id(conversation_id)
        persisted_message = db.get_message_by_id(message.persisted_message_id)
        assert conversation["scope_type"] == "workspace"
        assert conversation["workspace_id"] == "workspace-a"
        assert conversation["assistant_kind"] == "generic"
        assert conversation["assistant_id"] == "console"
        assert persisted_message["content"] == "hello"
        workspace_conversations = registry.list_workspace_conversations("workspace-a")
        assert [item.item_id for item in workspace_conversations] == [conversation_id]
    finally:
        db.close()


def test_store_persists_default_workspace_chat_without_runtime_access(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="test_client")
        )
        registry.ensure_default_workspace()
        store = ConsoleChatStore(
            persistence=ChatPersistenceService(db, workspace_registry=registry)
        )
        session = store.ensure_session(title="Chat 1", workspace_id=DEFAULT_WORKSPACE_ID)

        conversation_id = store.persist_session_if_needed(session.id)
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="default workspace chat remains usable",
            persist=True,
        )

        conversation = db.get_conversation_by_id(conversation_id)
        persisted_message = db.get_message_by_id(message.persisted_message_id)
        workspace_conversations = registry.list_workspace_conversations(DEFAULT_WORKSPACE_ID)
        assert conversation is not None
        assert persisted_message is not None
        assert conversation["scope_type"] == "workspace"
        assert conversation["workspace_id"] == DEFAULT_WORKSPACE_ID
        assert persisted_message["content"] == "default workspace chat remains usable"
        assert [item.item_id for item in workspace_conversations] == [conversation_id]
        assert registry.list_runtime_bindings(DEFAULT_WORKSPACE_ID) == ()
    finally:
        db.close()


def test_store_system_prompt_round_trips_through_real_chat_persistence_service(tmp_path):
    """Persistence round-trip: create, apply a system prompt, reload from the real DB.

    Covers the Task 0 persistence seam end to end: creating a conversation
    with a session-level system prompt, then changing it once the
    conversation is already saved (the update path Task 0 flagged as
    missing), then reading the raw DB row back -- independent of any
    in-memory store state -- to confirm the change is truly durable.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.ensure_session(
            title="Chat 1",
            settings=ConsoleSessionSettings(provider="llama_cpp", system_prompt="Be terse."),
        )

        conversation_id = store.persist_session_if_needed(session.id)
        assert db.get_conversation_by_id(conversation_id)["system_prompt"] == "Be terse."

        store.set_session_system_prompt(session.id, "Answer only in French.")

        # Read straight from the DB (not through the in-memory store) to
        # confirm the update is durable, the way a reload/reopen would see it.
        reloaded = db.get_conversation_by_id(conversation_id)
        assert reloaded["system_prompt"] == "Answer only in French."
        assert store.session_settings(session.id).system_prompt == "Answer only in French."
    finally:
        db.close()


def test_store_delays_empty_assistant_persistence_until_terminal_content_with_real_service(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.ensure_session(title="Chat 1")
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )

        assert store.get_message(assistant.id).persisted_message_id is None

        store.append_stream_chunk(assistant.id, "hel")
        store.append_stream_chunk(assistant.id, "lo")
        completed = store.mark_message_complete(assistant.id)

        assert completed.persisted_message_id is not None
        persisted_message = db.get_message_by_id(completed.persisted_message_id)
        assert persisted_message["content"] == "hello"
    finally:
        db.close()


def test_store_rejects_streaming_chunks_for_non_assistant_message():
    store = ConsoleChatStore()
    session = store.ensure_session()
    user_message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
    )

    with pytest.raises(ValueError, match="Only assistant messages"):
        store.append_stream_chunk(user_message.id, "nope")


def test_store_rejects_streaming_chunks_after_terminal_state():
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.mark_message_stopped(assistant.id)

    with pytest.raises(ValueError, match="Cannot append stream chunks"):
        store.append_stream_chunk(assistant.id, "late")


def test_store_returns_message_snapshots_not_mutable_internals():
    store = ConsoleChatStore()
    session = store.ensure_session()
    user_message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
    )

    user_message.content = "external mutation"
    listed = store.messages_for_session(session.id)
    listed[0].status = "failed"

    stored = store.get_message(user_message.id)
    assert stored.content == "hello"
    assert stored.status == "complete"


def test_create_session_records_updated_at():
    store = ConsoleChatStore()
    session = store.create_session()
    parsed = datetime.fromisoformat(session.updated_at)
    assert parsed.tzinfo is not None


def test_append_message_touches_session_updated_at():
    store = ConsoleChatStore()
    session = store.create_session()
    original = session.updated_at
    store._sessions[session.id].updated_at = "2020-01-01T00:00:00+00:00"

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")

    touched = store._sessions[session.id].updated_at
    assert touched != "2020-01-01T00:00:00+00:00"
    assert datetime.fromisoformat(touched) >= datetime.fromisoformat(original)


from tldw_chatbook.Chat.attachment_core import PendingAttachment


def _image_attachment(name="photo.png"):
    return PendingAttachment(
        file_path=f"/tmp/{name}",
        display_name=name,
        file_type="image",
        insert_mode="attachment",
        data=b"\x89PNG-bytes",
        mime_type="image/png",
        original_size=11,
        processed_size=11,
    )


class RecordingPersistence:
    def __init__(self):
        self.created = []
        self.updated = []
        self._counter = 0

    def create_conversation(self, **kwargs):
        return "conv-1"

    def create_message(self, **kwargs):
        self.created.append(kwargs)
        self._counter += 1
        return f"msg-{self._counter}"

    def update_message_content(self, **kwargs):
        self.updated.append(kwargs)
        return True


def test_pending_attachment_is_per_session():
    store = ConsoleChatStore()
    first = store.create_session(title="A")
    second = store.create_session(title="B")

    store.set_pending_attachment(first.id, _image_attachment())

    assert store.pending_attachment(first.id) is not None
    assert store.pending_attachment(second.id) is None

    store.clear_pending_attachment(first.id)
    assert store.pending_attachment(first.id) is None


def test_append_message_persists_image_fields():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()

    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="what is this?",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        attachment_label="photo.png · 11 B",
        persist=True,
    )

    assert message.image_data == b"\x89PNG-bytes"
    assert message.attachment_label == "photo.png · 11 B"
    assert persistence.created[-1]["image_data"] == b"\x89PNG-bytes"
    assert persistence.created[-1]["image_mime_type"] == "image/png"


def test_image_only_user_message_persists_immediately():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()

    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        persist=True,
    )

    assert len(persistence.created) == 1
    assert persistence.created[0]["content"] == ""
    assert persistence.created[0]["image_data"] == b"\x89PNG-bytes"


def test_editing_message_content_does_not_wipe_persisted_image():
    persistence = RecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="original",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        persist=True,
    )

    store.update_message_content(message.id, "edited")

    assert persistence.updated[-1]["image_data"] == b"\x89PNG-bytes"
    assert persistence.updated[-1]["image_mime_type"] == "image/png"
