import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleWorkspaceContext
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


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

    def create_conversation(self, **kwargs):
        self.created_conversations.append(kwargs)
        return "conv-1"

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
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
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
