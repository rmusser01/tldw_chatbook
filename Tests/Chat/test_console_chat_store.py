import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleWorkspaceContext
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
