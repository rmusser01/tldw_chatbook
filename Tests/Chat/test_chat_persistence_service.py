import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def client_id():
    return "test_chat_persistence_client"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_chat_persistence.sqlite"


@pytest.fixture
def db_instance(db_path, client_id):
    db = CharactersRAGDB(db_path, client_id)
    yield db
    db.close_connection()


@pytest.mark.integration
class TestChatPersistenceService:
    def test_persistence_service_never_uses_display_name_as_assistant_id(self, db_instance: CharactersRAGDB):
        service = ChatPersistenceService(db_instance)
        character_id = db_instance.add_character_card({"name": "Alice"})
        conversation_id = service.create_conversation(
            character_id=character_id,
            character_name="Alice",
            assistant_kind="character",
            assistant_id="char.local.alice",
            runtime_backend="local",
            discovery_owner="ccp_character",
            discovery_entity_id="char.local.alice",
        )
        conversation = db_instance.get_conversation_by_id(conversation_id)
        assert conversation["assistant_id"] == "char.local.alice"

    def test_update_message_content_preserves_topology_variant_and_feedback(self, db_instance: CharactersRAGDB):
        service = ChatPersistenceService(db_instance)
        char_id = db_instance.add_character_card({"name": "Preserver"})

        conversation_id = service.create_conversation(character_id=char_id, conversation_title="Chat with Preserver")

        root_message_id = db_instance.add_message({
            "id": "msg-root",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Original content",
            "client_id": db_instance.client_id,
        })
        variant_message_id = db_instance.create_message_variant(
            original_message_id=root_message_id,
            variant_content="Variant content",
            is_selected=True,
        )
        db_instance.update_message_feedback(variant_message_id, "1;liked", expected_version=1)

        variant_before = next(
            message
            for message in db_instance.get_messages_for_conversation(conversation_id)
            if message["id"] == variant_message_id
        )

        service.update_message_content(
            message_id=variant_message_id,
            content="Updated variant content",
            image_data=None,
            image_mime_type=None,
        )

        variant_after = next(
            message
            for message in db_instance.get_messages_for_conversation(conversation_id)
            if message["id"] == variant_message_id
        )

        assert variant_after["id"] == variant_before["id"]
        assert variant_after["content"] == "Updated variant content"
        assert variant_after["parent_message_id"] == variant_before["parent_message_id"]
        assert variant_after["variant_of"] == variant_before["variant_of"]
        assert variant_after["variant_number"] == variant_before["variant_number"]
        assert variant_after["is_selected_variant"] == variant_before["is_selected_variant"]
        assert variant_after["total_variants"] == variant_before["total_variants"]
        assert variant_after["feedback"] == variant_before["feedback"]

    def test_save_history_soft_deletes_messages_removed_from_resave(self, db_instance: CharactersRAGDB):
        service = ChatPersistenceService(db_instance)
        conversation_id = service.create_conversation(assistant_kind="persona", assistant_id="planner")

        service.save_history(
            conversation_id=conversation_id,
            chatbot_history=[
                {"id": "msg-user-1", "role": "user", "content": "First"},
                {
                    "id": "msg-assistant-1",
                    "role": "assistant",
                    "content": "Reply",
                    "parent_message_id": "msg-user-1",
                },
                {
                    "id": "msg-user-2",
                    "role": "user",
                    "content": "Trailing message",
                    "parent_message_id": "msg-assistant-1",
                },
            ],
        )

        service.save_history(
            conversation_id=conversation_id,
            chatbot_history=[
                {"id": "msg-user-1", "role": "user", "content": "First updated"},
                {
                    "id": "msg-assistant-1",
                    "role": "assistant",
                    "content": "Reply updated",
                    "parent_message_id": "msg-user-1",
                },
            ],
        )

        active_messages = db_instance.get_messages_for_conversation(conversation_id)
        assert [message["id"] for message in active_messages] == ["msg-user-1", "msg-assistant-1"]

        deleted_row = db_instance.execute_query(
            "SELECT deleted FROM messages WHERE id = ?",
            ("msg-user-2",),
        ).fetchone()
        assert deleted_row["deleted"] == 1

    def test_save_history_without_ids_skips_variant_rows_in_positional_fallback(self, db_instance: CharactersRAGDB):
        service = ChatPersistenceService(db_instance)
        conversation_id = service.create_conversation(assistant_kind="persona", assistant_id="planner")

        root_message_id = db_instance.add_message({
            "id": "msg-user-1",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "First",
            "client_id": db_instance.client_id,
        })
        assistant_message_id = db_instance.add_message({
            "id": "msg-assistant-1",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Reply",
            "parent_message_id": root_message_id,
            "client_id": db_instance.client_id,
        })
        variant_message_id = db_instance.create_message_variant(
            original_message_id=assistant_message_id,
            variant_content="Variant reply",
            is_selected=True,
        )
        later_user_message_id = db_instance.add_message({
            "id": "msg-user-2",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Later turn",
            "parent_message_id": assistant_message_id,
            "client_id": db_instance.client_id,
        })

        service.save_history(
            conversation_id=conversation_id,
            chatbot_history=[
                {"role": "user", "content": "First updated"},
                {"role": "assistant", "content": "Reply updated"},
                {"role": "user", "content": "Later updated"},
            ],
        )

        messages = {
            message["id"]: message
            for message in db_instance.get_messages_for_conversation(conversation_id)
        }
        assert set(messages) == {
            root_message_id,
            assistant_message_id,
            variant_message_id,
            later_user_message_id,
        }
        assert messages[root_message_id]["content"] == "First updated"
        assert messages[assistant_message_id]["content"] == "Reply updated"
        assert messages[variant_message_id]["content"] == "Variant reply"
        assert messages[later_user_message_id]["content"] == "Later updated"

    def test_save_history_tolerates_malformed_image_data_uris(self, db_instance: CharactersRAGDB):
        service = ChatPersistenceService(db_instance)
        conversation_id = service.create_conversation(assistant_kind="persona", assistant_id="planner")

        service.save_history(
            conversation_id=conversation_id,
            chatbot_history=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Broken image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,not-valid-base64"},
                        },
                    ],
                },
            ],
        )

        messages = db_instance.get_messages_for_conversation(conversation_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "Broken image\n<Error: Failed to decode image data from history>"
        assert messages[0]["image_data"] is None
        assert messages[0]["image_mime_type"] is None

    @pytest.mark.parametrize(
        ("character_id", "assistant_kind", "assistant_id", "expected_title"),
        [
            (None, "persona", "Archivist", "Chat with Archivist"),
            (None, None, None, "New Chat"),
        ],
    )
    def test_create_conversation_supports_non_default_creation_models(
        self,
        db_instance: CharactersRAGDB,
        character_id,
        assistant_kind,
        assistant_id,
        expected_title,
    ):
        service = ChatPersistenceService(db_instance)

        conversation_id = service.create_conversation(
            character_id=character_id,
            assistant_kind=assistant_kind,
            assistant_id=assistant_id,
        )

        conversation = db_instance.get_conversation_by_id(conversation_id)
        assert conversation is not None
        assert conversation["title"] == expected_title
        assert conversation["character_id"] == character_id
        assert conversation["assistant_kind"] == assistant_kind
        assert conversation["assistant_id"] == assistant_id
