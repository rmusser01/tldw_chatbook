import inspect

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService


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

    def test_update_message_content_with_none_image_data_preserves_persisted_image(
        self, db_instance: CharactersRAGDB
    ):
        """A metadata-only edit (image bytes unavailable, e.g. failed rehydration)
        must not NULL out an image that is already persisted for the message."""
        service = ChatPersistenceService(db_instance)
        conversation_id = service.create_conversation(assistant_kind="persona", assistant_id="planner")

        message_id = db_instance.add_message({
            "id": "msg-with-image",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "original",
            "image_data": b"\x89PNG-bytes",
            "image_mime_type": "image/png",
            "client_id": db_instance.client_id,
        })

        # Simulate the Console store editing a message whose in-memory image
        # bytes were never rehydrated (e.g. after a failed screen-state
        # restore), so it calls update_message_content with image_data=None.
        service.update_message_content(
            message_id=message_id,
            content="edited",
            image_data=None,
            image_mime_type=None,
        )

        message = db_instance.get_message_by_id(message_id)
        assert message["content"] == "edited"
        assert message["image_data"] == b"\x89PNG-bytes"
        assert message["image_mime_type"] == "image/png"

    def test_update_message_content_with_new_image_data_replaces_persisted_image(
        self, db_instance: CharactersRAGDB
    ):
        """Passing real image bytes must still update the persisted image."""
        service = ChatPersistenceService(db_instance)
        conversation_id = service.create_conversation(assistant_kind="persona", assistant_id="planner")

        message_id = db_instance.add_message({
            "id": "msg-with-image-2",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "original",
            "image_data": b"\x89PNG-old-bytes",
            "image_mime_type": "image/png",
            "client_id": db_instance.client_id,
        })

        service.update_message_content(
            message_id=message_id,
            content="edited",
            image_data=b"\x89PNG-new-bytes",
            image_mime_type="image/png",
        )

        message = db_instance.get_message_by_id(message_id)
        assert message["content"] == "edited"
        assert message["image_data"] == b"\x89PNG-new-bytes"
        assert message["image_mime_type"] == "image/png"

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

    def test_create_conversation_persists_system_prompt(self, db_instance: CharactersRAGDB):
        service = ChatPersistenceService(db_instance)

        conversation_id = service.create_conversation(
            assistant_kind="generic",
            assistant_id="console",
            conversation_title="Console chat",
            system_prompt="Answer as a pirate.",
        )

        conversation = db_instance.get_conversation_by_id(conversation_id)
        assert conversation["system_prompt"] == "Answer as a pirate."

    def test_create_conversation_defaults_system_prompt_to_none(self, db_instance: CharactersRAGDB):
        service = ChatPersistenceService(db_instance)

        conversation_id = service.create_conversation(
            assistant_kind="generic",
            assistant_id="console",
            conversation_title="Console chat without prompt",
        )

        conversation = db_instance.get_conversation_by_id(conversation_id)
        assert conversation["system_prompt"] is None

    def test_update_conversation_system_prompt_round_trips_through_real_db(
        self, db_instance: CharactersRAGDB
    ):
        service = ChatPersistenceService(db_instance)
        conversation_id = service.create_conversation(
            assistant_kind="generic",
            assistant_id="console",
            conversation_title="Console chat",
        )

        result = service.update_conversation_system_prompt(
            conversation_id=conversation_id,
            system_prompt="Be terse and cite sources.",
        )

        assert result is True
        reloaded = db_instance.get_conversation_by_id(conversation_id)
        assert reloaded["system_prompt"] == "Be terse and cite sources."

        # A second, independent read (simulating a fresh load/reopen) sees
        # the same persisted value.
        reloaded_again = db_instance.get_conversation_by_id(conversation_id)
        assert reloaded_again["system_prompt"] == "Be terse and cite sources."

    def test_update_conversation_system_prompt_raises_for_missing_conversation(
        self, db_instance: CharactersRAGDB
    ):
        service = ChatPersistenceService(db_instance)

        with pytest.raises(ValueError, match="not found"):
            service.update_conversation_system_prompt(
                conversation_id="missing-conversation",
                system_prompt="Anything",
            )

    def test_workspace_conversation_requires_existing_workspace(
        self,
        db_instance: CharactersRAGDB,
        tmp_path,
    ):
        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
        )
        service = ChatPersistenceService(db_instance, workspace_registry=registry)

        with pytest.raises(ValueError, match="Unknown workspace"):
            service.create_conversation(
                scope_type="workspace",
                workspace_id="missing",
                conversation_title="Missing workspace chat",
            )

    def test_workspace_conversation_links_membership(
        self,
        db_instance: CharactersRAGDB,
        tmp_path,
    ):
        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
        )
        registry.create_workspace(workspace_id="ws-a", name="Workspace A")
        service = ChatPersistenceService(db_instance, workspace_registry=registry)

        conversation_id = service.create_conversation(
            scope_type="workspace",
            workspace_id="ws-a",
            conversation_title="Workspace planning",
        )

        conversations = registry.list_workspace_conversations("ws-a")
        assert [conversation.item_id for conversation in conversations] == [conversation_id]
        assert conversations[0].title == "Workspace planning"

    def test_workspace_conversation_link_failure_soft_deletes_created_conversation(
        self,
        db_instance: CharactersRAGDB,
        tmp_path,
    ):
        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="client-1")
        )
        registry.create_workspace(workspace_id="ws-a", name="Workspace A")

        class FailingMembershipRegistry:
            def get_workspace(self, workspace_id: str):
                return registry.get_workspace(workspace_id)

            def link_membership(self, *args, **kwargs):
                raise RuntimeError("membership write failed")

        service = ChatPersistenceService(
            db_instance,
            workspace_registry=FailingMembershipRegistry(),
        )

        with pytest.raises(RuntimeError, match="membership write failed"):
            service.create_conversation(
                scope_type="workspace",
                workspace_id="ws-a",
                conversation_title="Partially linked workspace chat",
            )

        rows = db_instance.execute_query(
            "SELECT id, deleted FROM conversations WHERE title = ?",
            ("Partially linked workspace chat",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["deleted"] == 1
        assert db_instance.get_conversation_by_id(rows[0]["id"]) is None

    def test_fork_conversation_rejects_unresolved_workspace_scope_without_assert(
        self,
        db_instance: CharactersRAGDB,
        monkeypatch,
    ):
        service = ChatPersistenceService(db_instance, workspace_registry=object())
        conversation_id = service.create_conversation(
            assistant_kind="persona",
            assistant_id="planner",
        )
        monkeypatch.setattr(
            service,
            "_require_workspace_scope",
            lambda **_kwargs: None,
        )

        with pytest.raises(ValueError, match="valid workspace ID"):
            service.fork_conversation_into_workspace(
                conversation_id=conversation_id,
                target_workspace_id="ws-a",
            )

    def test_fork_conversation_into_workspace_documents_public_contract(self):
        docstring = inspect.getdoc(ChatPersistenceService.fork_conversation_into_workspace)

        assert docstring is not None
        assert "Args:" in docstring
        assert "Returns:" in docstring
        assert "Raises:" in docstring
