import inspect

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, MessageAttachment
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
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

    def test_create_message_splits_position_zero_and_rest(self, db_instance: CharactersRAGDB):
        service = ChatPersistenceService(db_instance)
        conv_id = service.create_conversation(
            assistant_kind="generic", assistant_id="console",
            conversation_title="t", workspace_id=None, scope_type="global",
        )
        attachments = [
            {"position": 0, "data": b"img-0", "mime_type": "image/png", "display_name": "a.png"},
            {"position": 1, "data": b"img-1", "mime_type": "image/jpeg", "display_name": "b.jpg"},
            {"position": 2, "data": b"img-2", "mime_type": "image/png", "display_name": "c.png"},
        ]
        msg_id = service.create_message(
            conversation_id=conv_id, sender="user", content="multi",
            image_data=None, image_mime_type=None, attachments=attachments,
        )
        row = db_instance.get_message_by_id(msg_id)
        assert row["image_data"] == b"img-0"
        assert row["image_mime_type"] == "image/png"
        extra = db_instance.get_attachments_for_messages([msg_id])[msg_id]
        assert [r["position"] for r in extra] == [1, 2]
        assert extra[0]["data"] == b"img-1"
        # The service-level batch read is a passthrough to the DB method.
        assert service.get_attachments_for_messages([msg_id]) == {msg_id: extra}

    def test_update_without_attachments_leaves_table_and_columns_alone(self, db_instance: CharactersRAGDB):
        service = ChatPersistenceService(db_instance)
        conv_id = service.create_conversation(
            assistant_kind="generic", assistant_id="console",
            conversation_title="t", workspace_id=None, scope_type="global",
        )
        msg_id = service.create_message(
            conversation_id=conv_id, sender="user", content="multi",
            image_data=None, image_mime_type=None,
            attachments=[
                {"position": 0, "data": b"img-0", "mime_type": "image/png", "display_name": "a.png"},
                {"position": 1, "data": b"img-1", "mime_type": "image/png", "display_name": "b.png"},
            ],
        )
        service.update_message_content(
            message_id=msg_id, content="edited",
            image_data=None, image_mime_type=None,
        )
        row = db_instance.get_message_by_id(msg_id)
        assert row["content"] == "edited"
        assert row["image_data"] == b"img-0"
        assert db_instance.get_attachments_for_messages([msg_id])[msg_id][0]["data"] == b"img-1"

    def test_update_with_position_zero_only_rewrites_columns_and_clears_table(
        self, db_instance: CharactersRAGDB
    ):
        """An explicit attachments list is an authoritative rewrite: a list
        with no >= 1 positions still calls through to the table write so
        stale rows are cleared (empty-list DELETE+INSERT)."""
        service = ChatPersistenceService(db_instance)
        conv_id = service.create_conversation(
            assistant_kind="generic", assistant_id="console",
            conversation_title="t", workspace_id=None, scope_type="global",
        )
        msg_id = service.create_message(
            conversation_id=conv_id, sender="user", content="multi",
            image_data=None, image_mime_type=None,
            attachments=[
                {"position": 0, "data": b"img-0", "mime_type": "image/png", "display_name": "a.png"},
                {"position": 1, "data": b"img-1", "mime_type": "image/png", "display_name": "b.png"},
            ],
        )
        service.update_message_content(
            message_id=msg_id, content="rewritten",
            image_data=None, image_mime_type=None,
            attachments=[
                {"position": 0, "data": b"img-new", "mime_type": "image/jpeg", "display_name": "new.jpg"},
            ],
        )
        row = db_instance.get_message_by_id(msg_id)
        assert row["content"] == "rewritten"
        assert row["image_data"] == b"img-new"
        assert row["image_mime_type"] == "image/jpeg"
        assert db_instance.get_attachments_for_messages([msg_id]) == {}

    def test_create_message_rolls_back_row_when_attachment_write_fails(
        self, db_instance: CharactersRAGDB, monkeypatch
    ):
        """The message insert and the >=1 attachment-table write must be one
        atomic unit: a failure writing the table must roll back the row."""
        service = ChatPersistenceService(db_instance)
        conv_id = service.create_conversation(
            assistant_kind="generic", assistant_id="console",
            conversation_title="t", workspace_id=None, scope_type="global",
        )

        def _boom(message_id, rows):
            raise RuntimeError("attachment write failed")

        monkeypatch.setattr(db_instance, "set_message_attachments", _boom)
        with pytest.raises(RuntimeError, match="attachment write failed"):
            service.create_message(
                conversation_id=conv_id, sender="user", content="multi",
                image_data=None, image_mime_type=None,
                message_id="msg-atomic-create",
                attachments=[
                    {"position": 0, "data": b"img-0", "mime_type": "image/png", "display_name": "a.png"},
                    {"position": 1, "data": b"img-1", "mime_type": "image/png", "display_name": "b.png"},
                ],
            )
        # get_message_by_id returns None for a missing row (per its contract),
        # proving the INSERT rolled back with the failed attachment write.
        assert db_instance.get_message_by_id("msg-atomic-create") is None

    def test_update_rolls_back_content_and_columns_when_attachment_write_fails(
        self, db_instance: CharactersRAGDB, monkeypatch
    ):
        """The message-row update and the >=1 attachment-table rewrite must be
        one atomic unit: a table-write failure rolls back content and the
        legacy image columns."""
        service = ChatPersistenceService(db_instance)
        conv_id = service.create_conversation(
            assistant_kind="generic", assistant_id="console",
            conversation_title="t", workspace_id=None, scope_type="global",
        )
        msg_id = service.create_message(
            conversation_id=conv_id, sender="user", content="before",
            image_data=b"img-old", image_mime_type="image/png",
        )

        def _boom(message_id, rows):
            raise RuntimeError("attachment write failed")

        monkeypatch.setattr(db_instance, "set_message_attachments", _boom)
        with pytest.raises(RuntimeError, match="attachment write failed"):
            service.update_message_content(
                message_id=msg_id, content="after",
                image_data=None, image_mime_type=None,
                attachments=[
                    {"position": 0, "data": b"img-new", "mime_type": "image/jpeg", "display_name": "n.jpg"},
                    {"position": 1, "data": b"img-1", "mime_type": "image/png", "display_name": "b.png"},
                ],
            )
        row = db_instance.get_message_by_id(msg_id)
        assert row["content"] == "before"
        assert row["image_data"] == b"img-old"
        assert row["image_mime_type"] == "image/png"

    def test_update_skips_attachment_write_when_row_update_returns_false(
        self, db_instance: CharactersRAGDB, monkeypatch
    ):
        """``update_message`` returning a falsy result (optimistic-lock miss
        reported without an exception, e.g. from a future/alternate db
        implementation) must short-circuit before ``set_message_attachments``
        runs. Without the guard, attachments would be rewritten even though
        the content/version update did not take -- attachments and content
        would drift out of sync."""
        service = ChatPersistenceService(db_instance)
        conv_id = service.create_conversation(
            assistant_kind="generic", assistant_id="console",
            conversation_title="t", workspace_id=None, scope_type="global",
        )
        msg_id = service.create_message(
            conversation_id=conv_id, sender="user", content="before",
            image_data=None, image_mime_type=None,
            attachments=[
                {"position": 0, "data": b"img-0", "mime_type": "image/png", "display_name": "a.png"},
                {"position": 1, "data": b"img-1", "mime_type": "image/png", "display_name": "b.png"},
            ],
        )

        set_attachments_calls = []
        original_set_attachments = db_instance.set_message_attachments

        def _tracking_set_attachments(message_id, rows):
            set_attachments_calls.append((message_id, rows))
            return original_set_attachments(message_id, rows)

        monkeypatch.setattr(db_instance, "set_message_attachments", _tracking_set_attachments)
        monkeypatch.setattr(db_instance, "update_message", lambda *args, **kwargs: False)

        result = service.update_message_content(
            message_id=msg_id, content="after",
            image_data=None, image_mime_type=None,
            attachments=[
                {"position": 0, "data": b"img-new", "mime_type": "image/jpeg", "display_name": "n.jpg"},
                {"position": 1, "data": b"img-1-new", "mime_type": "image/png", "display_name": "b2.png"},
            ],
        )

        assert result is False
        assert set_attachments_calls == []
        # The message_attachments table must be untouched -- still the
        # original position-1 row, not the rewritten one.
        extra = db_instance.get_attachments_for_messages([msg_id])[msg_id]
        assert extra[0]["data"] == b"img-1"

    # -- Regression coverage for the #217 P0 live crash --------------------
    #
    # ``ConsoleChatStore``'s persistence tests all wire in **kwargs-based
    # fakes (see ``RecordingPersistence`` in test_console_chat_store.py).
    # Those fakes silently swallowed a call shape the REAL
    # ``ChatPersistenceService.create_message`` rejected outright: the
    # store's multi-attachment branch omitted the keyword-only
    # ``image_data``/``image_mime_type`` arguments, which used to have no
    # defaults, so a real send with >= 2 attachments raised
    # ``TypeError: create_message() missing 2 required keyword-only
    # arguments: 'image_data' and 'image_mime_type'`` and crashed the whole
    # app. The store->fake seam never exercised the store against the real
    # service, so the gap went undetected. The tests below wire a REAL
    # ``ChatPersistenceService`` (backed by the ``db_instance`` fixture's
    # real in-memory-file SQLite) into ``ConsoleChatStore`` and drive
    # ``append_message(..., persist=True)`` -- the exact call path that
    # crashed live -- for zero, one, and two-or-more attachments.

    def test_console_store_real_service_persists_zero_attachment_message(
        self, db_instance: CharactersRAGDB
    ):
        """A plain text message (no attachments) persists cleanly through a
        real ``ChatPersistenceService`` wired into ``ConsoleChatStore``."""
        service = ChatPersistenceService(db_instance)
        store = ConsoleChatStore(persistence=service)
        session = store.ensure_session()

        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="hello",
            persist=True,
        )

        assert message.persisted_message_id is not None
        row = db_instance.get_message_by_id(message.persisted_message_id)
        assert row["content"] == "hello"
        assert row["image_data"] is None
        assert row["image_mime_type"] is None
        assert db_instance.get_attachments_for_messages([message.persisted_message_id]) == {}

    def test_console_store_real_service_persists_single_attachment_message(
        self, db_instance: CharactersRAGDB
    ):
        """A single attachment stays on the pre-split scalar columns (the
        store's ``len(attachments) > 1`` gate never engages split
        addressing for exactly one attachment); the real service must
        accept that call shape too."""
        service = ChatPersistenceService(db_instance)
        store = ConsoleChatStore(persistence=service)
        session = store.ensure_session()

        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="one file",
            attachments=(
                MessageAttachment(
                    data=b"img-0", mime_type="image/png", display_name="a.png", position=0
                ),
            ),
            persist=True,
        )

        assert message.persisted_message_id is not None
        row = db_instance.get_message_by_id(message.persisted_message_id)
        assert row["image_data"] == b"img-0"
        assert row["image_mime_type"] == "image/png"
        assert db_instance.get_attachments_for_messages([message.persisted_message_id]) == {}

    def test_console_store_real_service_persists_multi_attachment_message(
        self, db_instance: CharactersRAGDB
    ):
        """The exact P0 live-crash call path: sending >= 2 attachments
        through ``ConsoleChatStore.append_message(..., persist=True)``
        against a REAL ``ChatPersistenceService`` must not raise. Legacy
        columns hold position 0; the ``message_attachments`` table holds
        positions >= 1."""
        service = ChatPersistenceService(db_instance)
        store = ConsoleChatStore(persistence=service)
        session = store.ensure_session()

        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="two files",
            attachments=(
                MessageAttachment(
                    data=b"img-0", mime_type="image/png", display_name="a.png", position=0
                ),
                MessageAttachment(
                    data=b"img-1", mime_type="image/jpeg", display_name="b.jpg", position=1
                ),
            ),
            persist=True,
        )

        assert message.persisted_message_id is not None
        row = db_instance.get_message_by_id(message.persisted_message_id)
        # The real service derives the legacy columns from attachments[0],
        # overriding the store's explicit None scalars.
        assert row["image_data"] == b"img-0"
        assert row["image_mime_type"] == "image/png"
        extra = db_instance.get_attachments_for_messages([message.persisted_message_id])[
            message.persisted_message_id
        ]
        assert [entry["position"] for entry in extra] == [1]
        assert extra[0]["data"] == b"img-1"
        assert extra[0]["display_name"] == "b.jpg"
