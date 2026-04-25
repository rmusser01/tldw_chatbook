from tldw_chatbook.Character_Chat.local_character_persona_service import LocalCharacterPersonaService
from tldw_chatbook.tldw_api.character_persona_schemas import (
    CharacterChatSessionCreate,
    CharacterChatSessionUpdate,
    CharacterCreateRequest,
    CharacterUpdateRequest,
)


class FakeConversationDB:
    def __init__(self):
        self.client_id = "local-client"
        self.conversations = {}
        self.deleted = set()
        self.created_payloads = []
        self.updated_payloads = []
        self.character_cards = {
            7: {"id": 7, "name": "Ada", "version": 1, "deleted": 0},
        }
        self.next_character_id = 8
        self.updated_character_payloads = []
        self.deleted_character_calls = []
        self.restored_character_calls = []

    def list_character_cards(self, limit=100, offset=0):
        records = [dict(record) for record in self.character_cards.values() if not record.get("deleted")]
        return records[offset : offset + limit]

    def search_character_cards(self, search_term, limit=10):
        records = [
            dict(record)
            for record in self.character_cards.values()
            if not record.get("deleted") and search_term.lower() in record.get("name", "").lower()
        ]
        return records[:limit]

    def get_character_card_by_id(self, character_id):
        record = self.character_cards.get(character_id)
        if record is None or record.get("deleted"):
            return None
        return dict(record)

    def add_character_card(self, payload):
        character_id = self.next_character_id
        self.next_character_id += 1
        self.character_cards[character_id] = {
            "id": character_id,
            "version": 1,
            "deleted": 0,
            **payload,
        }
        return character_id

    def update_character_card(self, character_id, payload, expected_version):
        record = self.character_cards[character_id]
        if record["version"] != expected_version:
            return False
        self.updated_character_payloads.append((character_id, payload, expected_version))
        record.update(payload)
        record["version"] = expected_version + 1
        return True

    def soft_delete_character_card(self, character_id, expected_version):
        record = self.character_cards[character_id]
        if record["version"] != expected_version:
            return False
        self.deleted_character_calls.append((character_id, expected_version))
        record["deleted"] = 1
        record["version"] = expected_version + 1
        return True

    def restore_character_card(self, character_id, expected_version):
        record = self.character_cards[character_id]
        if record["version"] != expected_version:
            return False
        self.restored_character_calls.append((character_id, expected_version))
        record["deleted"] = 0
        record["version"] = expected_version + 1
        return True

    def add_conversation(self, payload):
        conversation_id = payload.get("id") or f"conv-{len(self.conversations) + 1}"
        record = {
            "id": conversation_id,
            "root_id": conversation_id,
            "title": payload.get("title"),
            "character_id": payload.get("character_id"),
            "assistant_kind": payload.get("assistant_kind"),
            "assistant_id": payload.get("assistant_id"),
            "persona_memory_mode": payload.get("persona_memory_mode"),
            "scope_type": payload.get("scope_type") or "global",
            "workspace_id": payload.get("workspace_id"),
            "state": payload.get("state") or "in-progress",
            "topic_label": payload.get("topic_label"),
            "cluster_id": payload.get("cluster_id"),
            "source": payload.get("source"),
            "external_ref": payload.get("external_ref"),
            "runtime_backend": payload.get("runtime_backend") or "local",
            "discovery_owner": payload.get("discovery_owner") or "general_chat",
            "discovery_entity_id": payload.get("discovery_entity_id"),
            "version": 1,
            "deleted": 0,
            "created_at": "2026-04-25T00:00:00Z",
            "last_modified": "2026-04-25T00:00:00Z",
        }
        self.created_payloads.append(payload)
        self.conversations[conversation_id] = record
        return conversation_id

    def get_conversation_by_id(self, conversation_id, include_deleted=False):
        record = self.conversations.get(conversation_id)
        if record is None:
            return None
        if record.get("deleted") and not include_deleted:
            return None
        return dict(record)

    def search_conversations_page(self, query, **kwargs):
        records = [
            dict(record)
            for record in self.conversations.values()
            if kwargs.get("include_deleted") or not record.get("deleted")
        ]
        if kwargs.get("character_id") is not None:
            records = [record for record in records if record.get("character_id") == kwargs["character_id"]]
        if kwargs.get("scope_type"):
            records = [record for record in records if record.get("scope_type") == kwargs["scope_type"]]
        if kwargs.get("workspace_id"):
            records = [record for record in records if record.get("workspace_id") == kwargs["workspace_id"]]
        if kwargs.get("state"):
            records = [record for record in records if record.get("state") == kwargs["state"]]
        if query:
            records = [record for record in records if query.lower() in (record.get("title") or "").lower()]
        return records, len(records), 0.0

    def count_messages_for_conversation(self, *args, **kwargs):
        return 0

    def count_messages_for_conversations(self, conversation_ids, **kwargs):
        return {conversation_id: 0 for conversation_id in conversation_ids}

    def get_keywords_for_conversation(self, conversation_id):
        return []

    def get_keywords_for_conversations(self, conversation_ids):
        return {conversation_id: [] for conversation_id in conversation_ids}

    def update_conversation(self, conversation_id, payload, expected_version):
        record = self.conversations[conversation_id]
        if record["version"] != expected_version:
            return False
        self.updated_payloads.append((conversation_id, payload, expected_version))
        record.update(payload)
        record["version"] = expected_version + 1
        return True

    def soft_delete_conversation(self, conversation_id, expected_version):
        record = self.conversations[conversation_id]
        if record["version"] != expected_version:
            return False
        record["deleted"] = 1
        record["version"] = expected_version + 1
        return True

    def restore_conversation(self, conversation_id, expected_version):
        record = self.conversations[conversation_id]
        if record["version"] != expected_version:
            return False
        record["deleted"] = 0
        record["version"] = expected_version + 1
        return True

    def get_messages_for_conversation(self, conversation_id, limit=100, offset=0, include_deleted=False):
        return [
            {
                "id": "msg-1",
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "Hello",
                "created_at": "2026-04-25T00:01:00Z",
            }
        ][offset : offset + limit]


def test_local_character_persona_service_routes_character_session_metadata_crud():
    db = FakeConversationDB()
    service = LocalCharacterPersonaService(db)

    created = service.create_character_chat_session(
        CharacterChatSessionCreate(
            character_id=7,
            title="Ada chat",
            scope_type="workspace",
            workspace_id="ws-1",
            topic_label="planning",
        )
    )
    listed = service.list_character_chat_sessions(character_id=7, scope_type="workspace", workspace_id="ws-1")
    updated = service.update_character_chat_session(
        created["id"],
        CharacterChatSessionUpdate(title="Ada chat renamed", state="resolved"),
        expected_version=created["version"],
    )
    exported = service.export_chat_history(created["id"], format="json")
    deleted = service.delete_character_chat_session(created["id"], expected_version=updated["version"])
    restored = service.restore_character_chat_session(created["id"], expected_version=updated["version"] + 1)

    assert created["record_id"] == f"local:character_chat_session:{created['id']}"
    assert created["backend"] == "local"
    assert created["assistant_kind"] == "character"
    assert created["assistant_id"] == "7"
    assert created["discovery_owner"] == "ccp_character"
    assert listed["total"] == 1
    assert listed["chats"][0]["workspace_id"] == "ws-1"
    assert updated["title"] == "Ada chat renamed"
    assert updated["state"] == "resolved"
    assert exported["messages"][0]["content"] == "Hello"
    assert deleted == {"status": "deleted", "chat_id": created["id"]}
    assert restored["deleted"] == 0
    assert restored["version"] == updated["version"] + 2


def test_local_character_persona_service_routes_character_card_crud():
    db = FakeConversationDB()
    service = LocalCharacterPersonaService(db)

    listed = service.list_characters()
    searched = service.search_characters("Ada")
    detail = service.get_character(7)
    created = service.create_character(CharacterCreateRequest(name="Local New", description="created"))
    updated = service.update_character(
        created["id"],
        CharacterUpdateRequest(name="Local New v2"),
        expected_version=created["version"],
    )
    deleted = service.delete_character(updated["id"], expected_version=updated["version"])
    restored = service.restore_character(updated["id"], expected_version=updated["version"] + 1)

    assert listed == [{"id": 7, "name": "Ada", "version": 1, "deleted": 0}]
    assert searched[0]["id"] == 7
    assert detail["name"] == "Ada"
    assert created["id"] == 8
    assert created["description"] == "created"
    assert updated["name"] == "Local New v2"
    assert updated["version"] == 2
    assert deleted == {"status": "deleted", "character_id": 8}
    assert restored["id"] == 8
    assert restored["deleted"] == 0
    assert restored["version"] == 4
    assert db.restored_character_calls == [(8, 3)]


def test_local_character_persona_service_supports_persona_session_metadata():
    db = FakeConversationDB()
    service = LocalCharacterPersonaService(db)

    created = service.create_character_chat_session(
        CharacterChatSessionCreate(
            assistant_kind="persona",
            assistant_id="persona-1",
            persona_memory_mode="read_write",
            title="Guide chat",
        )
    )
    listed = service.list_character_chat_sessions(assistant_kind="persona", assistant_id="persona-1")

    assert created["character_id"] is None
    assert created["assistant_kind"] == "persona"
    assert created["assistant_id"] == "persona-1"
    assert created["persona_memory_mode"] == "read_write"
    assert created["discovery_owner"] == "ccp_persona"
    assert listed["total"] == 1
    assert listed["chats"][0]["id"] == created["id"]
