import json
import uuid
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.legacy_conversation_schema import create_legacy_v12_conversations_db
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError


@pytest.fixture
def client_id():
    return "parity_client"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "chat_conversation_parity.sqlite"


@pytest.fixture
def db_instance(db_path, client_id):
    current_db_path = Path(db_path)
    for suffix in ["", "-wal", "-shm"]:
        Path(str(current_db_path) + suffix).unlink(missing_ok=True)

    db = CharactersRAGDB(current_db_path, client_id)
    try:
        yield db
    finally:
        db.close_connection()
        for suffix in ["", "-wal", "-shm"]:
            Path(str(current_db_path) + suffix).unlink(missing_ok=True)


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def _sample_character_id(db: CharactersRAGDB) -> int:
    return db.add_character_card(
        {
            "name": _unique_name("Parity Character"),
            "description": "Character for conversation parity tests.",
            "first_message": "Hello from parity.",
        }
    )


class TestConversationParity:
    def test_add_conversation_allows_persona_conversation_without_character_id(self, db_instance: CharactersRAGDB):
        conversation_id = db_instance.add_conversation(
            {
                "title": "Persona Memory",
                "assistant_kind": "persona",
                "assistant_id": "persona.alpha",
                "persona_memory_mode": "read_write",
            }
        )

        conversation = db_instance.get_conversation_by_id(conversation_id)
        assert conversation is not None
        assert conversation["character_id"] is None
        assert conversation["assistant_kind"] == "persona"
        assert conversation["assistant_id"] == "persona.alpha"
        assert conversation["persona_memory_mode"] == "read_write"
        assert conversation["scope_type"] == "global"
        assert conversation["workspace_id"] is None
        assert conversation["state"] == "in-progress"

    def test_add_conversation_allows_generic_conversation_without_assistant_identity(self, db_instance: CharactersRAGDB):
        conversation_id = db_instance.add_conversation({"title": "Generic Chat"})

        conversation = db_instance.get_conversation_by_id(conversation_id)
        assert conversation is not None
        assert conversation["character_id"] is None
        assert conversation["assistant_kind"] is None
        assert conversation["assistant_id"] is None
        assert conversation["persona_memory_mode"] is None
        assert conversation["scope_type"] == "global"
        assert conversation["workspace_id"] is None
        assert conversation["state"] == "in-progress"

    def test_add_conversation_validates_topic_label_source(self, db_instance: CharactersRAGDB):
        with pytest.raises(InputError, match="topic_label_source must be 'manual' or 'auto'"):
            db_instance.add_conversation(
                {
                    "title": "Invalid Topic Source",
                    "topic_label_source": "guessed",
                }
            )

    def test_legacy_rows_default_to_global_scope_and_null_workspace(self, db_path, client_id):
        legacy_conversation_id = str(uuid.uuid4())
        create_legacy_v12_conversations_db(
            db_path,
            [
                (
                    legacy_conversation_id,
                    legacy_conversation_id,
                    None,
                    None,
                    None,
                    "Legacy Generic",
                    None,
                    "2026-04-18T01:00:00Z",
                    "2026-04-18T01:00:00Z",
                    0,
                    client_id,
                    1,
                )
            ],
        )

        db = CharactersRAGDB(db_path, client_id)
        try:
            conversation = db.get_conversation_by_id(legacy_conversation_id)
            assert conversation is not None
            assert conversation["scope_type"] == "global"
            assert conversation["workspace_id"] is None
            assert conversation["state"] == "in-progress"
        finally:
            db.close_connection()

    def test_legacy_character_rows_backfill_character_assistant_identity(self, db_path, client_id):
        legacy_conversation_id = str(uuid.uuid4())
        create_legacy_v12_conversations_db(
            db_path,
            [
                (
                    legacy_conversation_id,
                    legacy_conversation_id,
                    None,
                    None,
                    77,
                    "Legacy Character",
                    None,
                    "2026-04-18T02:00:00Z",
                    "2026-04-18T02:00:00Z",
                    0,
                    client_id,
                    1,
                )
            ],
        )

        db = CharactersRAGDB(db_path, client_id)
        try:
            conversation = db.get_conversation_by_id(legacy_conversation_id)
            assert conversation is not None
            assert conversation["assistant_kind"] == "character"
            assert conversation["assistant_id"] == "77"
            assert conversation["character_id"] == 77
            sync_rows = db.get_connection().execute(
                "SELECT entity, operation FROM sync_log WHERE entity = 'conversations'"
            ).fetchall()
            assert sync_rows == []
        finally:
            db.close_connection()

    def test_search_conversations_page_filters_by_state_and_topic_and_excludes_deleted_rows_by_default(
        self,
        db_instance: CharactersRAGDB,
    ):
        included_id = db_instance.add_conversation(
            {
                "title": "Billing Resolution",
                "state": "resolved",
                "topic_label": "billing",
            }
        )
        deleted_id = db_instance.add_conversation(
            {
                "title": "Deleted Billing Resolution",
                "state": "resolved",
                "topic_label": "billing",
            }
        )
        wrong_state_id = db_instance.add_conversation(
            {
                "title": "Billing Backlog",
                "state": "backlog",
                "topic_label": "billing",
            }
        )
        wrong_topic_id = db_instance.add_conversation(
            {
                "title": "Support Resolution",
                "state": "resolved",
                "topic_label": "support",
            }
        )

        assert wrong_state_id != wrong_topic_id
        db_instance.soft_delete_conversation(deleted_id, expected_version=1)

        rows, total, max_bm25 = db_instance.search_conversations_page(
            None,
            state="resolved",
            topic_label="billing",
        )

        assert total == 1
        assert max_bm25 == 0.0
        assert [row["id"] for row in rows] == [included_id]
        assert deleted_id not in {row["id"] for row in rows}

    def test_workspace_rows_stay_out_of_global_history(self, db_instance: CharactersRAGDB):
        global_id = db_instance.add_conversation({"title": "Global History"})
        workspace_id = db_instance.add_conversation(
            {
                "title": "Workspace History",
                "scope_type": "workspace",
                "workspace_id": "ws-123",
            }
        )

        global_rows, global_total, _ = db_instance.search_conversations_page(None)
        workspace_rows, workspace_total, _ = db_instance.search_conversations_page(
            None,
            scope_type="workspace",
            workspace_id="ws-123",
        )

        assert global_total == 1
        assert [row["id"] for row in global_rows] == [global_id]
        assert workspace_id not in {row["id"] for row in global_rows}

        assert workspace_total == 1
        assert [row["id"] for row in workspace_rows] == [workspace_id]

    def test_list_all_active_conversations_defaults_to_global_history_only(self, db_instance: CharactersRAGDB):
        global_id = db_instance.add_conversation({"title": "Listable Global"})
        workspace_id = db_instance.add_conversation(
            {
                "title": "Listable Workspace",
                "scope_type": "workspace",
                "workspace_id": "ws-list",
            }
        )

        rows = db_instance.list_all_active_conversations()

        assert [row["id"] for row in rows] == [global_id]
        assert workspace_id not in {row["id"] for row in rows}

    def test_count_messages_for_conversations_returns_conversation_to_count_map(self, db_instance: CharactersRAGDB):
        first_conversation_id = db_instance.add_conversation({"title": "First Message Count"})
        second_conversation_id = db_instance.add_conversation({"title": "Second Message Count"})
        empty_conversation_id = db_instance.add_conversation({"title": "Empty Message Count"})

        db_instance.add_message(
            {
                "conversation_id": first_conversation_id,
                "sender": "user",
                "content": "first",
                "timestamp": "2026-04-19T00:00:01Z",
            }
        )
        db_instance.add_message(
            {
                "conversation_id": first_conversation_id,
                "sender": "assistant",
                "content": "second",
                "timestamp": "2026-04-19T00:00:02Z",
            }
        )
        db_instance.add_message(
            {
                "conversation_id": second_conversation_id,
                "sender": "user",
                "content": "third",
                "timestamp": "2026-04-19T00:00:03Z",
            }
        )

        counts = db_instance.count_messages_for_conversations(
            [first_conversation_id, second_conversation_id, empty_conversation_id]
        )

        assert counts == {
            first_conversation_id: 2,
            second_conversation_id: 1,
            empty_conversation_id: 0,
        }

    def test_update_conversation_persists_aligned_metadata_fields(self, db_instance: CharactersRAGDB):
        conversation_id = db_instance.add_conversation({"title": "Update Me"})

        assert db_instance.update_conversation(
            conversation_id,
            {
                "assistant_kind": "persona",
                "assistant_id": "persona.support",
                "persona_memory_mode": "read_only",
                "scope_type": "workspace",
                "workspace_id": "workspace-7",
                "state": "resolved",
                "topic_label": "billing",
                "topic_label_source": "manual",
                "topic_last_tagged_at": "2026-04-19T01:00:00Z",
                "topic_last_tagged_message_id": "msg-42",
                "cluster_id": "cluster-9",
                "source": "import",
                "external_ref": "external-123",
            },
            expected_version=1,
        ) is True

        conversation = db_instance.get_conversation_by_id(conversation_id)
        assert conversation is not None
        assert conversation["assistant_kind"] == "persona"
        assert conversation["assistant_id"] == "persona.support"
        assert conversation["persona_memory_mode"] == "read_only"
        assert conversation["scope_type"] == "workspace"
        assert conversation["workspace_id"] == "workspace-7"
        assert conversation["state"] == "resolved"
        assert conversation["topic_label"] == "billing"
        assert conversation["topic_label_source"] == "manual"
        assert conversation["topic_last_tagged_at"] == "2026-04-19T01:00:00Z"
        assert conversation["topic_last_tagged_message_id"] == "msg-42"
        assert conversation["cluster_id"] == "cluster-9"
        assert conversation["source"] == "import"
        assert conversation["external_ref"] == "external-123"
        assert conversation["version"] == 2

        sync_row = db_instance.get_connection().execute(
            """
            SELECT payload
            FROM sync_log
            WHERE entity = 'conversations' AND entity_id = ? AND operation = 'update'
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        assert sync_row is not None
        payload = json.loads(sync_row["payload"])
        assert payload["assistant_kind"] == "persona"
        assert payload["assistant_id"] == "persona.support"
        assert payload["persona_memory_mode"] == "read_only"
        assert payload["scope_type"] == "workspace"
        assert payload["workspace_id"] == "workspace-7"
        assert payload["state"] == "resolved"
        assert payload["topic_label"] == "billing"
        assert payload["topic_label_source"] == "manual"
        assert payload["topic_last_tagged_at"] == "2026-04-19T01:00:00Z"
        assert payload["topic_last_tagged_message_id"] == "msg-42"
        assert payload["cluster_id"] == "cluster-9"
        assert payload["source"] == "import"
        assert payload["external_ref"] == "external-123"

    def test_keyword_and_message_tree_helpers_match_server_shaped_contract(self, db_instance: CharactersRAGDB):
        character_id = _sample_character_id(db_instance)
        conversation_id = db_instance.add_conversation({"title": "Helper Coverage", "character_id": character_id})

        alpha_id = db_instance.add_keyword("alpha")
        beta_id = db_instance.add_keyword("beta")
        gamma_id = db_instance.add_keyword("gamma")

        db_instance.replace_keywords_for_conversation(conversation_id, [alpha_id, beta_id])
        keyword_map = db_instance.get_keywords_for_conversations([conversation_id])
        assert [item["keyword"] for item in keyword_map[conversation_id]] == ["alpha", "beta"]

        db_instance.replace_keywords_for_conversation(conversation_id, [gamma_id])
        keyword_map = db_instance.get_keywords_for_conversations([conversation_id])
        assert [item["keyword"] for item in keyword_map[conversation_id]] == ["gamma"]

        sync_rows = db_instance.get_connection().execute(
            """
            SELECT operation, entity_id, payload
            FROM sync_log
            WHERE entity = 'conversation_keywords'
            ORDER BY rowid ASC
            """
        ).fetchall()
        sync_ops = [(row["operation"], row["entity_id"]) for row in sync_rows]
        assert ( "create", f"{conversation_id}_{alpha_id}") in sync_ops
        assert ( "create", f"{conversation_id}_{beta_id}") in sync_ops
        assert ( "delete", f"{conversation_id}_{alpha_id}") in sync_ops
        assert ( "create", f"{conversation_id}_{gamma_id}") in sync_ops

        root_message_id = db_instance.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "root one",
                "timestamp": "2026-04-19T02:00:01Z",
            }
        )
        child_message_id = db_instance.add_message(
            {
                "conversation_id": conversation_id,
                "parent_message_id": root_message_id,
                "sender": "assistant",
                "content": "child one",
                "timestamp": "2026-04-19T02:00:02Z",
            }
        )
        second_root_message_id = db_instance.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "root two",
                "timestamp": "2026-04-19T02:00:03Z",
            }
        )
        db_instance.get_connection().execute(
            """
            UPDATE messages
            SET variant_of = ?, variant_number = ?, is_selected_variant = ?, total_variants = ?
            WHERE id = ?
            """,
            (root_message_id, 2, 1, 2, child_message_id),
        )
        db_instance.get_connection().commit()

        assert db_instance.count_messages_for_conversation(conversation_id) == 3
        assert db_instance.count_root_messages_for_conversation(conversation_id) == 2

        latest_message = db_instance.get_latest_message_for_conversation(conversation_id)
        assert latest_message is not None
        assert latest_message["id"] == second_root_message_id

        root_messages = db_instance.get_root_messages_for_conversation(
            conversation_id,
            limit=10,
            offset=0,
            order_by_timestamp="ASC",
        )
        assert [message["id"] for message in root_messages] == [root_message_id, second_root_message_id]

        child_messages = db_instance.get_messages_for_conversation_by_parent_ids(
            conversation_id,
            [root_message_id],
            order_by_timestamp="ASC",
        )
        assert [message["id"] for message in child_messages] == [child_message_id]
        assert child_messages[0]["conversation_id"] == conversation_id
        assert child_messages[0]["variant_of"] == root_message_id
        assert child_messages[0]["variant_number"] == 2
        assert child_messages[0]["is_selected_variant"] == 1
        assert child_messages[0]["total_variants"] == 2

    def test_deleted_conversation_helpers_require_explicit_opt_in(self, db_instance: CharactersRAGDB):
        conversation_id = db_instance.add_conversation({"title": "Deleted Helper Coverage"})
        root_message_id = db_instance.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "root",
                "timestamp": "2026-04-19T03:00:01Z",
            }
        )
        child_message_id = db_instance.add_message(
            {
                "conversation_id": conversation_id,
                "parent_message_id": root_message_id,
                "sender": "assistant",
                "content": "child",
                "timestamp": "2026-04-19T03:00:02Z",
            }
        )
        db_instance.soft_delete_conversation(conversation_id, expected_version=1)

        assert db_instance.count_messages_for_conversation(conversation_id) == 0
        assert db_instance.count_messages_for_conversations([conversation_id]) == {conversation_id: 0}
        assert db_instance.get_latest_message_for_conversation(conversation_id) is None
        assert db_instance.count_root_messages_for_conversation(conversation_id) == 0
        assert db_instance.get_root_messages_for_conversation(conversation_id, limit=10, offset=0) == []
        assert db_instance.get_messages_for_conversation_by_parent_ids(conversation_id, [root_message_id]) == []

        assert db_instance.count_messages_for_conversation(
            conversation_id,
            include_deleted_conversation=True,
        ) == 2
        assert db_instance.count_messages_for_conversations(
            [conversation_id],
            include_deleted_conversation=True,
        ) == {conversation_id: 2}
        latest_message = db_instance.get_latest_message_for_conversation(
            conversation_id,
            include_deleted_conversation=True,
        )
        assert latest_message is not None
        assert latest_message["id"] == child_message_id
        assert db_instance.count_root_messages_for_conversation(
            conversation_id,
            include_deleted_conversation=True,
        ) == 1
        root_messages = db_instance.get_root_messages_for_conversation(
            conversation_id,
            limit=10,
            offset=0,
            include_deleted_conversation=True,
        )
        assert [message["id"] for message in root_messages] == [root_message_id]
        child_messages = db_instance.get_messages_for_conversation_by_parent_ids(
            conversation_id,
            [root_message_id],
            include_deleted_conversation=True,
        )
        assert [message["id"] for message in child_messages] == [child_message_id]
