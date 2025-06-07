# test_chachanotes_db.py
#
#
# Imports
import pytest
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

#
# Third-Party Imports
from hypothesis import strategies as st
from hypothesis import given, settings, HealthCheck
#
# Local Imports
# --- UPDATED IMPORT PATH ---
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
    InputError,
    ConflictError
)


#
#######################################################################################################################
#
# Functions:


# --- Fixtures ---

@pytest.fixture
def client_id():
    """Provides a consistent client ID for tests."""
    return "test_client_001"


@pytest.fixture
def db_path(tmp_path):
    """Provides a temporary path for the database file for each test."""
    return tmp_path / "test_db.sqlite"


@pytest.fixture(scope="function")
def db_instance(db_path, client_id):
    """Creates a DB instance for each test, ensuring a fresh database."""
    current_db_path = Path(db_path)

    # Clean up any existing files from previous runs to be safe
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(current_db_path) + suffix)
        if p.exists():
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Could not unlink {p}: {e}")

    db = None
    try:
        db = CharactersRAGDB(current_db_path, client_id)
        yield db
    finally:
        if db:
            db.close_connection()
            # Additional cleanup after test completes
            for suffix in ["", "-wal", "-shm"]:
                p = Path(str(current_db_path) + suffix)
                if p.exists():
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass


@pytest.fixture
def mem_db_instance(client_id):
    """Creates an in-memory DB instance for tests that don't need file persistence."""
    db = CharactersRAGDB(":memory:", client_id)
    yield db
    db.close_connection()

@pytest.fixture
def sample_card(db_instance: CharactersRAGDB) -> dict:
    """A fixture that adds a sample card to the DB and returns its data."""
    card_data = _create_sample_card_data("FromFixture")
    card_id = db_instance.add_character_card(card_data)
    # Return the full record from the DB, which includes ID, version, etc.
    return db_instance.get_character_card_by_id(card_id)

# You can create similar fixtures for conversations, messages, etc.
@pytest.fixture
def sample_conv(db_instance: CharactersRAGDB, sample_card: dict) -> dict:
    """Adds a sample conversation linked to the sample_card."""
    conv_data = {
        "character_id": sample_card['id'],
        "title": "Conversation From Fixture"
    }
    conv_id = db_instance.add_conversation(conv_data)
    return db_instance.get_conversation_by_id(conv_id)


# --- Helper Functions ---
def get_current_utc_timestamp_iso():
    """Returns the current UTC time in ISO 8601 format, as used by the DB."""
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def _create_sample_card_data(name_suffix="", client_id_override=None):
    """Creates a sample character card data dictionary."""
    return {
        "name": f"Test Character {name_suffix}",
        "description": "A test character.",
        "personality": "Testy",
        "scenario": "A test scenario.",
        "image": b"testimagebytes",
        "first_message": "Hello, test!",
        "alternate_greetings": json.dumps(["Hi", "Hey"]),
        "tags": json.dumps(["test", "sample"]),
        "extensions": json.dumps({"custom_field": "value"}),
        "client_id": client_id_override
    }


# --- Test Cases ---

class TestDBInitialization:
    def test_db_creation_and_schema_version(self, db_path, client_id):
        current_db_path = Path(db_path)
        assert not current_db_path.exists()
        db = CharactersRAGDB(current_db_path, client_id)
        assert current_db_path.exists()
        assert db.client_id == client_id

        # Check schema version
        conn = db.get_connection()
        version_row = conn.execute("SELECT version FROM db_schema_version WHERE schema_name = ?",
                                   (db._SCHEMA_NAME,)).fetchone()
        assert version_row is not None
        assert version_row['version'] == db._CURRENT_SCHEMA_VERSION
        db.close_connection()

    def test_in_memory_db_initialization(self, client_id):
        db = CharactersRAGDB(":memory:", client_id)
        assert db.is_memory_db
        assert db.client_id == client_id
        conn = db.get_connection()
        version_row = conn.execute("SELECT version FROM db_schema_version WHERE schema_name = ?",
                                   (db._SCHEMA_NAME,)).fetchone()
        assert version_row is not None
        assert version_row['version'] == db._CURRENT_SCHEMA_VERSION
        db.close_connection()

    def test_initialization_with_missing_client_id(self, db_path):
        with pytest.raises(ValueError, match="Client ID cannot be empty or None."):
            CharactersRAGDB(db_path, "")
        with pytest.raises(ValueError, match="Client ID cannot be empty or None."):
            CharactersRAGDB(db_path, None)

    def test_reopening_db_preserves_schema(self, db_path, client_id):
        db1 = CharactersRAGDB(db_path, client_id)
        v1 = db1._get_db_version(db1.get_connection())
        db1.close_connection()

        db2 = CharactersRAGDB(db_path, "another_client")
        v2 = db2._get_db_version(db2.get_connection())
        assert v1 == v2
        assert v2 == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        db2.close_connection()

    def test_opening_db_with_newer_schema_raises_error(self, db_path, client_id):
        db = CharactersRAGDB(db_path, client_id)
        conn = db.get_connection()
        newer_version = CharactersRAGDB._CURRENT_SCHEMA_VERSION + 1
        conn.execute("UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
                     (newer_version, CharactersRAGDB._SCHEMA_NAME))
        conn.commit()
        db.close_connection()

        expected_message_part = f"version \\({newer_version}\\) is newer than supported by code \\({CharactersRAGDB._CURRENT_SCHEMA_VERSION}\\)"
        with pytest.raises(CharactersRAGDBError, match=expected_message_part):
            CharactersRAGDB(db_path, client_id)


class TestCharacterCards:
    def test_add_character_card(self, db_instance: CharactersRAGDB):
        card_data = _create_sample_card_data("Add")
        card_id = db_instance.add_character_card(card_data)
        assert isinstance(card_id, int)

        retrieved = db_instance.get_character_card_by_id(card_id)
        assert retrieved is not None
        assert retrieved["name"] == card_data["name"]
        assert retrieved["description"] == card_data["description"]
        assert retrieved["image"] == card_data["image"]
        assert isinstance(retrieved["alternate_greetings"], list)
        assert retrieved["alternate_greetings"] == json.loads(card_data["alternate_greetings"])
        assert retrieved["client_id"] == db_instance.client_id
        assert retrieved["version"] == 1
        assert not retrieved["deleted"]

    def test_add_character_card_with_missing_name_raises_error(self, db_instance: CharactersRAGDB):
        card_data = _create_sample_card_data("MissingName")
        del card_data["name"]
        with pytest.raises(InputError, match="Required field 'name' is missing"):
            db_instance.add_character_card(card_data)

    def test_add_character_card_with_duplicate_name_raises_error(self, db_instance: CharactersRAGDB):
        card_data = _create_sample_card_data("Duplicate")
        db_instance.add_character_card(card_data)
        with pytest.raises(ConflictError, match=f"Character card with name '{card_data['name']}' already exists"):
            db_instance.add_character_card(card_data)

    def test_get_character_card_by_id_not_found(self, db_instance: CharactersRAGDB):
        assert db_instance.get_character_card_by_id(999) is None

    def test_get_character_card_by_name(self, db_instance: CharactersRAGDB):
        card_data = _create_sample_card_data("ByName")
        card_id = db_instance.add_character_card(card_data)
        retrieved = db_instance.get_character_card_by_name(card_data["name"])
        assert retrieved is not None
        assert retrieved["id"] == card_id

    def test_list_character_cards(self, db_instance: CharactersRAGDB):
        # A new DB instance should contain exactly one default card.
        initial_cards = db_instance.list_character_cards()
        assert len(initial_cards) == 1
        assert initial_cards[0]['name'] == 'Default Assistant'

        card_data1 = _create_sample_card_data("List1")
        card_data2 = _create_sample_card_data("List2")
        db_instance.add_character_card(card_data1)
        db_instance.add_character_card(card_data2)

        # The list should now contain 3 cards (1 default + 2 new)
        cards = db_instance.list_character_cards()
        assert len(cards) == 3

        # You can still sort and check your added cards if you filter out the default one.
        added_card_names = {c['name'] for c in cards if c['name'] != 'Default Assistant'}
        assert added_card_names == {card_data1['name'], card_data2['name']}

    def test_update_character_card(self, db_instance: CharactersRAGDB, sample_card: dict):
        update_payload = {"description": "Updated Description"}
        updated = db_instance.update_character_card(
            sample_card['id'],
            update_payload,
            expected_version=sample_card['version']
        )
        assert updated is True

        retrieved = db_instance.get_character_card_by_id(sample_card['id'])
        assert retrieved["description"] == "Updated Description"
        assert retrieved["version"] == sample_card['version'] + 1

    def test_update_character_card_with_version_conflict_raises_error(self, db_instance: CharactersRAGDB):
        card_id = db_instance.add_character_card(_create_sample_card_data("VersionConflict"))

        # Simulate another client's update, bumping DB version to 2
        db_instance.update_character_card(card_id, {"description": "First update"}, expected_version=1)

        # Client tries to update with old expected_version=1
        update_payload = {"description": "Conflict Update"}
        expected_error_regex = r"version mismatch \(db has 2, client expected 1\)"
        with pytest.raises(ConflictError, match=expected_error_regex):
            db_instance.update_character_card(card_id, update_payload, expected_version=1)

    def test_update_character_card_not_found_raises_error(self, db_instance: CharactersRAGDB):
        with pytest.raises(ConflictError, match="Record not found in character_cards"):
            db_instance.update_character_card(999, {"description": "Not Found"}, expected_version=1)

    def test_soft_delete_character_card(self, db_instance: CharactersRAGDB, sample_card: dict):
        deleted = db_instance.soft_delete_character_card(
            sample_card['id'],
            expected_version=sample_card['version']
        )
        assert deleted is True
        assert db_instance.get_character_card_by_id(sample_card['id']) is None

    def test_soft_delete_is_idempotent(self, db_instance: CharactersRAGDB):
        card_id = db_instance.add_character_card(_create_sample_card_data("IdempotentDelete"))
        db_instance.soft_delete_character_card(card_id, expected_version=1)

        # Calling delete again on an already deleted record should succeed
        assert db_instance.soft_delete_character_card(card_id, expected_version=1) is True
        # Verify version didn't change again
        conn = db_instance.get_connection()
        raw_record = conn.execute("SELECT version FROM character_cards WHERE id = ?", (card_id,)).fetchone()
        assert raw_record["version"] == 2

    def test_search_character_cards(self, db_instance: CharactersRAGDB):
        card1_data = _create_sample_card_data("Search Alpha")
        card1_data["description"] = "Unique keyword: ZYX"
        card2_data = _create_sample_card_data("Search Beta")
        card2_data["system_prompt"] = "Also has ZYX"
        card3_data = _create_sample_card_data("Unsearchable")
        db_instance.add_character_card(card1_data)
        card2_id = db_instance.add_character_card(card2_data)
        db_instance.add_character_card(card3_data)

        results = db_instance.search_character_cards("ZYX")
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert card1_data["name"] in names
        assert card2_data["name"] in names

        # Test search after soft-deleting one of the results
        card2 = db_instance.get_character_card_by_id(card2_id)
        db_instance.soft_delete_character_card(card2["id"], expected_version=card2["version"])

        results_after_delete = db_instance.search_character_cards("ZYX")
        assert len(results_after_delete) == 1
        assert results_after_delete[0]["name"] == card1_data["name"]

    @pytest.mark.parametrize(
        "field_to_remove, expected_error, error_match",
        [
            ("name", InputError, "Required field 'name' is missing"),
            # Assuming you add a required 'creator' field later
            # ("creator", InputError, "Required field 'creator' is missing"),
        ]
    )
    def test_add_card_missing_required_fields(self, db_instance, field_to_remove, expected_error, error_match):
        card_data = _create_sample_card_data("MissingFields")
        del card_data[field_to_remove]
        with pytest.raises(expected_error, match=error_match):
            db_instance.add_character_card(card_data)

class TestConversationsAndMessages:
    @pytest.fixture
    def char_id(self, db_instance):
        card_id = db_instance.add_character_card(_create_sample_card_data("ConvChar"))
        return card_id

    def test_add_conversation(self, db_instance: CharactersRAGDB, char_id):
        conv_data = {"id": str(uuid.uuid4()), "character_id": char_id, "title": "Test Conversation"}
        conv_id = db_instance.add_conversation(conv_data)
        assert conv_id == conv_data["id"]

        retrieved = db_instance.get_conversation_by_id(conv_id)
        assert retrieved["title"] == "Test Conversation"
        assert retrieved["character_id"] == char_id
        assert retrieved["version"] == 1
        assert retrieved["client_id"] == db_instance.client_id

    def test_add_message_and_get_for_conversation(self, db_instance: CharactersRAGDB, char_id):
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "MsgConv"})
        msg1_id = db_instance.add_message(
            {"conversation_id": conv_id, "sender": "user", "content": "First", "timestamp": "2023-01-01T10:00:00Z"})
        msg2_id = db_instance.add_message(
            {"conversation_id": conv_id, "sender": "ai", "content": "Second", "timestamp": "2023-01-01T10:01:00Z"})

        messages_asc = db_instance.get_messages_for_conversation(conv_id, order_by_timestamp="ASC")
        assert len(messages_asc) == 2
        assert messages_asc[0]["id"] == msg1_id
        assert messages_asc[1]["id"] == msg2_id

        messages_desc = db_instance.get_messages_for_conversation(conv_id, order_by_timestamp="DESC")
        assert len(messages_desc) == 2
        assert messages_desc[0]["id"] == msg2_id
        assert messages_desc[1]["id"] == msg1_id

    def test_update_conversation_and_fts(self, db_instance: CharactersRAGDB, char_id: int):
        initial_title = "AlphaTitleOne"
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": initial_title})
        original_conv = db_instance.get_conversation_by_id(conv_id)

        # Verify FTS state before update
        assert len(db_instance.search_conversations_by_title(initial_title)) == 1

        # Perform update
        updated_title = "BetaTitleTwo"
        db_instance.update_conversation(conv_id, {"title": updated_title}, expected_version=original_conv['version'])

        # Verify FTS state after update
        assert len(db_instance.search_conversations_by_title(updated_title)) == 1
        assert len(db_instance.search_conversations_by_title(initial_title)) == 0, "FTS should not find the old title"

    def test_soft_delete_conversation_and_fts(self, db_instance: CharactersRAGDB, char_id):
        conv_title_for_delete_test = "DeleteConvForFTS"
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": conv_title_for_delete_test})
        original_conv = db_instance.get_conversation_by_id(conv_id)

        assert len(db_instance.search_conversations_by_title(conv_title_for_delete_test)) == 1

        db_instance.soft_delete_conversation(conv_id, expected_version=original_conv['version'])

        assert db_instance.get_conversation_by_id(conv_id) is None
        assert len(db_instance.search_conversations_by_title(
            conv_title_for_delete_test)) == 0, "FTS should not find soft-deleted conversation"

    def test_search_messages_by_content(self, db_instance: CharactersRAGDB, char_id):
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "MessageSearchConv"})
        msg1_data = {"id": str(uuid.uuid4()), "conversation_id": conv_id, "sender": "user",
                     "content": "UniqueMessageContentAlpha"}
        db_instance.add_message(msg1_data)

        results = db_instance.search_messages_by_content("UniqueMessageContentAlpha")
        assert len(results) == 1
        assert results[0]["id"] == msg1_data["id"]


    # @pytest.mark.parametrize(
    #     "msg_data, raises_error",
    #     [
    #         ({"content": "Hello", "image_data": None, "image_mime_type": None}, False),
    #         ({"content": "", "image_data": b'img', "image_mime_type": "image/png"}, False),
    #         ({"content": "Hello", "image_data": b'img', "image_mime_type": "image/png"}, False),
    #         # Failure cases
    #         ({"content": "", "image_data": None, "image_mime_type": None}, True),  # Both missing
    #         ({"content": None, "image_data": None, "image_mime_type": None}, True),  # Both missing
    #         ({"content": "", "image_data": b'img', "image_mime_type": None}, True),  # Mime type missing
    #     ]
    # )
    # def test_add_message_content_requirements(self, db_instance, sample_conv, msg_data, raises_error):
    #     full_payload = {
    #         "conversation_id": sample_conv['id'],
    #         "sender": "user",
    #         **msg_data
    #     }
    #
    #     if raises_error:
    #         with pytest.raises((InputError, TypeError)):  # TypeError if content is None
    #             db_instance.add_message(full_payload)
    #     else:
    #         msg_id = db_instance.add_message(full_payload)
    #         assert msg_id is not None



class TestNotesAndKeywords:
    def test_add_and_update_note(self, db_instance: CharactersRAGDB):
        note_id = db_instance.add_note("Original Title", "Original Content")
        assert isinstance(note_id, str)

        original_note = db_instance.get_note_by_id(note_id)
        updated = db_instance.update_note(note_id, {"title": "Updated Title"},
                                          expected_version=original_note['version'])
        assert updated is True

        retrieved = db_instance.get_note_by_id(note_id)
        assert retrieved["title"] == "Updated Title"
        assert retrieved["version"] == original_note['version'] + 1

    def test_add_keyword_and_undelete(self, db_instance: CharactersRAGDB):
        keyword_id = db_instance.add_keyword("TestKeyword")
        kw_v1 = db_instance.get_keyword_by_id(keyword_id)

        db_instance.soft_delete_keyword(keyword_id, expected_version=kw_v1['version'])
        assert db_instance.get_keyword_by_id(keyword_id) is None

        # Adding same keyword again should undelete it
        new_keyword_id = db_instance.add_keyword("TestKeyword")
        assert new_keyword_id == keyword_id

        retrieved = db_instance.get_keyword_by_id(keyword_id)
        assert not retrieved["deleted"]
        assert retrieved["version"] == 3  # 1(add) -> 2(delete) -> 3(undelete/update)

    def test_link_and_unlink_conversation_to_keyword(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card(_create_sample_card_data("LinkChar"))
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "LinkConv"})
        kw_id = db_instance.add_keyword("Linkable")

        assert db_instance.link_conversation_to_keyword(conv_id, kw_id) is True
        keywords = db_instance.get_keywords_for_conversation(conv_id)
        assert len(keywords) == 1
        assert keywords[0]["id"] == kw_id

        # Test idempotency of linking
        assert db_instance.link_conversation_to_keyword(conv_id, kw_id) is False

        assert db_instance.unlink_conversation_from_keyword(conv_id, kw_id) is True
        assert len(db_instance.get_keywords_for_conversation(conv_id)) == 0

        # Test idempotency of unlinking
        assert db_instance.unlink_conversation_from_keyword(conv_id, kw_id) is False


class TestSyncLog:
    def test_sync_log_entry_on_add_and_update_character(self, db_instance: CharactersRAGDB):
        initial_log_max_id = db_instance.get_latest_sync_log_change_id()
        card_data = _create_sample_card_data("SyncLogChar")
        card_id = db_instance.add_character_card(card_data)

        log_entries = db_instance.get_sync_log_entries(since_change_id=initial_log_max_id)
        create_entry = next((e for e in log_entries if e["entity"] == "character_cards" and e["operation"] == "create"),
                            None)
        assert create_entry is not None
        assert create_entry["entity_id"] == str(card_id)
        assert create_entry["payload"]["name"] == card_data["name"]

        # Test update
        latest_change_id_after_add = db_instance.get_latest_sync_log_change_id()
        db_instance.update_character_card(card_id, {"description": "Updated for Sync"}, expected_version=1)

        update_log_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id_after_add)
        update_entry = next(
            (e for e in update_log_entries if e["entity"] == "character_cards" and e["operation"] == "update"), None)
        assert update_entry is not None
        assert update_entry["payload"]["description"] == "Updated for Sync"
        assert update_entry["payload"]["version"] == 2

    def test_sync_log_on_soft_delete_character(self, db_instance: CharactersRAGDB):
        card_id = db_instance.add_character_card(_create_sample_card_data("SyncDeleteChar"))
        latest_change_id = db_instance.get_latest_sync_log_change_id()

        db_instance.soft_delete_character_card(card_id, expected_version=1)

        new_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id)
        delete_entry = next((e for e in new_entries if e["entity"] == "character_cards" and e["operation"] == "delete"),
                            None)
        assert delete_entry is not None
        assert delete_entry["entity_id"] == str(card_id)
        assert delete_entry["payload"]["deleted"] == 1  # Stored as integer
        assert delete_entry["payload"]["version"] == 2

    def test_sync_log_for_link_tables(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card(_create_sample_card_data("SyncLinkChar"))
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "SyncLinkConv"})
        kw_id = db_instance.add_keyword("SyncLinkable")
        latest_change_id = db_instance.get_latest_sync_log_change_id()

        db_instance.link_conversation_to_keyword(conv_id, kw_id)

        link_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id)
        link_entry = next(
            (e for e in link_entries if e["entity"] == "conversation_keywords" and e["operation"] == "create"), None)
        assert link_entry is not None
        assert link_entry["payload"]["conversation_id"] == conv_id
        assert link_entry["payload"]["keyword_id"] == kw_id

        # Test unlink
        latest_change_id_after_link = db_instance.get_latest_sync_log_change_id()
        db_instance.unlink_conversation_from_keyword(conv_id, kw_id)
        unlink_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id_after_link)
        unlink_entry = next(
            (e for e in unlink_entries if e["entity"] == "conversation_keywords" and e["operation"] == "delete"), None)
        assert unlink_entry is not None
        assert unlink_entry["entity_id"] == f"{conv_id}_{kw_id}"


class TestTransactions:
    def test_transaction_commit(self, db_instance: CharactersRAGDB):
        with db_instance.transaction() as conn:
            conn.execute("INSERT INTO character_cards (name, client_id) VALUES (?, ?)",
                         ("Trans1", db_instance.client_id))
            conn.execute("INSERT INTO character_cards (name, client_id) VALUES (?, ?)",
                         ("Trans2", db_instance.client_id))

        assert db_instance.get_character_card_by_name("Trans1") is not None
        assert db_instance.get_character_card_by_name("Trans2") is not None

    def test_transaction_rollback(self, db_instance: CharactersRAGDB):
        initial_count = len(db_instance.list_character_cards())
        with pytest.raises(sqlite3.IntegrityError):
            with db_instance.transaction() as conn:
                conn.execute("INSERT INTO character_cards (name, client_id) VALUES (?, ?)",
                             ("TransRollback", db_instance.client_id))
                # This will fail due to duplicate name, causing a rollback
                conn.execute("INSERT INTO character_cards (name, client_id) VALUES (?, ?)",
                             ("TransRollback", db_instance.client_id))

        assert len(db_instance.list_character_cards()) == initial_count
        assert db_instance.get_character_card_by_name("TransRollback") is None