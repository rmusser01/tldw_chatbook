# test_chachanotes_db.py
#
#
# Imports
import shutil

import pytest
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

#
# Third-Party Imports
#
# Local Imports
# --- UPDATED IMPORT PATH ---
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
    ConflictError,
)
from Tests.ChaChaNotesDB.historical_bootstrap import (
    chachanotes_db_at_version,
    open_current_chachanotes_from_legacy,
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
def db_instance(db_path, client_id, chachanotes_template_db):
    """Creates a DB instance for each test from the session template (task-1460)."""
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
        shutil.copyfile(chachanotes_template_db, current_db_path)
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
        "character_id": sample_card["id"],
        "title": "Conversation From Fixture",
    }
    conv_id = db_instance.add_conversation(conv_data)
    return db_instance.get_conversation_by_id(conv_id)


# --- Helper Functions ---
def get_current_utc_timestamp_iso():
    """Returns the current UTC time in ISO 8601 format, as used by the DB."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


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
        "client_id": client_id_override,
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
        version_row = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (db._SCHEMA_NAME,),
        ).fetchone()
        assert version_row is not None
        assert version_row["version"] == db._CURRENT_SCHEMA_VERSION
        db.close_connection()

    def test_in_memory_db_initialization(self, client_id):
        db = CharactersRAGDB(":memory:", client_id)
        assert db.is_memory_db
        assert db.client_id == client_id
        conn = db.get_connection()
        version_row = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (db._SCHEMA_NAME,),
        ).fetchone()
        assert version_row is not None
        assert version_row["version"] == db._CURRENT_SCHEMA_VERSION
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
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (newer_version, CharactersRAGDB._SCHEMA_NAME),
        )
        conn.commit()
        db.close_connection()

        expected_message_part = f"version \\({newer_version}\\) is newer than supported by code \\({CharactersRAGDB._CURRENT_SCHEMA_VERSION}\\)"
        with pytest.raises(CharactersRAGDBError, match=expected_message_part):
            CharactersRAGDB(db_path, client_id)

    def test_fresh_db_creates_conversations_system_prompt_column(
        self, db_path, client_id
    ):
        db = CharactersRAGDB(db_path, client_id)
        conn = db.get_connection()

        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
        }
        assert "system_prompt" in columns

        version_row = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (db._SCHEMA_NAME,),
        ).fetchone()
        assert version_row["version"] == db._CURRENT_SCHEMA_VERSION
        db.close_connection()

    def test_conversations_migrate_from_v17_to_v18_adds_system_prompt_column(
        self, db_path, client_id
    ):
        # Build a genuinely v17-shaped DB: the production migration chain
        # itself, run under a patched _CURRENT_SCHEMA_VERSION, stops and
        # stamps at 17 (task-16840; replaces the retired rollback registry,
        # which had to fake this state by dropping artifacts from a
        # current-version DB and could never carry the real v17 triggers).
        with chachanotes_db_at_version(db_path, 17, client_id=client_id) as db:
            conn = db.get_connection()

            # Guard the replay preconditions: the column the V17->V18
            # migration must add is absent, the conversations sync triggers
            # EXIST in their real pre-V18 form (none references
            # system_prompt) — so replay exercises the migration's genuine
            # redefine-live-triggers path, which the registry-era fixture
            # could not (it had to drop the triggers to drop the column) —
            # and no later migration's table exists (note_folders was the
            # artifact that broke the registry-era fixture in task-15765).
            columns_before = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            assert "system_prompt" not in columns_before
            trigger_sql = {
                row["name"]: row["sql"]
                for row in conn.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
                    "AND name LIKE 'conversations_sync_%'"
                ).fetchall()
            }
            assert set(trigger_sql) == {
                "conversations_sync_create",
                "conversations_sync_update",
                "conversations_sync_delete",
                "conversations_sync_undelete",
            }
            assert all(
                "system_prompt" not in sql for sql in trigger_sql.values()
            )
            table_names = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            assert "note_folders" not in table_names
            assert "note_folder_memberships" not in table_names
            version_before = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (db._SCHEMA_NAME,),
            ).fetchone()
            assert version_before["version"] == 17

        migrated = open_current_chachanotes_from_legacy(
            db_path, client_id=client_id
        )
        migrated_conn = migrated.get_connection()

        version_row = migrated_conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (migrated._SCHEMA_NAME,),
        ).fetchone()
        assert version_row["version"] == migrated._CURRENT_SCHEMA_VERSION

        columns = {
            row["name"]
            for row in migrated_conn.execute(
                "PRAGMA table_info(conversations)"
            ).fetchall()
        }
        assert "system_prompt" in columns

        # The redefined sync trigger must fire (and include the new column)
        # on a system_prompt-only update.
        char_id = migrated.add_character_card(
            _create_sample_card_data("MigrationCheck")
        )
        conv_id = migrated.add_conversation(
            {"character_id": char_id, "title": "Migration check"}
        )
        # task-19564 replaced the row-count proxy this used to assert. The
        # v45 retention triggers drop superseded `sync_log` versions, so the
        # total no longer grows by one -- but "the trigger fired, with the new
        # column in its payload" is what the test is actually for, and the
        # frontier row states that directly instead of by arithmetic.
        current = migrated.get_conversation_by_id(conv_id)
        migrated.update_conversation(
            conv_id,
            {"system_prompt": "Migrated prompt."},
            expected_version=current["version"],
        )
        latest_entry = migrated_conn.execute(
            "SELECT operation, version, payload FROM sync_log "
            "WHERE entity = 'conversations' AND entity_id = ? "
            "ORDER BY change_id DESC LIMIT 1",
            (conv_id,),
        ).fetchone()
        assert latest_entry["operation"] == "update"
        assert latest_entry["version"] == current["version"] + 1
        assert (
            json.loads(latest_entry["payload"])["system_prompt"] == "Migrated prompt."
        )
        assert (
            migrated.get_conversation_by_id(conv_id)["system_prompt"]
            == "Migrated prompt."
        )
        migrated.close_connection()


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
        assert retrieved["alternate_greetings"] == json.loads(
            card_data["alternate_greetings"]
        )
        assert retrieved["client_id"] == db_instance.client_id
        assert retrieved["version"] == 1
        assert not retrieved["deleted"]

    def test_add_character_card_with_missing_name_raises_error(
        self, db_instance: CharactersRAGDB
    ):
        card_data = _create_sample_card_data("MissingName")
        del card_data["name"]
        with pytest.raises(InputError, match="Required field 'name' is missing"):
            db_instance.add_character_card(card_data)

    def test_add_character_card_with_duplicate_name_raises_error(
        self, db_instance: CharactersRAGDB
    ):
        card_data = _create_sample_card_data("Duplicate")
        db_instance.add_character_card(card_data)
        with pytest.raises(
            ConflictError,
            match=f"Character card with name '{card_data['name']}' already exists",
        ):
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
        assert initial_cards[0]["name"] == "Default Assistant"

        card_data1 = _create_sample_card_data("List1")
        card_data2 = _create_sample_card_data("List2")
        db_instance.add_character_card(card_data1)
        db_instance.add_character_card(card_data2)

        # The list should now contain 3 cards (1 default + 2 new)
        cards = db_instance.list_character_cards()
        assert len(cards) == 3

        # You can still sort and check your added cards if you filter out the default one.
        added_card_names = {
            c["name"] for c in cards if c["name"] != "Default Assistant"
        }
        assert added_card_names == {card_data1["name"], card_data2["name"]}

    def test_list_character_cards_excludes_image_by_default(
        self, db_instance: CharactersRAGDB
    ):
        """task-15474: list/picker reads must not fetch the `image` BLOB
        column by default -- it drags up to `limit` raw images through
        SQLite/Python for callers that only render name/description rows."""
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-png-bytes" * 100
        card_id = db_instance.add_character_card(
            {"name": "Imaged Default List", "image": image_bytes}
        )
        cards = db_instance.list_character_cards(limit=100)
        assert cards
        for card in cards:
            assert "image" not in card
        assert any(c["id"] == card_id for c in cards)

    def test_list_character_cards_include_image_true_round_trips(
        self, db_instance: CharactersRAGDB
    ):
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-png-bytes" * 100
        card_id = db_instance.add_character_card(
            {"name": "Imaged Include List", "image": image_bytes}
        )
        cards = db_instance.list_character_cards(limit=100, include_image=True)
        card = next(c for c in cards if c["id"] == card_id)
        assert card["image"] == image_bytes

    def test_update_character_card(
        self, db_instance: CharactersRAGDB, sample_card: dict
    ):
        update_payload = {"description": "Updated Description"}
        updated = db_instance.update_character_card(
            sample_card["id"], update_payload, expected_version=sample_card["version"]
        )
        assert updated is True

        retrieved = db_instance.get_character_card_by_id(sample_card["id"])
        assert retrieved["description"] == "Updated Description"
        assert retrieved["version"] == sample_card["version"] + 1

    def test_update_character_card_with_version_conflict_raises_error(
        self, db_instance: CharactersRAGDB
    ):
        card_id = db_instance.add_character_card(
            _create_sample_card_data("VersionConflict")
        )

        # Simulate another client's update, bumping DB version to 2
        db_instance.update_character_card(
            card_id, {"description": "First update"}, expected_version=1
        )

        # Client tries to update with old expected_version=1
        update_payload = {"description": "Conflict Update"}
        expected_error_regex = r"version mismatch \(db has 2, client expected 1\)"
        with pytest.raises(ConflictError, match=expected_error_regex):
            db_instance.update_character_card(
                card_id, update_payload, expected_version=1
            )

    def test_update_character_card_not_found_raises_error(
        self, db_instance: CharactersRAGDB
    ):
        with pytest.raises(ConflictError, match="Record not found in character_cards"):
            db_instance.update_character_card(
                999, {"description": "Not Found"}, expected_version=1
            )

    def test_soft_delete_character_card(
        self, db_instance: CharactersRAGDB, sample_card: dict
    ):
        deleted = db_instance.soft_delete_character_card(
            sample_card["id"], expected_version=sample_card["version"]
        )
        assert deleted is True
        assert db_instance.get_character_card_by_id(sample_card["id"]) is None

    def test_soft_delete_is_idempotent(self, db_instance: CharactersRAGDB):
        card_id = db_instance.add_character_card(
            _create_sample_card_data("IdempotentDelete")
        )
        db_instance.soft_delete_character_card(card_id, expected_version=1)

        # Calling delete again on an already deleted record should succeed
        assert (
            db_instance.soft_delete_character_card(card_id, expected_version=1) is True
        )
        # Verify version didn't change again
        conn = db_instance.get_connection()
        raw_record = conn.execute(
            "SELECT version FROM character_cards WHERE id = ?", (card_id,)
        ).fetchone()
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
        db_instance.soft_delete_character_card(
            card2["id"], expected_version=card2["version"]
        )

        results_after_delete = db_instance.search_character_cards("ZYX")
        assert len(results_after_delete) == 1
        assert results_after_delete[0]["name"] == card1_data["name"]

    @pytest.mark.parametrize(
        "field_to_remove, expected_error, error_match",
        [
            ("name", InputError, "Required field 'name' is missing"),
            # Assuming you add a required 'creator' field later
            # ("creator", InputError, "Required field 'creator' is missing"),
        ],
    )
    def test_add_card_missing_required_fields(
        self, db_instance, field_to_remove, expected_error, error_match
    ):
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
        conv_data = {
            "id": str(uuid.uuid4()),
            "character_id": char_id,
            "title": "Test Conversation",
        }
        conv_id = db_instance.add_conversation(conv_data)
        assert conv_id == conv_data["id"]

        retrieved = db_instance.get_conversation_by_id(conv_id)
        assert retrieved["title"] == "Test Conversation"
        assert retrieved["character_id"] == char_id
        assert retrieved["version"] == 1
        assert retrieved["client_id"] == db_instance.client_id

    def test_add_message_and_get_for_conversation(
        self, db_instance: CharactersRAGDB, char_id
    ):
        conv_id = db_instance.add_conversation(
            {"character_id": char_id, "title": "MsgConv"}
        )
        msg1_id = db_instance.add_message(
            {
                "conversation_id": conv_id,
                "sender": "user",
                "content": "First",
                "timestamp": "2023-01-01T10:00:00Z",
            }
        )
        msg2_id = db_instance.add_message(
            {
                "conversation_id": conv_id,
                "sender": "ai",
                "content": "Second",
                "timestamp": "2023-01-01T10:01:00Z",
            }
        )

        messages_asc = db_instance.get_messages_for_conversation(
            conv_id, order_by_timestamp="ASC"
        )
        assert len(messages_asc) == 2
        assert messages_asc[0]["id"] == msg1_id
        assert messages_asc[1]["id"] == msg2_id

        messages_desc = db_instance.get_messages_for_conversation(
            conv_id, order_by_timestamp="DESC"
        )
        assert len(messages_desc) == 2
        assert messages_desc[0]["id"] == msg2_id
        assert messages_desc[1]["id"] == msg1_id

    def test_update_conversation_and_fts(
        self, db_instance: CharactersRAGDB, char_id: int
    ):
        initial_title = "AlphaTitleOne"
        conv_id = db_instance.add_conversation(
            {"character_id": char_id, "title": initial_title}
        )
        original_conv = db_instance.get_conversation_by_id(conv_id)

        # Verify FTS state before update
        assert len(db_instance.search_conversations_by_title(initial_title)) == 1

        # Perform update
        updated_title = "BetaTitleTwo"
        db_instance.update_conversation(
            conv_id, {"title": updated_title}, expected_version=original_conv["version"]
        )

        # Verify FTS state after update
        assert len(db_instance.search_conversations_by_title(updated_title)) == 1
        assert len(db_instance.search_conversations_by_title(initial_title)) == 0, (
            "FTS should not find the old title"
        )

    def test_add_conversation_defaults_system_prompt_to_none(
        self, db_instance: CharactersRAGDB, char_id
    ):
        conv_id = db_instance.add_conversation(
            {"character_id": char_id, "title": "NoSystemPrompt"}
        )

        retrieved = db_instance.get_conversation_by_id(conv_id)
        assert retrieved["system_prompt"] is None

    def test_add_conversation_persists_system_prompt(
        self, db_instance: CharactersRAGDB, char_id
    ):
        conv_id = db_instance.add_conversation(
            {
                "character_id": char_id,
                "title": "WithSystemPrompt",
                "system_prompt": "  Be concise.  ",
            }
        )

        retrieved = db_instance.get_conversation_by_id(conv_id)
        assert retrieved["system_prompt"] == "Be concise."

    def test_update_conversation_sets_system_prompt_and_bumps_version(
        self, db_instance: CharactersRAGDB, char_id
    ):
        conv_id = db_instance.add_conversation(
            {"character_id": char_id, "title": "UpdateSystemPrompt"}
        )
        original = db_instance.get_conversation_by_id(conv_id)

        result = db_instance.update_conversation(
            conv_id,
            {"system_prompt": "Speak like a pirate."},
            expected_version=original["version"],
        )

        assert result is True
        updated = db_instance.get_conversation_by_id(conv_id)
        assert updated["system_prompt"] == "Speak like a pirate."
        assert updated["version"] == original["version"] + 1

    def test_update_conversation_clears_system_prompt_with_none(
        self, db_instance: CharactersRAGDB, char_id
    ):
        conv_id = db_instance.add_conversation(
            {
                "character_id": char_id,
                "title": "ClearSystemPrompt",
                "system_prompt": "Initial prompt.",
            }
        )
        original = db_instance.get_conversation_by_id(conv_id)

        db_instance.update_conversation(
            conv_id,
            {"system_prompt": None},
            expected_version=original["version"],
        )

        updated = db_instance.get_conversation_by_id(conv_id)
        assert updated["system_prompt"] is None

    def test_update_conversation_preserves_system_prompt_when_untouched(
        self, db_instance: CharactersRAGDB, char_id
    ):
        conv_id = db_instance.add_conversation(
            {
                "character_id": char_id,
                "title": "PreserveSystemPrompt",
                "system_prompt": "Keep me around.",
            }
        )
        original = db_instance.get_conversation_by_id(conv_id)

        db_instance.update_conversation(
            conv_id,
            {"title": "PreserveSystemPromptRenamed"},
            expected_version=original["version"],
        )

        updated = db_instance.get_conversation_by_id(conv_id)
        assert updated["title"] == "PreserveSystemPromptRenamed"
        assert updated["system_prompt"] == "Keep me around."

    def test_soft_delete_conversation_and_fts(
        self, db_instance: CharactersRAGDB, char_id
    ):
        conv_title_for_delete_test = "DeleteConvForFTS"
        conv_id = db_instance.add_conversation(
            {"character_id": char_id, "title": conv_title_for_delete_test}
        )
        original_conv = db_instance.get_conversation_by_id(conv_id)

        assert (
            len(db_instance.search_conversations_by_title(conv_title_for_delete_test))
            == 1
        )

        db_instance.soft_delete_conversation(
            conv_id, expected_version=original_conv["version"]
        )

        assert db_instance.get_conversation_by_id(conv_id) is None
        assert (
            len(db_instance.search_conversations_by_title(conv_title_for_delete_test))
            == 0
        ), "FTS should not find soft-deleted conversation"

    def test_search_messages_by_content(self, db_instance: CharactersRAGDB, char_id):
        conv_id = db_instance.add_conversation(
            {"character_id": char_id, "title": "MessageSearchConv"}
        )
        msg1_data = {
            "id": str(uuid.uuid4()),
            "conversation_id": conv_id,
            "sender": "user",
            "content": "UniqueMessageContentAlpha",
        }
        db_instance.add_message(msg1_data)

        results = db_instance.search_messages_by_content("UniqueMessageContentAlpha")
        assert len(results) == 1
        assert results[0]["id"] == msg1_data["id"]

    def test_search_messages_by_content_unscoped_excludes_deleted_conversations(
        self, db_instance: CharactersRAGDB, char_id
    ):
        """task-19567 A: the shape that is one caller away from a leak.

        Soft-deleting a conversation leaves its messages at `deleted = 0`, and
        this method filtered only `m.deleted = 0` without joining
        `conversations` -- while its sibling `search_conversations_by_content`
        filtered both. It was not exploitable at the time only because the one
        live caller always passed a `conversation_id` obtained from that
        already-filtered sibling. Called UNSCOPED, as any new caller would,
        it returned the deleted conversation's messages.
        """
        needle = "UniqueDeletedConversationBodyOmega"
        kept_conv = db_instance.add_conversation(
            {"character_id": char_id, "title": "KeptConv"}
        )
        dropped_conv = db_instance.add_conversation(
            {"character_id": char_id, "title": "DroppedConv"}
        )
        kept_message = db_instance.add_message(
            {"conversation_id": kept_conv, "sender": "user", "content": needle}
        )
        db_instance.add_message(
            {"conversation_id": dropped_conv, "sender": "user", "content": needle}
        )
        assert len(db_instance.search_messages_by_content(needle)) == 2

        db_instance.soft_delete_conversation(dropped_conv, expected_version=1)

        unscoped = db_instance.search_messages_by_content(needle)
        assert [row["id"] for row in unscoped] == [kept_message]
        # ... and it now agrees with the sibling it used to diverge from.
        assert [
            row["id"] for row in db_instance.search_conversations_by_content(needle)
        ] == [kept_conv]
        # Scoping to the deleted conversation must not reopen the hole.
        assert (
            db_instance.search_messages_by_content(
                needle, conversation_id=dropped_conv
            )
            == []
        )

    def test_update_message_usage_local_leaves_version_and_last_modified_untouched(
        self, db_instance: CharactersRAGDB, char_id
    ):
        """Qodo round (Finding 4): a usage-only local write must not bump
        `version`/`last_modified` -- those two columns are exactly what the
        `messages_sync_update` trigger's WHEN clause watches, so bumping
        them on a write whose payload can never carry `usage_json` (the
        trigger's payload only ever includes syncable columns) would
        enqueue a cross-device `sync_log` row for a column that is
        local-only by design.
        """
        conv_id = db_instance.add_conversation(
            {"character_id": char_id, "title": "UsageLocalConv"}
        )
        msg_id = db_instance.add_message(
            {
                "conversation_id": conv_id,
                "sender": "assistant",
                "content": "the answer",
            }
        )
        before = db_instance.get_message_by_id(msg_id)
        assert before["usage_json"] is None
        latest_change_id = db_instance.get_latest_sync_log_change_id()

        result = db_instance.update_message_usage_local(
            msg_id, '{"uncached_input": 10, "output": 5}'
        )

        assert result is True
        after = db_instance.get_message_by_id(msg_id)
        assert after["usage_json"] == '{"uncached_input": 10, "output": 5}'
        assert after["version"] == before["version"]
        assert after["last_modified"] == before["last_modified"]

        new_entries = db_instance.get_sync_log_entries(
            since_change_id=latest_change_id, entity_type="messages"
        )
        assert new_entries == [], (
            "a usage-only local write must not enqueue a sync_log row "
            "for the messages entity"
        )

    def test_update_message_usage_local_unknown_id_returns_false(
        self, db_instance: CharactersRAGDB
    ):
        assert db_instance.update_message_usage_local("missing-id", "{}") is False

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


class TestGetAllConversationIds:
    """``get_all_conversation_ids`` -- the truncation-proof id source for
    Library chatbook export (see ``Library/library_export_scope.py``).

    Mirrors the WHERE clause ``search_conversations_page`` builds for the
    Library's conversations snapshot fetch (``ChatConversationService.
    list_conversations`` with ``scope_type='all'``, spanning global- and
    workspace-scoped rows): ``client_id = ? AND deleted = 0``, but with no
    page cap.
    """

    def test_returns_all_non_deleted_conversation_ids(
        self, db_instance: CharactersRAGDB
    ):
        conv_id_1 = db_instance.add_conversation({"title": "Conv 1"})
        conv_id_2 = db_instance.add_conversation({"title": "Conv 2"})
        conv_to_delete = db_instance.add_conversation({"title": "Conv to delete"})
        deleted_record = db_instance.get_conversation_by_id(conv_to_delete)
        db_instance.soft_delete_conversation(
            conv_to_delete, expected_version=deleted_record["version"]
        )

        ids = db_instance.get_all_conversation_ids()

        assert set(ids) == {conv_id_1, conv_id_2}

    def test_includes_workspace_scoped_conversations(
        self, db_instance: CharactersRAGDB
    ):
        """Console chats persisted inside a workspace session are workspace-scoped
        and must be exportable, matching the Library's all-scope listing (task-179)."""
        global_id = db_instance.add_conversation({"title": "Global conv"})
        workspace_id = db_instance.add_conversation(
            {
                "title": "Workspace conv",
                "scope_type": "workspace",
                "workspace_id": "ws-1",
            }
        )

        ids = db_instance.get_all_conversation_ids()

        assert set(ids) == {global_id, workspace_id}

    def test_excludes_conversations_from_a_different_client_id(
        self, db_instance: CharactersRAGDB
    ):
        own_id = db_instance.add_conversation({"title": "Own conv"})
        db_instance.add_conversation(
            {"title": "Other client conv", "client_id": "some-other-client"}
        )

        ids = db_instance.get_all_conversation_ids()

        assert ids == [own_id]

    def test_returns_every_row_beyond_a_50_row_page_cap(
        self, db_instance: CharactersRAGDB
    ):
        """The Library conversations snapshot caps at 50 rows -- this DB method must not."""
        seeded_ids = [
            db_instance.add_conversation({"title": f"Conv {i}"}) for i in range(55)
        ]

        ids = db_instance.get_all_conversation_ids()

        assert set(ids) == set(seeded_ids)
        assert len(ids) == 55

    def test_empty_db_returns_empty_list(self, db_instance: CharactersRAGDB):
        assert db_instance.get_all_conversation_ids() == []


class TestNotesAndKeywords:
    def test_keyword_and_link_cursor_follow_caller_transaction(
        self, db_instance: CharactersRAGDB
    ):
        note_id = db_instance.add_note("Cursor note", "content")
        with pytest.raises(RuntimeError, match="rollback"):
            with db_instance.transaction() as cursor:
                keyword_id = db_instance.add_keyword("Cursor keyword", cursor=cursor)
                db_instance.link_note_to_keyword(note_id, keyword_id, cursor=cursor)
                raise RuntimeError("rollback")

        assert db_instance.get_keyword_by_text("Cursor keyword") is None
        assert db_instance.get_keywords_for_note(note_id) == []

        with db_instance.transaction() as cursor:
            keyword_id = db_instance.add_keyword("Cursor keyword", cursor=cursor)
            assert db_instance.link_note_to_keyword(
                note_id, keyword_id, cursor=cursor
            )

        assert db_instance.get_keyword_by_id(keyword_id)["keyword"] == "Cursor keyword"
        assert [row["id"] for row in db_instance.get_keywords_for_note(note_id)] == [
            keyword_id
        ]

    def test_add_and_update_note(self, db_instance: CharactersRAGDB):
        note_id = db_instance.add_note("Original Title", "Original Content")
        assert isinstance(note_id, str)

        original_note = db_instance.get_note_by_id(note_id)
        updated = db_instance.update_note(
            note_id,
            {"title": "Updated Title"},
            expected_version=original_note["version"],
        )
        assert updated is True

        retrieved = db_instance.get_note_by_id(note_id)
        assert retrieved["title"] == "Updated Title"
        assert retrieved["version"] == original_note["version"] + 1

    def test_add_keyword_and_undelete(self, db_instance: CharactersRAGDB):
        keyword_id = db_instance.add_keyword("TestKeyword")
        kw_v1 = db_instance.get_keyword_by_id(keyword_id)

        db_instance.soft_delete_keyword(keyword_id, expected_version=kw_v1["version"])
        assert db_instance.get_keyword_by_id(keyword_id) is None

        # Adding same keyword again should undelete it
        new_keyword_id = db_instance.add_keyword("TestKeyword")
        assert new_keyword_id == keyword_id

        retrieved = db_instance.get_keyword_by_id(keyword_id)
        assert not retrieved["deleted"]
        assert retrieved["version"] == 3  # 1(add) -> 2(delete) -> 3(undelete/update)

    def test_link_and_unlink_conversation_to_keyword(
        self, db_instance: CharactersRAGDB
    ):
        char_id = db_instance.add_character_card(_create_sample_card_data("LinkChar"))
        conv_id = db_instance.add_conversation(
            {"character_id": char_id, "title": "LinkConv"}
        )
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


class TestKeywordCollections:
    """TASK-864: ``keyword_collections`` was omitted from
    ``sql_validation.VALID_TABLES['chachanotes']``, so ``update_keyword_collection``
    (and ``soft_delete_keyword_collection``) raised ``ValueError`` for every
    caller, unconditionally -- this feature had no test coverage at all until
    this class, which is exactly how the omission went unnoticed.
    """

    def test_add_then_update_keyword_collection(self, db_instance: CharactersRAGDB):
        collection_id = db_instance.add_keyword_collection("Coll A")
        assert isinstance(collection_id, int)

        updated = db_instance.update_keyword_collection(
            collection_id, {"name": "Coll B"}, expected_version=1
        )
        assert updated is True

        collections = db_instance.list_keyword_collections()
        names = {c["name"] for c in collections}
        assert "Coll B" in names
        assert "Coll A" not in names

    def test_add_then_soft_delete_keyword_collection(
        self, db_instance: CharactersRAGDB
    ):
        collection_id = db_instance.add_keyword_collection("Coll To Delete")

        deleted = db_instance.soft_delete_keyword_collection(
            collection_id, expected_version=1
        )
        assert deleted is True

        names = {c["name"] for c in db_instance.list_keyword_collections()}
        assert "Coll To Delete" not in names

    def test_collection_cursor_follows_caller_transaction_and_default_still_commits(
        self, db_instance: CharactersRAGDB
    ):
        keyword_id = db_instance.add_keyword("Collection keyword")
        with pytest.raises(RuntimeError, match="rollback"):
            with db_instance.transaction() as cursor:
                collection_id = db_instance.add_keyword_collection(
                    "Rolled back collection", cursor=cursor
                )
                db_instance.link_collection_to_keyword(
                    collection_id, keyword_id, cursor=cursor
                )
                raise RuntimeError("rollback")

        assert db_instance.get_keyword_collection_by_name("Rolled back collection") is None

        collection_id = db_instance.add_keyword_collection("Committed collection")
        assert db_instance.link_collection_to_keyword(collection_id, keyword_id)
        assert [row["id"] for row in db_instance.get_keywords_for_collection(collection_id)] == [
            keyword_id
        ]


class TestGetAllNoteIds:
    """``get_all_note_ids`` -- the truncation-proof id source for Library
    chatbook export (see ``Library/library_export_scope.py``).

    Mirrors ``list_notes``'/``count_notes``' visibility: ``deleted = 0``
    only -- notes are not ``client_id``-scoped the way conversations are
    (``_list_generic_items`` never filters on ``client_id``) -- but with no
    page cap.
    """

    def test_returns_all_non_deleted_note_ids(self, db_instance: CharactersRAGDB):
        note_id_1 = db_instance.add_note("Note 1", "Content 1")
        note_id_2 = db_instance.add_note("Note 2", "Content 2")
        note_to_delete = db_instance.add_note("Note to delete", "Content 3")
        deleted_record = db_instance.get_note_by_id(note_to_delete)
        db_instance.soft_delete_note(
            note_to_delete, expected_version=deleted_record["version"]
        )

        ids = db_instance.get_all_note_ids()

        assert set(ids) == {note_id_1, note_id_2}

    def test_returns_every_row_beyond_a_100_row_page_cap(
        self, db_instance: CharactersRAGDB
    ):
        """The Library notes snapshot caps at 100 rows -- this DB method must not."""
        seeded_ids = [
            db_instance.add_note(f"Note {i}", f"Content {i}") for i in range(105)
        ]

        ids = db_instance.get_all_note_ids()

        assert set(ids) == set(seeded_ids)
        assert len(ids) == 105

    def test_empty_db_returns_empty_list(self, db_instance: CharactersRAGDB):
        assert db_instance.get_all_note_ids() == []


class TestSyncLog:
    def test_sync_log_entry_on_add_and_update_character(
        self, db_instance: CharactersRAGDB
    ):
        initial_log_max_id = db_instance.get_latest_sync_log_change_id()
        card_data = _create_sample_card_data("SyncLogChar")
        card_id = db_instance.add_character_card(card_data)

        log_entries = db_instance.get_sync_log_entries(
            since_change_id=initial_log_max_id
        )
        create_entry = next(
            (
                e
                for e in log_entries
                if e["entity"] == "character_cards" and e["operation"] == "create"
            ),
            None,
        )
        assert create_entry is not None
        assert create_entry["entity_id"] == str(card_id)
        assert create_entry["payload"]["name"] == card_data["name"]

        # Test update
        latest_change_id_after_add = db_instance.get_latest_sync_log_change_id()
        db_instance.update_character_card(
            card_id, {"description": "Updated for Sync"}, expected_version=1
        )

        update_log_entries = db_instance.get_sync_log_entries(
            since_change_id=latest_change_id_after_add
        )
        update_entry = next(
            (
                e
                for e in update_log_entries
                if e["entity"] == "character_cards" and e["operation"] == "update"
            ),
            None,
        )
        assert update_entry is not None
        assert update_entry["payload"]["description"] == "Updated for Sync"
        assert update_entry["payload"]["version"] == 2

    def test_sync_log_on_soft_delete_character(self, db_instance: CharactersRAGDB):
        card_id = db_instance.add_character_card(
            _create_sample_card_data("SyncDeleteChar")
        )
        latest_change_id = db_instance.get_latest_sync_log_change_id()

        db_instance.soft_delete_character_card(card_id, expected_version=1)

        new_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id)
        delete_entry = next(
            (
                e
                for e in new_entries
                if e["entity"] == "character_cards" and e["operation"] == "delete"
            ),
            None,
        )
        assert delete_entry is not None
        assert delete_entry["entity_id"] == str(card_id)
        assert delete_entry["payload"]["deleted"] == 1  # Stored as integer
        assert delete_entry["payload"]["version"] == 2

    def test_sync_log_for_link_tables(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card(
            _create_sample_card_data("SyncLinkChar")
        )
        conv_id = db_instance.add_conversation(
            {"character_id": char_id, "title": "SyncLinkConv"}
        )
        kw_id = db_instance.add_keyword("SyncLinkable")
        latest_change_id = db_instance.get_latest_sync_log_change_id()

        db_instance.link_conversation_to_keyword(conv_id, kw_id)

        link_entries = db_instance.get_sync_log_entries(
            since_change_id=latest_change_id
        )
        link_entry = next(
            (
                e
                for e in link_entries
                if e["entity"] == "conversation_keywords" and e["operation"] == "create"
            ),
            None,
        )
        assert link_entry is not None
        assert link_entry["payload"]["conversation_id"] == conv_id
        assert link_entry["payload"]["keyword_id"] == kw_id

        # Test unlink
        latest_change_id_after_link = db_instance.get_latest_sync_log_change_id()
        db_instance.unlink_conversation_from_keyword(conv_id, kw_id)
        unlink_entries = db_instance.get_sync_log_entries(
            since_change_id=latest_change_id_after_link
        )
        unlink_entry = next(
            (
                e
                for e in unlink_entries
                if e["entity"] == "conversation_keywords" and e["operation"] == "delete"
            ),
            None,
        )
        assert unlink_entry is not None
        assert unlink_entry["entity_id"] == f"{conv_id}_{kw_id}"


class TestTransactions:
    def test_transaction_commit(self, db_instance: CharactersRAGDB):
        with db_instance.transaction() as conn:
            conn.execute(
                "INSERT INTO character_cards (name, client_id) VALUES (?, ?)",
                ("Trans1", db_instance.client_id),
            )
            conn.execute(
                "INSERT INTO character_cards (name, client_id) VALUES (?, ?)",
                ("Trans2", db_instance.client_id),
            )

        assert db_instance.get_character_card_by_name("Trans1") is not None
        assert db_instance.get_character_card_by_name("Trans2") is not None

    def test_transaction_rollback(self, db_instance: CharactersRAGDB):
        initial_count = len(db_instance.list_character_cards())
        with pytest.raises(sqlite3.IntegrityError):
            with db_instance.transaction() as conn:
                conn.execute(
                    "INSERT INTO character_cards (name, client_id) VALUES (?, ?)",
                    ("TransRollback", db_instance.client_id),
                )
                # This will fail due to duplicate name, causing a rollback
                conn.execute(
                    "INSERT INTO character_cards (name, client_id) VALUES (?, ?)",
                    ("TransRollback", db_instance.client_id),
                )

        assert len(db_instance.list_character_cards()) == initial_count
        assert db_instance.get_character_card_by_name("TransRollback") is None


def _make_conversation_with_message(db):
    conv_id = db.add_conversation({"title": "att", "client_id": db.client_id})
    msg_id = db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "hello",
            "client_id": db.client_id,
        }
    )
    return conv_id, msg_id


class TestMessageAttachmentsTable:
    def test_schema_v19_creates_empty_attachments_table(self, db_instance):
        with db_instance.transaction() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_attachments'"
            )
            assert cursor.fetchone() is not None
            cursor.execute("SELECT COUNT(*) FROM message_attachments")
            assert cursor.fetchone()[0] == 0

    def test_set_and_batch_get_attachments(self, db_instance):
        _conv, msg_id = _make_conversation_with_message(db_instance)
        rows = [
            {
                "position": 1,
                "data": b"img-1",
                "mime_type": "image/png",
                "display_name": "a.png",
            },
            {
                "position": 2,
                "data": b"img-2",
                "mime_type": "image/jpeg",
                "display_name": "b.jpg",
            },
        ]
        db_instance.set_message_attachments(msg_id, rows)

        fetched = db_instance.get_attachments_for_messages([msg_id])
        assert list(fetched.keys()) == [msg_id]
        assert [r["position"] for r in fetched[msg_id]] == [1, 2]
        assert fetched[msg_id][0]["data"] == b"img-1"
        assert fetched[msg_id][1]["display_name"] == "b.jpg"

        # Replace semantics: a second set replaces, not appends.
        db_instance.set_message_attachments(
            msg_id,
            [
                {
                    "position": 1,
                    "data": b"img-3",
                    "mime_type": "image/png",
                    "display_name": "c.png",
                }
            ],
        )
        fetched = db_instance.get_attachments_for_messages([msg_id])
        assert [r["display_name"] for r in fetched[msg_id]] == ["c.png"]

    def test_position_zero_rejected(self, db_instance):
        _conv, msg_id = _make_conversation_with_message(db_instance)
        import sqlite3 as _sqlite3

        import pytest as _pytest

        with _pytest.raises((ValueError, _sqlite3.IntegrityError, Exception)):
            db_instance.set_message_attachments(
                msg_id,
                [
                    {
                        "position": 0,
                        "data": b"x",
                        "mime_type": "image/png",
                        "display_name": "z.png",
                    }
                ],
            )

    def test_hard_delete_cascades_attachments(self, db_instance):
        _conv, msg_id = _make_conversation_with_message(db_instance)
        db_instance.set_message_attachments(
            msg_id,
            [
                {
                    "position": 1,
                    "data": b"img",
                    "mime_type": "image/png",
                    "display_name": "a.png",
                }
            ],
        )
        with db_instance.transaction() as cursor:
            cursor.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
            cursor.execute(
                "SELECT COUNT(*) FROM message_attachments WHERE message_id = ?",
                (msg_id,),
            )
            assert cursor.fetchone()[0] == 0

    def test_get_attachments_empty_and_unknown_ids(self, db_instance):
        assert db_instance.get_attachments_for_messages([]) == {}
        assert db_instance.get_attachments_for_messages(["nope"]) == {}
