# tests/test_media_db_v2.py
# Description: Unit tests for SQLite database operations, including CRUD, transactions, and sync log management.
# This version is self-contained and does not require a conftest.py file.
#
# Standard Library Imports:
import json
import os
import pytest
import shutil
import sys
import time
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
#
# --- Path Setup (Replaces conftest.py logic) ---
# Add the project root to the Python path to allow importing the library.
# This assumes the tests are in a 'tests' directory at the project root.
try:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    # If your source code is in a 'src' directory, you might need:
    # sys.path.insert(0, str(project_root / "src"))
except (NameError, IndexError):
    # Fallback for environments where __file__ is not defined
    pass
#
# Local imports (from the main project)
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase as Database, ConflictError
#
#######################################################################################################################
#
# Helper Functions (for use in tests)
#

def get_log_count(db: Database, entity_uuid: str) -> int:
    """Helper to get sync log entries for assertions."""
    cursor = db.execute_query("SELECT COUNT(*) FROM sync_log WHERE entity_uuid = ?", (entity_uuid,))
    return cursor.fetchone()[0]


def get_latest_log(db: Database, entity_uuid: str) -> dict | None:
    """Helper to get the most recent sync log for an entity."""
    cursor = db.execute_query(
        "SELECT * FROM sync_log WHERE entity_uuid = ? ORDER BY change_id DESC LIMIT 1",
        (entity_uuid,)
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def get_entity_version(db: Database, entity_table: str, uuid: str) -> int | None:
    """Helper to get the current version of an entity."""
    cursor = db.execute_query(f"SELECT version FROM {entity_table} WHERE uuid = ?", (uuid,))
    row = cursor.fetchone()
    return row['version'] if row else None


#######################################################################################################################
#
# Pytest Fixtures (Moved from conftest.py)
#

@pytest.fixture(scope="function")
def memory_db_factory():
    """Factory fixture to create in-memory Database instances with automatic connection closing."""
    created_dbs = []

    def _create_db(client_id="test_client"):
        db = Database(db_path=":memory:", client_id=client_id)
        created_dbs.append(db)
        return db

    yield _create_db
    # Teardown: close connections for all created in-memory DBs
    for db in created_dbs:
        try:
            db.close_connection()
        except Exception:  # Ignore errors during cleanup
            pass


@pytest.fixture(scope="function")
def temp_db_path(tmp_path: Path) -> str:
    """Creates a temporary directory and returns a unique DB path string within it."""
    # The built-in tmp_path fixture handles directory creation and cleanup.
    return str(tmp_path / "test_db.sqlite")


@pytest.fixture(scope="function")
def file_db(temp_db_path: str):
    """Creates a file-based Database instance using a temporary path with automatic connection closing."""
    db = Database(db_path=temp_db_path, client_id="file_client")
    yield db
    db.close_connection()


@pytest.fixture(scope="function")
def db_instance(memory_db_factory):
    """Provides a fresh, isolated in-memory DB for a single test."""
    return memory_db_factory("crud_client")


@pytest.fixture(scope="class")
def search_db(tmp_path_factory):
    """Sets up a single DB with predictable data for all search tests in a class."""
    db_path = tmp_path_factory.mktemp("search_tests") / "search.db"
    db = Database(db_path, "search_client")

    # Add a predictable set of media items
    db.add_media_with_keywords(
        title="Alpha One", content="Content about Python and programming.", media_type="article",
        keywords=["python", "programming"], ingestion_date="2023-01-15T12:00:00Z"
    )  # ID 1
    db.add_media_with_keywords(
        title="Beta Two", content="A video about data science.", media_type="video",
        keywords=["python", "data science"], ingestion_date="2023-02-20T12:00:00Z"
    )  # ID 2
    db.add_media_with_keywords(
        title="Gamma Three (TRASH)", content="Old news.", media_type="article",
        keywords=["news"], ingestion_date="2023-03-10T12:00:00Z"
    )  # ID 3
    db.mark_as_trash(3)

    yield db
    db.close_connection()


#######################################################################################################################
#
# Test Classes
#

@pytest.mark.integration
class TestDatabaseInitialization:
    def test_memory_db_creation(self, memory_db_factory):
        """Test creating an in-memory database."""
        db = memory_db_factory("client_mem")
        assert db.is_memory_db
        assert db.client_id == "client_mem"
        cursor = db.execute_query("SELECT name FROM sqlite_master WHERE type='table' AND name='Media'")
        assert cursor.fetchone() is not None
        db.close_connection()

    def test_file_db_creation(self, file_db, temp_db_path):
        """Test creating a file-based database."""
        assert not file_db.is_memory_db
        assert file_db.client_id == "file_client"
        assert os.path.exists(temp_db_path)
        cursor = file_db.execute_query("SELECT name FROM sqlite_master WHERE type='table' AND name='Media'")
        assert cursor.fetchone() is not None
        # file_db fixture handles closure

    def test_missing_client_id(self):
        """Test that ValueError is raised if client_id is missing."""
        with pytest.raises(ValueError, match="Client ID cannot be empty"):
            Database(db_path=":memory:", client_id="")
        with pytest.raises(ValueError, match="Client ID cannot be empty"):
            Database(db_path=":memory:", client_id=None)


@pytest.mark.integration
class TestDatabaseTransactions:
    def test_transaction_commit(self, memory_db_factory):
        db = memory_db_factory()
        keyword = "commit_test"
        with db.transaction():
            db.add_keyword(keyword)
        cursor = db.execute_query("SELECT keyword FROM Keywords WHERE keyword = ?", (keyword,))
        assert cursor.fetchone()['keyword'] == keyword

    def test_transaction_rollback(self, memory_db_factory):
        db = memory_db_factory()
        keyword = "rollback_test"
        initial_count_cursor = db.execute_query("SELECT COUNT(*) FROM Keywords")
        initial_count = initial_count_cursor.fetchone()[0]
        try:
            with db.transaction():
                new_uuid = db._generate_uuid()
                db.execute_query(
                    "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, 1, ?, 0)",
                    (keyword, new_uuid, db._get_current_utc_timestamp_str(), db.client_id),
                    commit=False
                )
                cursor_inside = db.execute_query("SELECT COUNT(*) FROM Keywords")
                assert cursor_inside.fetchone()[0] == initial_count + 1
                raise ValueError("Simulating error to trigger rollback")
        except ValueError:
            pass  # Expected error
        except Exception as e:
            pytest.fail(f"Unexpected exception during rollback test: {e}")

        final_count_cursor = db.execute_query("SELECT COUNT(*) FROM Keywords")
        assert final_count_cursor.fetchone()[0] == initial_count


@pytest.mark.integration
class TestSearchFunctionality:
    # The 'search_db' fixture is now defined at the module level
    # and provides a shared database for all tests in this class.

    def test_some_search_function(self, search_db):
        """A placeholder test demonstrating usage of the search_db fixture."""
        # Example: Search for items with the keyword "python"
        # results = search_db.search(keywords=["python"])
        # assert len(results) == 2
        pass  # Add actual search tests here


@pytest.mark.integration
class TestDatabaseCRUDAndSync:
    # The 'db_instance' fixture is now defined at the module level
    # and provides a fresh in-memory DB for each test in this class.

    def test_add_keyword(self, db_instance):
        keyword = " test keyword "
        expected_keyword = "test keyword"
        kw_id, kw_uuid = db_instance.add_keyword(keyword)

        assert kw_id is not None
        assert kw_uuid is not None

        cursor = db_instance.execute_query("SELECT * FROM Keywords WHERE id = ?", (kw_id,))
        row = cursor.fetchone()
        assert row['keyword'] == expected_keyword
        assert row['uuid'] == kw_uuid

        log_entry = get_latest_log(db_instance, kw_uuid)
        assert log_entry['operation'] == 'create'
        assert log_entry['entity'] == 'Keywords'

    def test_add_existing_keyword(self, db_instance):
        keyword = "existing"
        kw_id1, kw_uuid1 = db_instance.add_keyword(keyword)
        log_count1 = get_log_count(db_instance, kw_uuid1)
        kw_id2, kw_uuid2 = db_instance.add_keyword(keyword)
        log_count2 = get_log_count(db_instance, kw_uuid1)

        assert kw_id1 == kw_id2
        assert kw_uuid1 == kw_uuid2
        assert log_count1 == log_count2

    def test_soft_delete_keyword(self, db_instance):
        keyword = "to_delete"
        kw_id, kw_uuid = db_instance.add_keyword(keyword)
        initial_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        assert db_instance.soft_delete_keyword(keyword) is True

        cursor = db_instance.execute_query("SELECT deleted, version FROM Keywords WHERE id = ?", (kw_id,))
        row = cursor.fetchone()
        assert row['deleted'] == 1
        assert row['version'] == initial_version + 1

        log_entry = get_latest_log(db_instance, kw_uuid)
        assert log_entry['operation'] == 'delete'
        assert log_entry['version'] == initial_version + 1

    def test_undelete_keyword(self, db_instance):
        keyword = "to_undelete"
        kw_id, kw_uuid = db_instance.add_keyword(keyword)
        db_instance.soft_delete_keyword(keyword)
        deleted_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        undelete_id, undelete_uuid = db_instance.add_keyword(keyword)

        assert undelete_id == kw_id
        cursor = db_instance.execute_query("SELECT deleted, version FROM Keywords WHERE id = ?", (kw_id,))
        row = cursor.fetchone()
        assert row['deleted'] == 0
        assert row['version'] == deleted_version + 1

        log_entry = get_latest_log(db_instance, kw_uuid)
        assert log_entry['operation'] == 'update'
        assert log_entry['version'] == deleted_version + 1

    def test_add_media_with_keywords_create(self, db_instance):
        title = "Test Media Create"
        content = "Some unique content for create."
        keywords = ["create_kw1", "create_kw2"]

        media_id, media_uuid, msg = db_instance.add_media_with_keywords(
            title=title, media_type="article", content=content, keywords=keywords
        )
        assert media_id is not None
        assert f"Media '{title}' added." in msg

        cursor = db_instance.execute_query("SELECT uuid, version FROM Media WHERE id = ?", (media_id,))
        media_row = cursor.fetchone()
        assert media_row['uuid'] == media_uuid
        assert media_row['version'] == 1

        log_entry = get_latest_log(db_instance, media_uuid)
        assert log_entry['operation'] == 'create'

    def test_add_media_with_keywords_update(self, db_instance):
        """Test updating a media item with new content, title, and keywords."""
        # Initial media setup
        title1 = "Test Media Original"
        title2 = "Test Media Updated"
        content1 = "Initial content."
        content2 = "Updated content."
        keywords1 = ["update_kw1"]
        keywords2 = ["update_kw2", "update_kw3"]
        media_type = "article"  # Use consistent media_type across tests

        # Create initial media item
        media_id, media_uuid, msg1 = db_instance.add_media_with_keywords(
            title=title1, media_type=media_type, content=content1, keywords=keywords1
        )
        assert "added" in msg1.lower(), f"Expected 'added' in message, got: {msg1}"
        initial_version = get_entity_version(db_instance, "Media", media_uuid)
        assert initial_version == 1, f"Expected initial version 1, got {initial_version}"

        # Fetch the created media to get its URL (stable identifier)
        created_media = db_instance.get_media_by_id(media_id)
        assert created_media is not None, "Failed to retrieve created media item"
        url_to_update = created_media['url']

        # Update the media item
        updated_id, updated_uuid, msg2 = db_instance.add_media_with_keywords(
            title=title2,
            media_type=media_type,
            content=content2,
            keywords=keywords2,
            overwrite=True,
            url=url_to_update
        )

        # Verify update operation returned correct values
        assert updated_id == media_id, "Update returned different media ID"
        assert updated_uuid == media_uuid, "Update returned different UUID"
        assert "updated" in msg2.lower(), f"Expected 'updated' in message, got: {msg2}"

        # Verify content was updated
        cursor = db_instance.execute_query("SELECT content, title, version FROM Media WHERE id = ?", (media_id,))
        media_row = cursor.fetchone()
        assert media_row['content'] == content2, "Content was not updated"
        assert media_row['title'] == title2, "Title was not updated"
        assert media_row['version'] == initial_version + 1, f"Version not incremented, expected {initial_version + 1}, got {media_row['version']}"

        # Verify keywords were updated
        cursor = db_instance.execute_query(
            """
            SELECT k.keyword FROM MediaKeywords mk
            JOIN Keywords k ON mk.keyword_id = k.id
            WHERE mk.media_id = ?
            """,
            (media_id,)
        )
        linked_keywords = [row['keyword'] for row in cursor.fetchall()]
        assert set(kw.lower() for kw in linked_keywords) == set(kw.lower() for kw in keywords2), "Keywords were not updated correctly"

        # Verify sync log was created for the update
        log_entry = get_latest_log(db_instance, media_uuid)
        assert log_entry['operation'] == 'update', f"Expected 'update' operation, got {log_entry['operation']}"
        assert log_entry['version'] == initial_version + 1, f"Log version mismatch: {log_entry['version']} vs {initial_version + 1}"
        assert log_entry['entity'] == 'Media', f"Expected 'Media' entity, got {log_entry['entity']}"

    def test_soft_delete_media_cascade(self, db_instance):
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(
            title="Cascade Test", content="Cascade content", keywords=["cascade1"], media_type="article"
        )
        media_version = get_entity_version(db_instance, "Media", media_uuid)

        assert db_instance.soft_delete_media(media_id, cascade=True) is True

        cursor = db_instance.execute_query("SELECT deleted, version FROM Media WHERE id = ?", (media_id,))
        assert dict(cursor.fetchone()) == {'deleted': 1, 'version': media_version + 1}

        cursor = db_instance.execute_query("SELECT COUNT(*) FROM MediaKeywords WHERE media_id = ?", (media_id,))
        assert cursor.fetchone()[0] == 0

        media_log = get_latest_log(db_instance, media_uuid)
        assert media_log['operation'] == 'delete'

    def test_optimistic_locking_prevents_update_with_stale_version(self, db_instance):
        kw_id, kw_uuid = db_instance.add_keyword("conflict_test")
        original_version = 1

        db_instance.execute_query(
            "UPDATE Keywords SET version = ?, client_id = ? WHERE id = ?",
            (original_version + 1, "external_client", kw_id), commit=True
        )

        cursor = db_instance.execute_query(
            "UPDATE Keywords SET keyword='stale_update', version=?, client_id=? WHERE id=? AND version=?",
            (original_version + 1, db_instance.client_id, kw_id, original_version), commit=True
        )
        assert cursor.rowcount == 0

    def test_version_validation_trigger(self, db_instance):
        kw_id, kw_uuid = db_instance.add_keyword("validation_test")
        current_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        with pytest.raises(sqlite3.IntegrityError, match="Version must increment by exactly 1"):
            db_instance.execute_query(
                "UPDATE Keywords SET version = ? WHERE id = ?",
                (current_version + 2, kw_id), commit=True
            )

    def test_client_id_validation_trigger(self, db_instance):
        kw_id, kw_uuid = db_instance.add_keyword("clientid_test")
        current_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        with pytest.raises(sqlite3.IntegrityError, match="Client ID cannot be NULL or empty"):
            db_instance.execute_query(
                "UPDATE Keywords SET version = ?, client_id = NULL WHERE id = ?",
                (current_version + 1, kw_id), commit=True
            )


@pytest.mark.integration
class TestSyncLogManagement:
    @pytest.fixture(autouse=True)
    def setup_db(self, db_instance):
        """Use autouse to provide the db_instance to every test in this class."""
        # Add some initial data to generate logs
        db_instance.add_keyword("log_kw_1")
        time.sleep(0.01)
        db_instance.add_keyword("log_kw_2")
        time.sleep(0.01)
        db_instance.add_keyword("log_kw_3")
        db_instance.soft_delete_keyword("log_kw_2")
        self.db = db_instance

    def test_get_sync_log_entries_all(self):
        logs = self.db.get_sync_log_entries()
        assert len(logs) == 4
        assert logs[0]['change_id'] == 1

    def test_get_sync_log_entries_since(self):
        logs = self.db.get_sync_log_entries(since_change_id=2)
        assert len(logs) == 2
        assert logs[0]['change_id'] == 3

    def test_get_sync_log_entries_limit(self):
        logs = self.db.get_sync_log_entries(limit=2)
        assert len(logs) == 2
        assert logs[0]['change_id'] == 1
        assert logs[1]['change_id'] == 2

    def test_delete_sync_log_entries_specific(self):
        initial_logs = self.db.get_sync_log_entries()
        ids_to_delete = [initial_logs[1]['change_id'], initial_logs[2]['change_id']]
        deleted_count = self.db.delete_sync_log_entries(ids_to_delete)
        assert deleted_count == 2
        remaining_ids = {log['change_id'] for log in self.db.get_sync_log_entries()}
        assert remaining_ids == {1, 4}

    def test_delete_sync_log_entries_before(self):
        deleted_count = self.db.delete_sync_log_entries_before(3)
        assert deleted_count == 3
        remaining_logs = self.db.get_sync_log_entries()
        assert len(remaining_logs) == 1
        assert remaining_logs[0]['change_id'] == 4

    def test_delete_sync_log_entries_invalid_id(self):
        with pytest.raises(ValueError):
            self.db.delete_sync_log_entries([1, "two", 3])

#
# End of test_media_db_v2.py
########################################################################################################################

