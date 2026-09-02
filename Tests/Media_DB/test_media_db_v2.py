# tests/test_media_db_v2.py
# Description: Unit tests for SQLite database operations, including CRUD, transactions, and sync log management.
# This version is self-contained and does not require a conftest.py file.
#
# Standard Library Imports:
import io
import json
import logging
import os
import pytest
import sys
import time
import sqlite3
from pathlib import Path

from loguru import logger

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
from tldw_chatbook.DB.Client_Media_DB_v2 import (
    DatabaseError,
    MediaDatabase as Database,
    permanently_delete_item,
)
from Tests.DB.historical_bootstrap_v6 import media_db_at_version
#
#######################################################################################################################
#
# Helper Functions (for use in tests)
#


def get_log_count(db: Database, entity_uuid: str) -> int:
    """Helper to get sync log entries for assertions."""
    cursor = db.execute_query(
        "SELECT COUNT(*) FROM sync_log WHERE entity_uuid = ?", (entity_uuid,)
    )
    return cursor.fetchone()[0]


def get_latest_log(db: Database, entity_uuid: str):
    """Helper to get the most recent sync log for an entity."""
    cursor = db.execute_query(
        "SELECT * FROM sync_log WHERE entity_uuid = ? ORDER BY change_id DESC LIMIT 1",
        (entity_uuid,),
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def get_entity_version(db: Database, entity_table: str, uuid: str):
    """Helper to get the current version of an entity."""
    cursor = db.execute_query(
        f"SELECT version FROM {entity_table} WHERE uuid = ?", (uuid,)
    )
    row = cursor.fetchone()
    return row["version"] if row else None


def get_document_version_count(db: Database, media_id: int) -> int:
    """Helper to count document versions for a media item."""
    cursor = db.execute_query(
        "SELECT COUNT(*) FROM DocumentVersions WHERE media_id = ?", (media_id,)
    )
    return cursor.fetchone()[0]


def _capture_permanent_delete_logs(caplog, operation):
    """Capture both logging stacks for one already-initialized DB operation."""
    caplog.clear()
    caplog.set_level(logging.DEBUG)
    loguru_output = io.StringIO()
    sink_id = logger.add(loguru_output, format="{message}", level="DEBUG")
    try:
        operation()
    finally:
        logger.remove(sink_id)
    return f"{loguru_output.getvalue()}\n{caplog.text}"


def _assert_permanent_delete_log_privacy(
    rendered_logs: str,
    *,
    target_id: int,
    db_path: Path,
    private_values: tuple[str, ...],
) -> None:
    identifier_fragments = (
        f"Media ID {target_id}",
        f"Media ID: {target_id}",
        f"local:media:{target_id}",
        f"Media {target_id}",
        f"media {target_id}",
        f"media_id={target_id}",
        f"media_id: {target_id}",
    )
    path_fragments = (
        str(db_path.resolve()),
        str(Path(f"{db_path.resolve()}-wal")),
        str(Path(f"{db_path.resolve()}-shm")),
    )
    assert "Traceback (most recent call last)" not in rendered_logs
    for private_value in (*identifier_fragments, *path_fragments, *private_values):
        assert private_value not in rendered_logs


def test_permanent_delete_privacy_guard_rejects_former_fts_success_line(
    tmp_path: Path,
) -> None:
    """Mutation proof: the former identifier-bearing FTS line is forbidden."""
    target_id = 7
    rendered_logs = f"Deleted FTS entry for Media ID {target_id}"
    with pytest.raises(AssertionError):
        _assert_permanent_delete_log_privacy(
            rendered_logs,
            target_id=target_id,
            db_path=tmp_path / "private.sqlite",
            private_values=(),
        )


def get_schema_version(db: Database) -> int:
    """Helper to fetch the current schema version."""
    cursor = db.execute_query("SELECT version FROM schema_version LIMIT 1")
    return cursor.fetchone()[0]


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
        title="Alpha One",
        content="Content about Python and programming.",
        media_type="article",
        keywords=["python", "programming"],
        ingestion_date="2023-01-15T12:00:00Z",
    )  # ID 1
    db.add_media_with_keywords(
        title="Beta Two",
        content="A video about data science.",
        media_type="video",
        keywords=["python", "data science"],
        ingestion_date="2023-02-20T12:00:00Z",
    )  # ID 2
    db.add_media_with_keywords(
        title="Gamma Three (TRASH)",
        content="Old news.",
        media_type="article",
        keywords=["news"],
        ingestion_date="2023-03-10T12:00:00Z",
    )  # ID 3
    db.mark_as_trash(3)

    yield db
    db.close_connection()


def test_permanent_delete_success_logs_only_fixed_metadata(tmp_path, caplog):
    db_path = tmp_path / "private-media-success.sqlite"
    private_title = "PRIVATE PERMANENT DELETE TITLE"
    private_url = "private-delete://success-url"
    private_client = "private-delete-success-client"
    db = Database(db_path, private_client)
    try:
        target_id, _uuid, _message = db.add_media_with_keywords(
            url=private_url,
            title=private_title,
            media_type="article",
            content="private delete content",
            keywords=[],
        )
        assert db.mark_as_trash(target_id)

        rendered_logs = _capture_permanent_delete_logs(
            caplog,
            lambda: permanently_delete_item(db, target_id),
        )

        assert db.get_media_by_id(target_id, include_trash=True) is None
        _assert_permanent_delete_log_privacy(
            rendered_logs,
            target_id=target_id,
            db_path=db_path,
            private_values=(private_title, private_url, private_client),
        )
        permanent_lines = tuple(
            line
            for line in rendered_logs.splitlines()
            if "operation=permanent_delete" in line
        )
        assert permanent_lines == (
            "Media mutation operation=permanent_delete status=started count=1",
            "Media mutation operation=permanent_delete status=committed count=1",
        )
        assert (
            "Media FTS mutation operation=delete status=committed count=1"
            in tuple(record.getMessage() for record in caplog.records)
        )
    finally:
        db.close_connection()


def test_permanent_delete_sqlite_failure_logs_category_without_private_values(
    tmp_path, caplog
):
    db_path = tmp_path / "private-media-failure.sqlite"
    private_title = "PRIVATE FAILED DELETE TITLE"
    private_url = "private-delete://failure-url"
    private_client = "private-delete-failure-client"
    private_exception = "PRIVATE SQLITE DELETE FAILURE"
    db = Database(db_path, private_client)
    try:
        target_id, _uuid, _message = db.add_media_with_keywords(
            url=private_url,
            title=private_title,
            media_type="article",
            content="private failed delete content",
            keywords=[],
        )
        assert db.mark_as_trash(target_id)
        with db.transaction() as conn:
            conn.execute(
                "CREATE TRIGGER block_private_media_delete "
                "BEFORE DELETE ON Media BEGIN "
                f"SELECT RAISE(ABORT, '{private_exception}'); END"
            )

        def fail_delete() -> None:
            with pytest.raises(DatabaseError):
                permanently_delete_item(db, target_id)

        rendered_logs = _capture_permanent_delete_logs(caplog, fail_delete)

        assert db.get_media_by_id(target_id, include_trash=True) is not None
        _assert_permanent_delete_log_privacy(
            rendered_logs,
            target_id=target_id,
            db_path=db_path,
            private_values=(
                private_title,
                private_url,
                private_client,
                private_exception,
            ),
        )
        permanent_lines = tuple(
            line
            for line in rendered_logs.splitlines()
            if "operation=permanent_delete" in line
        )
        assert permanent_lines == (
            "Media mutation operation=permanent_delete status=started count=1",
            "Media mutation operation=permanent_delete status=failed count=0 "
            "category=IntegrityError",
        )
    finally:
        db.close_connection()


def test_permanent_delete_fts_failure_logs_categories_without_private_values(
    tmp_path, caplog
):
    """Exercise the real FTS sink after Media deletion starts, then roll back."""
    db_path = tmp_path / "private-media-fts-failure.sqlite"
    private_title = "PRIVATE FTS FAILED DELETE TITLE"
    private_url = "private-delete://fts-failure-url"
    private_client = "private-delete-fts-failure-client"
    private_exception = "PRIVATE FTS DELETE FAILURE"
    private_sql = "DELETE FROM media_fts WHERE rowid = PRIVATE TARGET"
    db = Database(db_path, private_client)
    try:
        target_id, _uuid, _message = db.add_media_with_keywords(
            url=private_url,
            title=private_title,
            media_type="article",
            content="private FTS failure content",
            keywords=[],
        )
        assert db.mark_as_trash(target_id)
        original_delete_fts = db._delete_fts_media
        reached_fts_sink: list[int] = []

        class _FailingFTSConnection:
            def execute(self, sql, params):
                assert sql == "DELETE FROM media_fts WHERE rowid = ?"
                assert params == (target_id,)
                reached_fts_sink.append(target_id)
                raise sqlite3.OperationalError(
                    f"{private_exception} | {private_sql}"
                )

        def fail_after_media_delete(conn, media_id):
            assert conn.execute(
                "SELECT 1 FROM Media WHERE id = ?", (media_id,)
            ).fetchone() is None
            return original_delete_fts(_FailingFTSConnection(), media_id)

        db._delete_fts_media = fail_after_media_delete

        def fail_delete() -> None:
            with pytest.raises(DatabaseError):
                permanently_delete_item(db, target_id)

        rendered_logs = _capture_permanent_delete_logs(caplog, fail_delete)

        assert reached_fts_sink == [target_id]
        assert db.get_media_by_id(target_id, include_trash=True) is not None
        _assert_permanent_delete_log_privacy(
            rendered_logs,
            target_id=target_id,
            db_path=db_path,
            private_values=(
                private_title,
                private_url,
                private_client,
                private_exception,
                private_sql,
            ),
        )
        assert (
            "Media FTS mutation operation=delete status=failed count=0 "
            "category=OperationalError"
            in tuple(record.getMessage() for record in caplog.records)
        )
        permanent_lines = tuple(
            line
            for line in rendered_logs.splitlines()
            if "operation=permanent_delete" in line
        )
        assert permanent_lines == (
            "Media mutation operation=permanent_delete status=started count=1",
            "Media mutation operation=permanent_delete status=failed count=0 "
            "category=DatabaseError",
        )
    finally:
        db.close_connection()


def test_permanent_delete_privacy_guard_rejects_raw_fts_failure_detail(
    tmp_path: Path,
) -> None:
    """Mutation proof: category metadata may not carry raw FTS failure text."""
    target_id = 9
    private_exception = "PRIVATE RAW FTS FAILURE"
    rendered_logs = (
        "Media FTS mutation operation=delete status=failed count=0 "
        f"category=OperationalError detail={private_exception}"
    )
    with pytest.raises(AssertionError):
        _assert_permanent_delete_log_privacy(
            rendered_logs,
            target_id=target_id,
            db_path=tmp_path / "private.sqlite",
            private_values=(private_exception,),
        )


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
        cursor = db.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='Media'"
        )
        assert cursor.fetchone() is not None
        db.close_connection()

    def test_file_db_creation(self, file_db, temp_db_path):
        """Test creating a file-based database."""
        assert not file_db.is_memory_db
        assert file_db.client_id == "file_client"
        assert os.path.exists(temp_db_path)
        cursor = file_db.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='Media'"
        )
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
        cursor = db.execute_query(
            "SELECT keyword FROM Keywords WHERE keyword = ?", (keyword,)
        )
        assert cursor.fetchone()["keyword"] == keyword

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
                    (
                        keyword,
                        new_uuid,
                        db._get_current_utc_timestamp_str(),
                        db.client_id,
                    ),
                    commit=False,
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

        cursor = db_instance.execute_query(
            "SELECT * FROM Keywords WHERE id = ?", (kw_id,)
        )
        row = cursor.fetchone()
        assert row["keyword"] == expected_keyword
        assert row["uuid"] == kw_uuid

        log_entry = get_latest_log(db_instance, kw_uuid)
        assert log_entry["operation"] == "create"
        assert log_entry["entity"] == "Keywords"

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

        cursor = db_instance.execute_query(
            "SELECT deleted, version FROM Keywords WHERE id = ?", (kw_id,)
        )
        row = cursor.fetchone()
        assert row["deleted"] == 1
        assert row["version"] == initial_version + 1

        log_entry = get_latest_log(db_instance, kw_uuid)
        assert log_entry["operation"] == "delete"
        assert log_entry["version"] == initial_version + 1

    def test_undelete_keyword(self, db_instance):
        keyword = "to_undelete"
        kw_id, kw_uuid = db_instance.add_keyword(keyword)
        db_instance.soft_delete_keyword(keyword)
        deleted_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        undelete_id, undelete_uuid = db_instance.add_keyword(keyword)

        assert undelete_id == kw_id
        cursor = db_instance.execute_query(
            "SELECT deleted, version FROM Keywords WHERE id = ?", (kw_id,)
        )
        row = cursor.fetchone()
        assert row["deleted"] == 0
        assert row["version"] == deleted_version + 1

        log_entry = get_latest_log(db_instance, kw_uuid)
        assert log_entry["operation"] == "update"
        assert log_entry["version"] == deleted_version + 1

    def test_add_media_with_keywords_create(self, db_instance):
        title = "Test Media Create"
        content = "Some unique content for create."
        keywords = ["create_kw1", "create_kw2"]

        media_id, media_uuid, msg = db_instance.add_media_with_keywords(
            title=title, media_type="article", content=content, keywords=keywords
        )
        assert media_id is not None
        assert f"Media '{title}' added." in msg

        cursor = db_instance.execute_query(
            "SELECT uuid, version FROM Media WHERE id = ?", (media_id,)
        )
        media_row = cursor.fetchone()
        assert media_row["uuid"] == media_uuid
        assert media_row["version"] == 1

        log_entry = get_latest_log(db_instance, media_uuid)
        assert log_entry["operation"] == "create"

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
        assert initial_version == 1, (
            f"Expected initial version 1, got {initial_version}"
        )

        # Fetch the created media to get its URL (stable identifier)
        created_media = db_instance.get_media_by_id(media_id)
        assert created_media is not None, "Failed to retrieve created media item"
        url_to_update = created_media["url"]

        # Update the media item
        updated_id, updated_uuid, msg2 = db_instance.add_media_with_keywords(
            title=title2,
            media_type=media_type,
            content=content2,
            keywords=keywords2,
            overwrite=True,
            url=url_to_update,
        )

        # Verify update operation returned correct values
        assert updated_id == media_id, "Update returned different media ID"
        assert updated_uuid == media_uuid, "Update returned different UUID"
        assert "updated" in msg2.lower(), f"Expected 'updated' in message, got: {msg2}"

        # Verify content was updated
        cursor = db_instance.execute_query(
            "SELECT content, title, version FROM Media WHERE id = ?", (media_id,)
        )
        media_row = cursor.fetchone()
        assert media_row["content"] == content2, "Content was not updated"
        assert media_row["title"] == title2, "Title was not updated"
        assert media_row["version"] == initial_version + 1, (
            f"Version not incremented, expected {initial_version + 1}, got {media_row['version']}"
        )

        # Verify keywords were updated
        cursor = db_instance.execute_query(
            """
            SELECT k.keyword FROM MediaKeywords mk
            JOIN Keywords k ON mk.keyword_id = k.id
            WHERE mk.media_id = ?
            """,
            (media_id,),
        )
        linked_keywords = [row["keyword"] for row in cursor.fetchall()]
        assert set(kw.lower() for kw in linked_keywords) == set(
            kw.lower() for kw in keywords2
        ), "Keywords were not updated correctly"

        # Verify sync log was created for the update
        log_entry = get_latest_log(db_instance, media_uuid)
        assert log_entry["operation"] == "update", (
            f"Expected 'update' operation, got {log_entry['operation']}"
        )
        assert log_entry["version"] == initial_version + 1, (
            f"Log version mismatch: {log_entry['version']} vs {initial_version + 1}"
        )
        assert log_entry["entity"] == "Media", (
            f"Expected 'Media' entity, got {log_entry['entity']}"
        )

    def test_soft_delete_media_cascade(self, db_instance):
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(
            title="Cascade Test",
            content="Cascade content",
            keywords=["cascade1"],
            media_type="article",
        )
        media_version = get_entity_version(db_instance, "Media", media_uuid)

        assert db_instance.soft_delete_media(media_id, cascade=True) is True

        cursor = db_instance.execute_query(
            "SELECT deleted, version FROM Media WHERE id = ?", (media_id,)
        )
        assert dict(cursor.fetchone()) == {"deleted": 1, "version": media_version + 1}

        cursor = db_instance.execute_query(
            "SELECT COUNT(*) FROM MediaKeywords WHERE media_id = ?", (media_id,)
        )
        assert cursor.fetchone()[0] == 0

        media_log = get_latest_log(db_instance, media_uuid)
        assert media_log["operation"] == "delete"

    def test_undelete_sync_payload_carries_fts_source_fields(self, db_instance):
        title = "Restored sync title"
        content = "Restored sync content"
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(
            title=title,
            content=content,
            keywords=[],
            media_type="article",
        )
        assert db_instance.soft_delete_media(media_id, cascade=False) is True

        assert db_instance.undelete_media(media_id, cascade=False) is True

        media_log = get_latest_log(db_instance, media_uuid)
        payload = json.loads(media_log["payload"])
        assert media_log["operation"] == "update"
        assert payload["deleted"] == 0
        assert payload["title"] == title
        assert payload["content"] == content

    def test_optimistic_locking_prevents_update_with_stale_version(self, db_instance):
        kw_id, kw_uuid = db_instance.add_keyword("conflict_test")
        original_version = 1

        db_instance.execute_query(
            "UPDATE Keywords SET version = ?, client_id = ? WHERE id = ?",
            (original_version + 1, "external_client", kw_id),
            commit=True,
        )

        cursor = db_instance.execute_query(
            "UPDATE Keywords SET keyword='stale_update', version=?, client_id=? WHERE id=? AND version=?",
            (original_version + 1, db_instance.client_id, kw_id, original_version),
            commit=True,
        )
        assert cursor.rowcount == 0

    def test_version_validation_trigger(self, db_instance):
        kw_id, kw_uuid = db_instance.add_keyword("validation_test")
        current_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        with pytest.raises(
            sqlite3.IntegrityError, match="Version must increment by exactly 1"
        ):
            db_instance.execute_query(
                "UPDATE Keywords SET version = ? WHERE id = ?",
                (current_version + 2, kw_id),
                commit=True,
            )

    def test_client_id_validation_trigger(self, db_instance):
        kw_id, kw_uuid = db_instance.add_keyword("clientid_test")
        current_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        with pytest.raises(
            sqlite3.IntegrityError, match="Client ID cannot be NULL or empty"
        ):
            db_instance.execute_query(
                "UPDATE Keywords SET version = ?, client_id = NULL WHERE id = ?",
                (current_version + 1, kw_id),
                commit=True,
            )

    def test_reading_progress_round_trip(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(
            title="Reading Progress Round Trip",
            media_type="article",
            content="Round trip content for local reading progress.",
            keywords=["reading", "progress"],
        )

        payload = {
            "current_page": 4,
            "total_pages": 12,
            "view_mode": "single",
            "zoom_level": 1.25,
            "cfi": "epubcfi(/6/4[chapter]!/4/2/6)",
            "percentage": 33.3,
        }
        written = db_instance.upsert_reading_progress(media_id, payload)
        fetched = db_instance.get_reading_progress(media_id)

        assert written["media_id"] == media_id
        assert fetched is not None
        assert fetched["media_id"] == media_id
        assert fetched["current_page"] == 4
        assert fetched["total_pages"] == 12
        assert fetched["view_mode"] == "single"
        assert fetched["zoom_level"] == 1.25
        assert fetched["cfi"] == "epubcfi(/6/4[chapter]!/4/2/6)"
        assert fetched["percentage"] == 33.3

    def test_reading_progress_delete_removes_row(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(
            title="Reading Progress Delete",
            media_type="article",
            content="Delete content for local reading progress.",
            keywords=["reading", "delete"],
        )

        db_instance.upsert_reading_progress(
            media_id, {"current_page": 2, "total_pages": 5}
        )

        assert db_instance.delete_reading_progress(media_id) is True
        assert db_instance.get_reading_progress(media_id) is None
        assert db_instance.delete_reading_progress(media_id) is False

    def test_reading_progress_upsert_is_local_only(self, db_instance):
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(
            title="Reading Progress Local Only",
            media_type="article",
            content="Local-only content for reading progress.",
            keywords=["reading", "local"],
        )

        media_version_before = get_entity_version(db_instance, "Media", media_uuid)
        sync_log_before = get_log_count(db_instance, media_uuid)
        document_versions_before = get_document_version_count(db_instance, media_id)

        db_instance.upsert_reading_progress(
            media_id,
            {"current_page": 7, "total_pages": 11, "view_mode": "single"},
        )

        assert (
            get_entity_version(db_instance, "Media", media_uuid) == media_version_before
        )
        assert get_log_count(db_instance, media_uuid) == sync_log_before
        assert (
            get_document_version_count(db_instance, media_id)
            == document_versions_before
        )

    def test_reading_progress_reopens_through_versioned_migration(self, temp_db_path):
        # A genuinely historical v2 database, built by the real migration chain
        # (base v1 + v1->v2), NOT by stamping "2" onto a current one. The old
        # hand-degraded fixture had to be taught about every artifact each new
        # migration added, and broke on the first one it had not been told about
        # (v5->v6's chunk_engine_version). See TASK-21594.
        with media_db_at_version(temp_db_path, 2, client_id="schema_client") as old_db:
            assert get_schema_version(old_db) == 2
            # The historical preconditions the migration under test depends on:
            # neither local-only store exists yet at v2.
            for absent_table in ("ReadingProgress", "MediaReadItLaterState"):
                assert (
                    old_db.execute_query(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type = 'table' AND name = ?",
                        (absent_table,),
                    ).fetchone()
                    is None
                ), f"{absent_table} already exists at v2"

            # Seeded with the historical v1/v2 Media column set. The shipped
            # writer cannot be used here: add_media_with_keywords() targets the
            # current schema and fails with "no such column:
            # transcription_provenance_json" (added at v4->v5) against a real
            # v2 database -- the same reason this fixture needs a bootstrap
            # module rather than production code.
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with old_db.transaction():
                cursor = old_db.execute_query(
                    """
                    INSERT INTO Media (
                        title, type, content, content_hash, uuid,
                        ingestion_date, last_modified, client_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "Reading Progress Migration",
                        "article",
                        "Migration content for reading progress.",
                        "hash-reading-progress-migration",
                        "11111111-2222-3333-4444-555555555555",
                        now,
                        now,
                        "schema_client",
                    ),
                )
                media_id = cursor.lastrowid
            assert media_id is not None

        reopened_db = Database(db_path=temp_db_path, client_id="schema_client")
        try:
            assert get_schema_version(reopened_db) == Database._CURRENT_SCHEMA_VERSION
            reopened_db.upsert_reading_progress(
                media_id,
                {
                    "current_page": 8,
                    "total_pages": 20,
                    "view_mode": "single",
                    "zoom_level": 1.1,
                    "cfi": "epubcfi(/6/2[chapter]!/4/2/6)",
                    "percentage": 40.0,
                },
            )
            fetched = reopened_db.get_reading_progress(media_id)
            assert fetched is not None
            assert fetched["media_id"] == media_id
            assert fetched["current_page"] == 8
            assert fetched["zoom_level"] == 1.1
            assert fetched["cfi"] == "epubcfi(/6/2[chapter]!/4/2/6)"
            assert fetched["percentage"] == 40.0

            saved_state = reopened_db.save_media_to_read_it_later(media_id)
            fetched_saved_state = reopened_db.get_media_read_it_later_state(media_id)
            assert saved_state["media_id"] == media_id
            assert saved_state["is_read_it_later"] is True
            assert fetched_saved_state is not None
            assert fetched_saved_state["media_id"] == media_id
            assert fetched_saved_state["is_read_it_later"] is True
        finally:
            reopened_db.close_connection()

    def test_failed_migration_rolls_back_schema_version_and_is_retryable(
        self, db_instance
    ):
        conn = db_instance.get_connection()
        original_sql = db_instance._READING_PROGRESS_TABLE_SQL
        conn.execute("DROP TABLE IF EXISTS ReadingProgress")
        conn.execute("DROP TABLE IF EXISTS migration_atomicity_probe")
        conn.execute("UPDATE schema_version SET version = 2")
        conn.commit()

        db_instance._READING_PROGRESS_TABLE_SQL = f"""
            {original_sql}
            CREATE TABLE migration_atomicity_probe(id INTEGER);
            INSERT INTO table_that_does_not_exist VALUES (1);
        """
        try:
            with pytest.raises(DatabaseError, match="Migration v2->v3 failed"):
                db_instance._apply_migration_v2_to_v3(conn)
        finally:
            db_instance._READING_PROGRESS_TABLE_SQL = original_sql

        assert get_schema_version(db_instance) == 2
        for table_name in ("ReadingProgress", "migration_atomicity_probe"):
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone()
            assert row is None, f"{table_name} survived the failed migration"

        db_instance._apply_migration_v2_to_v3(conn)
        assert get_schema_version(db_instance) == 3
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'ReadingProgress'"
            ).fetchone()
            is not None
        )

    def test_v1_to_v2_seed_failure_rolls_back_ddl_and_version(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "v1-seed-failure.sqlite"
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        Database._execute_transactional_script(
            conn,
            f"""
            {Database._TABLES_SQL_V1}
            {Database._INDICES_SQL_V1}
            {Database._TRIGGERS_SQL_V1}
            {Database._SCHEMA_UPDATE_VERSION_SQL_V1}
            """,
        )
        conn.commit()
        conn.close()

        template = {
            "name": "duplicate-seed",
            "description": "forces unique failure",
            "template_json": {"name": "duplicate-seed"},
            "is_system": 1,
        }
        monkeypatch.setattr(
            Database,
            "_DEFAULT_CHUNKING_TEMPLATES",
            [template, template],
        )

        with pytest.raises(DatabaseError, match="Migration v1->v2 failed"):
            Database(db_path=path, client_id="migration-failure")

        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            assert conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()[0] == 1
            assert (
                conn.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'ChunkingTemplates'"
                ).fetchone()
                is None
            )
            media_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(Media)")
            }
            assert "chunking_config" not in media_columns
        finally:
            conn.close()

        monkeypatch.setattr(Database, "_DEFAULT_CHUNKING_TEMPLATES", [template])
        migrated = Database(db_path=path, client_id="migration-retry")
        conn = migrated.get_connection()
        try:
            assert (
                conn.execute("SELECT version FROM schema_version").fetchone()[0]
                == Database._CURRENT_SCHEMA_VERSION
            )
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM ChunkingTemplates "
                    # task-7 (v7 rebuild): the retry replays through the
                    # ChunkingTemplates migration, and this seed's body
                    # (name only — no chunk stage, no base_method) is
                    # unconvertible, so it survives under its quarantined
                    # name instead of the original.
                    "WHERE name IN ('duplicate-seed', "
                    "'duplicate-seed (needs review)')"
                ).fetchone()[0]
                == 1
            )
        finally:
            migrated.close_connection()

    def test_hard_delete_failure_rolls_back_the_entire_batch(
        self, db_instance, monkeypatch
    ):
        items = []
        for suffix in ("first", "second"):
            media_id, media_uuid, _ = db_instance.add_media_with_keywords(
                title=f"Atomic hard delete {suffix}",
                content=f"{suffix} source must survive a failed hard-delete batch.",
                keywords=["atomic-delete"],
                media_type="document",
            )
            assert db_instance.soft_delete_media(media_id)
            items.append((media_id, media_uuid))

        original_log_sync_event = db_instance._log_sync_event
        hard_delete_events = 0

        def fail_final_audit_event(
            conn, entity, entity_uuid, operation, version, payload
        ):
            nonlocal hard_delete_events
            if operation == "delete":
                hard_delete_events += 1
                if hard_delete_events == 2:
                    raise sqlite3.OperationalError(
                        "injected hard-delete audit failure"
                    )
            return original_log_sync_event(
                conn, entity, entity_uuid, operation, version, payload
            )

        monkeypatch.setattr(
            db_instance, "_log_sync_event", fail_final_audit_event
        )

        with pytest.raises(DatabaseError, match="Failed to perform hard deletion"):
            db_instance.hard_delete_old_media(days_old=-1)

        for media_id, media_uuid in items:
            row = db_instance.execute_query(
                "SELECT uuid, deleted FROM Media WHERE id = ?", (media_id,)
            ).fetchone()
            assert dict(row) == {"uuid": media_uuid, "deleted": 1}

    def test_read_it_later_state_round_trips_for_local_media(self, db_instance):
        media_id, _, _ = db_instance.add_media_with_keywords(
            title="Reader",
            content="Hello",
            media_type="article",
            keywords=[],
        )

        saved = db_instance.save_media_to_read_it_later(media_id)

        assert saved["media_id"] == media_id
        assert saved["is_read_it_later"] is True
        assert saved["saved_at"] is not None
        assert (
            db_instance.get_media_read_it_later_state(media_id)["is_read_it_later"]
            is True
        )

    def test_soft_deleted_local_media_is_hidden_from_saved_view_by_default(
        self, db_instance
    ):
        media_id, _, _ = db_instance.add_media_with_keywords(
            title="Reader",
            content="Hello",
            media_type="article",
            keywords=[],
        )
        db_instance.save_media_to_read_it_later(media_id)
        db_instance.soft_delete_media(media_id)

        ids = db_instance.list_read_it_later_media_ids()
        assert media_id not in ids

    def test_trashed_local_media_is_hidden_from_saved_view_by_default(
        self, db_instance
    ):
        media_id, _, _ = db_instance.add_media_with_keywords(
            title="Reader",
            content="Hello",
            media_type="article",
            keywords=[],
        )
        db_instance.save_media_to_read_it_later(media_id)
        db_instance.mark_as_trash(media_id)

        ids = db_instance.list_read_it_later_media_ids()
        assert media_id not in ids


@pytest.mark.integration
class TestReimportAfterTrash:
    """task-4022: re-importing a file whose Media row was moved to trash
    must restore that row, not silently skip the write and leave the item
    permanently un-importable.

    Reproduced live at dev `4d0232358`: import a file -> Media (Select) ->
    check it -> Delete selected -> confirm -> re-import the SAME file. The
    ingest row reads "matched · short.txt / Already in Library -- matched
    an existing item; nothing new was imported." while ``Media (1)`` and
    the item is absent from every list -- the trashed row was never
    excluded from the dedup match, so the write silently skipped
    (media_id=None) and the row stayed trashed forever. Uses a real
    file-backed DB (``file_db``), not an in-memory mock, per the task's
    live-DB requirement.

    task-4022 (review round 2): restore is now opt-in via
    ``restore_trashed=True`` rather than unconditional for any trashed
    match (see ``TestRestoreTrashedIsOptIn`` below for the contract test
    itself). Every test in this class passes ``restore_trashed=True``
    explicitly, mirroring what the real Library ingest writer
    (``persist_parsed_media``) now does.
    """

    def test_reimport_by_url_restores_trashed_row_instead_of_skipping(self, file_db):
        # Mirrors the real ingest writer's URL shape for a local file
        # (``local_file_ingestion.py``: ``f"file://{file_path.absolute()}"``)
        # and its actual call (no ``overwrite`` kwarg -- defaults to False;
        # ``persist_parsed_media`` does pass ``restore_trashed=True``).
        url = "file:///Users/example/short.txt"
        media_id, media_uuid, msg1 = file_db.add_media_with_keywords(
            title="short.txt",
            media_type="document",
            content="hello world",
            keywords=[],
            url=url,
        )
        assert media_id is not None, f"initial import failed: {msg1!r}"
        assert file_db.mark_as_trash(media_id) is True

        # Sanity check, not the bug under test: the trashed row is already
        # excluded from the normal active lookup.
        assert file_db.get_media_by_url(url) is None

        reimported_id, reimported_uuid, msg2 = file_db.add_media_with_keywords(
            title="short.txt",
            media_type="document",
            content="hello world",
            keywords=[],
            url=url,
            restore_trashed=True,
        )

        # The observed bug: reimported_id was None and msg2 said "already
        # exists. Overwrite not enabled." -- nothing new was ever written,
        # and the row stayed trashed forever with no way to reach it.
        assert reimported_id == media_id, (
            f"expected the SAME row to be restored on re-import, got "
            f"{reimported_id!r} (msg={msg2!r})"
        )
        assert reimported_uuid == media_uuid
        assert "restored" in msg2.lower(), msg2

        row = file_db.get_media_by_id(media_id, include_trash=True)
        assert row["is_trash"] == 0
        assert row["trash_date"] is None
        assert row["deleted"] == 0

        # The item is present exactly once (AC#1/AC#4): the normal active
        # lookup finds it again, and there is exactly one Media row for
        # this url -- re-importing never created a second row.
        found = file_db.get_media_by_url(url)
        assert found is not None and found["id"] == media_id
        cursor = file_db.execute_query(
            "SELECT COUNT(*) FROM Media WHERE url = ?", (url,)
        )
        assert cursor.fetchone()[0] == 1

    def test_reimport_by_hash_restores_trashed_row_when_url_differs(self, file_db):
        """Same bytes at a different path -- falls through to the
        content_hash fallback match (the second SELECT in
        ``_add_media_with_keywords_impl``), the other dedup leg named in
        the task brief (``get_media_by_hash``)."""
        content = "identical bytes, different path"
        media_id, media_uuid, _ = file_db.add_media_with_keywords(
            title="a.txt",
            media_type="document",
            content=content,
            keywords=[],
            url="file:///a/copy.txt",
        )
        assert file_db.mark_as_trash(media_id) is True

        reimported_id, reimported_uuid, msg = file_db.add_media_with_keywords(
            title="a.txt",
            media_type="document",
            content=content,
            keywords=[],
            url="file:///b/copy.txt",
            restore_trashed=True,
        )

        assert reimported_id == media_id
        assert reimported_uuid == media_uuid
        assert "restored" in msg.lower(), msg
        row = file_db.get_media_by_id(media_id, include_trash=True)
        assert row["is_trash"] == 0
        assert row["trash_date"] is None
        # task-4022 review round 2 (I3): the existing url
        # ("file:///a/copy.txt") is already a real, non-``local://`` url,
        # so the restore-time canonicalization rule (auto-generated
        # ``local://...`` -> a real url, never the reverse) must NOT fire
        # here -- the row keeps its original url rather than being
        # silently reassigned to whatever path this particular re-import
        # happened to use. This is what actually makes the comment above
        # ("never touches url") true; before the I3 fix this assertion
        # failed (the row came back at "file:///b/copy.txt").
        assert row["url"] == "file:///a/copy.txt"
        # What matters most here is there's still exactly one row, not two.
        cursor = file_db.execute_query("SELECT COUNT(*) FROM Media")
        assert cursor.fetchone()[0] == 1

    def test_reimport_of_active_duplicate_is_still_skipped(self, file_db):
        """Guard rail: an ACTIVE (non-trashed) duplicate must still be
        skipped exactly as before -- this fix only changes behavior for a
        match that is sitting in trash."""
        url = "file:///still/active.txt"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="active.txt",
            media_type="document",
            content="hi",
            keywords=[],
            url=url,
        )

        reimported_id, reimported_uuid, msg = file_db.add_media_with_keywords(
            title="active.txt",
            media_type="document",
            content="hi",
            keywords=[],
            url=url,
        )

        assert reimported_id is None
        assert reimported_uuid is None
        assert "already exists" in msg.lower()
        cursor = file_db.execute_query(
            "SELECT COUNT(*) FROM Media WHERE url = ?", (url,)
        )
        assert cursor.fetchone()[0] == 1

    def test_reimport_identical_content_at_new_url_canonicalizes_url(
        self, file_db
    ):
        """Review round 1 (Important #1): the identical-content restore path
        (A.1.a, metadata-only update) must ALSO canonicalize ``url`` to the
        just-imported path, not just reset ``is_trash``/``trash_date``.

        Reproduced case: a row first created with no explicit url (so it
        gets the auto-generated ``local://<type>/<hash>``), trashed, then
        re-imported at a REAL path with byte-identical content -- the
        content-hash match takes the metadata-only branch. Before this fix,
        that branch restored the row (is_trash=0) but left its url at the
        stale ``local://...`` value, so ``get_media_by_url`` on the real
        path the user just imported returned ``None`` for a live, un-
        trashed item -- confirmed reproduced against this exact fixture
        before the fix landed.
        """
        content = "identical bytes for the canonicalization regression"
        media_id, media_uuid, _ = file_db.add_media_with_keywords(
            title="auto-url.txt",
            media_type="document",
            content=content,
            keywords=[],
            # No explicit url -- add_media_with_keywords auto-generates
            # "local://<type>/<hash>".
        )
        created = file_db.get_media_by_id(media_id)
        assert created["url"].startswith("local://")
        stale_url = created["url"]
        assert file_db.mark_as_trash(media_id) is True

        real_url = "file:///real/path.txt"
        reimported_id, reimported_uuid, msg = file_db.add_media_with_keywords(
            title="auto-url.txt",
            media_type="document",
            content=content,
            keywords=[],
            url=real_url,
            restore_trashed=True,
        )

        assert reimported_id == media_id
        assert reimported_uuid == media_uuid
        assert "restored" in msg.lower(), msg

        row = file_db.get_media_by_id(media_id, include_trash=True)
        assert row["is_trash"] == 0
        assert row["trash_date"] is None
        assert row["url"] == real_url

        # The regression itself: a live item must be findable by the url it
        # was just (re-)imported at.
        found = file_db.get_media_by_url(real_url)
        assert found is not None and found["id"] == media_id
        # And the stale url must no longer resolve to it (it moved).
        assert file_db.get_media_by_url(stale_url) is None

        cursor = file_db.execute_query("SELECT COUNT(*) FROM Media")
        assert cursor.fetchone()[0] == 1


class TestRestoreTrashedIsOptIn:
    """task-4022 review round 2 (I1): restoring a trashed match on
    re-import must be opt-in (``restore_trashed=True``), not unconditional
    for any caller that happens to hit a trashed row. Round 1 made restore
    fire for EVERY trashed match regardless of ``overwrite``, which
    silently resurrected rows for callers that never asked for it --
    chatbook SKIP-conflict imports, reading-list bulk imports matching on
    content hash, and Console "save message as media" (which dedups
    purely by content hash, no url at all). This class pins the DB-layer
    half of that contract directly; the per-caller regression tests live
    in ``Tests/Chatbooks/test_chatbook_importer.py`` and
    ``Tests/Media/test_local_media_reading_service.py``.
    """

    def test_default_restore_trashed_leaves_url_match_untouched(self, file_db):
        url = "file:///opt-in/by-url.txt"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="by-url.txt",
            media_type="document",
            content="hello",
            keywords=[],
            url=url,
        )
        assert file_db.mark_as_trash(media_id) is True

        # No restore_trashed kwarg at all -- defaults to False.
        reimported_id, reimported_uuid, msg = file_db.add_media_with_keywords(
            title="by-url.txt",
            media_type="document",
            content="hello",
            keywords=[],
            url=url,
        )

        assert reimported_id is None
        assert reimported_uuid is None
        # Pin rewritten by task-4026: the trashed skip used to reuse the
        # generic "already exists. Overwrite not enabled." message, whose
        # advice became a lie once overwrite=True stopped touching trashed
        # rows -- the skip now names Trash and the actual remedy
        # (restore_trashed=True). Live-row duplicate skips keep the old
        # message unchanged.
        assert "trash" in msg.lower(), msg
        assert "restore_trashed" in msg, msg
        row = file_db.get_media_by_id(media_id, include_trash=True)
        assert row["is_trash"] == 1, "a non-opted-in caller must not restore the row"

    def test_default_restore_trashed_leaves_content_hash_match_untouched(
        self, file_db
    ):
        """Mirrors I1(b)/I1(c): a caller that only ever matches by content
        hash (no url leg, or a url leg that missed) must not resurrect a
        trashed row either, when it doesn't opt in."""
        content = "opt-in by hash, not by url"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="by-hash.txt",
            media_type="document",
            content=content,
            keywords=[],
            url="file:///opt-in/hash-a.txt",
        )
        assert file_db.mark_as_trash(media_id) is True

        reimported_id, reimported_uuid, msg = file_db.add_media_with_keywords(
            title="by-hash.txt",
            media_type="document",
            content=content,
            keywords=[],
            url="file:///opt-in/hash-b.txt",
        )

        assert reimported_id is None
        assert reimported_uuid is None
        row = file_db.get_media_by_id(media_id, include_trash=True)
        assert row["is_trash"] == 1
        # No second row was created for the new url either.
        cursor = file_db.execute_query("SELECT COUNT(*) FROM Media")
        assert cursor.fetchone()[0] == 1


class TestOverwriteDoesNotTouchTrashedRows:
    """task-4026: the ``overwrite=True`` trash contract, made explicit.

    A trashed match is NEVER mutated by ``add_media_with_keywords`` unless
    the caller also passes ``restore_trashed=True`` -- ``overwrite``
    governs live rows only. ``overwrite=True, restore_trashed=False``
    against a trashed row is a duplicate-style SKIP (``(None, None,
    <message naming trash and the restore flag>)``), not a hidden
    in-place update and not a resurrection. Both flags true =
    restore-and-overwrite (task-4022's ``restoring_from_trash`` path).

    Before this task the code was internally incoherent: an
    identical-content ``overwrite=True`` left the row trashed but still
    mutated its title/keywords/chunks in place (invisible to the user --
    no Trash surface exists), while a different-content ``overwrite=True``
    silently RESURRECTED the row via ``_media_payload``'s hardcoded
    ``is_trash: 0`` with no restore decision anywhere in the chain.

    Real file-backed DB per the programme's DB-layer requirement.
    """

    @staticmethod
    def _keywords_for(file_db, media_id: int) -> list:
        return sorted(
            r["keyword"]
            for r in file_db.execute_query(
                "SELECT k.keyword FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id "
                "WHERE mk.media_id = ? AND k.deleted = 0",
                (media_id,),
            ).fetchall()
        )

    @staticmethod
    def _chunk_texts(file_db, media_id: int) -> set:
        return {
            r["chunk_text"]
            for r in file_db.execute_query(
                "SELECT chunk_text FROM UnvectorizedMediaChunks WHERE media_id = ? AND deleted = 0",
                (media_id,),
            ).fetchall()
        }

    def test_overwrite_true_alone_skips_trashed_match_with_different_content(
        self, file_db
    ):
        """The headline bug: different content + ``overwrite=True`` used to
        resurrect the trashed row (full-update path writes ``is_trash=0``).
        It must now skip, leaving trash state AND content untouched."""
        url = "file:///task-4026/different-content.txt"
        media_id, media_uuid, _ = file_db.add_media_with_keywords(
            title="original.txt",
            media_type="document",
            content="original content",
            keywords=[],
            url=url,
        )
        assert file_db.mark_as_trash(media_id) is True
        before = file_db.get_media_by_id(media_id, include_trash=True)
        assert before["is_trash"] == 1 and before["trash_date"] is not None

        result_id, result_uuid, msg = file_db.add_media_with_keywords(
            title="rewritten.txt",
            media_type="document",
            content="completely different content",
            keywords=[],
            url=url,
            overwrite=True,
        )

        assert result_id is None, (
            f"overwrite=True alone must not touch a trashed row, got id={result_id!r} "
            f"(msg={msg!r})"
        )
        assert result_uuid is None
        assert "trash" in msg.lower(), msg
        assert "restore_trashed" in msg, msg

        after = file_db.get_media_by_id(media_id, include_trash=True)
        assert after["is_trash"] == 1, "row must stay trashed"
        assert after["trash_date"] == before["trash_date"]
        assert after["content"] == "original content", "content must not change"
        assert after["title"] == "original.txt"
        assert after["version"] == before["version"], "no versioned write may occur"
        cursor = file_db.execute_query(
            "SELECT COUNT(*) FROM Media WHERE url = ?", (url,)
        )
        assert cursor.fetchone()[0] == 1, "no second row may be created"

    def test_overwrite_true_alone_leaves_trashed_metadata_and_keywords_untouched(
        self, file_db
    ):
        """The quieter half of the old incoherence: identical content +
        ``overwrite=True`` left the row trashed but still rewrote its
        title/keywords in place. A trashed row is now untouched entirely."""
        url = "file:///task-4026/identical-content.txt"
        content = "identical content"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="curated.txt",
            media_type="document",
            content=content,
            keywords=["mine", "curated"],
            url=url,
        )
        assert self._keywords_for(file_db, media_id) == ["curated", "mine"]
        assert file_db.mark_as_trash(media_id) is True
        before = file_db.get_media_by_id(media_id, include_trash=True)

        result_id, _, msg = file_db.add_media_with_keywords(
            title="renamed.txt",
            media_type="document",
            content=content,
            keywords=["other"],
            url=url,
            overwrite=True,
        )

        assert result_id is None, msg
        assert "trash" in msg.lower(), msg
        after = file_db.get_media_by_id(media_id, include_trash=True)
        assert after["is_trash"] == 1
        assert after["title"] == "curated.txt", "metadata must not change"
        assert after["version"] == before["version"]
        assert self._keywords_for(file_db, media_id) == ["curated", "mine"], (
            "keywords must not change on a trashed row"
        )

    def test_overwrite_true_alone_leaves_trashed_chunks_untouched(self, file_db):
        """Chunked variant (per this batch's chunked-content requirement):
        the skip must also leave the trashed row's stored chunk rows alone
        -- the old code deleted and replaced them via ``_persist_chunks``
        with ``replace_existing=overwrite``."""
        url = "file:///task-4026/chunked-skip.txt"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="chunked.txt",
            media_type="document",
            content="chunked original content",
            keywords=[],
            url=url,
            chunks=[
                {"text": "old chunk one", "chunk_type": "text"},
                {"text": "old chunk two", "chunk_type": "text"},
            ],
        )
        assert self._chunk_texts(file_db, media_id) == {
            "old chunk one",
            "old chunk two",
        }
        assert file_db.mark_as_trash(media_id) is True

        result_id, _, msg = file_db.add_media_with_keywords(
            title="chunked.txt",
            media_type="document",
            content="chunked replacement content",
            keywords=[],
            url=url,
            chunks=[{"text": "new chunk", "chunk_type": "text"}],
            overwrite=True,
        )

        assert result_id is None, msg
        after = file_db.get_media_by_id(media_id, include_trash=True)
        assert after["is_trash"] == 1
        assert after["content"] == "chunked original content"
        assert self._chunk_texts(file_db, media_id) == {
            "old chunk one",
            "old chunk two",
        }, "stored chunks must not be replaced on a trashed skip"

    def test_overwrite_plus_restore_trashed_restores_and_overwrites(self, file_db):
        """The two flags compose: ``overwrite=True, restore_trashed=True``
        is an explicit restore-and-overwrite. (Green before and after this
        task -- pinned so the compose case can't regress while the
        overwrite-alone case is locked down.)"""
        url = "file:///task-4026/restore-and-overwrite.txt"
        media_id, media_uuid, _ = file_db.add_media_with_keywords(
            title="original.txt",
            media_type="document",
            content="original content",
            keywords=[],
            url=url,
        )
        assert file_db.mark_as_trash(media_id) is True

        result_id, result_uuid, msg = file_db.add_media_with_keywords(
            title="restored.txt",
            media_type="document",
            content="fresh content",
            keywords=[],
            url=url,
            overwrite=True,
            restore_trashed=True,
        )

        assert result_id == media_id, msg
        assert result_uuid == media_uuid
        assert "restored" in msg.lower(), msg
        after = file_db.get_media_by_id(media_id, include_trash=True)
        assert after["is_trash"] == 0
        assert after["trash_date"] is None
        assert after["deleted"] == 0
        assert after["content"] == "fresh content"
        assert after["title"] == "restored.txt"

    def test_overwrite_plus_restore_trashed_replaces_chunks(self, file_db):
        """Chunked compose variant: an explicit restore-and-overwrite
        replaces the stored chunks with the fresh set (no
        ``UNIQUE(media_id, chunk_index, chunk_type)`` collision)."""
        url = "file:///task-4026/chunked-restore.txt"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="chunked.txt",
            media_type="document",
            content="chunked original content",
            keywords=[],
            url=url,
            chunks=[
                {"text": "old chunk one", "chunk_type": "text"},
                {"text": "old chunk two", "chunk_type": "text"},
            ],
        )
        assert file_db.mark_as_trash(media_id) is True

        result_id, _, msg = file_db.add_media_with_keywords(
            title="chunked.txt",
            media_type="document",
            content="chunked replacement content",
            keywords=[],
            url=url,
            chunks=[{"text": "new chunk", "chunk_type": "text"}],
            overwrite=True,
            restore_trashed=True,
        )

        assert result_id == media_id, msg
        assert "restored" in msg.lower(), msg
        after = file_db.get_media_by_id(media_id, include_trash=True)
        assert after["is_trash"] == 0
        assert after["content"] == "chunked replacement content"
        assert self._chunk_texts(file_db, media_id) == {"new chunk"}


class TestReimportAfterTrashChunks:
    """task-4022 review round 2 (C1, Critical): a chunked re-import of a
    trashed row must not raise ``sqlite3.IntegrityError``. ``_persist_chunks``
    used to gate its ``DELETE FROM UnvectorizedMediaChunks`` on the raw
    ``overwrite`` flag; the restore path enters chunk-writing with
    ``overwrite=False``/``restoring_from_trash=True`` (the real ingest
    writer never passes ``overwrite=True``), so the old chunk rows
    survived and the fresh INSERTs collided with
    ``UNIQUE(media_id, chunk_index, chunk_type)``. Every existing
    ``TestReimportAfterTrash`` test omits ``chunks`` entirely, so this
    path was never exercised by that class (``_persist_chunks`` early-
    returns on ``chunks is None``)."""

    @staticmethod
    def _chunk_count(file_db, media_id: int) -> int:
        cursor = file_db.execute_query(
            "SELECT COUNT(*) FROM UnvectorizedMediaChunks WHERE media_id = ? AND deleted = 0",
            (media_id,),
        )
        return cursor.fetchone()[0]

    def test_chunked_reimport_on_identical_content_restore_does_not_raise(
        self, file_db
    ):
        url = "file:///chunks/identical.txt"
        content = "identical content, chunked"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="identical.txt",
            media_type="document",
            content=content,
            keywords=[],
            url=url,
            # ``chunk_type`` must be an explicit, shared value: SQLite's
            # UNIQUE index treats NULL as always distinct from any other
            # value (including another NULL), so an omitted chunk_type
            # would never actually collide and this test would pass for
            # the wrong reason. Real ingest chunks always carry a concrete
            # chunk_type.
            chunks=[
                {"text": "old chunk one", "chunk_type": "text"},
                {"text": "old chunk two", "chunk_type": "text"},
            ],
        )
        assert self._chunk_count(file_db, media_id) == 2
        assert file_db.mark_as_trash(media_id) is True

        # Same content -> takes the metadata-only (A.1.a) restore branch,
        # with a DIFFERENT set of chunks than what's already stored, at
        # OVERLAPPING (chunk_index, chunk_type) pairs (0/1, both "text")
        # so an un-deleted old row collides on
        # UNIQUE(media_id, chunk_index, chunk_type) exactly as C1 describes.
        reimported_id, _, msg = file_db.add_media_with_keywords(
            title="identical.txt",
            media_type="document",
            content=content,
            keywords=[],
            url=url,
            chunks=[
                {"text": "new chunk one", "chunk_type": "text"},
                {"text": "new chunk two", "chunk_type": "text"},
                {"text": "new chunk three", "chunk_type": "text"},
            ],
            restore_trashed=True,
        )

        assert reimported_id == media_id, msg
        assert "restored" in msg.lower(), msg
        row = file_db.get_media_by_id(media_id, include_trash=True)
        assert row["is_trash"] == 0
        # The OLD chunk rows must have been replaced, not left to collide
        # with the new ones.
        assert self._chunk_count(file_db, media_id) == 3
        texts = {
            r["chunk_text"]
            for r in file_db.execute_query(
                "SELECT chunk_text FROM UnvectorizedMediaChunks WHERE media_id = ? AND deleted = 0",
                (media_id,),
            ).fetchall()
        }
        assert texts == {"new chunk one", "new chunk two", "new chunk three"}

    def test_chunked_reimport_on_full_update_restore_does_not_raise(self, file_db):
        url = "file:///chunks/full-update.txt"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="full-update.txt",
            media_type="document",
            content="original content",
            keywords=[],
            url=url,
            chunks=[
                {"text": "old chunk one", "chunk_type": "text"},
                {"text": "old chunk two", "chunk_type": "text"},
            ],
        )
        assert self._chunk_count(file_db, media_id) == 2
        assert file_db.mark_as_trash(media_id) is True

        # DIFFERENT content this time -> the full-content-update (A.1.b)
        # restore branch, the other sub-path C1 named. Same chunk_type at
        # chunk_index=0 as the old row, so an un-deleted old row collides.
        reimported_id, _, msg = file_db.add_media_with_keywords(
            title="full-update.txt",
            media_type="document",
            content="brand new content",
            keywords=[],
            url=url,
            chunks=[{"text": "fresh chunk", "chunk_type": "text"}],
            restore_trashed=True,
        )

        assert reimported_id == media_id, msg
        assert "restored" in msg.lower(), msg
        row = file_db.get_media_by_id(media_id, include_trash=True)
        assert row["is_trash"] == 0
        assert row["content"] == "brand new content"
        assert self._chunk_count(file_db, media_id) == 1
        remaining = file_db.execute_query(
            "SELECT chunk_text FROM UnvectorizedMediaChunks WHERE media_id = ? AND deleted = 0",
            (media_id,),
        ).fetchone()
        assert remaining["chunk_text"] == "fresh chunk"


class TestReimportAfterTrashKeywords:
    """task-4022 review round 2 (I2, Important), corrected by the P1
    re-critique (finding 2): restoring a trashed row must not silently wipe
    the user's curated keywords just because the caller re-importing it
    never supplied a ``keywords`` argument at all (``keywords=None``, the
    default) -- most restore callers simply have no opinion on keywords.
    An explicit ``keywords=[]`` is a different signal (the caller DOES want
    them cleared) and must not be conflated with "not supplied" -- see
    ``TestReimportAfterTrashKeywords.test_restore_with_overwrite_and_
    explicit_empty_keywords_clears_them`` below for that contract. The I2
    fix originally kept both signals indistinguishable (``keywords_norm``
    is ``[]`` either way); this class now covers all three: not supplied
    (preserve), non-empty (apply), and explicit empty (clear)."""

    def test_restore_with_keywords_omitted_preserves_existing_keywords(
        self, file_db
    ):
        url = "file:///keywords/preserve.txt"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="preserve.txt",
            media_type="document",
            content="content with curated keywords",
            keywords=["mine", "important"],
            url=url,
        )
        before = sorted(
            r["keyword"]
            for r in file_db.execute_query(
                "SELECT k.keyword FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id "
                "WHERE mk.media_id = ? AND k.deleted = 0",
                (media_id,),
            ).fetchall()
        )
        assert before == ["important", "mine"]
        assert file_db.mark_as_trash(media_id) is True

        # ``keywords`` is genuinely omitted here (defaults to ``None``) --
        # NOT an explicit ``keywords=[]`` -- the two must no longer behave
        # the same way (see finding 2).
        reimported_id, _, msg = file_db.add_media_with_keywords(
            title="preserve.txt",
            media_type="document",
            content="content with curated keywords",
            url=url,
            restore_trashed=True,
        )

        assert reimported_id == media_id, msg
        assert "restored" in msg.lower(), msg
        after = sorted(
            r["keyword"]
            for r in file_db.execute_query(
                "SELECT k.keyword FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id "
                "WHERE mk.media_id = ? AND k.deleted = 0",
                (media_id,),
            ).fetchall()
        )
        assert after == ["important", "mine"], (
            "restore must not wipe curated keywords just because the "
            "re-import call didn't supply a keywords argument at all"
        )

    def test_restore_with_nonempty_keywords_still_applies_them(self, file_db):
        """Guard rail: the I2 fix must only skip the sync when the
        INCOMING keyword list is empty -- a restore that DOES supply
        keywords must still apply them normally (replace, not merge;
        unchanged, pre-existing behavior)."""
        url = "file:///keywords/replace.txt"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="replace.txt",
            media_type="document",
            content="content",
            keywords=["old-keyword"],
            url=url,
        )
        assert file_db.mark_as_trash(media_id) is True

        reimported_id, _, msg = file_db.add_media_with_keywords(
            title="replace.txt",
            media_type="document",
            content="content",
            keywords=["new-keyword"],
            url=url,
            restore_trashed=True,
        )

        assert reimported_id == media_id, msg
        after = sorted(
            r["keyword"]
            for r in file_db.execute_query(
                "SELECT k.keyword FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id "
                "WHERE mk.media_id = ? AND k.deleted = 0",
                (media_id,),
            ).fetchall()
        )
        assert after == ["new-keyword"]

    def test_restore_with_overwrite_and_explicit_empty_keywords_clears_them(
        self, file_db
    ):
        """P1 re-critique finding 2: the I2 guard above must not make an
        explicit clear impossible. ``keywords_norm`` alone can't tell
        "caller didn't pass keywords" (``None``) apart from "caller wants
        them all gone" (``[]``) -- both normalise to the same empty list.
        With ``overwrite=True`` and an explicit ``keywords=[]``, a restore
        must still clear the row's existing keywords, exactly as a plain
        (non-restore) ``overwrite=True`` + ``keywords=[]`` call already
        does."""
        url = "file:///keywords/explicit-clear.txt"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="explicit-clear.txt",
            media_type="document",
            content="content with keywords to be explicitly cleared",
            keywords=["stale", "obsolete"],
            url=url,
        )
        before = sorted(
            r["keyword"]
            for r in file_db.execute_query(
                "SELECT k.keyword FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id "
                "WHERE mk.media_id = ? AND k.deleted = 0",
                (media_id,),
            ).fetchall()
        )
        assert before == ["obsolete", "stale"]
        assert file_db.mark_as_trash(media_id) is True

        reimported_id, _, msg = file_db.add_media_with_keywords(
            title="explicit-clear.txt",
            media_type="document",
            content="content with keywords to be explicitly cleared",
            keywords=[],
            url=url,
            overwrite=True,
            restore_trashed=True,
        )

        assert reimported_id == media_id, msg
        assert "restored" in msg.lower(), msg
        after = file_db.execute_query(
            "SELECT k.keyword FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id "
            "WHERE mk.media_id = ? AND k.deleted = 0",
            (media_id,),
        ).fetchall()
        assert after == [], (
            "overwrite=True + an explicit keywords=[] must clear keywords "
            "during a restore, exactly as it does outside a restore"
        )


class TestReimportAfterTrashUrlCanonicalization:
    """task-4022 review round 2 (I3, Important): round 1's identical-
    content restore branch wrote ``url`` unconditionally, reversing the
    pre-existing ``is_canonicalisation`` rule's deliberate one direction
    (auto-generated ``local://...`` -> a real url, never the reverse).
    Reproduces the reviewer's own probe: a row imported from a real,
    canonical source url, trashed, then re-imported from a local file path
    with byte-identical content must NOT have its canonical source url
    overwritten by the local path."""

    def test_canonical_source_url_survives_restore_from_local_path(self, file_db):
        canonical_url = "https://example.com/canonical-article"
        content = "identical bytes, canonical source vs. local re-import"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="canonical-article",
            media_type="article",
            content=content,
            keywords=[],
            url=canonical_url,
        )
        assert file_db.mark_as_trash(media_id) is True

        local_url = "file:///Users/me/Downloads/article.txt"
        reimported_id, _, msg = file_db.add_media_with_keywords(
            title="canonical-article",
            media_type="article",
            content=content,
            keywords=[],
            url=local_url,
            restore_trashed=True,
        )

        assert reimported_id == media_id, msg
        assert "restored" in msg.lower(), msg
        row = file_db.get_media_by_id(media_id, include_trash=True)
        assert row["is_trash"] == 0
        assert row["url"] == canonical_url, (
            "the row's canonical source url must survive a restore "
            "triggered by a less-canonical (local file) re-import"
        )


class TestReimportAfterTrashCombined:
    """The reviewer's own suggestion for coverage: ONE real-DB test
    exercising restore with ``chunks=[...]`` AND pre-existing keywords AND
    a canonical existing url together -- this combination would have
    caught C1 (chunk UNIQUE collision), I2 (keyword wipe), and I3 (url
    reversal) all at once, where the per-finding tests above each isolate
    a single dimension."""

    def test_restore_with_chunks_keywords_and_canonical_url_all_at_once(
        self, file_db
    ):
        canonical_url = "https://example.com/deep-dive"
        content = "identical bytes for the combined C1+I2+I3 regression"
        media_id, _, _ = file_db.add_media_with_keywords(
            title="deep-dive",
            media_type="article",
            content=content,
            keywords=["mine", "curated"],
            url=canonical_url,
            # Explicit chunk_type: SQLite treats a NULL chunk_type as
            # always distinct, so an omitted chunk_type would never
            # actually exercise the UNIQUE(media_id, chunk_index,
            # chunk_type) collision C1 describes.
            chunks=[
                {"text": "stale chunk one", "chunk_type": "text"},
                {"text": "stale chunk two", "chunk_type": "text"},
            ],
        )
        assert file_db.mark_as_trash(media_id) is True

        local_url = "file:///Users/me/Downloads/deep-dive.txt"
        reimported_id, reimported_uuid, msg = file_db.add_media_with_keywords(
            title="deep-dive",
            media_type="article",
            content=content,
            # I2, corrected by the P1 re-critique (finding 2): ``keywords``
            # is genuinely OMITTED here, not an explicit ``keywords=[]`` --
            # only "no keywords argument at all" preserves the existing
            # curated set; an explicit empty list now clears them (see
            # ``TestReimportAfterTrashKeywords.test_restore_with_
            # overwrite_and_explicit_empty_keywords_clears_them``).
            url=local_url,  # I3: must not clobber the canonical https url
            chunks=[  # C1: must not IntegrityError against the old chunks
                {"text": "fresh chunk one", "chunk_type": "text"},
                {"text": "fresh chunk two", "chunk_type": "text"},
                {"text": "fresh chunk three", "chunk_type": "text"},
            ],
            restore_trashed=True,
        )

        assert reimported_id == media_id, msg
        assert "restored" in msg.lower(), msg

        row = file_db.get_media_by_id(media_id, include_trash=True)
        assert row["is_trash"] == 0
        assert row["trash_date"] is None
        # I3: canonical url survives a restore from a less-canonical path.
        assert row["url"] == canonical_url

        # I2: pre-existing curated keywords survive an omitted keywords arg.
        keywords = sorted(
            r["keyword"]
            for r in file_db.execute_query(
                "SELECT k.keyword FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id "
                "WHERE mk.media_id = ? AND k.deleted = 0",
                (media_id,),
            ).fetchall()
        )
        assert keywords == ["curated", "mine"]

        # C1: the new chunks replaced the old ones without an
        # IntegrityError, and there's exactly the new set, not a union.
        chunk_texts = sorted(
            r["chunk_text"]
            for r in file_db.execute_query(
                "SELECT chunk_text FROM UnvectorizedMediaChunks WHERE media_id = ? AND deleted = 0",
                (media_id,),
            ).fetchall()
        )
        assert chunk_texts == [
            "fresh chunk one",
            "fresh chunk three",
            "fresh chunk two",
        ]

        cursor = file_db.execute_query("SELECT COUNT(*) FROM Media")
        assert cursor.fetchone()[0] == 1


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
        assert logs[0]["change_id"] == 1

    def test_get_sync_log_entries_since(self):
        logs = self.db.get_sync_log_entries(since_change_id=2)
        assert len(logs) == 2
        assert logs[0]["change_id"] == 3

    def test_get_sync_log_entries_limit(self):
        logs = self.db.get_sync_log_entries(limit=2)
        assert len(logs) == 2
        assert logs[0]["change_id"] == 1
        assert logs[1]["change_id"] == 2

    def test_delete_sync_log_entries_specific(self):
        initial_logs = self.db.get_sync_log_entries()
        ids_to_delete = [initial_logs[1]["change_id"], initial_logs[2]["change_id"]]
        deleted_count = self.db.delete_sync_log_entries(ids_to_delete)
        assert deleted_count == 2
        remaining_ids = {log["change_id"] for log in self.db.get_sync_log_entries()}
        assert remaining_ids == {1, 4}

    def test_delete_sync_log_entries_before(self):
        deleted_count = self.db.delete_sync_log_entries_before(3)
        assert deleted_count == 3
        remaining_logs = self.db.get_sync_log_entries()
        assert len(remaining_logs) == 1
        assert remaining_logs[0]["change_id"] == 4

    def test_delete_sync_log_entries_invalid_id(self):
        with pytest.raises(ValueError):
            self.db.delete_sync_log_entries([1, "two", 3])


@pytest.mark.integration
class TestGetAllActiveMediaIds:
    """``get_all_active_media_ids`` -- the truncation-proof id source for
    Library chatbook export (see ``Library/library_export_scope.py``).

    Mirrors ``get_paginated_files``'s ``WHERE deleted = 0 AND is_trash = 0``
    visibility, but returns every matching id (no page cap).
    """

    def test_returns_only_active_non_deleted_non_trashed_ids(self, db_instance):
        active_id, _, _ = db_instance.add_media_with_keywords(
            title="Active", content="active content", media_type="article"
        )
        deleted_id, _, _ = db_instance.add_media_with_keywords(
            title="Deleted", content="deleted content", media_type="article"
        )
        trashed_id, _, _ = db_instance.add_media_with_keywords(
            title="Trashed", content="trashed content", media_type="article"
        )
        db_instance.soft_delete_media(deleted_id)
        db_instance.mark_as_trash(trashed_id)

        ids = db_instance.get_all_active_media_ids()

        assert ids == [active_id]

    def test_filters_by_type_when_given(self, db_instance):
        video_id, _, _ = db_instance.add_media_with_keywords(
            title="V1", content="video content", media_type="video"
        )
        db_instance.add_media_with_keywords(
            title="A1", content="article content", media_type="article"
        )

        ids = db_instance.get_all_active_media_ids(media_type="video")

        assert ids == [video_id]

    def test_no_media_type_returns_all_active_types(self, db_instance):
        video_id, _, _ = db_instance.add_media_with_keywords(
            title="V1", content="video content", media_type="video"
        )
        article_id, _, _ = db_instance.add_media_with_keywords(
            title="A1", content="article content", media_type="article"
        )

        ids = db_instance.get_all_active_media_ids()

        assert set(ids) == {video_id, article_id}

    def test_returns_every_row_beyond_a_50_row_page_cap(self, db_instance):
        """The Library media snapshot caps at 50 rows -- this DB method must not."""
        seeded_ids = []
        for i in range(55):
            media_id, _, _ = db_instance.add_media_with_keywords(
                title=f"Media {i}", content=f"content {i}", media_type="article"
            )
            seeded_ids.append(media_id)

        ids = db_instance.get_all_active_media_ids()

        assert set(ids) == set(seeded_ids)
        assert len(ids) == 55

    def test_empty_db_returns_empty_list(self, db_instance):
        assert db_instance.get_all_active_media_ids() == []


#
# End of test_media_db_v2.py
########################################################################################################################
