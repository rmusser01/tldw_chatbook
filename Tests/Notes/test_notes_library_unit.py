# tldw_Server_API/tests/Notes/test_notes_library_unit.py
import unittest
import os
import stat
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
from loguru import logger
import sqlite3

from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryMigrationSeed
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDBError as Actual_CharactersRAGDBError,
    ConflictError as Actual_ConflictError,
    CharactersRAGDB,
)
from tldw_chatbook.Notes.Notes_Library import NotesInteropService

MODULE_PATH_PREFIX_CHACHA_DB = "tldw_chatbook.DB.ChaChaNotes_DB"
NOTES_LIBRARY_MODULE_PATH = "tldw_chatbook.Notes.Notes_Library"
CHARACHERS_RAGDB_CLASS_PATCH_TARGET = f"{NOTES_LIBRARY_MODULE_PATH}.CharactersRAGDB"


class TestNotesInteropService(unittest.TestCase):
    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="notes_service_test_")
        self.addCleanup(self.temp_dir_obj.cleanup)
        self.base_db_dir = Path(self.temp_dir_obj.name).resolve()
        self.api_client_id = "test_api_client_v1"

        # autospec (not bare ``spec``) so the double enforces the *real*
        # ``CharactersRAGDB.__init__`` signature at call time. A production
        # signature change now raises a TypeError here instead of silently
        # drifting away from the expected-call literal below (TASK-21531).
        self.mock_ragdb_class_patcher = patch(
            CHARACHERS_RAGDB_CLASS_PATCH_TARGET, autospec=True
        )
        self.MockCharactersRAGDB_class = self.mock_ragdb_class_patcher.start()
        self.addCleanup(self.mock_ragdb_class_patcher.stop)

        # The migration seed is loaded from config by production; pin it to a
        # non-default sentinel so the constructor assertion proves the loaded
        # value is forwarded, not that *some* seed object was passed.
        self.migration_seed = ConsoleLibraryMigrationSeed(auto_retrieve_on_send=True)
        self.mock_seed_loader_patcher = patch(
            f"{NOTES_LIBRARY_MODULE_PATH}.load_console_library_migration_seed",
            return_value=self.migration_seed,
        )
        self.mock_seed_loader = self.mock_seed_loader_patcher.start()
        self.addCleanup(self.mock_seed_loader_patcher.stop)

        self.mock_notes_library_logger_patcher = patch(
            f"{NOTES_LIBRARY_MODULE_PATH}.logger", spec=True
        )
        self.mock_notes_library_logger = self.mock_notes_library_logger_patcher.start()
        self.addCleanup(self.mock_notes_library_logger_patcher.stop)

        self.mock_db_instance = MagicMock(spec=CharactersRAGDB)
        self.MockCharactersRAGDB_class.return_value = self.mock_db_instance

        # Create a mock global DB template
        self.mock_global_db = MagicMock(spec=CharactersRAGDB)
        self.mock_global_db.db_path_str = str(self.base_db_dir / "unified.db")

        self.service = NotesInteropService(
            base_db_directory=str(self.base_db_dir),
            api_client_id=self.api_client_id,
            global_db_to_use=self.mock_global_db,
        )

    def tearDown(self):
        if hasattr(self, "service") and self.service:
            try:
                self.service.close_all_user_connections()
            except Exception as e:
                logger.error(
                    f"Error during service.close_all_user_connections() in tearDown: {e}",
                    exc_info=True,
                )

    def test_initialization(self):
        self.assertTrue(self.base_db_dir.exists())
        self.assertEqual(self.service.api_client_id, self.api_client_id)
        # The selected parent is verified in place rather than created.
        self.mock_notes_library_logger.info.assert_any_call(
            f"NotesInteropService: Verified base directory: {self.base_db_dir}"
        )

    def test_initialization_requires_existing_base_directory(self):
        missing_base = self.base_db_dir / "missing"
        expected_msg_part = f"Failed to verify base DB directory {missing_base}:"
        with self.assertRaises(Actual_CharactersRAGDBError) as cm:
            NotesInteropService(
                base_db_directory=str(missing_base),
                api_client_id="fail_client",
                global_db_to_use=self.mock_global_db,
            )
        self.assertIn(expected_msg_part, str(cm.exception))
        self.assertFalse(missing_base.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX mode contract")
    def test_initialization_does_not_mutate_existing_base_directory_mode(self):
        self.base_db_dir.chmod(0o751)

        service = NotesInteropService(
            base_db_directory=self.base_db_dir,
            api_client_id="mode-test",
            global_db_to_use=self.mock_global_db,
        )

        self.addCleanup(service.close_all_user_connections)
        self.assertEqual(stat.S_IMODE(self.base_db_dir.stat().st_mode), 0o751)

    def test_get_db_new_instance(self):
        user_id = "user1"
        db_instance = self.service._get_db(user_id)
        # The per-user instance points at the unified DB path, uses user_id as
        # its client_id, and carries the config-loaded legacy migration seed
        # (TASK-19900) so a reopened legacy DB migrates with the user's real
        # pre-upgrade automatic-retrieval value.
        self.MockCharactersRAGDB_class.assert_called_once_with(
            db_path=self.mock_global_db.db_path_str,
            client_id=user_id,
            console_library_migration_seed=self.migration_seed,
        )
        self.mock_seed_loader.assert_called_once_with()
        self.assertIs(db_instance, self.mock_db_instance)

    def test_get_db_cached_instance(self):
        user_id = "user1"
        self.service._get_db(user_id)
        self.MockCharactersRAGDB_class.assert_called_once()
        self.MockCharactersRAGDB_class.reset_mock()
        db_instance_cached = self.service._get_db(user_id)
        self.MockCharactersRAGDB_class.assert_not_called()
        self.assertIs(db_instance_cached, self.mock_db_instance)

    def test_get_db_invalid_user_id_empty(self):
        with self.assertRaisesRegex(ValueError, "user_id must be a non-empty string."):
            self.service._get_db("")

    def test_get_db_invalid_user_id_whitespace(self):
        # This test relies on Notes_Library.py's _get_db user_id validation being:
        # `if not isinstance(user_id, str) or not user_id.strip():`
        user_id_whitespace = "   "
        with self.assertRaisesRegex(ValueError, "user_id must be a non-empty string."):
            self.service._get_db(user_id_whitespace)

    def test_get_db_invalid_user_id_none(self):
        with self.assertRaisesRegex(ValueError, "user_id must be a non-empty string."):
            self.service._get_db(None)

    def test_get_db_init_failure_ragdb_error(self):
        db_error_message = "DB init failed via class from RAGDBError"
        db_error_instance = Actual_CharactersRAGDBError(db_error_message)
        self.MockCharactersRAGDB_class.side_effect = db_error_instance
        user_id = "user_fail_ragdb"
        with self.assertRaises(Actual_CharactersRAGDBError) as cm:
            self.service._get_db(user_id)
        self.assertIs(cm.exception, db_error_instance)

        # Expecting the log message from the except (CharactersRAGDBError, SchemaError, sqlite3.Error) block
        expected_log_message = f"Failed to initialize dynamic CharactersRAGDB instance for user '{user_id}': {db_error_message}"
        self.mock_notes_library_logger.error.assert_called_once_with(
            expected_log_message, exc_info=True
        )

    def test_get_db_init_failure_sqlite_error(self):
        sqlite_error_message = "SQLite connection failed from sqlite3.Error"
        sqlite_error_instance = sqlite3.Error(sqlite_error_message)
        self.MockCharactersRAGDB_class.side_effect = sqlite_error_instance
        user_id = "user_fail_sqlite"
        with self.assertRaises(sqlite3.Error) as cm:
            self.service._get_db(user_id)
        self.assertIs(cm.exception, sqlite_error_instance)

        # Expecting the log message from the except (CharactersRAGDBError, SchemaError, sqlite3.Error) block
        expected_log_message = f"Failed to initialize dynamic CharactersRAGDB instance for user '{user_id}': {sqlite_error_message}"
        self.mock_notes_library_logger.error.assert_called_once_with(
            expected_log_message, exc_info=True
        )

    def test_get_db_init_failure_unexpected_error(self):
        self.MockCharactersRAGDB_class.side_effect = Exception("Unexpected boom")
        user_id = "user_generic_fail"
        with self.assertRaisesRegex(
            Actual_CharactersRAGDBError,
            f"Unexpected error initializing DB instance for user {user_id}: Unexpected boom",
        ):
            self.service._get_db(user_id)
        self.mock_notes_library_logger.error.assert_called_once_with(
            f"Unexpected error initializing dynamic CharactersRAGDB for user '{user_id}': Unexpected boom",
            exc_info=True,
        )

    def test_add_note(self):
        user_id, title, content, expected_note_id = (
            "user1",
            "Test Note",
            "Test Content",
            "note_uuid_1",
        )
        self.mock_db_instance.add_note.return_value = expected_note_id
        note_id = self.service.add_note(user_id, title, content)
        self.mock_db_instance.add_note.assert_called_once_with(
            title=title, content=content, note_id=None
        )
        self.assertEqual(note_id, expected_note_id)

    def test_add_note_with_provided_id(self):
        user_id, title, content, provided_note_id = (
            "user1",
            "Test Note",
            "Test Content",
            "client_note_id",
        )
        self.mock_db_instance.add_note.return_value = provided_note_id
        note_id = self.service.add_note(
            user_id, title, content, note_id=provided_note_id
        )
        self.mock_db_instance.add_note.assert_called_once_with(
            title=title, content=content, note_id=provided_note_id
        )
        self.assertEqual(note_id, provided_note_id)

    def test_add_note_returns_none_unexpectedly(self):
        user_id, title, content = "user1", "Test Note", "Test Content"
        self.mock_db_instance.add_note.return_value = None
        with self.assertRaisesRegex(
            Actual_CharactersRAGDBError,
            "Failed to create note, received None ID unexpectedly",
        ):
            self.service.add_note(user_id, title, content)
        self.mock_notes_library_logger.error.assert_called_once_with(
            f"add_note for user_id '{user_id}' (as client_id) returned None unexpectedly for title '{title}'."
        )

    def test_get_note_by_id(self):
        user_id, note_id_val = "user1", "note_uuid_1"
        expected_data = {"id": note_id_val, "title": "Test"}
        self.mock_db_instance.get_note_by_id.return_value = expected_data
        note = self.service.get_note_by_id(user_id, note_id_val)
        self.mock_db_instance.get_note_by_id.assert_called_once_with(
            note_id=note_id_val
        )
        self.assertEqual(note, expected_data)

    def test_list_notes(self):
        user_id, expected_notes = "user1", [{"id": "1"}, {"id": "2"}]
        self.mock_db_instance.list_notes.return_value = expected_notes
        notes = self.service.list_notes(user_id, limit=10, offset=0)
        self.mock_db_instance.list_notes.assert_called_once_with(limit=10, offset=0)
        self.assertEqual(notes, expected_notes)

    def test_update_note(self):
        user_id, note_id_val, update_data, expected_version = (
            "user1",
            "note_uuid_1",
            {"title": "New Title"},
            1,
        )
        self.mock_db_instance.update_note.return_value = True
        success = self.service.update_note(
            user_id, note_id_val, update_data, expected_version
        )
        self.mock_db_instance.update_note.assert_called_once_with(
            note_id=note_id_val,
            update_data=update_data,
            expected_version=expected_version,
        )
        self.assertTrue(success)

    def test_update_note_conflict(self):
        user_id, note_id_val, update_data, expected_version = (
            "user1",
            "note_uuid_1",
            {"title": "New Title"},
            1,
        )
        conflict_error_instance = Actual_ConflictError(
            "DB version mismatch", entity="notes", entity_id=note_id_val
        )
        self.mock_db_instance.update_note.side_effect = conflict_error_instance
        with self.assertRaises(Actual_ConflictError) as cm:
            self.service.update_note(
                user_id, note_id_val, update_data, expected_version
            )
        self.mock_db_instance.update_note.assert_called_once_with(
            note_id=note_id_val,
            update_data=update_data,
            expected_version=expected_version,
        )
        self.assertIs(cm.exception, conflict_error_instance)

    def test_soft_delete_note(self):
        user_id, note_id_val, expected_version = "user1", "note_uuid_1", 2
        self.mock_db_instance.soft_delete_note.return_value = True
        success = self.service.soft_delete_note(user_id, note_id_val, expected_version)
        self.mock_db_instance.soft_delete_note.assert_called_once_with(
            note_id=note_id_val, expected_version=expected_version
        )
        self.assertTrue(success)

    def test_soft_delete_note_conflict(self):
        user_id, note_id_val, expected_version = "user1", "note_uuid_1", 2
        conflict_error_instance = Actual_ConflictError(
            "Cannot delete", entity="notes", entity_id=note_id_val
        )
        self.mock_db_instance.soft_delete_note.side_effect = conflict_error_instance
        with self.assertRaises(Actual_ConflictError) as cm:
            self.service.soft_delete_note(user_id, note_id_val, expected_version)
        self.mock_db_instance.soft_delete_note.assert_called_once_with(
            note_id=note_id_val, expected_version=expected_version
        )
        self.assertIs(cm.exception, conflict_error_instance)

    def test_search_notes(self):
        user_id, term = "user1", "search term"
        expected_results = [{"id": "1", "content": "Contains search term"}]
        self.mock_db_instance.search_notes.return_value = expected_results
        results = self.service.search_notes(user_id, term, limit=5)
        self.mock_db_instance.search_notes.assert_called_once_with(
            search_term=term, limit=5
        )
        self.assertEqual(results, expected_results)

    def test_add_keyword(self):
        user_id, keyword_text, expected_keyword_id = "user1", "test_keyword", 1
        self.mock_db_instance.add_keyword.return_value = expected_keyword_id
        keyword_id = self.service.add_keyword(user_id, keyword_text)
        self.mock_db_instance.add_keyword.assert_called_once_with(
            keyword_text=keyword_text
        )
        self.assertEqual(keyword_id, expected_keyword_id)

    def test_link_note_to_keyword(self):
        user_id, note_id_val, keyword_id_val = "user1", "note_uuid_1", 1
        self.mock_db_instance.link_note_to_keyword.return_value = True
        success = self.service.link_note_to_keyword(
            user_id, note_id_val, keyword_id_val
        )
        self.mock_db_instance.link_note_to_keyword.assert_called_once_with(
            note_id=note_id_val, keyword_id=keyword_id_val
        )
        self.assertTrue(success)

    def test_close_user_connection(self):
        user_id = "user1"
        db_mock = self.service._get_db(user_id)
        self.assertIs(db_mock, self.mock_db_instance)
        self.assertIn(user_id, self.service._db_instances)
        self.service.close_user_connection(user_id)
        self.mock_db_instance.close_connection.assert_called_once()
        self.assertNotIn(user_id, self.service._db_instances)
        self.mock_notes_library_logger.info.assert_any_call(
            f"Closed and removed DB instance for user context '{user_id}'."
        )

    def test_close_user_connection_not_exist(self):
        user_id = "non_existent_user"
        self.service.close_user_connection(user_id)
        self.mock_db_instance.close_connection.assert_not_called()
        self.mock_notes_library_logger.debug.assert_any_call(
            f"No active DB instance found in cache for user context '{user_id}' to close."
        )

    def test_close_all_user_connections(self):
        user1_id, user2_id = "user1_for_close_all", "user2_for_close_all"
        mock_db_1_instance, mock_db_2_instance = (
            MagicMock(spec=CharactersRAGDB),
            MagicMock(spec=CharactersRAGDB),
        )
        self.MockCharactersRAGDB_class.side_effect = [
            mock_db_1_instance,
            mock_db_2_instance,
        ]
        db_instance1_ret, db_instance2_ret = (
            self.service._get_db(user1_id),
            self.service._get_db(user2_id),
        )
        self.assertIs(db_instance1_ret, mock_db_1_instance)
        self.assertIs(db_instance2_ret, mock_db_2_instance)
        self.service.close_all_user_connections()
        mock_db_1_instance.close_connection.assert_called_once()
        mock_db_2_instance.close_connection.assert_called_once()
        self.assertEqual(len(self.service._db_instances), 0)
        self.mock_notes_library_logger.info.assert_any_call(
            "All cached user-context DB instances have been processed for closure."
        )
        self.MockCharactersRAGDB_class.side_effect = None
        self.MockCharactersRAGDB_class.return_value = self.mock_db_instance

    def test_close_connection_exception(self):
        user_id = "user_close_fail"
        db_mock = self.service._get_db(user_id)
        self.assertIs(db_mock, self.mock_db_instance)
        self.mock_db_instance.close_connection.side_effect = Exception(
            "Failed to close"
        )
        self.service.close_user_connection(user_id)
        self.assertNotIn(user_id, self.service._db_instances)
        self.mock_notes_library_logger.error.assert_called_with(
            f"Error closing DB instance for user context '{user_id}': Failed to close",
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Library query seams (task-1337 plan Task 2)
# ---------------------------------------------------------------------------


def _make_library_notes_db():
    return CharactersRAGDB(db_path=":memory:", client_id="library_test_client")


def _seed_library_note(db, *, title, content=None, keywords=(), last_modified=None):
    note_id = db.add_note(
        title=title, content=content if content is not None else f"body for {title}"
    )
    for keyword in keywords:
        keyword_id = db.add_keyword(keyword)
        db.link_note_to_keyword(note_id, keyword_id)
    if last_modified is not None:
        db.execute_query(
            "UPDATE notes SET last_modified = ?, version = version + 1 WHERE id = ?",
            (last_modified, note_id),
        )
    return note_id


def test_library_notes_page_lists_active_with_stable_order():
    db = _make_library_notes_db()
    try:
        first_id = _seed_library_note(
            db, title="First", last_modified="2026-01-01 00:00:00"
        )
        second_id = _seed_library_note(
            db, title="Second", last_modified="2026-01-03 00:00:00"
        )
        third_id = _seed_library_note(
            db, title="Third", last_modified="2026-01-02 00:00:00"
        )
        deleted_id = _seed_library_note(db, title="Deleted")
        db.soft_delete_note(deleted_id, expected_version=1)

        page_one = db.list_library_notes_page(limit=2, offset=0)
        assert page_one["total"] == 3
        assert [item["id"] for item in page_one["items"]] == [second_id, third_id]

        page_two = db.list_library_notes_page(limit=2, offset=2)
        assert page_two["total"] == 3
        assert [item["id"] for item in page_two["items"]] == [first_id]

        beyond = db.list_library_notes_page(limit=10, offset=50)
        assert beyond["total"] == 3
        assert beyond["items"] == []
    finally:
        db.close_connection()


def test_library_notes_page_projection_and_keyword_cap():
    db = _make_library_notes_db()
    try:
        keywords = [f"kw{index:02d}" for index in range(25)]
        _seed_library_note(db, title="Heavy", content="secret body " * 100, keywords=keywords)

        item = db.list_library_notes_page(limit=10, offset=0)["items"][0]
        assert "content" not in item
        assert len(item["preview"]) <= 241
        assert len(item["keywords"]) == 20
        assert item["keyword_total"] == 25
        assert item["keywords_truncated"] is True
    finally:
        db.close_connection()


def test_library_notes_search_exact_title_first_and_distinct_total():
    db = _make_library_notes_db()
    try:
        exact_id = _seed_library_note(
            db,
            title="Quarterly",
            content="nothing relevant here",
            keywords=("quarterly", "quarterly-finance"),
            last_modified="2026-01-01 00:00:00",
        )
        content_id = _seed_library_note(
            db,
            title="Other",
            content="a quarterly deep dive",
            last_modified="2026-02-01 00:00:00",
        )

        payload = db.search_library_notes_page(query="quarterly", limit=10, offset=0)
        assert payload["total"] == 2
        assert [item["id"] for item in payload["items"]] == [exact_id, content_id]
        exact_item, content_item = payload["items"]
        assert "title" in exact_item["matched_fields"]
        assert "keywords" in exact_item["matched_fields"]
        assert "quarterly" in exact_item["matched_keywords"]
        assert "quarterly-finance" in exact_item["matched_keywords"]
        assert "content" in content_item["matched_fields"]
    finally:
        db.close_connection()


def test_library_notes_search_treats_wildcards_and_operators_literally():
    db = _make_library_notes_db()
    try:
        target_id = _seed_library_note(db, title="100% ready_now", content="plain")
        _seed_library_note(db, title="readyXnow decoy", content="plain decoy body")

        percent = db.search_library_notes_page(query="100%", limit=10, offset=0)
        assert [item["id"] for item in percent["items"]] == [target_id]

        underscore = db.search_library_notes_page(query="ready_now", limit=10, offset=0)
        assert [item["id"] for item in underscore["items"]] == [target_id]

        for hostile in ('"unclosed', "ready OR", "AND )(", "ready*", "NEAR/1"):
            result = db.search_library_notes_page(query=hostile, limit=10, offset=0)
            assert isinstance(result["total"], int)
            assert isinstance(result["items"], list)
    finally:
        db.close_connection()


def test_library_notes_detail_windows_text_and_missing_returns_none():
    db = _make_library_notes_db()
    try:
        content = "abcdef" * 900  # 5400 chars
        note_id = _seed_library_note(db, title="Long read", content=content)

        detail = db.get_library_note_text(note_id, start=1200, max_chars=2000)
        assert detail is not None
        assert detail["id"] == note_id
        assert detail["title"] == "Long read"
        assert detail["total_chars"] == len(content)
        assert detail["start"] == 1200
        assert detail["returned_chars"] == 2000
        assert detail["has_more"] is True
        assert detail["text"] == content[1200:3200]
        assert "content" not in detail

        tail = db.get_library_note_text(note_id, start=5000, max_chars=2000)
        assert tail["text"] == content[5000:]
        assert tail["has_more"] is False

        assert db.get_library_note_text("no-such-id", start=0, max_chars=100) is None
    finally:
        db.close_connection()


def test_library_note_detail_read_runs_inside_transaction():
    db = _make_library_notes_db()
    try:
        note_id = _seed_library_note(db, title="Transactional", content="body")
        conn = db.get_connection()
        conn.commit()
        observed: list[bool] = []

        def record_transaction_state(sql: str) -> None:
            if "FROM notes" in sql and "AS total_chars" in sql:
                observed.append(conn.in_transaction)

        conn.set_trace_callback(record_transaction_state)
        try:
            assert db.get_library_note_text(note_id, start=0, max_chars=20) is not None
        finally:
            conn.set_trace_callback(None)

        assert observed == [True]
    finally:
        db.close_connection()


class TestLibraryNotesInteropDelegates(TestNotesInteropService):
    """NotesInteropService delegates for the Library read seams (task-1337)."""

    def test_list_library_notes_delegates_and_echoes_pagination(self):
        self.mock_db_instance.list_library_notes_page.return_value = {
            "items": [{"id": "n-1", "title": "One"}],
            "total": 7,
        }

        payload = self.service.list_library_notes("user-1", limit=5, offset=10)

        self.mock_db_instance.list_library_notes_page.assert_called_once_with(
            limit=5, offset=10
        )
        self.assertEqual(payload["total"], 7)
        self.assertEqual(payload["offset"], 10)
        self.assertEqual(payload["limit"], 5)
        self.assertEqual(payload["items"], [{"id": "n-1", "title": "One"}])

    def test_search_library_notes_delegates_query_and_pagination(self):
        self.mock_db_instance.search_library_notes_page.return_value = {
            "items": [],
            "total": 0,
        }

        payload = self.service.search_library_notes(
            "user-1", query="quarterly", limit=3, offset=6
        )

        self.mock_db_instance.search_library_notes_page.assert_called_once_with(
            query="quarterly", limit=3, offset=6
        )
        self.assertEqual(payload["total"], 0)
        self.assertEqual(payload["offset"], 6)
        self.assertEqual(payload["limit"], 3)

    def test_get_library_note_text_delegates_window(self):
        self.mock_db_instance.get_library_note_text.return_value = {
            "id": "n-1",
            "text": "segment",
        }

        detail = self.service.get_library_note_text(
            "user-1", "n-1", start=120, max_chars=400
        )

        self.mock_db_instance.get_library_note_text.assert_called_once_with(
            "n-1", start=120, max_chars=400
        )
        self.assertEqual(detail["text"], "segment")

    def test_get_library_note_text_passes_through_missing(self):
        self.mock_db_instance.get_library_note_text.return_value = None
        self.assertIsNone(
            self.service.get_library_note_text("user-1", "missing", start=0, max_chars=10)
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
