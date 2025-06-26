# tldw_Server_API/tests/Notes/test_notes_library_unit.py
import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import tempfile
import re
import logging
import sqlite3
from typing import Any, Optional, Dict

from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDBError as Actual_CharactersRAGDBError,
    SchemaError as Actual_SchemaError,
    InputError as Actual_InputError,
    ConflictError as Actual_ConflictError, CharactersRAGDB
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

        self.mock_ragdb_class_patcher = patch(CHARACHERS_RAGDB_CLASS_PATCH_TARGET, spec=True)
        self.MockCharactersRAGDB_class = self.mock_ragdb_class_patcher.start()
        self.addCleanup(self.mock_ragdb_class_patcher.stop)

        self.mock_notes_library_logger_patcher = patch(f'{NOTES_LIBRARY_MODULE_PATH}.logger', spec=True)
        self.mock_notes_library_logger = self.mock_notes_library_logger_patcher.start()
        self.addCleanup(self.mock_notes_library_logger_patcher.stop)

        self.mock_db_instance = MagicMock(spec=CharactersRAGDB)
        self.MockCharactersRAGDB_class.return_value = self.mock_db_instance
        
        # Create a mock global DB template
        self.mock_global_db = MagicMock(spec=CharactersRAGDB)
        self.mock_global_db.db_path_str = str(self.base_db_dir / "unified.db")

        self.service = NotesInteropService(base_db_directory=str(self.base_db_dir),
                                           api_client_id=self.api_client_id,
                                           global_db_to_use=self.mock_global_db)

    def tearDown(self):
        if hasattr(self, 'service') and self.service:
            try:
                self.service.close_all_user_connections()
            except Exception as e:
                logger.error(f"Error during service.close_all_user_connections() in tearDown: {e}",
                                                  exc_info=True)

    def test_initialization(self):
        self.assertTrue(self.base_db_dir.exists())
        self.assertEqual(self.service.api_client_id, self.api_client_id)
        # Check for the actual log messages from the implementation
        self.mock_notes_library_logger.info.assert_any_call(
            f"NotesInteropService: Ensured base directory exists: {self.base_db_dir}")

    @patch(f'{NOTES_LIBRARY_MODULE_PATH}.Path.mkdir')
    def test_initialization_failure_os_error(self, mock_mkdir):
        mock_mkdir.side_effect = OSError("Permission denied")
        expected_msg_part = f"Failed to create base DB directory {self.base_db_dir}: Permission denied"
        with self.assertRaises(Actual_CharactersRAGDBError) as cm:
            NotesInteropService(base_db_directory=str(self.base_db_dir), api_client_id="fail_client")
        self.assertIn(expected_msg_part, str(cm.exception))
        self.mock_notes_library_logger.error.assert_called_with(
            f"Failed to create base DB directory {self.base_db_dir}: Permission denied"
        )

    def test_get_db_new_instance(self):
        user_id = "user1"
        db_instance = self.service._get_db(user_id)
        # The new implementation creates a DB instance with the unified DB path and user_id as client_id
        self.MockCharactersRAGDB_class.assert_called_once_with(
            db_path=self.mock_global_db.db_path_str, client_id=user_id
        )
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
        with self.assertRaisesRegex(Actual_CharactersRAGDBError,
                                    f"Unexpected error initializing DB instance for user {user_id}: Unexpected boom"):
            self.service._get_db(user_id)
        self.mock_notes_library_logger.error.assert_called_once_with(
            f"Unexpected error initializing dynamic CharactersRAGDB for user '{user_id}': Unexpected boom",
            exc_info=True
        )

    def test_add_note(self):
        user_id, title, content, expected_note_id = "user1", "Test Note", "Test Content", "note_uuid_1"
        self.mock_db_instance.add_note.return_value = expected_note_id
        note_id = self.service.add_note(user_id, title, content)
        self.mock_db_instance.add_note.assert_called_once_with(title=title, content=content, note_id=None)
        self.assertEqual(note_id, expected_note_id)

    def test_add_note_with_provided_id(self):
        user_id, title, content, provided_note_id = "user1", "Test Note", "Test Content", "client_note_id"
        self.mock_db_instance.add_note.return_value = provided_note_id
        note_id = self.service.add_note(user_id, title, content, note_id=provided_note_id)
        self.mock_db_instance.add_note.assert_called_once_with(title=title, content=content, note_id=provided_note_id)
        self.assertEqual(note_id, provided_note_id)

    def test_add_note_returns_none_unexpectedly(self):
        user_id, title, content = "user1", "Test Note", "Test Content"
        self.mock_db_instance.add_note.return_value = None
        with self.assertRaisesRegex(Actual_CharactersRAGDBError,
                                    "Failed to create note, received None ID unexpectedly"):
            self.service.add_note(user_id, title, content)
        self.mock_notes_library_logger.error.assert_called_once_with(
            f"add_note for user_id '{user_id}' (as client_id) returned None unexpectedly for title '{title}'."
        )

    def test_get_note_by_id(self):
        user_id, note_id_val = "user1", "note_uuid_1"
        expected_data = {"id": note_id_val, "title": "Test"}
        self.mock_db_instance.get_note_by_id.return_value = expected_data
        note = self.service.get_note_by_id(user_id, note_id_val)
        self.mock_db_instance.get_note_by_id.assert_called_once_with(note_id=note_id_val)
        self.assertEqual(note, expected_data)

    def test_list_notes(self):
        user_id, expected_notes = "user1", [{"id": "1"}, {"id": "2"}]
        self.mock_db_instance.list_notes.return_value = expected_notes
        notes = self.service.list_notes(user_id, limit=10, offset=0)
        self.mock_db_instance.list_notes.assert_called_once_with(limit=10, offset=0)
        self.assertEqual(notes, expected_notes)

    def test_update_note(self):
        user_id, note_id_val, update_data, expected_version = "user1", "note_uuid_1", {"title": "New Title"}, 1
        self.mock_db_instance.update_note.return_value = True
        success = self.service.update_note(user_id, note_id_val, update_data, expected_version)
        self.mock_db_instance.update_note.assert_called_once_with(
            note_id=note_id_val, update_data=update_data, expected_version=expected_version
        )
        self.assertTrue(success)

    def test_update_note_conflict(self):
        user_id, note_id_val, update_data, expected_version = "user1", "note_uuid_1", {"title": "New Title"}, 1
        conflict_error_instance = Actual_ConflictError("DB version mismatch", entity="notes", entity_id=note_id_val)
        self.mock_db_instance.update_note.side_effect = conflict_error_instance
        with self.assertRaises(Actual_ConflictError) as cm:
            self.service.update_note(user_id, note_id_val, update_data, expected_version)
        self.mock_db_instance.update_note.assert_called_once_with(
            note_id=note_id_val, update_data=update_data, expected_version=expected_version
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
        conflict_error_instance = Actual_ConflictError("Cannot delete", entity="notes", entity_id=note_id_val)
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
        self.mock_db_instance.search_notes.assert_called_once_with(search_term=term, limit=5)
        self.assertEqual(results, expected_results)

    def test_add_keyword(self):
        user_id, keyword_text, expected_keyword_id = "user1", "test_keyword", 1
        self.mock_db_instance.add_keyword.return_value = expected_keyword_id
        keyword_id = self.service.add_keyword(user_id, keyword_text)
        self.mock_db_instance.add_keyword.assert_called_once_with(keyword_text=keyword_text)
        self.assertEqual(keyword_id, expected_keyword_id)

    def test_link_note_to_keyword(self):
        user_id, note_id_val, keyword_id_val = "user1", "note_uuid_1", 1
        self.mock_db_instance.link_note_to_keyword.return_value = True
        success = self.service.link_note_to_keyword(user_id, note_id_val, keyword_id_val)
        self.mock_db_instance.link_note_to_keyword.assert_called_once_with(note_id=note_id_val,
                                                                           keyword_id=keyword_id_val)
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
            f"Closed and removed DB instance for user context '{user_id}'.")

    def test_close_user_connection_not_exist(self):
        user_id = "non_existent_user"
        self.service.close_user_connection(user_id)
        self.mock_db_instance.close_connection.assert_not_called()
        self.mock_notes_library_logger.debug.assert_any_call(
            f"No active DB instance found in cache for user context '{user_id}' to close.")

    def test_close_all_user_connections(self):
        user1_id, user2_id = "user1_for_close_all", "user2_for_close_all"
        mock_db_1_instance, mock_db_2_instance = MagicMock(spec=CharactersRAGDB), MagicMock(spec=CharactersRAGDB)
        self.MockCharactersRAGDB_class.side_effect = [mock_db_1_instance, mock_db_2_instance]
        db_instance1_ret, db_instance2_ret = self.service._get_db(user1_id), self.service._get_db(user2_id)
        self.assertIs(db_instance1_ret, mock_db_1_instance)
        self.assertIs(db_instance2_ret, mock_db_2_instance)
        self.service.close_all_user_connections()
        mock_db_1_instance.close_connection.assert_called_once()
        mock_db_2_instance.close_connection.assert_called_once()
        self.assertEqual(len(self.service._db_instances), 0)
        self.mock_notes_library_logger.info.assert_any_call(
            "All cached user-context DB instances have been processed for closure.")
        self.MockCharactersRAGDB_class.side_effect = None
        self.MockCharactersRAGDB_class.return_value = self.mock_db_instance

    def test_close_connection_exception(self):
        user_id = "user_close_fail"
        db_mock = self.service._get_db(user_id)
        self.assertIs(db_mock, self.mock_db_instance)
        self.mock_db_instance.close_connection.side_effect = Exception("Failed to close")
        self.service.close_user_connection(user_id)
        self.assertNotIn(user_id, self.service._db_instances)
        self.mock_notes_library_logger.error.assert_called_with(
            f"Error closing DB instance for user context '{user_id}': Failed to close", exc_info=True
        )


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)