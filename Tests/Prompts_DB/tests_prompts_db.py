# test_Prompts_DB.py
import sqlite3
import unittest
import uuid
import os
import shutil
import tempfile
import threading
from pathlib import Path

# The module to be tested
from tldw_chatbook.DB.Prompts_DB import (
    PromptsDatabase,
    DatabaseError,
    SchemaError,
    InputError,
    ConflictError,
    add_or_update_prompt,
    load_prompt_details_for_ui,
    export_prompt_keywords_to_csv,
    view_prompt_keywords_markdown,
    export_prompts_formatted,
)


# --- Test Case Base ---
class BaseTestCase(unittest.TestCase):
    """Base class for tests, handles temporary DB setup and teardown."""

    def setUp(self):
        """Set up a new in-memory database for each test."""
        self.client_id = "test_client_1"
        self.db = PromptsDatabase(':memory:', client_id=self.client_id)
        # For tests requiring a file-based DB
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_prompts.db"
        # ---- Add a list to track all created file DBs ----
        self.file_db_instances = []

    def tearDown(self):
        """Close connection and clean up resources."""
        if hasattr(self, 'db') and self.db:
            self.db.close_connection()
        # ---- Close all tracked file DB instances ----
        for instance in self.file_db_instances:
            instance.close_connection()
        # Now it's safe to remove the directory
        shutil.rmtree(self.temp_dir)

    def _get_file_db(self):
        """Helper to get a file-based database instance, ensuring it's tracked for cleanup."""
        # ---- Track the new instance ----
        instance = PromptsDatabase(self.db_path, client_id=f"test_client_file_{len(self.file_db_instances)}")
        self.file_db_instances.append(instance)
        return instance

    def _add_sample_data(self, db_instance):
        """Helper to populate a given database instance with some initial data."""
        # Note: All arguments are provided to match the fixed library code
        db_instance.add_prompt(
            name="Recipe Generator",
            author="ChefAI",
            details="Creates recipes for cooking",
            system_prompt="You are a chef.",
            user_prompt="Give me a recipe for pasta.",
            keywords=["food", "cooking"],
            overwrite=True
        )
        db_instance.add_prompt(
            name="Code Explainer",
            author="DevHelper",
            details="Explains python code snippets",
            system_prompt="You are a senior dev.",
            user_prompt="Explain this python code.",
            keywords=["code", "python"],
            overwrite=True
        )
        db_instance.add_prompt(
            name="Poem Writer",
            author="BardBot",
            details="Writes poems about nature",
            system_prompt="You are a poet.",
            user_prompt="Write a poem about the sea.",
            keywords=["writing", "poetry"],
            overwrite=True
        )
        # Add a deleted prompt for testing filters
        pid, _, _ = db_instance.add_prompt(
            name="Old Prompt",
            author="Old",
            details="Old details",
            overwrite=True
        )
        db_instance.soft_delete_prompt(pid)


# --- Test Suites ---
class TestDatabaseInitialization(BaseTestCase):

    def test_init_success_in_memory(self):
        self.assertIsNotNone(self.db)
        self.assertIsInstance(self.db, PromptsDatabase)
        self.assertTrue(self.db.is_memory_db)
        self.assertEqual(self.db.client_id, self.client_id)

    def test_init_success_file_based(self):
        file_db = self._get_file_db()
        self.assertTrue(self.db_path.exists())
        self.assertFalse(file_db.is_memory_db)
        conn = file_db.get_connection()
        # WAL mode is set for file-based dbs
        cursor = conn.execute("PRAGMA journal_mode;")
        self.assertEqual(cursor.fetchone()[0].lower(), 'wal')

    def test_init_failure_no_client_id(self):
        with self.assertRaises(ValueError):
            PromptsDatabase(':memory:', client_id=None)
        with self.assertRaises(ValueError):
            PromptsDatabase(':memory:', client_id="")

    def test_schema_version_check(self):
        conn = self.db.get_connection()
        version = conn.execute("SELECT version FROM schema_version").fetchone()['version']
        self.assertEqual(version, self.db._CURRENT_SCHEMA_VERSION)

    def test_fts_tables_created(self):
        conn = self.db.get_connection()
        try:
            conn.execute("SELECT * FROM prompts_fts LIMIT 1")
            conn.execute("SELECT * FROM prompt_keywords_fts LIMIT 1")
        except Exception as e:
            self.fail(f"FTS tables not created or queryable: {e}")

    def test_thread_safety_connections(self):
        """Verify that different threads get different connection objects."""
        connections = {}
        db_instance = self._get_file_db()

        def get_conn(thread_id):
            conn = db_instance.get_connection()
            connections[thread_id] = id(conn)
            db_instance.close_connection()

        thread1 = threading.Thread(target=get_conn, args=(1,))
        thread2 = threading.Thread(target=get_conn, args=(2,))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        self.assertIn(1, connections)
        self.assertIn(2, connections)
        self.assertNotEqual(connections[1], connections[2],
                            "Connections for different threads should be different objects")


class TestCrudOperations(BaseTestCase):

    def test_add_keyword(self):
        kw_id, kw_uuid = self.db.add_keyword("  Test Keyword 1 ")
        self.assertIsNotNone(kw_id)
        self.assertIsNotNone(kw_uuid)

        # Verify it was added correctly and normalized
        kw_data = self.db.get_active_keyword_by_text("test keyword 1")
        self.assertIsNotNone(kw_data)
        self.assertEqual(kw_data['keyword'], "test keyword 1")
        self.assertEqual(kw_data['id'], kw_id)

        # Verify sync log
        sync_logs = self.db.get_sync_log_entries()
        self.assertEqual(len(sync_logs), 1)
        log_entry = sync_logs[0]
        self.assertEqual(log_entry['entity'], 'PromptKeywordsTable')
        self.assertEqual(log_entry['entity_uuid'], kw_uuid)
        self.assertEqual(log_entry['operation'], 'create')
        self.assertEqual(log_entry['version'], 1)

        # Verify FTS
        res = self.db.execute_query(
            "SELECT rowid FROM prompt_keywords_fts WHERE prompt_keywords_fts MATCH ?",
            ("test",)
        ).fetchone()
        self.assertIsNotNone(res)
        self.assertEqual(res['rowid'], kw_id)

    def test_add_existing_keyword(self):
        kw_id1, kw_uuid1 = self.db.add_keyword("duplicate")
        kw_id2, kw_uuid2 = self.db.add_keyword("duplicate")
        self.assertEqual(kw_id1, kw_id2)
        self.assertEqual(kw_uuid1, kw_uuid2)
        sync_logs = self.db.get_sync_log_entries()
        self.assertEqual(len(sync_logs), 1)  # Should only log the creation once

    def test_add_prompt(self):
        pid, puuid, msg = self.db.add_prompt("My Prompt", "Me", "Details here", keywords=["tag1", "tag2"])
        self.assertIsNotNone(pid)
        self.assertIsNotNone(puuid)
        self.assertIn("added", msg)

        prompt = self.db.get_prompt_by_id(pid)
        self.assertEqual(prompt['name'], "My Prompt")
        self.assertEqual(prompt['version'], 1)

        keywords = self.db.fetch_keywords_for_prompt(pid)
        self.assertIn("tag1", keywords)
        self.assertIn("tag2", keywords)

        # Check sync logs - 1 for prompt, 2 for keywords, 2 for links
        sync_logs = self.db.get_sync_log_entries()
        prompt_create_logs = [l for l in sync_logs if l['entity'] == 'Prompts' and l['operation'] == 'create']
        self.assertEqual(len(prompt_create_logs), 1)
        prompt_create_log = prompt_create_logs[0]
        self.assertEqual(prompt_create_log['entity_uuid'], puuid)

        # Verify FTS
        res = self.db.execute_query(
            "SELECT rowid FROM prompts_fts WHERE prompts_fts MATCH ?",
            ("My Prompt",)
            ).fetchone()
        self.assertIsNotNone(res)
        self.assertEqual(res['rowid'], pid)

    def test_add_prompt_conflict(self):
        self.db.add_prompt("Conflict Prompt", "Author", "Details")
        with self.assertRaises(ConflictError):
            self.db.add_prompt("Conflict Prompt", "Author", "Details", overwrite=False)

    def test_add_prompt_overwrite(self):
        pid1, _, _ = self.db.add_prompt("Overwrite Me", "Author1", "Details1")
        pid2, _, msg = self.db.add_prompt("Overwrite Me", "Author2", "Details2", overwrite=True)
        self.assertEqual(pid1, pid2)
        self.assertIn("updated", msg)

        prompt = self.db.get_prompt_by_id(pid1)
        self.assertEqual(prompt['author'], "Author2")
        self.assertEqual(prompt['version'], 2)

    def test_soft_delete_prompt(self):
        pid, puuid, _ = self.db.add_prompt("To Be Deleted", "Author", "Details", keywords=["temp"])

        self.assertIsNotNone(self.db.get_prompt_by_id(pid))

        success = self.db.soft_delete_prompt(pid)
        self.assertTrue(success)

        self.assertIsNone(self.db.get_prompt_by_id(pid))

        deleted_prompt = self.db.get_prompt_by_id(pid, include_deleted=True)
        self.assertIsNotNone(deleted_prompt)
        self.assertEqual(deleted_prompt['deleted'], 1)
        self.assertEqual(deleted_prompt['version'], 2)

        res = self.db.execute_query(
            "SELECT rowid FROM prompts_fts WHERE prompts_fts MATCH ?",
            ("To Be Deleted",)
        ).fetchone()
        self.assertIsNone(res)

        link_exists = self.db.execute_query("SELECT 1 FROM PromptKeywordLinks WHERE prompt_id=?", (pid,)).fetchone()
        self.assertIsNone(link_exists)

        sync_logs = self.db.get_sync_log_entries()  # Check all logs
        delete_log = next(l for l in sync_logs if l['entity'] == 'Prompts' and l['operation'] == 'delete')
        unlink_log = next(l for l in sync_logs if l['entity'] == 'PromptKeywordLinks' and l['operation'] == 'unlink')
        self.assertEqual(delete_log['entity_uuid'], puuid)
        self.assertIn(puuid, unlink_log['entity_uuid'])

    def test_soft_delete_keyword(self):
        kw_id, kw_uuid = self.db.add_keyword("ephemeral")
        self.db.add_prompt("Test Prompt", "Author", "Some details", keywords=["ephemeral"])

        success = self.db.soft_delete_keyword("ephemeral")
        self.assertTrue(success)

        self.assertIsNone(self.db.get_active_keyword_by_text("ephemeral"))

        res = self.db.execute_query(
            "SELECT rowid FROM prompt_keywords_fts WHERE prompt_keywords_fts MATCH ?",
            ("ephemeral",)
        ).fetchone()
        self.assertIsNone(res)

        prompt = self.db.get_prompt_by_name("Test Prompt")
        keywords = self.db.fetch_keywords_for_prompt(prompt['id'])
        self.assertNotIn("ephemeral", keywords)

    def test_update_prompt_by_id(self):
        pid, puuid, _ = self.db.add_prompt("Initial Name", "Author", "Details", keywords=["old_kw"])

        update_data = {"name": "Updated Name", "details": "New details", "keywords": ["new_kw", "another_kw"]}

        updated_uuid, msg = self.db.update_prompt_by_id(pid, update_data)
        self.assertEqual(updated_uuid, puuid)
        self.assertIn("updated successfully", msg)

        prompt = self.db.get_prompt_by_id(pid)
        self.assertEqual(prompt['name'], "Updated Name")
        self.assertEqual(prompt['details'], "New details")
        self.assertEqual(prompt['version'], 2)

        keywords = self.db.fetch_keywords_for_prompt(pid)
        self.assertIn("new_kw", keywords)
        self.assertIn("another_kw", keywords)
        self.assertNotIn("old_kw", keywords)

        res = self.db.execute_query(
            "SELECT rowid FROM prompts_fts WHERE prompts_fts MATCH ?",
            ("Updated Name",)
            ).fetchone()
        self.assertIsNotNone(res)
        self.assertEqual(res['rowid'], pid)


class TestQueryOperations(BaseTestCase):

    def setUp(self):
        super().setUp()
        self._add_sample_data(self.db)

    def test_list_prompts(self):
        prompts, total_pages, page, total_items = self.db.list_prompts(page=1, per_page=2)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(total_items, 3)  # 3 active prompts
        self.assertEqual(total_pages, 2)
        self.assertEqual(page, 1)

    def test_list_prompts_include_deleted(self):
        _, _, _, total_items = self.db.list_prompts(include_deleted=True)
        self.assertEqual(total_items, 4)

    def test_fetch_prompt_details(self):
        details = self.db.fetch_prompt_details("Recipe Generator")
        self.assertIsNotNone(details)
        self.assertEqual(details['name'], "Recipe Generator")
        self.assertIn("food", details['keywords'])
        self.assertIn("cooking", details['keywords'])

    def test_fetch_all_keywords(self):
        keywords = self.db.fetch_all_keywords()
        expected = sorted(["food", "cooking", "code", "python", "writing", "poetry"])
        self.assertEqual(sorted(keywords), expected)

    def test_search_prompts_by_name(self):
        results, total = self.db.search_prompts("Recipe")
        self.assertEqual(total, 1)
        self.assertEqual(results[0]['name'], "Recipe Generator")

    def test_search_prompts_by_details(self):
        results, total = self.db.search_prompts("python code", search_fields=['details', 'user_prompt'])
        self.assertEqual(total, 1)
        self.assertEqual(results[0]['name'], "Code Explainer")

    def test_search_prompts_by_keyword(self):
        results, total = self.db.search_prompts("poetry", search_fields=['keywords'])
        self.assertEqual(total, 1)
        self.assertEqual(results[0]['name'], "Poem Writer")


class TestUtilitiesAndAdvancedFeatures(BaseTestCase):

    def test_backup_database(self):
        file_db = self._get_file_db()
        file_db.add_prompt("Backup Test", "Tester", "Details")

        backup_path = self.db_path.with_suffix('.backup.db')

        success = file_db.backup_database(str(backup_path))
        self.assertTrue(success)
        self.assertTrue(backup_path.exists())

        backup_db = PromptsDatabase(backup_path, client_id="backup_verifier")
        prompt = backup_db.get_prompt_by_name("Backup Test")
        self.assertIsNotNone(prompt)
        self.assertEqual(prompt['author'], "Tester")

        backup_db.close_connection()

    def test_backup_database_same_file_fails(self):
        file_db = self._get_file_db()
        # The function catches this ValueError and returns False
        success = file_db.backup_database(str(self.db_path))
        self.assertFalse(success)

    def test_transaction_rollback(self):
        pid, _, _ = self.db.add_prompt("Initial", "Auth", "Det")

        try:
            with self.db.transaction():
                self.db.execute_query("UPDATE Prompts SET name = ?, version = version + 1, client_id='t' WHERE id = ?",
                                      ("Updated", pid), commit=False)
                prompt_inside = self.db.get_prompt_by_id(pid)
                self.assertEqual(prompt_inside['name'], "Updated")
                raise ValueError("Intentional failure to trigger rollback")
        except ValueError:
            pass  # Expected exception

        prompt_outside = self.db.get_prompt_by_id(pid)
        self.assertEqual(prompt_outside['name'], "Initial")

    def test_delete_sync_log_entries(self):
        self.db.add_keyword("kw1")
        self.db.add_keyword("kw2")

        logs = self.db.get_sync_log_entries()
        self.assertEqual(len(logs), 2)
        log_ids_to_delete = [log['change_id'] for log in logs]

        deleted_count = self.db.delete_sync_log_entries(log_ids_to_delete)
        self.assertEqual(deleted_count, 2)

        remaining_logs = self.db.get_sync_log_entries()
        self.assertEqual(len(remaining_logs), 0)

    def test_update_prompt_version_conflict(self):
        """
        Tests that the database trigger prevents updates that violate the versioning rule.
        This is a direct test of the database integrity layer.
        """
        db = self.db
        pid, _, _ = db.add_prompt("Trigger Test", "Author", "Details")  # Prompt is now at version 1

        db.update_prompt_by_id(pid, {'details': 'New Details'})
        prompt = db.get_prompt_by_id(pid)
        self.assertEqual(prompt['version'], 2, "Version should be 2 after the first update.")

        conn = db.get_connection()

        with self.assertRaises(sqlite3.IntegrityError) as cm:
            with conn:
                conn.execute("UPDATE Prompts SET version = 2, client_id='raw' WHERE id = ?", (pid,))

        self.assertIn("Version must increment by exactly 1", str(cm.exception))

        final_prompt = db.get_prompt_by_id(pid)
        self.assertEqual(final_prompt['version'], 2, "Version should remain 2 after the failed update.")


class TestStandaloneFunctions(BaseTestCase):
    def setUp(self):
        super().setUp()
        self._add_sample_data(self.db)

    def test_add_or_update_prompt(self):
        # Test update
        pid, _, msg = add_or_update_prompt(self.db, "Recipe Generator", "New Chef", "New details", keywords=["italian"])
        self.assertIn("updated", msg)
        prompt = self.db.get_prompt_by_id(pid)
        self.assertEqual(prompt['author'], "New Chef")
        keywords = self.db.fetch_keywords_for_prompt(pid)
        self.assertIn("italian", keywords)

        # Test add
        pid_new, _, msg_new = add_or_update_prompt(self.db, "New Standalone Prompt", "Tester", "Details")
        self.assertIn("added", msg_new)
        self.assertIsNotNone(self.db.get_prompt_by_id(pid_new))

    def test_load_prompt_details_for_ui(self):
        name, author, details, sys_p, user_p, kws = load_prompt_details_for_ui(self.db, "Code Explainer")
        self.assertEqual(name, "Code Explainer")
        self.assertEqual(author, "DevHelper")
        self.assertEqual(sys_p, "You are a senior dev.")
        self.assertEqual(kws, "code, python")

    def test_load_prompt_details_not_found(self):
        result = load_prompt_details_for_ui(self.db, "Non Existent")
        self.assertEqual(result, ("", "", "", "", "", ""))

    def test_view_prompt_keywords_markdown(self):
        md_output = view_prompt_keywords_markdown(self.db)
        self.assertIn("### Current Active Prompt Keywords:", md_output)
        self.assertIn("- code (1 active prompts)", md_output)
        self.assertIn("- cooking (1 active prompts)", md_output)

    def test_export_keywords_to_csv(self):
        file_db = self._get_file_db()
        add_or_update_prompt(file_db, "Prompt 1", "Auth", "Det", keywords=["a", "b"])
        add_or_update_prompt(file_db, "Prompt 2", "Auth", "Det", keywords=["b", "c"])

        status, file_path = export_prompt_keywords_to_csv(file_db)
        self.assertIn("Successfully exported", status)
        self.assertTrue(os.path.exists(file_path))

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("Keyword,Associated Prompts", content)
            self.assertIn("a,Prompt 1,1", content)
            self.assertIn("c,Prompt 2,1", content)
            # Handle potential ordering difference in GROUP_CONCAT
            self.assertTrue("b,\"Prompt 1,Prompt 2\",2" in content or "b,\"Prompt 2,Prompt 1\",2" in content)

        os.remove(file_path)

    def test_export_prompts_formatted_csv(self):
        file_db = self._get_file_db()
        self._add_sample_data(file_db)

        status, file_path = export_prompts_formatted(file_db, export_format='csv')
        self.assertIn("Successfully exported 3 prompts", status)
        self.assertTrue(os.path.exists(file_path))

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("Name,UUID,Author,Details,System Prompt,User Prompt,Keywords", content)
            self.assertIn("Recipe Generator", content)
            # FIX: Keywords are sorted by the fetch method, so 'cooking' comes before 'food'.
            self.assertIn('"cooking, food"', content)

        os.remove(file_path)

    def test_export_prompts_formatted_markdown(self):
        file_db = self._get_file_db()
        self._add_sample_data(file_db)

        status, file_path = export_prompts_formatted(file_db, export_format='markdown')
        self.assertIn("Successfully exported 3 prompts to Markdown", status)
        self.assertTrue(os.path.exists(file_path))
        # A more thorough test could unzip and verify content, but file creation is a good start.
        os.remove(file_path)


class TestDatabaseIntegrityAndSchema(BaseTestCase):
    """
    Tests focused on low-level database integrity, schema rules, and triggers.
    These tests may involve direct SQL execution to bypass library methods.
    """

    def test_database_version_too_high_raises_error(self):
        """Ensure initializing a DB with a future schema version fails gracefully."""
        file_db = self._get_file_db()
        conn = file_db.get_connection()
        # Manually set the schema version to a higher number
        conn.execute("UPDATE schema_version SET version = 99")
        conn.commit()
        # FIX: We must close the connection so the file handle is released before the next open attempt
        file_db.close_connection()
        # The instance is now closed, remove it from tracking to avoid double-closing
        self.file_db_instances.remove(file_db)

        # Now, trying to create a new instance pointing to this DB should fail
        # FIX: The __init__ method wraps SchemaError in a DatabaseError.
        with self.assertRaisesRegex(DatabaseError, "newer than supported"):
            PromptsDatabase(self.db_path, client_id="test_client_fail")

    def test_trigger_prevents_bad_version_update(self):
        """Verify the SQL trigger prevents updates that don't increment version by 1."""
        pid, _, _ = self.db.add_prompt("Trigger Test", "T", "D")  # Version is 1
        conn = self.db.get_connection()

        with self.assertRaises(sqlite3.IntegrityError) as cm:
            conn.execute("UPDATE Prompts SET version = 3 WHERE id = ?", (pid,))
        self.assertIn("Version must increment by exactly 1", str(cm.exception))

        # This should succeed
        conn.execute("UPDATE Prompts SET version = 2, client_id='raw_sql' WHERE id = ?", (pid,))
        conn.commit()
        prompt = self.db.get_prompt_by_id(pid)
        self.assertEqual(prompt['version'], 2)

    def test_trigger_prevents_uuid_change(self):
        """Verify the SQL trigger prevents changing a UUID on update."""
        pid, original_uuid, _ = self.db.add_prompt("UUID Lock", "T", "D")
        conn = self.db.get_connection()
        new_uuid = str(uuid.uuid4())

        with self.assertRaises(sqlite3.IntegrityError) as cm:
            # Try to update the UUID (and correctly increment version)
            conn.execute("UPDATE Prompts SET uuid = ?, version = 2, client_id='raw' WHERE id = ?", (new_uuid, pid))
        self.assertIn("UUID cannot be changed", str(cm.exception))

        # Verify UUID is unchanged
        prompt = self.db.get_prompt_by_id(pid)
        self.assertEqual(prompt['uuid'], original_uuid)


class TestAdvancedBehaviorsAndEdgeCases(BaseTestCase):
    """
    Tests for more nuanced behaviors and specific edge cases not covered
    in standard CRUD operations.
    """

    def test_reopen_closed_connection(self):
        """Test that the library can reopen a connection that was explicitly closed."""
        file_db = self._get_file_db()

        conn1 = file_db.get_connection()
        self.assertIsNotNone(conn1)
        conn1.close()

        conn2 = file_db.get_connection()
        self.assertIsNotNone(conn2)
        self.assertNotEqual(id(conn1), id(conn2))

        file_db.add_keyword("test_after_reopen")
        self.assertIsNotNone(file_db.get_active_keyword_by_text("test_after_reopen"))

    def test_undelete_keyword(self):
        """Test that adding an already soft-deleted keyword undeletes it."""
        self.db.add_keyword("to be deleted and restored")
        self.db.soft_delete_keyword("to be deleted and restored")
        self.assertIsNone(self.db.get_active_keyword_by_text("to be deleted and restored"))

        # Now, add it again
        kw_id, kw_uuid = self.db.add_keyword("to be deleted and restored")

        # Verify it's active again
        restored_kw = self.db.get_active_keyword_by_text("to be deleted and restored")
        self.assertIsNotNone(restored_kw)
        self.assertEqual(restored_kw['id'], kw_id)
        self.assertEqual(restored_kw['version'], 3)  # 1: create, 2: delete, 3: undelete (update)

        # Check sync log for the 'update' operation
        sync_logs = self.db.get_sync_log_entries()
        undelete_log = next(log for log in sync_logs if log['version'] == 3)
        self.assertEqual(undelete_log['operation'], 'update')
        self.assertEqual(undelete_log['payload']['deleted'], 0)

    def test_update_prompt_name_to_existing_name_conflict(self):
        """Ensure updating a prompt's name to another existing prompt's name fails."""
        self.db.add_prompt("Prompt A", "Author", "Details")
        pid_b, _, _ = self.db.add_prompt("Prompt B", "Author", "Details")

        with self.assertRaises(ConflictError):
            self.db.update_prompt_by_id(pid_b, {"name": "Prompt A"})

    def test_nested_transactions(self):
        pid, _, _ = self.db.add_prompt("Transaction Test", "T", "D")

        try:
            with self.db.transaction():  # Outer transaction
                self.db.execute_query("UPDATE Prompts SET name = 'Outer Update', version=2, client_id='t' WHERE id = ?",
                                      (pid,))

                with self.db.transaction():  # Inner transaction
                    self.db.execute_query(
                        "UPDATE Prompts SET author = 'Inner Update', version=3, client_id='t' WHERE id = ?", (pid,))

                prompt_inside = self.db.get_prompt_by_id(pid)
                self.assertEqual(prompt_inside['author'], 'Inner Update')

                raise ValueError("Force rollback of outer transaction")

        except ValueError:
            pass

        prompt_outside = self.db.get_prompt_by_id(pid)
        self.assertEqual(prompt_outside['name'], "Transaction Test")
        self.assertEqual(prompt_outside['author'], "T")
        self.assertEqual(prompt_outside['version'], 1)

    def test_soft_delete_prompt_by_name_and_uuid(self):
        """Test soft deletion using name and UUID identifiers."""
        _, p1_uuid, _ = self.db.add_prompt("Deletable By Name", "A", "D")
        p2_id, p2_uuid, _ = self.db.add_prompt("Deletable By UUID", "B", "E")

        # Delete by name
        self.assertTrue(self.db.soft_delete_prompt("Deletable By Name"))
        self.assertIsNone(self.db.get_prompt_by_name("Deletable By Name"))

        # Delete by UUID
        self.assertTrue(self.db.soft_delete_prompt(p2_uuid))
        self.assertIsNone(self.db.get_prompt_by_id(p2_id))


class TestSearchFunctionality(BaseTestCase):
    """More detailed tests for the search_prompts function."""

    def setUp(self):
        super().setUp()
        self._add_sample_data(self.db)
        self.db.add_prompt("Shared Term Prompt", "Author", "This prompt contains python.", keywords=["generic"])
        self.db.add_prompt("Another Code Prompt", "DevHelper", "More code things.", keywords=["python"])

    def test_search_with_no_query_returns_all_active(self):
        """Searching with no query should act like listing all active prompts."""
        results, total = self.db.search_prompts(None)
        # 3 from _add_sample_data + 2 added in this setUp = 5 active prompts
        self.assertEqual(total, 5)
        self.assertEqual(len(results), 5)

    def test_search_with_no_results(self):
        """Ensure a search with no matches returns an empty list and zero total."""
        results, total = self.db.search_prompts("nonexistentxyz")
        self.assertEqual(total, 0)
        self.assertEqual(len(results), 0)

    def test_search_pagination(self):
        """Test if pagination works correctly on search results."""
        results, total = self.db.search_prompts("python", search_fields=['details', 'keywords'], page=1,
                                                results_per_page=2)
        self.assertEqual(total, 3)  # Code Explainer, Shared Term, Another Code
        self.assertEqual(len(results), 2)

        results_p2, total_p2 = self.db.search_prompts("python", search_fields=['details', 'keywords'], page=2,
                                                      results_per_page=2)
        self.assertEqual(total_p2, 3)
        self.assertEqual(len(results_p2), 1)

    def test_search_across_multiple_fields(self):
        """Test searching in both details and keywords simultaneously."""
        results, total = self.db.search_prompts("python", search_fields=['details', 'keywords'])
        self.assertEqual(total, 3)
        names = {r['name'] for r in results}
        self.assertIn("Code Explainer", names)
        self.assertIn("Shared Term Prompt", names)
        self.assertIn("Another Code Prompt", names)

    def test_search_with_invalid_fts_syntax_raises_error(self):
        """Verify that malformed FTS queries raise a DatabaseError."""
        # An unclosed quote is invalid syntax
        with self.assertRaises(DatabaseError):
            self.db.search_prompts('invalid "syntax', search_fields=['name'])


class TestStandaloneFunctionExports(BaseTestCase):
    """Tests for variations in the standalone export functions."""

    def setUp(self):
        super().setUp()
        self.file_db = self._get_file_db()
        self._add_sample_data(self.file_db)

    def test_export_with_no_matching_prompts(self):
        """Test export when the filter criteria yields no results."""
        status, file_path = export_prompts_formatted(
            self.file_db,
            filter_keywords=["nonexistent_keyword"]
        )
        self.assertIn("No prompts found", status)
        self.assertEqual(file_path, "None")

    def test_export_csv_minimal_columns(self):
        """Test CSV export with most boolean flags turned off."""
        status, file_path = export_prompts_formatted(
            self.file_db,
            export_format='csv',
            include_details=False,
            include_system=False,
            include_user=False,
            include_associated_keywords=False
        )
        self.assertIn("Successfully exported", status)
        self.assertTrue(os.path.exists(file_path))

        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            self.assertEqual(header, "Name,UUID,Author")

        os.remove(file_path)

    def test_export_markdown_with_different_template(self):
        """Test Markdown export using a non-default template."""
        status, file_path = export_prompts_formatted(
            self.file_db,
            export_format='markdown',
            markdown_template_name="Detailed Template"
        )
        self.assertIn("Successfully exported", status)
        self.assertTrue(os.path.exists(file_path))
        os.remove(file_path)


class TestConcurrencyAndDataIntegrity(BaseTestCase):
    """
    Tests for race conditions, data encoding, and integrity under stress.
    """

    def test_concurrent_updates_to_same_prompt(self):
        """
        Simulate a race condition where two threads update the same prompt.
        One should succeed, the other should fail with a ConflictError.
        """
        file_db = self._get_file_db()
        pid, _, _ = file_db.add_prompt("Race Condition", "Initial", "Data")

        results = {}
        barrier = threading.Barrier(2, timeout=5)

        def worker(user_id):
            try:
                db_instance = PromptsDatabase(self.db_path, client_id=f"worker_{user_id}")
                barrier.wait()
                db_instance.update_prompt_by_id(pid, {'details': f'Updated by {user_id}'})
                results[user_id] = "success"
            except ConflictError:
                results[user_id] = "conflict"
            except DatabaseError as e:
                if "locked" in str(e).lower() or "conflict" in str(e).lower():
                    results[user_id] = "conflict"
                else:
                    results[user_id] = e
            except Exception as e:
                results[user_id] = e
            finally:
                if 'db_instance' in locals():
                    db_instance.close_connection()

        thread1 = threading.Thread(target=worker, args=(1,))
        thread2 = threading.Thread(target=worker, args=(2,))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        self.assertIn("success", results.values())
        self.assertIn("conflict", results.values())

        final_prompt = file_db.get_prompt_by_id(pid)
        self.assertEqual(final_prompt['version'], 2)
        self.assertTrue(final_prompt['details'].startswith("Updated by"))

    def test_unicode_character_support(self):
        """Ensure all text fields correctly handle Unicode characters."""
        unicode_name = "こんにちは世界"
        unicode_author = "Александр"
        unicode_details = "Testing emoji support 👍 and special characters ç, é, à."
        unicode_keywords = ["你好", "世界", "prüfung"]

        pid, _, _ = self.db.add_prompt(
            name=unicode_name,
            author=unicode_author,
            details=unicode_details,
            keywords=unicode_keywords
        )

        prompt = self.db.fetch_prompt_details(pid)
        self.assertEqual(prompt['name'], unicode_name)
        self.assertEqual(prompt['author'], unicode_author)
        self.assertEqual(prompt['details'], unicode_details)
        self.assertEqual(sorted(prompt['keywords']), sorted(unicode_keywords))

        results, total = self.db.search_prompts("你好", search_fields=['keywords'])
        self.assertEqual(total, 1)
        self.assertEqual(results[0]['name'], unicode_name)

        status, file_path = export_prompts_formatted(self.db, 'csv')
        self.assertTrue(os.path.exists(file_path))
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn(unicode_name, content)
            self.assertIn(unicode_author, content)
            self.assertIn("prüfung", content)
        os.remove(file_path)

    def test_fts_desynchronization_by_direct_sql(self):
        """
        Demonstrates that direct SQL updates will de-sync the FTS table.
        """
        unique_term = "zzyzx"
        pid, _, _ = self.db.add_prompt("FTS Sync Test", "Author", f"Details with {unique_term}")

        results, total = self.db.search_prompts(unique_term)
        self.assertEqual(total, 1)

        conn = self.db.get_connection()
        conn.execute(
            "UPDATE Prompts SET details='Details are now different', version=2, client_id='raw' WHERE id=?",
            (pid,)
        )
        conn.commit()

        prompt = self.db.get_prompt_by_id(pid)
        self.assertEqual(prompt['details'], "Details are now different")

        # FTS search for the *new* term will FAIL because FTS was not updated
        results_new, total_new = self.db.search_prompts("different")
        self.assertEqual(total_new, 0)

        # FTS search for the *old* term will SUCCEED because FTS is stale
        results_old, total_old = self.db.search_prompts(unique_term)
        self.assertEqual(total_old, 1)


# Additional tests from the failing set, now expected to pass
class TestAdvancedStateTransitions(BaseTestCase):
    def test_update_soft_deleted_prompt_restores_it(self):
        pid, _, _ = self.db.add_prompt("To be deleted", "Author", "Old Details")
        self.db.soft_delete_prompt(pid)
        self.assertIsNone(self.db.get_prompt_by_id(pid))

        update_data = {"details": "New, Restored Details"}
        self.db.update_prompt_by_id(pid, update_data)

        restored_prompt = self.db.get_prompt_by_id(pid)
        self.assertIsNotNone(restored_prompt)
        self.assertEqual(restored_prompt['deleted'], 0)
        self.assertEqual(restored_prompt['details'], "New, Restored Details")
        self.assertEqual(restored_prompt['version'], 3)

    def test_handling_of_corrupted_sync_log_payload(self):
        self.db.add_keyword("good_payload")
        log_entry = self.db.get_sync_log_entries()[0]
        change_id = log_entry['change_id']

        conn = self.db.get_connection()
        conn.execute(
            "UPDATE sync_log SET payload = ? WHERE change_id = ?",
            ("{'bad_json': this_is_not_valid}", change_id)
        )
        conn.commit()

        all_logs = self.db.get_sync_log_entries()
        corrupted_log = next(log for log in all_logs if log['change_id'] == change_id)
        self.assertIsNone(corrupted_log['payload'])

    def test_add_keyword_with_only_whitespace_fails(self):
        """Test that adding a keyword that is only whitespace raises an InputError."""
        with self.assertRaises(InputError):
            self.db.add_keyword("    ")

        # Verify no sync log was created
        self.assertEqual(len(self.db.get_sync_log_entries()), 0)

    def test_add_prompt_with_only_whitespace_name_fails(self):
        """Test that adding a prompt with a whitespace-only name raises an InputError."""
        with self.assertRaises(InputError):
            self.db.add_prompt("   ", "Author", "Details")

        # Verify no sync log was created
        self.assertEqual(len(self.db.get_sync_log_entries()), 0)


class TestSyncLogManagement(BaseTestCase):
    """
    In-depth tests for the sync_log table management and access methods.
    """

    def test_get_sync_log_with_since_change_id_and_limit(self):
        """Verify that fetching logs with a starting ID and a limit works correctly."""
        # Add 5 items, which should generate at least 5 logs
        self.db.add_keyword("kw1")
        self.db.add_keyword("kw2")
        self.db.add_keyword("kw3")
        self.db.add_keyword("kw4")
        self.db.add_keyword("kw5")

        all_logs = self.db.get_sync_log_entries()
        self.assertEqual(len(all_logs), 5)

        # Get logs since change_id 2 (should get 3, 4, 5)
        logs_since_2 = self.db.get_sync_log_entries(since_change_id=2)
        self.assertEqual(len(logs_since_2), 3)
        self.assertEqual(logs_since_2[0]['change_id'], 3)

        # Get logs since change_id 2 with a limit of 1 (should only get 3)
        logs_limited = self.db.get_sync_log_entries(since_change_id=2, limit=1)
        self.assertEqual(len(logs_limited), 1)
        self.assertEqual(logs_limited[0]['change_id'], 3)

    def test_delete_sync_log_with_nonexistent_ids(self):
        """Ensure deleting a mix of existing and non-existent log IDs works as expected."""
        self.db.add_keyword("kw1")  # change_id 1
        self.db.add_keyword("kw2")  # change_id 2

        # Attempt to delete IDs 1 and 9999 (which doesn't exist)
        deleted_count = self.db.delete_sync_log_entries([1, 9999])

        self.assertEqual(deleted_count, 1)

        remaining_logs = self.db.get_sync_log_entries()
        self.assertEqual(len(remaining_logs), 1)
        self.assertEqual(remaining_logs[0]['change_id'], 2)


class TestComplexStateAndInputInteractions(BaseTestCase):
    """
    Tests for nuanced interactions between methods and states.
    """

    def test_add_prompt_with_overwrite_false_on_deleted_prompt(self):
        """
        Verify the specific behavior of add_prompt(overwrite=False) when a prompt
        is soft-deleted. It should not restore it and should return a specific message.
        """
        self.db.add_prompt("Deleted but Exists", "Author", "Details")
        self.db.soft_delete_prompt("Deleted but Exists")

        # Attempt to add it again without overwrite flag
        pid, puuid, msg = self.db.add_prompt("Deleted but Exists", "New Author", "New Details", overwrite=False)

        self.assertIn("exists but is soft-deleted", msg)

        # Verify it was not restored or updated
        prompt = self.db.get_prompt_by_name("Deleted but Exists", include_deleted=True)
        self.assertEqual(prompt['author'], "Author")  # Should be the original author
        self.assertEqual(prompt['deleted'], 1)  # Should remain deleted

    def test_soft_delete_nonexistent_item_returns_false(self):
        """Ensure attempting to delete non-existent items returns False and doesn't error."""
        result_prompt = self.db.soft_delete_prompt("non-existent prompt")
        self.assertFalse(result_prompt)

        result_keyword = self.db.soft_delete_keyword("non-existent keyword")
        self.assertFalse(result_keyword)

    def test_update_keywords_for_prompt_with_empty_list_removes_all(self):
        """Updating keywords with an empty list should remove all existing keywords."""
        pid, _, _ = self.db.add_prompt("Keyword Test", "A", "D", keywords=["kw1", "kw2"])
        self.assertEqual(len(self.db.fetch_keywords_for_prompt(pid)), 2)

        # Update with an empty list
        self.db.update_keywords_for_prompt(pid, [])

        self.assertEqual(len(self.db.fetch_keywords_for_prompt(pid)), 0)

        # Verify unlink events were logged
        unlink_logs = [l for l in self.db.get_sync_log_entries() if l['operation'] == 'unlink']
        self.assertEqual(len(unlink_logs), 2)

    def test_update_keywords_for_prompt_is_idempotent(self):
        """Running update_keywords with the same list should result in no changes or new logs."""
        pid, _, _ = self.db.add_prompt("Idempotent Test", "A", "D", keywords=["kw1", "kw2"])

        initial_log_count = len(self.db.get_sync_log_entries())

        # Rerun with the same keywords
        self.db.update_keywords_for_prompt(pid, ["kw1", "kw2"])

        final_log_count = len(self.db.get_sync_log_entries())
        self.assertEqual(initial_log_count, final_log_count,
                         "No new sync logs should be created for an idempotent update")

    def test_update_keywords_for_prompt_handles_duplicates_and_whitespace(self):
        """Ensure keyword lists are properly normalized before processing."""
        pid, _, _ = self.db.add_prompt("Normalization Test", "A", "D")

        messy_keywords = ["  Tag A  ", "tag b", "Tag A", "   ", "tag c  "]
        self.db.update_keywords_for_prompt(pid, messy_keywords)

        final_keywords = self.db.fetch_keywords_for_prompt(pid)
        self.assertEqual(sorted(final_keywords), ["tag a", "tag b", "tag c"])


class TestBulkOperationsAndScale(BaseTestCase):
    """
    Tests focusing on bulk methods and behavior with a larger number of records.
    """

    def test_execute_many_success(self):
        """Test successful bulk insertion with execute_many."""
        keywords_to_add = [
            (f"bulk_keyword_{i}", str(uuid.uuid4()), self.db._get_current_utc_timestamp_str(), 1, self.client_id, 0)
            for i in range(50)
        ]
        sql = "INSERT INTO PromptKeywordsTable (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, ?)"

        with self.db.transaction():
            self.db.execute_many(sql, keywords_to_add)

        count = self.db.execute_query("SELECT COUNT(*) FROM PromptKeywordsTable").fetchone()[0]
        self.assertEqual(count, 50)

    def test_execute_many_failure_with_integrity_error_rolls_back(self):
        """Ensure a failing execute_many call within a transaction rolls back the entire batch."""
        keywords_to_add = [
            ("unique_1", str(uuid.uuid4()), self.db._get_current_utc_timestamp_str(), 1, self.client_id, 0),
            ("not_unique", str(uuid.uuid4()), self.db._get_current_utc_timestamp_str(), 1, self.client_id, 0),
            ("not_unique", str(uuid.uuid4()), self.db._get_current_utc_timestamp_str(), 1, self.client_id, 0),
            # Fails here
            ("unique_2", str(uuid.uuid4()), self.db._get_current_utc_timestamp_str(), 1, self.client_id, 0),
        ]
        sql = "INSERT INTO PromptKeywordsTable (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, ?)"

        with self.assertRaises(DatabaseError) as cm:
            with self.db.transaction():
                self.db.execute_many(sql, keywords_to_add)

        self.assertIn("UNIQUE constraint failed", str(cm.exception))

        # Verify rollback
        count = self.db.execute_query("SELECT COUNT(*) FROM PromptKeywordsTable").fetchone()[0]
        self.assertEqual(count, 0)

    def test_dependency_integrity_on_delete(self):
        """
        Verify that deleting a prompt doesn't delete a keyword used by other prompts.
        """
        # "common_kw" is used by both prompts
        p1_id, _, _ = self.db.add_prompt("Prompt 1", "A", "D", keywords=["p1_kw", "common_kw"])
        self.db.add_prompt("Prompt 2", "B", "E", keywords=["p2_kw", "common_kw"])

        # Soft delete Prompt 1
        self.db.soft_delete_prompt(p1_id)

        # Check that "common_kw" still exists and is active
        kw_data = self.db.get_active_keyword_by_text("common_kw")
        self.assertIsNotNone(kw_data)

        # Check that Prompt 2 still has its link to "common_kw"
        p2_details = self.db.fetch_prompt_details("Prompt 2")
        self.assertIn("common_kw", p2_details['keywords'])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


#
# End of tests_prompts_db.py
#######################################################################################################################
