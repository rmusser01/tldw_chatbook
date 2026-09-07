# test_prompts_db_pytest.py
# Pytest-based tests for Prompts_DB

"""
Integration tests for Prompts_DB using real SQLite database instances.

These tests verify the complete functionality of the PromptsDatabase class
including schema creation, CRUD operations, and data integrity.
"""

import inspect
import sqlite3
import json

import pytest
from pathlib import Path

from tldw_chatbook.DB.Prompts_DB import (
    PromptsDatabase,
    DatabaseError,
    InputError,
    ConflictError,
    ExpectedVersionConflictError,
    add_or_update_prompt,
    load_prompt_details_for_ui,
    export_prompt_keywords_to_csv,
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    yield str(tmp_path / "test_prompts.db")


@pytest.fixture
def in_memory_db():
    """Create an in-memory database for testing."""
    db = PromptsDatabase(":memory:", client_id="test_client")
    yield db
    db.close_connection()


@pytest.fixture
def file_db(temp_db_path):
    """Create a file-based database for testing."""
    db = PromptsDatabase(temp_db_path, client_id="test_client")
    yield db
    db.close_connection()


def test_transaction_declares_connection_iterator_return_type():
    """The public transaction context manager exposes its yielded type."""
    annotation = inspect.signature(PromptsDatabase.transaction).return_annotation

    assert annotation is not inspect.Signature.empty
    assert "Iterator" in str(annotation)
    assert "sqlite3.Connection" in str(annotation)


def test_list_prompts_does_not_interpolate_sql_fragments():
    """Prompt listing keeps every SQL statement static for future safety."""
    source = inspect.getsource(PromptsDatabase.list_prompts)

    assert "{where_clause}" not in source


class TestPromptsDBInitialization:
    """Test database initialization and schema creation."""

    def test_memory_db_initialization(self):
        """Test in-memory database initialization."""
        db = PromptsDatabase(":memory:", client_id="test_client")
        assert db.is_memory_db is True
        assert db.db_path_str == ":memory:"
        assert db.client_id == "test_client"
        db.close_connection()

    def test_file_db_initialization(self, temp_db_path):
        """Test file-based database initialization."""
        db = PromptsDatabase(temp_db_path, client_id="test_client")
        # Handle macOS path resolution differences
        assert db.db_path_str == temp_db_path or Path(db.db_path_str).samefile(
            temp_db_path
        )
        assert Path(temp_db_path).exists()
        db.close_connection()

    def test_schema_creation(self, in_memory_db):
        """Test that all required tables are created."""
        conn = in_memory_db._get_thread_connection()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            "Prompts",
            "PromptKeywordsTable",
            "PromptKeywordLinks",
            "sync_log",
        ]
        for table in expected_tables:
            assert table in tables

    def test_fts_tables_creation(self, in_memory_db):
        """Test that FTS5 virtual tables are created."""
        conn = in_memory_db._get_thread_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'"
        )
        fts_tables = [row[0] for row in cursor.fetchall()]
        assert "prompts_fts" in fts_tables


class TestPromptOperations:
    """Test CRUD operations for prompts."""

    def test_add_prompt_basic(self, in_memory_db):
        """Test basic prompt creation."""
        name = "Test Prompt"
        result = in_memory_db.add_prompt(
            name=name,
            author="Test Author",
            details=None,
            system_prompt="Test system prompt",
            user_prompt="Test user prompt",
        )

        prompt_id, prompt_uuid, action = result
        assert prompt_id is not None
        assert isinstance(prompt_id, int)
        assert prompt_uuid is not None
        assert "added successfully" in action

        # Verify prompt was created
        prompt = in_memory_db.get_prompt_by_id(prompt_id)
        assert prompt is not None
        assert prompt["name"] == name

    def test_add_prompt_with_keywords(self, in_memory_db):
        """Test prompt creation with keywords."""
        result = in_memory_db.add_prompt(
            name="Test Prompt with Keywords",
            author="Test Author",
            details=None,
            system_prompt="System prompt",
            user_prompt="User prompt",
            keywords=["test", "keywords", "example"],
        )

        prompt_id, _, _ = result

        # Verify the prompt was created
        prompt = in_memory_db.get_prompt_by_id(prompt_id)
        assert prompt is not None
        assert prompt["name"] == "Test Prompt with Keywords"

    def test_update_prompt(self, in_memory_db):
        """Test prompt update."""
        # Create prompt
        result = in_memory_db.add_prompt(
            name="Original Name", author="Original Author", details=None
        )
        prompt_id, _, _ = result

        # Update it
        in_memory_db.update_prompt_by_id(
            prompt_id,
            {"name": "Updated Name", "system_prompt": "Updated system prompt"},
        )

        # Verify update
        prompt = in_memory_db.get_prompt_by_id(prompt_id)
        assert prompt["name"] == "Updated Name"
        assert prompt["system_prompt"] == "Updated system prompt"
        assert prompt["author"] == "Original Author"  # Should not change

    def test_delete_prompt(self, in_memory_db):
        """Test prompt deletion (soft delete)."""
        result = in_memory_db.add_prompt(name="To Delete", author=None, details=None)
        prompt_id, _, _ = result

        # Delete it
        in_memory_db.soft_delete_prompt(prompt_id)

        # Should not be in active prompts
        prompts = in_memory_db.get_all_prompts()
        assert not any(p["id"] == prompt_id for p in prompts)

    def test_restore_deleted_prompt_preserves_artifact_and_keywords(self, in_memory_db):
        """Undo resurrects the exact tombstone as a new current version."""
        prompt_id, _uuid, _message = in_memory_db.add_prompt(
            name="Restore recipe",
            author="Author",
            details="Details",
            system_prompt="System lane",
            user_prompt="User lane",
            keywords=["Alpha", "Beta"],
            artifact_type="recipe",
        )
        assert in_memory_db.soft_delete_prompt(prompt_id) is True

        restored = in_memory_db.restore_deleted_prompt(
            prompt_id, expected_version=2
        )

        assert restored["id"] == prompt_id
        assert restored["deleted"] == 0
        assert restored["version"] == 3
        assert restored["artifact_type"] == "recipe"
        assert restored["system_prompt"] == "System lane"
        assert restored["user_prompt"] == "User lane"
        assert in_memory_db.fetch_keywords_for_prompt(prompt_id) == ["alpha", "beta"]
        events = in_memory_db.get_sync_log_entries()
        prompt_events = [event for event in events if event["entity"] == "Prompts"]
        assert prompt_events[-1]["operation"] == "update"
        assert prompt_events[-1]["payload"]["deleted"] == 0
        assert prompt_events[-1]["payload"]["keywords"] == ["alpha", "beta"]

    def test_restore_deleted_prompt_rejects_stale_expected_version(self, in_memory_db):
        """A stale receipt cannot resurrect a newer tombstone."""
        prompt_id, _uuid, _message = in_memory_db.add_prompt(
            name="Stale restore", author=None, details=None
        )
        assert in_memory_db.soft_delete_prompt(prompt_id) is True

        with pytest.raises(ExpectedVersionConflictError):
            in_memory_db.restore_deleted_prompt(prompt_id, expected_version=1)

        tombstone = in_memory_db.fetch_prompt_details(prompt_id, include_deleted=True)
        assert tombstone is not None
        assert tombstone["deleted"] == 1
        assert tombstone["version"] == 2

    def test_soft_delete_prompt_rejects_stale_expected_version(self, in_memory_db):
        """A stale editor cannot create a tombstone its receipt cannot restore."""
        prompt_id, _uuid, _message = in_memory_db.add_prompt(
            name="Concurrent delete", author=None, details=None
        )
        in_memory_db.update_prompt_by_id(
            prompt_id,
            {"details": "changed elsewhere"},
            expected_version=1,
        )

        with pytest.raises(ExpectedVersionConflictError):
            in_memory_db.soft_delete_prompt(prompt_id, expected_version=1)

        current = in_memory_db.fetch_prompt_details(prompt_id, include_deleted=True)
        assert current is not None
        assert current["deleted"] == 0
        assert current["version"] == 2

    def test_duplicate_prompt_name(self, in_memory_db):
        """Test that duplicate prompt names are rejected."""
        name = "Unique Prompt"
        in_memory_db.add_prompt(name=name, author=None, details=None)

        with pytest.raises(ConflictError):
            in_memory_db.add_prompt(name=name, author=None, details=None)

    def test_prompt_and_recipe_briefs_preserve_artifact_type_and_lane_flags(
        self, in_memory_db
    ):
        """Library rows expose artifact identity and compiled-lane presence."""
        prompt_definition = {
            "kind": "block_prompt",
            "schema_version": 2,
            "lanes": [],
        }
        recipe_definition = {
            "kind": "block_recipe",
            "schema_version": 2,
            "lanes": [],
        }
        in_memory_db.add_prompt(
            name="Executable Prompt",
            author=None,
            details=None,
            system_prompt="system lane",
            user_prompt="",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=prompt_definition,
            artifact_type="prompt",
        )
        recipe_id, recipe_uuid, _ = in_memory_db.add_prompt(
            name="Reusable Recipe",
            author=None,
            details=None,
            system_prompt="",
            user_prompt="compiled recipe lane",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=recipe_definition,
            artifact_type="recipe",
        )

        briefs, _pages, _page, _total = in_memory_db.list_prompts(per_page=10)
        by_name = {brief["name"]: brief for brief in briefs}
        assert by_name["Executable Prompt"]["artifact_type"] == "prompt"
        assert by_name["Executable Prompt"]["has_system_prompt"] == 1
        assert by_name["Executable Prompt"]["has_user_prompt"] == 0
        assert by_name["Reusable Recipe"]["artifact_type"] == "recipe"
        assert by_name["Reusable Recipe"]["has_system_prompt"] == 0
        assert by_name["Reusable Recipe"]["has_user_prompt"] == 1

        detail = in_memory_db.fetch_prompt_details(recipe_uuid, include_deleted=True)
        assert detail["id"] == recipe_id
        assert detail["artifact_type"] == "recipe"
        assert json.loads(detail["prompt_definition"]) == recipe_definition

        searched, total = in_memory_db.search_prompts("compiled")
        assert total == 1
        assert searched[0]["name"] == "Reusable Recipe"
        assert searched[0]["artifact_type"] == "recipe"
        assert searched[0]["has_user_prompt"] == 1

    @pytest.mark.parametrize("artifact_type", ["Recipe", "unknown", "", 7])
    def test_invalid_artifact_type_is_rejected_at_prompt_db_boundary(
        self, in_memory_db, artifact_type
    ):
        with pytest.raises(InputError, match="artifact_type"):
            in_memory_db.add_prompt(
                name="Invalid Type",
                author=None,
                details=None,
                artifact_type=artifact_type,
            )

    def test_expected_version_is_checked_inside_update_transaction(
        self, temp_db_path
    ):
        """A stale writer from a second database instance cannot alter a row."""
        first = PromptsDatabase(temp_db_path, client_id="writer-one")
        second = PromptsDatabase(temp_db_path, client_id="writer-two")
        try:
            prompt_id, prompt_uuid, _ = first.add_prompt(
                name="Concurrent Prompt",
                author="Author",
                details="original details",
                system_prompt="original system",
                user_prompt="original user",
            )
            captured_version = first.fetch_prompt_details(prompt_uuid)["version"]

            first.update_prompt_by_id(
                prompt_id,
                {"details": "first writer"},
                expected_version=captured_version,
            )
            after_first_write = first.fetch_prompt_details(
                prompt_uuid, include_deleted=True
            )

            with pytest.raises(ConflictError, match="changed after it was opened"):
                second.update_prompt_by_id(
                    prompt_id,
                    {"details": "stale writer", "user_prompt": "must not persist"},
                    expected_version=captured_version,
                )

            stored = second.fetch_prompt_details(prompt_uuid, include_deleted=True)
            assert stored == after_first_write
        finally:
            first.close_connection()
            second.close_connection()

    def test_busy_snapshot_from_a_second_writer_is_reported_as_conflict(
        self, temp_db_path
    ):
        """A WAL snapshot race raises ConflictError and leaves the winner intact."""
        first = PromptsDatabase(temp_db_path, client_id="writer-one")
        second = PromptsDatabase(temp_db_path, client_id="writer-two")
        try:
            prompt_id, prompt_uuid, _ = first.add_prompt(
                name="WAL Race Prompt",
                author="Author",
                details="original details",
                system_prompt="original system",
                user_prompt="original user",
            )
            expected_version = first.fetch_prompt_details(prompt_uuid)["version"]
            raced = False

            def commit_second_writer_before_first_update(statement):
                nonlocal raced
                if raced or not statement.startswith("UPDATE Prompts SET"):
                    return
                raced = True
                second.update_prompt_by_id(
                    prompt_id,
                    {"details": "winning writer"},
                    expected_version=expected_version,
                )

            first.get_connection().set_trace_callback(
                commit_second_writer_before_first_update
            )
            try:
                with pytest.raises(ConflictError, match="version race"):
                    first.update_prompt_by_id(
                        prompt_id,
                        {"details": "stale writer", "user_prompt": "must not persist"},
                        expected_version=expected_version,
                    )
            finally:
                first.get_connection().set_trace_callback(None)

            assert raced
            stored = first.fetch_prompt_details(prompt_uuid, include_deleted=True)
            assert stored["details"] == "winning writer"
            assert stored["user_prompt"] == "original user"
            assert stored["version"] == expected_version + 1
        finally:
            first.close_connection()
            second.close_connection()

    def test_busy_snapshot_during_create_is_reported_as_conflict(self, temp_db_path):
        """A WAL snapshot race during create must not escape as a database error."""
        first = PromptsDatabase(temp_db_path, client_id="writer-one")
        second = PromptsDatabase(temp_db_path, client_id="writer-two")
        try:
            raced = False

            def commit_second_writer_before_first_insert(statement):
                nonlocal raced
                if raced or not statement.startswith("INSERT INTO Prompts"):
                    return
                raced = True
                second.add_prompt(
                    name="WAL Create Race Prompt",
                    author="Second writer",
                    details="winning create",
                    system_prompt="winner system",
                    user_prompt="winner user",
                )

            first.get_connection().set_trace_callback(
                commit_second_writer_before_first_insert
            )
            try:
                with pytest.raises(ConflictError, match="snapshot race"):
                    first.add_prompt(
                        name="WAL Create Race Prompt",
                        author="First writer",
                        details="stale create",
                        system_prompt="stale system",
                        user_prompt="stale user",
                    )
            finally:
                first.get_connection().set_trace_callback(None)

            assert raced
            stored = second.get_prompt_by_name("WAL Create Race Prompt")
            assert stored["author"] == "Second writer"
            assert stored["details"] == "winning create"
            assert stored["system_prompt"] == "winner system"
            assert stored["user_prompt"] == "winner user"
        finally:
            first.close_connection()
            second.close_connection()


class TestKeywordOperations:
    """Test keyword management."""

    def test_add_keyword(self, in_memory_db):
        """Test adding keywords."""
        keyword_id = in_memory_db.add_keyword("test-keyword")
        assert keyword_id is not None

        # Adding same keyword should return same ID
        keyword_id2 = in_memory_db.add_keyword("test-keyword")
        assert keyword_id == keyword_id2

    def test_get_all_keywords(self, in_memory_db):
        """Test retrieving all keywords."""
        # Add some keywords
        keywords = ["python", "testing", "database"]
        for kw in keywords:
            in_memory_db.add_keyword(kw)

        all_keywords = in_memory_db.get_all_keywords()
        keyword_names = [kw["name"] for kw in all_keywords]

        for kw in keywords:
            assert kw in keyword_names


class TestSearchFunctionality:
    """Test search and filtering operations."""

    def test_search_prompts_by_keyword(self, in_memory_db):
        """Test searching prompts by keyword."""
        # Create prompts with different keywords
        in_memory_db.add_prompt(
            name="Python Tutorial",
            author=None,
            details=None,
            keywords=["python", "tutorial"],
        )
        in_memory_db.add_prompt(
            name="SQL Guide", author=None, details=None, keywords=["sql", "database"]
        )
        in_memory_db.add_prompt(
            name="Python Database",
            author=None,
            details=None,
            keywords=["python", "database"],
        )

        # Search for python prompts
        results = in_memory_db.search_prompts_by_keyword("python")
        assert len(results) == 2

        # Search for database prompts
        results = in_memory_db.search_prompts_by_keyword("database")
        assert len(results) == 2

    def test_search_prompts_by_text(self, in_memory_db):
        """Test full-text search."""
        in_memory_db.add_prompt(
            name="Code Review Assistant",
            author=None,
            details=None,
            system_prompt="Help review code for best practices",
        )
        in_memory_db.add_prompt(
            name="Writing Helper",
            author=None,
            details=None,
            system_prompt="Assist with writing and editing",
        )

        # Search for "review"
        results = in_memory_db.search_prompts_by_text("review")
        assert len(results) == 1
        assert results[0]["name"] == "Code Review Assistant"


class TestStandaloneFunctions:
    """Test standalone utility functions."""

    def test_add_or_update_prompt(self, temp_db_path):
        """Test the add_or_update_prompt standalone function."""
        db = PromptsDatabase(temp_db_path, client_id="test_client")
        try:
            # First add
            result = add_or_update_prompt(
                db,
                "Test Standalone",
                author="Tester",
                details=None,
                system_prompt=None,
                user_prompt=None,
                keywords=["test"],
            )
            prompt_id, prompt_uuid, action = result
            assert prompt_id is not None

            # Update (same name)
            result2 = add_or_update_prompt(
                db,
                "Test Standalone",
                author=None,
                details=None,
                system_prompt="Updated system",
                user_prompt=None,
                keywords=None,
            )
            prompt_id2, _, _ = result2
            assert prompt_id == prompt_id2
        finally:
            db.close_connection()

    def test_load_prompt_details(self, temp_db_path):
        """Test loading prompt details for UI."""
        db = PromptsDatabase(temp_db_path, client_id="test_client")
        try:
            # Add a prompt
            result = add_or_update_prompt(
                db,
                "UI Test",
                author=None,
                details=None,
                system_prompt=None,
                user_prompt=None,
                keywords=["ui", "test"],
            )
            prompt_id, _, _ = result

            # Load details by name (not ID)
            name, author, details, system, user, keywords_str = (
                load_prompt_details_for_ui(db, "UI Test")
            )
            assert name == "UI Test"
            assert "ui" in keywords_str and "test" in keywords_str
        finally:
            db.close_connection()

    def test_export_keywords_csv(self, temp_db_path, tmp_path):
        """Test exporting keywords to CSV."""
        db = PromptsDatabase(temp_db_path, client_id="test_client")
        try:
            # Add prompts with keywords
            add_or_update_prompt(
                db,
                "Prompt 1",
                author=None,
                details=None,
                system_prompt=None,
                user_prompt=None,
                keywords=["python", "test"],
            )
            add_or_update_prompt(
                db,
                "Prompt 2",
                author=None,
                details=None,
                system_prompt=None,
                user_prompt=None,
                keywords=["python", "database"],
            )

            # Export
            csv_file = tmp_path / "keywords.csv"
            export_prompt_keywords_to_csv(db, str(csv_file))

            assert csv_file.exists()
            content = csv_file.read_text()
            assert "python" in content
            assert "2" in content  # Usage count
        finally:
            db.close_connection()


class TestErrorHandling:
    """Test error conditions and validation."""

    def test_invalid_prompt_name(self, in_memory_db):
        """Test validation of prompt names."""
        with pytest.raises(InputError):
            in_memory_db.add_prompt(name="", author=None, details=None)  # Empty name

        with pytest.raises(InputError):
            in_memory_db.add_prompt(
                name="   ", author=None, details=None
            )  # Whitespace only

    def test_nonexistent_prompt(self, in_memory_db):
        """Test operations on non-existent prompts."""
        result = in_memory_db.get_prompt_by_id(9999)
        assert result is None

        # Update non-existent prompt should not fail
        in_memory_db.update_prompt(9999, name="Won't work")

        # Delete non-existent prompt should not fail
        in_memory_db.delete_prompt(9999)

    def test_concurrent_access(self, file_db):
        """Test thread-safe access."""
        import threading
        import time

        results = []

        def add_prompt(name):
            # Retry up to 3 times with exponential backoff
            for attempt in range(3):
                try:
                    result = file_db.add_prompt(name=name, author=None, details=None)
                    prompt_id = result[0]
                    results.append(("success", prompt_id, name))
                    return
                except (DatabaseError, sqlite3.OperationalError) as e:
                    if "database is locked" in str(e) and attempt < 2:
                        time.sleep(0.1 * (2**attempt))  # Exponential backoff
                        continue
                    results.append(("error", str(e)))
                    return
                except Exception as e:
                    results.append(("error", str(e)))
                    return

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_prompt, args=(f"Thread {i}",))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check results
        successes = [r for r in results if r[0] == "success"]
        [r for r in results if r[0] == "error"]

        # With retry logic, we expect most threads to succeed
        # At minimum, we should have more successes than errors
        assert len(successes) >= 3  # At least 3 out of 5 should succeed

        # Verify all successful prompts were actually created
        all_prompts = file_db.get_all_prompts()
        prompt_names = [p["name"] for p in all_prompts]
        for success in successes:
            _, _, thread_name = success
            assert thread_name in prompt_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



# ---------------------------------------------------------------------------
# Library query seams (task-1337 plan Task 3)
# ---------------------------------------------------------------------------


def _set_prompt_timestamps(db, prompt_id, last_modified):
    # Prompts sync triggers require version to increment by exactly 1 per UPDATE.
    db.execute_query(
        "UPDATE Prompts SET last_modified = ?, version = version + 1 WHERE id = ?",
        (last_modified, prompt_id),
    )
    # Seed helpers must not leak an ambient transaction into public mutations,
    # which own their BEGIN IMMEDIATE and durable commit boundary.
    db.get_connection().commit()


def _seed_library_prompt(
    db,
    *,
    name,
    details=None,
    system_prompt=None,
    user_prompt=None,
    keywords=None,
    prompt_definition=None,
    author="author",
    last_modified="2026-01-01 00:00:00",
):
    prompt_id, prompt_uuid, _ = db.add_prompt(
        name=name,
        author=author,
        details=details if details is not None else f"details for {name}",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        keywords=keywords or [],
        prompt_definition=prompt_definition,
    )
    assert prompt_id is not None, f"seed for {name!r} failed"
    _set_prompt_timestamps(db, prompt_id, last_modified)
    return prompt_id, prompt_uuid


def test_library_prompts_page_lists_active_with_stable_order(in_memory_db):
    db = in_memory_db
    first_id, _ = _seed_library_prompt(
        db, name="First", last_modified="2026-01-01 00:00:00"
    )
    second_id, _ = _seed_library_prompt(
        db, name="Second", last_modified="2026-01-03 00:00:00"
    )
    third_id, _ = _seed_library_prompt(
        db, name="Third", last_modified="2026-01-02 00:00:00"
    )
    deleted_id, _ = _seed_library_prompt(db, name="Deleted")
    assert db.get_connection().in_transaction is False
    db.soft_delete_prompt(deleted_id)

    page_one = db.list_library_prompts_page(limit=2, offset=0)
    assert page_one["total"] == 3
    assert [item["id"] for item in page_one["items"]] == [second_id, third_id]

    page_two = db.list_library_prompts_page(limit=2, offset=2)
    assert page_two["total"] == 3
    assert [item["id"] for item in page_two["items"]] == [first_id]

    beyond = db.list_library_prompts_page(limit=10, offset=50)
    assert beyond["total"] == 3
    assert beyond["items"] == []


def test_library_prompts_page_projection_is_bounded(in_memory_db):
    db = in_memory_db
    keywords = [f"kw{index:02d}" for index in range(25)]
    _, prompt_uuid = _seed_library_prompt(
        db,
        name="Projected",
        details="secret details " * 100,
        system_prompt="sys " * 500,
        user_prompt="usr " * 500,
        prompt_definition={"messages": [{"role": "user", "content": "hi"}]},
        keywords=keywords,
    )

    item = db.list_library_prompts_page(limit=10, offset=0)["items"][0]
    assert item["uuid"] == prompt_uuid
    assert item["name"] == "Projected"
    assert len(item["details_preview"]) <= 241
    assert item["has_system_prompt"] == 1
    assert item["has_user_prompt"] == 1
    assert item["has_prompt_definition"] == 1
    assert len(item["keywords"]) == 20
    assert item["keyword_total"] == 25
    assert item["keywords_truncated"] is True
    # Full section text belongs to the detail seam only.
    for forbidden in ("system_prompt", "user_prompt", "prompt_definition", "details"):
        assert forbidden not in item


def test_library_prompts_search_exact_name_first_and_distinct_total(in_memory_db):
    db = in_memory_db
    exact_id, _ = _seed_library_prompt(
        db,
        name="Quarterly",
        details="nothing relevant here",
        keywords=["quarterly", "quarterly-finance"],
        last_modified="2026-01-01 00:00:00",
    )
    body_id, _ = _seed_library_prompt(
        db,
        name="Other",
        system_prompt="a quarterly deep dive",
        last_modified="2026-02-01 00:00:00",
    )

    payload = db.search_library_prompts_page(query="quarterly", limit=10, offset=0)
    assert payload["total"] == 2
    assert [item["id"] for item in payload["items"]] == [exact_id, body_id]
    exact_item, body_item = payload["items"]
    assert "name" in exact_item["matched_fields"]
    assert "keywords" in exact_item["matched_fields"]
    assert "quarterly" in exact_item["matched_keywords"]
    assert "quarterly-finance" in exact_item["matched_keywords"]
    assert "system_prompt" in body_item["matched_fields"]


def test_library_prompts_search_covers_every_section_and_literal_wildcards(in_memory_db):
    db = in_memory_db
    details_id, _ = _seed_library_prompt(db, name="D", details="needle in details")
    user_id, _ = _seed_library_prompt(db, name="U", user_prompt="needle in user")
    definition_id, _ = _seed_library_prompt(
        db, name="J", prompt_definition='{"text": "needle in definition"}'
    )
    target_id, _ = _seed_library_prompt(db, name="100% ready_now", details="plain")
    _seed_library_prompt(db, name="readyXnow decoy", details="plain decoy body")

    for expected_id in (details_id, user_id, definition_id):
        payload = db.search_library_prompts_page(query="needle", limit=10, offset=0)
        assert expected_id in [item["id"] for item in payload["items"]]

    percent = db.search_library_prompts_page(query="100%", limit=10, offset=0)
    assert [item["id"] for item in percent["items"]] == [target_id]

    underscore = db.search_library_prompts_page(query="ready_now", limit=10, offset=0)
    assert [item["id"] for item in underscore["items"]] == [target_id]

    for hostile in ('"unclosed', "ready OR", "AND )(", "ready*", "NEAR/1"):
        result = db.search_library_prompts_page(query=hostile, limit=10, offset=0)
        assert isinstance(result["total"], int)
        assert isinstance(result["items"], list)


def test_library_prompt_overview_bounds_every_section(in_memory_db):
    db = in_memory_db
    _, prompt_uuid = _seed_library_prompt(
        db,
        name="Overview",
        details="d" * 1000,
        system_prompt="s" * 2000,
        user_prompt="u" * 3000,
        prompt_definition="x" * 4000,
    )

    overview = db.get_library_prompt_overview(prompt_uuid)
    assert overview is not None
    assert overview["uuid"] == prompt_uuid
    assert overview["name"] == "Overview"
    assert isinstance(overview["version"], int)
    sections = overview["sections"]
    assert sections["details"]["total_chars"] == 1000
    assert sections["system_prompt"]["total_chars"] == 2000
    assert sections["user_prompt"]["total_chars"] == 3000
    assert sections["prompt_definition"]["total_chars"] == 4000
    for section in sections.values():
        assert len(section["preview"]) <= 241
    # No full section text and no version-history expansion.
    assert "system_prompt" not in overview
    assert "user_prompt" not in overview
    assert "prompt_definition" not in overview
    assert "versions" not in overview
    assert "history" not in overview


def test_library_prompt_section_windows_text(in_memory_db):
    db = in_memory_db
    body = "abcdef" * 900  # 5400 chars
    _, prompt_uuid = _seed_library_prompt(db, name="Sectioned", system_prompt=body)

    detail = db.get_library_prompt_section(
        prompt_uuid, section="system_prompt", start=1200, max_chars=2000
    )
    assert detail is not None
    assert detail["uuid"] == prompt_uuid
    assert detail["section"] == "system_prompt"
    assert detail["total_chars"] == len(body)
    assert detail["start"] == 1200
    assert detail["returned_chars"] == 2000
    assert detail["has_more"] is True
    assert detail["text"] == body[1200:3200]

    tail = db.get_library_prompt_section(
        prompt_uuid, section="system_prompt", start=5000, max_chars=2000
    )
    assert tail["text"] == body[5000:]
    assert tail["has_more"] is False

    missing = db.get_library_prompt_section(
        "no-such-uuid", section="system_prompt", start=0, max_chars=100
    )
    assert missing is None


def test_library_prompt_detail_reads_run_inside_transaction(in_memory_db):
    db = in_memory_db
    _, prompt_uuid = _seed_library_prompt(
        db, name="Transactional", system_prompt="bounded body"
    )
    conn = db.get_connection()
    conn.commit()
    observed: list[bool] = []

    def record_transaction_state(sql: str) -> None:
        if "FROM Prompts" in sql and (
            "AS details_total" in sql or "AS total_chars" in sql
        ):
            observed.append(conn.in_transaction)

    conn.set_trace_callback(record_transaction_state)
    try:
        assert db.get_library_prompt_overview(prompt_uuid) is not None
        assert db.get_library_prompt_section(
            prompt_uuid, section="system_prompt", start=0, max_chars=20
        ) is not None
    finally:
        conn.set_trace_callback(None)

    assert observed == [True, True]


def test_library_prompt_section_rejects_invalid_section(in_memory_db):
    db = in_memory_db
    _, prompt_uuid = _seed_library_prompt(db, name="Guarded", system_prompt="body")
    with pytest.raises((InputError, ValueError)):
        db.get_library_prompt_section(
            prompt_uuid, section="sync_log", start=0, max_chars=100
        )
