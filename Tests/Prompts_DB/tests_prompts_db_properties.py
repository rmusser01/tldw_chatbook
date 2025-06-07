# test_prompts_db_properties.py
#
# Property-based tests for the Prompts_DB_v2 library using Hypothesis.

# Imports
import uuid
import pytest
import json
from pathlib import Path
import sqlite3
import threading
import time

# Third-Party Imports
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, Bundle

# Local Imports
# Assuming Prompts_DB_v2.py is in a location Python can find.
# For example, in the same directory or in a package.
from tldw_chatbook.DB.Prompts_DB import (
    PromptsDatabase,
    InputError,
    DatabaseError,
    ConflictError
)

########################################################################################################################
#
# Hypothesis Setup:

# A custom profile for DB tests to avoid timeouts on complex operations.
settings.register_profile(
    "db_friendly",
    deadline=1500,  # Increased deadline for potentially slow DB I/O
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture
    ]
)
settings.load_profile("db_friendly")


# --- Fixtures ---

@pytest.fixture
def client_id():
    """Provides a consistent client ID for tests."""
    return "hypothesis_client"


@pytest.fixture
def db_path(tmp_path):
    """Provides a temporary path for the database file for each test."""
    return tmp_path / "prop_test_prompts_db.sqlite"


@pytest.fixture(scope="function")
def db_instance(db_path, client_id):
    """Creates a fresh PromptsDatabase instance for each test function."""
    current_db_path = Path(db_path)
    # Ensure no leftover files from a failed previous run (important for WAL mode)
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(current_db_path) + suffix)
        if p.exists():
            p.unlink(missing_ok=True)

    db = PromptsDatabase(current_db_path, client_id)
    yield db
    db.close_connection()


# --- Hypothesis Strategies ---

# Strategy for text fields that cannot be empty or just whitespace.
st_required_text = st.text(min_size=1, max_size=100).filter(lambda s: s.strip())

# Strategy for optional text fields.
st_optional_text = st.one_of(st.none(), st.text(max_size=500))


@st.composite
def st_prompt_data(draw):
    """A composite strategy to generate a dictionary of prompt data."""
    # Generate keywords that are unique after normalization
    keywords = draw(st.lists(st_required_text, max_size=5, unique_by=lambda s: s.strip().lower()))

    return {
        "name": draw(st_required_text),
        "author": draw(st_optional_text),
        "details": draw(st_optional_text),
        "system_prompt": draw(st_optional_text),
        "user_prompt": draw(st_optional_text),
        "keywords": keywords,
    }


# A strategy for a non-one integer to test version validation triggers.
st_bad_version_offset = st.integers().filter(lambda x: x != 1)


# --- Test Classes ---

class TestPromptProperties:
    """Property-based tests for core Prompt operations."""

    @given(prompt_data=st_prompt_data())
    def test_prompt_roundtrip(self, db_instance: PromptsDatabase, prompt_data: dict):
        """
        Property: If we add a prompt, retrieving it should return the same data,
        accounting for any normalization (e.g., stripping name, normalizing keywords).
        """
        try:
            # Use overwrite=False to test the create path; ConflictError is a valid outcome.
            prompt_id, prompt_uuid, _ = db_instance.add_prompt(**prompt_data, overwrite=False)
        except ConflictError:
            return  # Hypothesis generated a name collision, which is not a failure.

        assert prompt_id is not None
        assert prompt_uuid is not None

        retrieved_prompt = db_instance.fetch_prompt_details(prompt_id)
        assert retrieved_prompt is not None

        # Compare basic fields
        assert retrieved_prompt["name"] == prompt_data["name"].strip()
        assert retrieved_prompt["author"] == prompt_data["author"]
        assert retrieved_prompt["details"] == prompt_data["details"]
        assert retrieved_prompt["system_prompt"] == prompt_data["system_prompt"]
        assert retrieved_prompt["user_prompt"] == prompt_data["user_prompt"]
        assert retrieved_prompt["version"] == 1
        assert not retrieved_prompt["deleted"]

        # Compare keywords, which are normalized by the database
        expected_keywords = sorted([db_instance._normalize_keyword(k) for k in prompt_data["keywords"]])
        retrieved_keywords = sorted(retrieved_prompt["keywords"])
        assert retrieved_keywords == expected_keywords

    @given(initial_prompt=st_prompt_data(), update_payload=st_prompt_data())
    def test_update_increments_version_and_changes_data(self, db_instance: PromptsDatabase, initial_prompt: dict,
                                                        update_payload: dict):
        """
        Property: A successful update must increment the version number by exactly 1
        and correctly apply the new data, including keywords.
        """
        try:
            prompt_id, _, _ = db_instance.add_prompt(**initial_prompt)
        except ConflictError:
            return  # Skip if initial name conflicts

        original_prompt = db_instance.get_prompt_by_id(prompt_id)

        try:
            # update_prompt_by_id handles fetching current version and incrementing it.
            uuid, msg = db_instance.update_prompt_by_id(prompt_id, update_payload)
            assert uuid is not None
        except ConflictError as e:
            # A legitimate failure if the new name is already taken by another prompt.
            assert "already exists" in str(e)
            return

        updated_prompt = db_instance.fetch_prompt_details(prompt_id)
        assert updated_prompt is not None
        assert updated_prompt['version'] == original_prompt['version'] + 1

        # Verify the payload was applied
        assert updated_prompt['name'] == update_payload['name'].strip()
        assert updated_prompt['author'] == update_payload['author']
        expected_keywords = sorted([db_instance._normalize_keyword(k) for k in update_payload["keywords"]])
        assert sorted(updated_prompt['keywords']) == expected_keywords

    @given(prompt_data=st_prompt_data())
    def test_soft_delete_makes_item_unfindable(self, db_instance: PromptsDatabase, prompt_data: dict):
        """
        Property: After soft-deleting a prompt, it should not be retrievable by
        default methods, but should exist in the DB with deleted=1.
        """
        try:
            prompt_id, _, _ = db_instance.add_prompt(**prompt_data)
        except ConflictError:
            return

        # Perform the soft delete
        success = db_instance.soft_delete_prompt(prompt_id)
        assert success is True

        # Assert it's no longer findable via public methods by default
        assert db_instance.get_prompt_by_id(prompt_id) is None
        assert db_instance.fetch_prompt_details(prompt_id) is None

        all_prompts, _, _, _ = db_instance.list_prompts()
        assert prompt_id not in [p['id'] for p in all_prompts]

        # Assert it CAN be found when explicitly requested
        deleted_record = db_instance.get_prompt_by_id(prompt_id, include_deleted=True)
        assert deleted_record is not None
        assert deleted_record['deleted'] == 1
        assert deleted_record['version'] == 2  # 1=create, 2=delete

    @given(initial_prompt=st_prompt_data(), update_name=st_required_text, version_offset=st_bad_version_offset)
    def test_update_with_stale_version_fails_via_trigger(self, db_instance: PromptsDatabase, initial_prompt: dict,
                                                         update_name: str, version_offset: int):
        """
        Property: Attempting a direct DB update with a version that does not increment
        by exactly 1 must be rejected by the database trigger.
        """
        try:
            prompt_id, _, _ = db_instance.add_prompt(**initial_prompt)
        except ConflictError:
            return

        original_prompt = db_instance.get_prompt_by_id(prompt_id)

        # Attempt a direct DB update with a bad version number.
        # This tests the 'prompts_validate_sync_update' trigger.
        with pytest.raises(DatabaseError) as excinfo:
            db_instance.execute_query(
                "UPDATE Prompts SET name = ?, version = ? WHERE id = ?",
                (update_name, original_prompt['version'] + version_offset, prompt_id),
                commit=True
            )
        assert "version must increment by exactly 1" in str(excinfo.value).lower()


class TestKeywordAndLinkingProperties:
    """Property-based tests for Keywords and their linking to Prompts."""

    @given(keyword_text=st_required_text)
    def test_keyword_normalization_and_roundtrip(self, db_instance: PromptsDatabase, keyword_text: str):
        """
        Property: Adding a keyword normalizes it (lowercase, stripped).
        Retrieving it returns the normalized version.
        """
        kw_id, kw_uuid = db_instance.add_keyword(keyword_text)
        assert kw_id is not None
        assert kw_uuid is not None

        retrieved_kw = db_instance.get_active_keyword_by_text(keyword_text)
        assert retrieved_kw is not None
        assert retrieved_kw['keyword'] == db_instance._normalize_keyword(keyword_text)

    @given(keyword=st_required_text)
    def test_add_keyword_is_idempotent_on_undelete(self, db_instance: PromptsDatabase, keyword: str):
        """
        Property: Adding a keyword that was previously soft-deleted should reactivate
        it (not create a new one), and its version should be correctly incremented.
        """
        # 1. Add for the first time
        kw_id_v1, _ = db_instance.add_keyword(keyword)
        assert db_instance.get_prompt_by_id(kw_id_v1) is not None  # Using wrong get method in original code
        kw_v1 = db_instance.get_active_keyword_by_text(keyword)
        assert kw_v1['version'] == 1

        # 2. Soft delete it
        success = db_instance.soft_delete_keyword(keyword)
        assert success is True

        # Check raw state
        raw_kw = db_instance.execute_query("SELECT * FROM PromptKeywordsTable WHERE id=?", (kw_id_v1,)).fetchone()
        assert raw_kw['deleted'] == 1
        assert raw_kw['version'] == 2

        # 3. Add it again (should trigger undelete)
        kw_id_v3, _ = db_instance.add_keyword(keyword)

        # Assert it's the same record
        assert kw_id_v3 == kw_id_v1

        kw_v3 = db_instance.get_active_keyword_by_text(keyword)
        assert kw_v3 is not None
        assert not db_instance.get_prompt_by_id(kw_v3['id'], include_deleted=True)['deleted']
        # The version should be 3 (1=create, 2=delete, 3=undelete/update)
        assert kw_v3['version'] == 3

    @given(
        prompt_data=st_prompt_data(),
        new_keywords=st.lists(st_required_text, max_size=5, unique_by=lambda s: s.strip().lower())
    )
    def test_update_keywords_for_prompt_links_and_unlinks(self, db_instance: PromptsDatabase, prompt_data: dict,
                                                          new_keywords: list):
        """
        Property: Updating keywords for a prompt correctly adds new links,
        removes old ones, and leaves unchanged ones alone.
        """
        try:
            prompt_id, _, _ = db_instance.add_prompt(**prompt_data)
        except ConflictError:
            return

        # Initial state check
        initial_expected_kws = sorted([db_instance._normalize_keyword(k) for k in prompt_data['keywords']])
        assert sorted(db_instance.fetch_keywords_for_prompt(prompt_id)) == initial_expected_kws

        # Update the keywords
        db_instance.update_keywords_for_prompt(prompt_id, new_keywords)

        # Final state check
        final_expected_kws = sorted([db_instance._normalize_keyword(k) for k in new_keywords])
        assert sorted(db_instance.fetch_keywords_for_prompt(prompt_id)) == final_expected_kws


class TestAdvancedProperties:
    """Tests for FTS, Sync Log, and other complex interactions."""

    @given(prompt_data=st_prompt_data())
    def test_soft_deleted_item_is_not_in_fts(self, db_instance: PromptsDatabase, prompt_data: dict):
        """
        Property: Once a prompt is soft-deleted, it must not appear in FTS search results.
        """
        # Ensure the name has a unique, searchable term.
        unique_term = str(uuid.uuid4())
        prompt_data['name'] = f"{prompt_data['name']} {unique_term}"

        try:
            prompt_id, _, _ = db_instance.add_prompt(**prompt_data)
        except ConflictError:
            return

        # 1. Verify it IS searchable before deletion
        results_before, total_before = db_instance.search_prompts(unique_term)
        assert total_before == 1
        assert results_before[0]['id'] == prompt_id

        # 2. Soft-delete the prompt
        db_instance.soft_delete_prompt(prompt_id)

        # 3. Verify it is NOT searchable after deletion
        results_after, total_after = db_instance.search_prompts(unique_term)
        assert total_after == 0

    @given(prompt_data=st_prompt_data())
    def test_add_creates_correct_sync_log_entries(self, db_instance: PromptsDatabase, prompt_data: dict):
        """
        Property: Adding a new prompt must create the correct 'create' and 'link'
        operations in the sync_log.
        """
        latest_change_id_before = db_instance.get_sync_log_entries(limit=1)[-1][
            'change_id'] if db_instance.get_sync_log_entries(limit=1) else 0

        try:
            prompt_id, prompt_uuid, _ = db_instance.add_prompt(**prompt_data)
        except ConflictError:
            return

        new_logs = db_instance.get_sync_log_entries(since_change_id=latest_change_id_before)

        # Verify 'create' log for the prompt itself
        prompt_create_logs = [log for log in new_logs if log['entity'] == 'Prompts' and log['operation'] == 'create']
        assert len(prompt_create_logs) == 1
        assert prompt_create_logs[0]['entity_uuid'] == prompt_uuid
        assert prompt_create_logs[0]['version'] == 1

        # Verify 'link' logs for the keywords
        normalized_keywords = {db_instance._normalize_keyword(k) for k in prompt_data['keywords']}
        link_logs = [log for log in new_logs if log['entity'] == 'PromptKeywordLinks' and log['operation'] == 'link']
        assert len(link_logs) == len(normalized_keywords)

        # The payload should contain the composite UUID of prompt_uuid + keyword_uuid
        for log in link_logs:
            assert log['payload']['prompt_uuid'] == prompt_uuid
            assert log['entity_uuid'].startswith(prompt_uuid)

    @given(prompt_data=st_prompt_data())
    def test_delete_creates_correct_sync_log_entries(self, db_instance: PromptsDatabase, prompt_data: dict):
        """
        Property: Soft-deleting a prompt must create a 'delete' log for the prompt
        and 'unlink' logs for all its keyword connections.
        """
        try:
            prompt_id, prompt_uuid, _ = db_instance.add_prompt(**prompt_data)
        except ConflictError:
            return

        num_keywords = len(prompt_data['keywords'])
        latest_change_id_before = db_instance.get_sync_log_entries(limit=1)[-1]['change_id']

        # Action: Soft delete
        db_instance.soft_delete_prompt(prompt_id)

        new_logs = db_instance.get_sync_log_entries(since_change_id=latest_change_id_before)

        # Verify 'delete' log for the prompt
        prompt_delete_logs = [log for log in new_logs if log['entity'] == 'Prompts' and log['operation'] == 'delete']
        assert len(prompt_delete_logs) == 1
        assert prompt_delete_logs[0]['entity_uuid'] == prompt_uuid
        assert prompt_delete_logs[0]['version'] == 2

        # Verify 'unlink' logs for the keywords
        unlink_logs = [log for log in new_logs if
                       log['entity'] == 'PromptKeywordLinks' and log['operation'] == 'unlink']
        assert len(unlink_logs) == num_keywords
        for log in unlink_logs:
            assert log['payload']['prompt_uuid'] == prompt_uuid


class TestDataIntegrityAndConcurrency:
    """Tests for database constraints and thread safety."""

    def test_add_prompt_with_conflicting_name_fails(self, db_instance: PromptsDatabase):
        """
        Property: Adding a prompt with a name that already exists (and overwrite=False)
        must raise a ConflictError.
        """
        prompt_data = {"name": "Unique Prompt Name", "author": "Tester"}
        db_instance.add_prompt(**prompt_data)

        # Attempt to add again with the same name
        with pytest.raises(ConflictError):
            db_instance.add_prompt(**prompt_data, overwrite=False)

    def test_update_prompt_to_conflicting_name_fails(self, db_instance: PromptsDatabase):
        """
        Property: Updating a prompt's name to a name that is already used by
        another active prompt must raise a ConflictError.
        """
        p1_id, _, _ = db_instance.add_prompt(name="Prompt One", author="A")
        db_instance.add_prompt(name="Prompt Two", author="B")  # The conflicting name

        update_payload = {"name": "Prompt Two"}
        with pytest.raises(ConflictError):
            db_instance.update_prompt_by_id(p1_id, update_payload)

    def test_each_thread_gets_a_separate_connection(self, db_instance: PromptsDatabase):
        """
        Property: The `get_connection` method must provide a unique
        connection object for each thread, via threading.local.
        """
        connection_ids = set()
        lock = threading.Lock()

        def get_and_store_conn_id():
            conn = db_instance.get_connection()
            with lock:
                connection_ids.add(id(conn))

        threads = [threading.Thread(target=get_and_store_conn_id) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # If threading.local is working, there should be 5 unique connection IDs.
        assert len(connection_ids) == 5

    def test_wal_mode_allows_concurrent_reads_during_write_transaction(self, db_instance: PromptsDatabase):
        """
        Property: In WAL mode, one thread can read from the DB while another
        thread has an open write transaction.
        """
        prompt_id, _, _ = db_instance.add_prompt(name="Concurrent Read Test", details="Original")

        write_transaction_started = threading.Event()
        read_result = []

        def writer_thread():
            # The update method opens its own transaction
            with db_instance.transaction():
                db_instance.execute_query("UPDATE Prompts SET details = 'Updated' WHERE id = ?", (prompt_id,))
                write_transaction_started.set()  # Signal that the transaction is open
                time.sleep(0.2)  # Hold the transaction open
            # Transaction commits here

        def reader_thread():
            write_transaction_started.wait()  # Wait until the writer is in its transaction
            # This read should succeed immediately and read the state BEFORE the commit.
            prompt = db_instance.get_prompt_by_id(prompt_id)
            read_result.append(prompt)

        w = threading.Thread(target=writer_thread)
        r = threading.Thread(target=reader_thread)

        w.start()
        r.start()
        w.join()
        r.join()

        # The reader thread should have completed successfully and read the *original* state.
        assert len(read_result) == 1
        assert read_result[0] is not None
        assert read_result[0]['details'] == "Original"  # It read the state before the writer committed.


# --- State Machine Tests ---

class PromptLifecycleMachine(RuleBasedStateMachine):
    """
    Models the lifecycle of a single Prompt: create, update, delete.
    Hypothesis will try to find sequences of these actions that break invariants.
    """

    def __init__(self):
        super().__init__()
        self.db = None  # Injected by the test class fixture
        # State for a single prompt's lifecycle
        self.prompt_id = None
        self.prompt_name = None
        self.is_deleted = True

    prompts = Bundle('prompts')

    @rule(target=prompts, data=st_prompt_data())
    def create_prompt(self, data):
        # We only want to test the lifecycle of one prompt per machine run for simplicity.
        if self.prompt_id is not None:
            return

        try:
            new_id, _, _ = self.db.add_prompt(**data, overwrite=False)
        except ConflictError:
            # Hypothesis might generate a duplicate name. We treat this as "no action taken".
            return

        self.prompt_id = new_id
        self.prompt_name = data['name'].strip()
        self.is_deleted = False

        retrieved = self.db.get_prompt_by_id(self.prompt_id)
        assert retrieved is not None
        assert retrieved['name'] == self.prompt_name
        assert retrieved['version'] == 1
        return self.prompt_id

    @rule(prompt_id=prompts, update_data=st_prompt_data())
    def update_prompt(self, prompt_id, update_data):
        if self.prompt_id is None or self.is_deleted:
            return

        original_version = self.db.get_prompt_by_id(prompt_id, include_deleted=True)['version']

        try:
            self.db.update_prompt_by_id(prompt_id, update_data)
            # If successful, update our internal state
            self.prompt_name = update_data['name'].strip()
        except ConflictError as e:
            # This is a valid outcome if the new name is taken.
            assert "already exists" in str(e)
            # The state of our prompt hasn't changed.
            return

        retrieved = self.db.get_prompt_by_id(self.prompt_id)
        assert retrieved is not None
        assert retrieved['version'] == original_version + 1
        assert retrieved['name'] == self.prompt_name

    @rule(prompt_id=prompts)
    def soft_delete_prompt(self, prompt_id):
        if self.prompt_id is None or self.is_deleted:
            return

        original_version = self.db.get_prompt_by_id(prompt_id, include_deleted=True)['version']

        success = self.db.soft_delete_prompt(prompt_id)
        assert success
        self.is_deleted = True

        # Verify it's gone from standard lookups
        assert self.db.get_prompt_by_id(self.prompt_id) is None
        assert self.db.get_prompt_by_name(self.prompt_name) is None

        # Verify its deleted state
        raw_record = self.db.get_prompt_by_id(self.prompt_id, include_deleted=True)
        assert raw_record['deleted'] == 1
        assert raw_record['version'] == original_version + 1


# This is the actual test class that pytest discovers and runs.
# It inherits the rules and provides the `db_instance` fixture.
@settings(max_examples=50, stateful_step_count=20)
class TestPromptLifecycleAsTest(PromptLifecycleMachine):

    @pytest.fixture(autouse=True)
    def inject_db(self, db_instance):
        """Injects the clean db_instance fixture into the state machine."""
        self.db = db_instance