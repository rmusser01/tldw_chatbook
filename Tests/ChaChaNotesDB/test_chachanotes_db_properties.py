# test_chachanotes_db_properties.py
#
# Property-based tests for the ChaChaNotes_DB library using Hypothesis.
#
# Imports
import uuid
import pytest
import json
from pathlib import Path
import sqlite3
import threading
import time
#
# Third-Party Imports
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, Bundle
#
# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    InputError,
    CharactersRAGDBError,
    ConflictError
)
#
########################################################################################################################
#
# Functions:
# --- Hypothesis Tests ---

settings.register_profile(
    "db_friendly",
    deadline=1000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture  # <--- THIS IS THE FIX
    ]
)
settings.load_profile("db_friendly")

# Strategy for generating a valid character card dictionary
# The `.map(lambda t: ...)` part is to assemble the parts into a dictionary
st_character_card_data = st.tuples(
    st.text(min_size=1, max_size=100),  # name
    st.one_of(st.none(), st.text(max_size=500)), # description
    st.one_of(st.none(), st.text(max_size=500)), # personality
    st.one_of(st.none(), st.binary(max_size=1024)), # image
    st.one_of(st.none(), st.lists(st.text(max_size=50)).map(json.dumps)), # alternate_greetings as json string
    st.one_of(st.none(), st.lists(st.text(max_size=20)).map(json.dumps))  # tags as json string
).map(lambda t: {
    "name": t[0],
    "description": t[1],
    "personality": t[2],
    "image": t[3],
    "alternate_greetings": t[4],
    "tags": t[5],
})

# Define a strategy for a non-zero integer to add to the version
st_version_offset = st.integers().filter(lambda x: x != 0)

# To prevent tests from being too slow on complex data, we can set a deadline.
# We also disable the 'too_slow' health check as DB operations can sometimes be slow.
settings.register_profile("db_friendly", deadline=1000, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("db_friendly")

# --- Fixtures (Copied from your existing test file for a self-contained example) ---

@pytest.fixture
def client_id():
    """Provides a consistent client ID for tests."""
    return "hypothesis_client"


@pytest.fixture
def db_path(tmp_path):
    """Provides a temporary path for the database file for each test."""
    return tmp_path / "prop_test_db.sqlite"


@pytest.fixture(scope="function")
def db_instance(db_path, client_id):
    """Creates a DB instance for each test, ensuring a fresh database."""
    current_db_path = Path(db_path)
    # Ensure no leftover files from a failed previous run
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(current_db_path) + suffix)
        if p.exists():
            p.unlink(missing_ok=True)

    db = CharactersRAGDB(current_db_path, client_id)
    yield db
    db.close_connection()


# --- Hypothesis Strategies ---
# These strategies define how to generate random, valid data for our database objects.

# A strategy for text fields that cannot be empty or just whitespace.
st_required_text = st.text(min_size=1, max_size=100).filter(lambda s: s.strip())

# A strategy for optional text or binary fields.
st_optional_text = st.one_of(st.none(), st.text(max_size=500))
st_optional_binary = st.one_of(st.none(), st.binary(max_size=1024))

# A strategy for fields that are stored as JSON strings in the DB.
# We generate a Python list/dict and then map it to a JSON string.
st_json_list = st.lists(st.text(max_size=50)).map(json.dumps)
st_json_dict = st.dictionaries(st.text(max_size=20), st.text(max_size=100)).map(json.dumps)


@st.composite
def st_character_card_data(draw):
    """A composite strategy to generate a dictionary of character card data."""
    # `draw` is a function that pulls a value from another strategy.
    name = draw(st_required_text)

    # To avoid conflicts with the guaranteed 'Default Assistant', we filter it out.
    if name == 'Default Assistant':
        name += "_hypothesis"  # Just ensure it's not the exact name

    return {
        "name": name,
        "description": draw(st_optional_text),
        "personality": draw(st_optional_text),
        "scenario": draw(st_optional_text),
        "system_prompt": draw(st_optional_text),
        "image": draw(st_optional_binary),
        "post_history_instructions": draw(st_optional_text),
        "first_message": draw(st_optional_text),
        "message_example": draw(st_optional_text),
        "creator_notes": draw(st_optional_text),
        "alternate_greetings": draw(st.one_of(st.none(), st_json_list)),
        "tags": draw(st.one_of(st.none(), st_json_list)),
        "creator": draw(st_optional_text),
        "character_version": draw(st_optional_text),
        "extensions": draw(st.one_of(st.none(), st_json_dict)),
    }


@st.composite
def st_note_data(draw):
    """Generates a dictionary for a note (title and content)."""
    return {
        "title": draw(st_required_text),
        "content": draw(st.text(max_size=2000))  # Content can be empty
    }


# --- Property Test Classes ---

class TestCharacterCardProperties:
    """Property-based tests for Character Cards."""

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(card_data=st_character_card_data())
    def test_character_card_roundtrip(self, db_instance: CharactersRAGDB, card_data: dict):
        """
        Property: If we add a character card, retrieving it should return the exact same data.
        This is a "round-trip" test.
        """
        try:
            card_id = db_instance.add_character_card(card_data)
        except ConflictError:
            # Hypothesis might generate the same name twice. This is not a failure of the
            # DB logic, so we just skip this test case.
            return

        retrieved_card = db_instance.get_character_card_by_id(card_id)

        assert retrieved_card is not None
        assert retrieved_card["version"] == 1
        assert not retrieved_card["deleted"]

        # Compare original data with retrieved data
        for key, value in card_data.items():
            if key in db_instance._CHARACTER_CARD_JSON_FIELDS and value is not None:
                # JSON fields are deserialized, so we compare to the parsed version
                assert retrieved_card[key] == json.loads(value)
            else:
                assert retrieved_card[key] == value

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(initial_card=st_character_card_data(), update_payload=st_character_card_data())
    def test_update_increments_version_and_changes_data(self, db_instance: CharactersRAGDB, initial_card: dict,
                                                        update_payload: dict):
        """
        Property: A successful update must increment the version number by exactly 1
        and correctly apply the new data.
        """
        try:
            card_id = db_instance.add_character_card(initial_card)
        except ConflictError:
            return  # Skip if initial card name conflicts

        original_card = db_instance.get_character_card_by_id(card_id)

        try:
            success = db_instance.update_character_card(card_id, update_payload,
                                                        expected_version=original_card['version'])
        except ConflictError as e:
            # An update can legitimately fail if the new name is already taken.
            # We accept this as a valid outcome.
            assert "already exists" in str(e)
            return

        assert success is True

        updated_card = db_instance.get_character_card_by_id(card_id)
        assert updated_card is not None
        assert updated_card['version'] == original_card['version'] + 1

        # Verify the payload was applied
        assert updated_card['name'] == update_payload['name']
        assert updated_card['description'] == update_payload['description']

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(card_data=st_character_card_data())
    def test_soft_delete_makes_item_unfindable(self, db_instance: CharactersRAGDB, card_data: dict):
        """
        Property: After soft-deleting an item, it should not be retrievable by
        the standard `get` or `list` methods, but should exist in the DB with deleted=1.
        """
        try:
            card_id = db_instance.add_character_card(card_data)
        except ConflictError:
            return

        original_card = db_instance.get_character_card_by_id(card_id)

        # Perform the soft delete
        success = db_instance.soft_delete_character_card(card_id, expected_version=original_card['version'])
        assert success is True

        # Assert it's no longer findable via public methods
        assert db_instance.get_character_card_by_id(card_id) is None

        all_cards = db_instance.list_character_cards()
        assert card_id not in [c['id'] for c in all_cards]

        # Assert its raw state in the DB is correct
        conn = db_instance.get_connection()
        raw_record = conn.execute("SELECT deleted, version FROM character_cards WHERE id = ?", (card_id,)).fetchone()
        assert raw_record is not None
        assert raw_record['deleted'] == 1
        assert raw_record['version'] == original_card['version'] + 1

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        initial_card=st_character_card_data(),
        update_payload=st_character_card_data(),
        version_offset=st_version_offset
    )
    def test_update_with_stale_version_always_fails(self, db_instance: CharactersRAGDB, initial_card: dict,
                                                    update_payload: dict, version_offset: int):
        """
        Property: Attempting to update a record with an incorrect `expected_version`
        must always raise a ConflictError.
        """
        try:
            card_id = db_instance.add_character_card(initial_card)
        except ConflictError:
            return  # Skip if initial card name conflicts

        original_card = db_instance.get_character_card_by_id(card_id)

        # Use the generated non-zero offset to create a stale version
        stale_version = original_card['version'] + version_offset

        with pytest.raises(ConflictError):
            db_instance.update_character_card(card_id, update_payload, expected_version=stale_version)

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        initial_card=st_character_card_data(),
        update_payload=st_character_card_data()
    )
    def test_update_does_not_change_immutable_fields(self, db_instance: CharactersRAGDB, initial_card: dict,
                                                     update_payload: dict):
        """
        Property: The `update` method must not change immutable fields like `id` and `created_at`,
        even if they are passed in the payload.
        """
        try:
            card_id = db_instance.add_character_card(initial_card)
        except ConflictError:
            return

        original_card = db_instance.get_character_card_by_id(card_id)

        # Add immutable fields to the update payload to try and change them
        malicious_payload = update_payload.copy()
        malicious_payload['id'] = 99999  # Try to change the ID
        malicious_payload['created_at'] = "1999-01-01T00:00:00Z"  # Try to change creation time

        try:
            db_instance.update_character_card(card_id, malicious_payload, expected_version=original_card['version'])
        except ConflictError:
            # This can happen if the update_payload name conflicts, which is a valid outcome.
            return

        updated_card = db_instance.get_character_card_by_id(card_id)

        # Assert that the immutable fields did NOT change.
        assert updated_card['id'] == original_card['id']
        assert updated_card['created_at'] == original_card['created_at']


class TestNoteAndKeywordProperties:
    """Property-based tests for Notes, Keywords, and their linking."""

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(note_data=st_note_data())
    def test_note_roundtrip(self, db_instance: CharactersRAGDB, note_data: dict):
        """
        Property: A created note, when retrieved, has the same data,
        accounting for any sanitization (like stripping whitespace).
        """
        note_id = db_instance.add_note(**note_data)
        assert note_id is not None

        retrieved = db_instance.get_note_by_id(note_id)

        assert retrieved is not None
        # Compare the retrieved title to the STRIPPED version of the original title
        assert retrieved['title'] == note_data['title'].strip()  # <-- The fix
        assert retrieved['content'] == note_data['content']
        assert retrieved['version'] == 1

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(keyword=st_required_text)
    def test_add_keyword_is_idempotent_on_undelete(self, db_instance: CharactersRAGDB, keyword: str):
        """
        Property: Adding a keyword that was previously soft-deleted should reactivate
        it, not create a new one, and its version should be correctly incremented.
        """
        # 1. Add for the first time
        try:
            kw_id_v1 = db_instance.add_keyword(keyword)
        except ConflictError:
            return

        kw_v1 = db_instance.get_keyword_by_id(kw_id_v1)
        assert kw_v1['version'] == 1

        # 2. Soft delete it
        db_instance.soft_delete_keyword(kw_id_v1, expected_version=1)
        kw_v2_raw = db_instance.get_connection().execute("SELECT * FROM keywords WHERE id = ?", (kw_id_v1,)).fetchone()
        assert kw_v2_raw['deleted'] == 1
        assert kw_v2_raw['version'] == 2

        # 3. Add it again (should trigger undelete)
        kw_id_v3 = db_instance.add_keyword(keyword)

        # Assert it's the same record
        assert kw_id_v3 == kw_id_v1

        kw_v3 = db_instance.get_keyword_by_id(kw_id_v3)
        assert kw_v3 is not None
        assert not kw_v3['deleted']
        # The version should be 3 (1=create, 2=delete, 3=undelete)
        assert kw_v3['version'] == 3

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(note_data=st_note_data(), keyword_text=st_required_text)
    def test_linking_and_unlinking_properties(self, db_instance: CharactersRAGDB, note_data: dict, keyword_text: str):
        """
        Property: Linking two items should make them appear in each other's "get_links"
        methods, and unlinking should remove them.
        """
        try:
            note_id = db_instance.add_note(**note_data)
            keyword_id = db_instance.add_keyword(keyword_text)
        except ConflictError:
            return  # Skip if hypothesis generates conflicting data

        # Initially, no links should exist
        assert db_instance.get_keywords_for_note(note_id) == []
        assert db_instance.get_notes_for_keyword(keyword_id) == []

        # --- Test Linking ---
        link_success = db_instance.link_note_to_keyword(note_id, keyword_id)
        assert link_success is True

        # Check that linking again is idempotent (returns False)
        link_again_success = db_instance.link_note_to_keyword(note_id, keyword_id)
        assert link_again_success is False

        # Verify the link exists from both sides
        keywords_for_note = db_instance.get_keywords_for_note(note_id)
        assert len(keywords_for_note) == 1
        assert keywords_for_note[0]['id'] == keyword_id

        notes_for_keyword = db_instance.get_notes_for_keyword(keyword_id)
        assert len(notes_for_keyword) == 1
        assert notes_for_keyword[0]['id'] == note_id

        # --- Test Unlinking ---
        unlink_success = db_instance.unlink_note_from_keyword(note_id, keyword_id)
        assert unlink_success is True

        # Check that unlinking again is idempotent (returns False)
        unlink_again_success = db_instance.unlink_note_from_keyword(note_id, keyword_id)
        assert unlink_again_success is False

        # Verify the link is gone
        assert db_instance.get_keywords_for_note(note_id) == []
        assert db_instance.get_notes_for_keyword(keyword_id) == []


class TestAdvancedProperties:
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(note_data=st_note_data())
    def test_soft_deleted_item_is_not_in_fts(self, db_instance: CharactersRAGDB, note_data: dict):
        """
        Property: Once an item is soft-deleted, it must not appear in FTS search results.
        """
        # Ensure the title has a unique, searchable term.
        unique_term = str(uuid.uuid4())
        note_data['title'] = f"{note_data['title']} {unique_term}"

        note_id = db_instance.add_note(**note_data)
        original_note = db_instance.get_note_by_id(note_id)

        # 1. Verify it IS searchable before deletion
        search_results_before = db_instance.search_notes(unique_term)
        assert len(search_results_before) == 1
        assert search_results_before[0]['id'] == note_id

        # 2. Soft-delete the note
        db_instance.soft_delete_note(note_id, expected_version=original_note['version'])

        # 3. Verify it is NOT searchable after deletion
        search_results_after = db_instance.search_notes(unique_term)
        assert len(search_results_after) == 0

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(note_data=st_note_data())
    def test_add_creates_sync_log_entry(self, db_instance: CharactersRAGDB, note_data: dict):
        """
        Property: Adding a new item must create exactly one 'create' operation
        in the sync_log for that item.
        """
        latest_change_id_before = db_instance.get_latest_sync_log_change_id()

        # Add the note (this action should be logged by a trigger)
        note_id = db_instance.add_note(**note_data)

        new_log_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id_before)

        # There should be exactly one new entry
        assert len(new_log_entries) == 1

        log_entry = new_log_entries[0]
        assert log_entry['entity'] == 'notes'
        assert log_entry['entity_id'] == note_id
        assert log_entry['operation'] == 'create'
        assert log_entry['client_id'] == db_instance.client_id
        assert log_entry['version'] == 1
        assert log_entry['payload']['title'] == note_data['title'].strip()  # The log stores the stripped version
        assert log_entry['payload']['content'] == note_data['content']

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(note_data=st_note_data(), update_content=st.text(max_size=500))
    def test_update_creates_sync_log_entry(self, db_instance: CharactersRAGDB, note_data: dict, update_content: str):
        """
        Property: Updating an item must create exactly one 'update' operation
        in the sync_log for that item.
        """
        note_id = db_instance.add_note(**note_data)
        latest_change_id_before = db_instance.get_latest_sync_log_change_id()

        # Update the note
        update_payload = {'content': update_content}
        db_instance.update_note(note_id, update_payload, expected_version=1)

        new_log_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id_before)

        assert len(new_log_entries) == 1
        log_entry = new_log_entries[0]
        assert log_entry['entity'] == 'notes'
        assert log_entry['entity_id'] == note_id
        assert log_entry['operation'] == 'update'
        assert log_entry['version'] == 2
        assert log_entry['payload']['content'] == update_content

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(note_data=st_note_data())
    def test_delete_creates_sync_log_entry(self, db_instance: CharactersRAGDB, note_data: dict):
        """
        Property: Soft-deleting an item must create exactly one 'delete' operation
        in the sync_log for that item.
        """
        note_id = db_instance.add_note(**note_data)
        latest_change_id_before = db_instance.get_latest_sync_log_change_id()

        # Delete the note
        db_instance.soft_delete_note(note_id, expected_version=1)

        new_log_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id_before)

        assert len(new_log_entries) == 1
        log_entry = new_log_entries[0]
        assert log_entry['entity'] == 'notes'
        assert log_entry['entity_id'] == note_id
        assert log_entry['operation'] == 'delete'
        assert log_entry['version'] == 2
        assert log_entry['payload']['deleted'] == 1

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(note_data=st_note_data(), keyword_text=st_required_text)
    def test_link_action_creates_correct_sync_log(self, db_instance: CharactersRAGDB, note_data: dict,
                                                  keyword_text: str):
        """
        Property: The `link_note_to_keyword` action must create a sync_log entry
        with the correct entity, IDs, and operation in its payload.
        """
        try:
            note_id = db_instance.add_note(**note_data)
            kw_id = db_instance.add_keyword(keyword_text)
        except ConflictError:
            return

        latest_change_id_before = db_instance.get_latest_sync_log_change_id()

        # Action
        db_instance.link_note_to_keyword(note_id, kw_id)

        # There should be exactly one new log entry from the linking.
        # We ignore the create logs for the note and keyword.
        link_log_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id_before,
                                                            entity_type='note_keywords')
        assert len(link_log_entries) == 1

        log_entry = link_log_entries[0]
        assert log_entry['entity'] == 'note_keywords'
        assert log_entry['operation'] == 'create'
        assert log_entry['entity_id'] == f"{note_id}_{kw_id}"
        assert log_entry['payload']['note_id'] == note_id
        assert log_entry['payload']['keyword_id'] == kw_id


    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(note_data=st_note_data(), keyword_text=st_required_text)
    def test_unlink_action_creates_correct_sync_log(self, db_instance: CharactersRAGDB, note_data: dict,
                                                    keyword_text: str):
        """
        Property: The `unlink_note_to_keyword` action must create a sync_log entry
        with the correct entity, IDs, and operation.
        """
        try:
            note_id = db_instance.add_note(**note_data)
            kw_id = db_instance.add_keyword(keyword_text)
        except ConflictError:
            return

        db_instance.link_note_to_keyword(note_id, kw_id)
        latest_change_id_before = db_instance.get_latest_sync_log_change_id()

        # Action
        db_instance.unlink_note_from_keyword(note_id, kw_id)

        unlink_log_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id_before,
                                                              entity_type='note_keywords')
        assert len(unlink_log_entries) == 1

        log_entry = unlink_log_entries[0]
        assert log_entry['entity'] == 'note_keywords'
        assert log_entry['operation'] == 'delete'
        assert log_entry['entity_id'] == f"{note_id}_{kw_id}"
        assert log_entry['payload']['note_id'] == note_id
        assert log_entry['payload']['keyword_id'] == kw_id


@pytest.fixture
def populated_conversation(db_instance: CharactersRAGDB):
    """A fixture to create a character, conversation, and message for cascade tests."""
    card_id = db_instance.add_character_card({'name': 'Cascade Test Character'})
    card = db_instance.get_character_card_by_id(card_id)

    conv_id = db_instance.add_conversation({'character_id': card['id'], 'title': 'Cascade Conv'})
    conv = db_instance.get_conversation_by_id(conv_id)

    msg_id = db_instance.add_message({'conversation_id': conv['id'], 'sender': 'user', 'content': 'Cascade Msg'})
    msg = db_instance.get_message_by_id(msg_id)

    return {"card": card, "conv": conv, "msg": msg}


class TestCascadeAndLinkingProperties:
    def test_soft_deleting_conversation_makes_messages_unfindable(self, db_instance: CharactersRAGDB,
                                                                  populated_conversation):
        """
        Property: After a conversation is soft-deleted, its messages should not be
        returned by get_messages_for_conversation.
        """
        conv = populated_conversation['conv']
        msg = populated_conversation['msg']

        # 1. Verify message and conversation exist before
        assert db_instance.get_message_by_id(msg['id']) is not None
        assert len(db_instance.get_messages_for_conversation(conv['id'])) == 1

        # 2. Soft-delete the parent conversation
        db_instance.soft_delete_conversation(conv['id'], expected_version=conv['version'])

        # 3. Verify the messages are now un-findable via the main query method
        messages = db_instance.get_messages_for_conversation(conv['id'])
        assert messages == []

        # 4. As per our current design, the message record itself still exists (orphaned).
        # This is an important check of the current behavior.
        assert db_instance.get_message_by_id(msg['id']) is not None

    def test_hard_deleting_conversation_cascades_to_messages(self, db_instance: CharactersRAGDB,
                                                             populated_conversation):
        """
        Property: A hard DELETE on a conversation should cascade and delete its
        messages, enforcing the FOREIGN KEY ... ON DELETE CASCADE constraint.
        """
        conv = populated_conversation['conv']
        msg = populated_conversation['msg']

        # 1. Verify message exists before
        assert db_instance.get_message_by_id(msg['id']) is not None

        # 2. Perform a hard delete in a clean transaction
        with db_instance.transaction() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv['id'],))

        # 3. Now the message should be truly gone.
        assert db_instance.get_message_by_id(msg['id']) is None

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(keyword_text=st_required_text)
    def test_deleting_keyword_cascades_to_link_tables(self, db_instance: CharactersRAGDB, populated_conversation,
                                                      keyword_text: str):
        """
        Property: Deleting a keyword should remove all links to it from linking tables
        due to ON DELETE CASCADE.
        """
        conv = populated_conversation['conv']
        try:
            kw_id = db_instance.add_keyword(keyword_text)
        except ConflictError:
            return

        # Link the conversation and keyword
        db_instance.link_conversation_to_keyword(conv['id'], kw_id)

        # Verify link exists
        keywords = db_instance.get_keywords_for_conversation(conv['id'])
        assert len(keywords) == 1
        assert keywords[0]['id'] == kw_id

        # Soft-delete the keyword
        keyword = db_instance.get_keyword_by_id(kw_id)
        db_instance.soft_delete_keyword(kw_id, keyword['version'])

        # The link should now be gone when we retrieve it, because the JOIN will fail
        # on `k.deleted = 0`. This tests the query logic.
        keywords_after_delete = db_instance.get_keywords_for_conversation(conv['id'])
        assert keywords_after_delete == []


    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(note_data=st_note_data(), keyword_text=st_required_text)
    def test_linking_is_idempotent(self, db_instance: CharactersRAGDB, note_data: dict, keyword_text: str):
        """
        Property: Calling a link function multiple times has the same effect as calling it once.
        The first call should return True (1 row affected), subsequent calls should return False (0 rows affected).
        """
        try:
            note_id = db_instance.add_note(**note_data)
            kw_id = db_instance.add_keyword(keyword_text)
        except ConflictError:
            return

        # First call should succeed and return True
        assert db_instance.link_note_to_keyword(note_id, kw_id) is True

        # Second call should do nothing and return False
        assert db_instance.link_note_to_keyword(note_id, kw_id) is False

        # Third call should also do nothing and return False
        assert db_instance.link_note_to_keyword(note_id, kw_id) is False

        # Verify there is still only one link
        keywords = db_instance.get_keywords_for_note(note_id)
        assert len(keywords) == 1


    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(note_data=st_note_data(), keyword_text=st_required_text)
    def test_unlinking_is_idempotent(self, db_instance: CharactersRAGDB, note_data: dict, keyword_text: str):
        """
        Property: Calling an unlink function on a non-existent link does nothing.
        Calling it on an existing link works once, then does nothing on subsequent calls.
        """
        try:
            note_id = db_instance.add_note(**note_data)
            kw_id = db_instance.add_keyword(keyword_text)
        except ConflictError:
            return

        # 1. Unlinking a non-existent link should return False
        assert db_instance.unlink_note_from_keyword(note_id, kw_id) is False

        # 2. Create the link
        db_instance.link_note_to_keyword(note_id, kw_id)
        assert len(db_instance.get_keywords_for_note(note_id)) == 1

        # 3. First unlink should succeed and return True
        assert db_instance.unlink_note_from_keyword(note_id, kw_id) is True

        # 4. Second unlink should fail and return False
        assert db_instance.unlink_note_from_keyword(note_id, kw_id) is False

        # Verify the link is gone
        assert len(db_instance.get_keywords_for_note(note_id)) == 0


# ==========================================================
# ==  STATE MACHINE SECTION
# ==========================================================

class NoteLifecycleMachine(RuleBasedStateMachine):
    """
    This class defines the rules and state for our test.
    It is not run directly by pytest.
    """

    def __init__(self):
        super().__init__()
        self.db = None  # This will be injected by the test class
        self.note_id = None
        self.expected_version = 0
        self.is_deleted = True

    notes = Bundle('notes')

    @rule(target=notes, note_data=st_note_data())
    def create_note(self, note_data):
        # We only want to test the lifecycle of one note per machine run.
        if self.note_id is not None:
            return

        self.note_id = self.db.add_note(**note_data)
        self.is_deleted = False
        self.expected_version = 1

        retrieved = self.db.get_note_by_id(self.note_id)
        assert retrieved is not None
        return self.note_id

    @rule(note_id=notes, update_data=st_note_data())
    def update_note(self, note_id, update_data):
        if self.note_id is None or self.is_deleted:
            return

        success = self.db.update_note(note_id, update_data, self.expected_version)
        assert success
        self.expected_version += 1

        retrieved = self.db.get_note_by_id(self.note_id)
        assert retrieved is not None
        assert retrieved['version'] == self.expected_version

    @rule(note_id=notes)
    def soft_delete_note(self, note_id):
        if self.note_id is None or self.is_deleted:
            return

        success = self.db.soft_delete_note(note_id, self.expected_version)
        assert success
        self.expected_version += 1
        self.is_deleted = True

        assert self.db.get_note_by_id(self.note_id) is None


# This class IS the test. pytest will discover it.
# It inherits our rules and provides the `db_instance` fixture.
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], max_examples=50)
class NoteLifecycleStateMachine(NoteLifecycleMachine):

    @pytest.fixture(autouse=True)
    def inject_db(self, db_instance):
        """Injects the clean db_instance fixture into the state machine for each test run."""
        self.db = db_instance


# ==========================================================
# ==  Character Card State Machine
# ==========================================================

class CharacterCardLifecycleMachine(RuleBasedStateMachine):
    """Models the lifecycle of a CharacterCard."""

    def __init__(self):
        super().__init__()
        self.db = None
        self.card_id = None
        self.card_name = None
        self.expected_version = 0
        self.is_deleted = True

    cards = Bundle('cards')

    @rule(target=cards, card_data=st_character_card_data())
    def create_card(self, card_data):
        # Only create one card per machine run for simplicity.
        if self.card_id is not None:
            return

        try:
            new_id = self.db.add_character_card(card_data)
        except ConflictError:
            # It's possible for hypothesis to generate a duplicate name
            # in its sequence. We treat this as "no action taken".
            return

        self.card_id = new_id
        self.card_name = card_data['name']
        self.expected_version = 1
        self.is_deleted = False

        retrieved = self.db.get_character_card_by_id(self.card_id)
        assert retrieved is not None
        assert retrieved['name'] == self.card_name
        return self.card_id

    @rule(card_id=cards, update_data=st_character_card_data())
    def update_card(self, card_id, update_data):
        if self.card_id is None or self.is_deleted:
            return

        try:
            success = self.db.update_character_card(card_id, update_data, self.expected_version)
            assert success
            self.expected_version += 1
            self.card_name = update_data['name']  # Name can change
        except ConflictError as e:
            # Update can fail legitimately if the new name is already taken.
            assert "already exists" in str(e)
            # The state of our card hasn't changed, so we just return.
            return

        retrieved = self.db.get_character_card_by_id(self.card_id)
        assert retrieved is not None
        assert retrieved['version'] == self.expected_version
        assert retrieved['name'] == self.card_name

    @rule(card_id=cards)
    def soft_delete_card(self, card_id):
        if self.card_id is None or self.is_deleted:
            return

        success = self.db.soft_delete_character_card(card_id, self.expected_version)
        assert success
        self.expected_version += 1
        self.is_deleted = True

        assert self.db.get_character_card_by_id(self.card_id) is None
        assert self.db.get_character_card_by_name(self.card_name) is None


# The pytest test class that runs the machine
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], max_examples=50)
class CharacterCardLifecycleStateMachine(CharacterCardLifecycleMachine):

    @pytest.fixture(autouse=True)
    def inject_db(self, db_instance):
        self.db = db_instance


st_message_content = st.text(max_size=1000)


# ==========================================================
# ==  Conversation/Message State Machine
# ==========================================================

class ConversationMachine(RuleBasedStateMachine):
    """
    Models creating a conversation and adding messages to it.
    This machine tests the integrity of a single conversation over time.
    """

    def __init__(self):
        super().__init__()
        self.db = None
        self.card_id = None
        self.conv_id = None
        self.message_count = 0

    @precondition(lambda self: self.card_id is None)
    @rule()
    def create_character(self):
        """Setup step: create a character to host the conversation."""
        self.card_id = self.db.add_character_card({'name': 'Chat Host Character'})
        assert self.card_id is not None

    @precondition(lambda self: self.card_id is not None and self.conv_id is None)
    @rule()
    def create_conversation(self):
        """Create the main conversation for this test run."""
        self.conv_id = self.db.add_conversation({'character_id': self.card_id, 'title': 'Test Chat'})
        assert self.conv_id is not None

    @precondition(lambda self: self.conv_id is not None)
    @rule(content=st_message_content, sender=st.sampled_from(['user', 'ai']))
    def add_message(self, content, sender):
        """Add a new message to the existing conversation."""
        if not content and not sender:  # Ensure message has some substance
            return

        msg_id = self.db.add_message({
            'conversation_id': self.conv_id,
            'sender': sender,
            'content': content
        })
        assert msg_id is not None
        self.message_count += 1

    def teardown(self):
        """
        This method is called at the end of a state machine run.
        We use it to check the final state of the system.
        """
        if self.conv_id is not None:
            messages = self.db.get_messages_for_conversation(self.conv_id)
            assert len(messages) == self.message_count


# The pytest test class that runs the machine
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow], max_examples=20,
          stateful_step_count=50)
class ConversationInteractionsStateMachine(ConversationMachine):

    @pytest.fixture(autouse=True)
    def inject_db(self, db_instance):
        self.db = db_instance


class TestDataIntegrity:
    def test_add_conversation_with_nonexistent_character_fails(self, db_instance: CharactersRAGDB):
        """
        Property: Cannot create a conversation for a character_id that does not exist.
        This tests the FOREIGN KEY constraint.
        """
        non_existent_char_id = 99999
        conv_data = {"character_id": non_existent_char_id, "title": "Orphan Conversation"}

        # The database will raise an IntegrityError. Your wrapper should catch this
        # and raise a custom error.
        with pytest.raises(CharactersRAGDBError, match="FOREIGN KEY constraint failed"):
            db_instance.add_conversation(conv_data)

    def test_add_message_to_nonexistent_conversation_fails(self, db_instance: CharactersRAGDB):
        """
        Property: Cannot add a message to a conversation_id that does not exist.
        """
        non_existent_conv_id = "a-fake-uuid-string"
        msg_data = {
            "conversation_id": non_existent_conv_id,
            "sender": "user",
            "content": "Message to nowhere"
        }

        # Your `add_message` has a pre-flight check for this, which should raise InputError.
        # This tests your application-level check.
        with pytest.raises(InputError, match="Conversation ID .* not found or deleted"):
            db_instance.add_message(msg_data)

    def test_rating_outside_range_fails(self, db_instance: CharactersRAGDB):
        """
        Property: Conversation rating must be between 1 and 5.
        This tests the CHECK constraint via the public API.
        """
        card_id = db_instance.add_character_card({'name': 'Rating Test Character'})

        # Test the application-level check in `update_conversation`
        conv_id = db_instance.add_conversation({'character_id': card_id, "title": "Rating Conv"})
        with pytest.raises(InputError, match="Rating must be between 1 and 5"):
            db_instance.update_conversation(conv_id, {"rating": 0}, expected_version=1)
        with pytest.raises(InputError, match="Rating must be between 1 and 5"):
            db_instance.update_conversation(conv_id, {"rating": 6}, expected_version=1)

        # Test the DB-level CHECK constraint directly by calling the wrapped execute_query
        with pytest.raises(CharactersRAGDBError, match="Database constraint violation"):
            # We start a transaction to ensure atomicity, but call the DB method
            # that handles exception wrapping.
            with db_instance.transaction():
                db_instance.execute_query("UPDATE conversations SET rating = 10 WHERE id = ?", (conv_id,))


class TestConcurrency:
    def test_each_thread_gets_a_separate_connection(self, db_instance: CharactersRAGDB):
        """
        Property: The `_get_thread_connection` method must provide a unique
        connection object for each thread.
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

    def test_wal_mode_allows_concurrent_reads_during_write_transaction(self, db_instance: CharactersRAGDB):
        """
        Property: In WAL mode, one thread can read from the DB while another
        thread has an open write transaction.
        """
        card_id = db_instance.add_character_card({'name': 'Concurrent Read Test'})

        # A threading.Event to signal when the write transaction has started
        write_transaction_started = threading.Event()
        read_result = []

        def writer_thread():
            with db_instance.transaction():
                db_instance.update_character_card(card_id, {'description': 'long update'}, 1)
                write_transaction_started.set()  # Signal that the transaction is open
                time.sleep(0.2)  # Hold the transaction open
            # Transaction commits here

        def reader_thread():
            write_transaction_started.wait()  # Wait until the writer is in its transaction
            # This read should succeed immediately and not be blocked by the writer.
            card = db_instance.get_character_card_by_id(card_id)
            read_result.append(card)

        w = threading.Thread(target=writer_thread)
        r = threading.Thread(target=reader_thread)

        w.start()
        r.start()
        w.join()
        r.join()

        # The reader thread should have completed successfully and read the *original* state.
        assert len(read_result) == 1
        assert read_result[0] is not None
        assert read_result[0]['description'] is None  # It read the state before the writer committed.


class TestComplexQueries:
    def test_get_keywords_for_conversation_filters_deleted_keywords(self, db_instance: CharactersRAGDB):
        """
        Property: When fetching keywords for a conversation, soft-deleted
        keywords must be excluded from the results.
        """
        card_id = db_instance.add_character_card({'name': 'Filter Test'})
        conv_id = db_instance.add_conversation({'character_id': card_id})

        kw1_id = db_instance.add_keyword("Active Keyword")
        kw2_id = db_instance.add_keyword("Keyword to be Deleted")
        kw2 = db_instance.get_keyword_by_id(kw2_id)

        db_instance.link_conversation_to_keyword(conv_id, kw1_id)
        db_instance.link_conversation_to_keyword(conv_id, kw2_id)

        # Verify both are present initially
        assert len(db_instance.get_keywords_for_conversation(conv_id)) == 2

        # Soft-delete one of the keywords
        db_instance.soft_delete_keyword(kw2_id, kw2['version'])

        # Fetch again and verify only the active one remains
        remaining_keywords = db_instance.get_keywords_for_conversation(conv_id)
        assert len(remaining_keywords) == 1
        assert remaining_keywords[0]['id'] == kw1_id
        assert remaining_keywords[0]['keyword'] == "Active Keyword"


class TestDBOperations:
    def test_backup_and_restore_correctness(self, db_instance: CharactersRAGDB, tmp_path: Path):
        """
        Property: A database created from a backup file must contain the exact
        same data as the original database at the time of backup.
        """
        # 1. Populate the original database with known data
        card_data = {'name': 'Backup Test Card', 'description': 'Data to be saved'}
        card_id = db_instance.add_character_card(card_data)
        original_card = db_instance.get_character_card_by_id(card_id)

        # 2. Perform the backup
        backup_path = tmp_path / "test_backup.db"
        assert db_instance.backup_database(str(backup_path)) is True
        assert backup_path.exists()

        # 3. Close the original DB connection
        db_instance.close_connection()

        # 4. Open the backup database as a new instance
        backup_db = CharactersRAGDB(backup_path, "restore_client")

        # 5. Verify the data
        restored_card = backup_db.get_character_card_by_id(card_id)
        assert restored_card is not None

        # Compare the entire dictionaries
        # sqlite.Row objects need to be converted to dicts for direct comparison
        assert dict(restored_card) == dict(original_card)

        # Also check list methods
        all_cards = backup_db.list_character_cards()
        # Remember the default card!
        assert len(all_cards) == 2

        backup_db.close_connection()


#
# End of test_chachanotes_db_properties.py
########################################################################################################################
