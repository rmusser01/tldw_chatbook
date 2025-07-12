# test_media_db_v2_properties.py
#
# Property-based tests for the Media_DB_v2 library using Hypothesis.
# These tests verify the logical correctness and invariants of the database
# operations across a wide range of generated data.
#
# Imports
from datetime import datetime, timezone, timedelta
from typing import Iterator, Callable, Any, Generator
import pytest
import uuid
from pathlib import Path
#
# Third-Party Imports
from hypothesis import given, strategies as st, settings, HealthCheck, assume
#
# Local Imports
# Adjust the import path based on your project structure
from tldw_chatbook.DB.Client_Media_DB_v2 import (
    MediaDatabase,
    InputError,
    DatabaseError,
    ConflictError, fetch_keywords_for_media, empty_trash
)
#
#######################################################################################################################
#
# --- Hypothesis Settings ---

# A custom profile for database-intensive tests.
# It increases the deadline and suppresses health checks that are common
# but expected in I/O-heavy testing scenarios.
settings.register_profile(
    "db_test_suite",
    deadline=2000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.data_too_large,
    ]
)
settings.load_profile("db_test_suite")


# --- Pytest Fixtures ---


@pytest.fixture
def db_factory(tmp_path: Path) -> Generator[Callable[[], MediaDatabase], Any, None]:
    """
    A factory that creates fresh, isolated MediaDatabase instances on demand.
    Manages cleanup of all created instances.
    """
    created_dbs = []

    def _create_db_instance() -> MediaDatabase:
        db_file = tmp_path / f"prop_test_{uuid.uuid4().hex}.db"
        client_id = f"client_{uuid.uuid4().hex[:8]}"
        db = MediaDatabase(db_path=db_file, client_id=client_id)
        created_dbs.append(db)
        return db

    yield _create_db_instance

    # Teardown: close all connections that were created by the factory
    for db in created_dbs:
        db.close_connection()

@pytest.fixture
def db_instance(db_factory: Callable[[], MediaDatabase]) -> MediaDatabase:
    """
    Provides a single, fresh MediaDatabase instance for a test function.
    This fixture uses the `db_factory` to create and manage the instance.
    """
    return db_factory()

# --- Hypothesis Strategies ---

# Strategy for generating text that is guaranteed to have non-whitespace content and no null bytes.
st_required_text = st.text(min_size=1, max_size=50).map(lambda s: s.strip().replace('\x00', '')).filter(lambda s: len(s) > 0)

# Strategy for a single, clean keyword.
st_keyword_text = st.text(
    alphabet=st.characters(whitelist_categories=["L", "N", "S", "P"]),
    min_size=2,
    max_size=20
).map(lambda s: s.strip()).filter(lambda s: len(s) > 0)

# Strategy for generating a list of unique, case-insensitive keywords.
st_keywords_list = st.lists(
    st_keyword_text,
    min_size=1,
    max_size=5,
    unique_by=lambda s: s.lower()
).filter(lambda l: len(l) > 0)  # Ensure list is not empty after filtering


# A composite strategy to generate a valid dictionary of media data for creation.
@st.composite
def st_media_data(draw):
    """Generates a dictionary of plausible data for a new media item."""
    # Generate content that excludes null bytes (\x00) which break SQLite FTS5
    content = draw(st.text(min_size=10, max_size=500))
    # Remove any null bytes that might have been generated
    content = content.replace('\x00', '')
    
    return {
        "title": draw(st_required_text),
        "content": content,
        "media_type": draw(st.sampled_from(['article', 'video', 'obsidian_note', 'pdf'])),
        "author": draw(st.one_of(st.none(), st.text(min_size=3, max_size=30))),
        "keywords": draw(st_keywords_list)
    }


# --- Property Test Classes ---

class TestMediaItemProperties:
    """Property-based tests for the core Media item lifecycle."""

    @given(media_data=st_media_data())
    def test_media_item_roundtrip(self, db_instance: MediaDatabase, media_data: dict):
        """
        Property: A media item, once added, should be retrievable with the same data.
        """
        media_data["content"] += f" {uuid.uuid4().hex}"

        media_id, media_uuid, msg = db_instance.add_media_with_keywords(**media_data)

        assert "added" in msg
        assert media_id is not None
        assert media_uuid is not None

        retrieved = db_instance.get_media_by_id(media_id)
        assert retrieved is not None

        assert retrieved['title'] == media_data['title']
        assert retrieved['content'] == media_data['content']
        assert retrieved['type'] == media_data['media_type']
        assert retrieved['author'] == media_data['author']
        assert retrieved['version'] == 1
        assert not retrieved['deleted']

        linked_keywords = {kw.lower().strip() for kw in fetch_keywords_for_media(db_instance, media_id)}
        expected_keywords = {kw.lower().strip() for kw in media_data['keywords']}
        assert linked_keywords == expected_keywords

        # FIX: The get_all_document_versions function defaults to NOT including content.
        # We must explicitly request it for the assertion to work.
        doc_versions = db_instance.get_all_document_versions(media_id, include_content=True)
        assert len(doc_versions) == 1
        assert doc_versions[0]['version_number'] == 1
        assert doc_versions[0]['content'] == media_data['content']

    # ... other tests in this class are correct ...
    @given(initial_media=st_media_data(), update_media=st_media_data())
    def test_update_increments_version_and_changes_data(self, db_instance: MediaDatabase, initial_media: dict,
                                                        update_media: dict):
        """
        Property: Updating an existing media item must increment its version by exactly 1
        and correctly apply the new data.
        """
        # Generate unique content strings to avoid unintentional hash collisions
        test_id = uuid.uuid4().hex
        initial_content = f"Initial content for update test {test_id}"
        update_content = f"Updated content for update test {test_id}"
        
        # Make sure the titles are different to verify the update
        assume(initial_media['title'] != update_media['title'])
        
        # Set our controlled content
        initial_media['content'] = initial_content
        update_media['content'] = update_content

        # Add the initial media item
        media_id, media_uuid, msg1 = db_instance.add_media_with_keywords(**initial_media)
        assert media_id is not None, "Failed to add initial media"
        assert "added" in msg1.lower(), f"Expected 'added' in message, got: {msg1}"

        # Fetch the original record to get its URL and version
        original = db_instance.get_media_by_id(media_id)
        assert original is not None, "Failed to retrieve the added media item"
        assert original['version'] == 1, f"Expected initial version to be 1, got {original['version']}"
        
        # Use the URL to identify the item for update
        url_to_update = original['url']
        
        # Update the media item
        media_id_up, media_uuid_up, msg2 = db_instance.add_media_with_keywords(
            url=url_to_update,  # This is crucial for identifying which item to update
            overwrite=True,     # This flag enables updating instead of skipping
            **update_media      # New data to apply
        )
        
        # Verify that we got back the same IDs
        assert media_id_up == media_id, f"Update returned different ID: {media_id_up} vs original {media_id}"
        assert media_uuid_up == media_uuid, f"Update returned different UUID"
        
        # Verify the message indicates an update
        assert "updated" in msg2.lower(), f"Expected 'updated' in message, got: {msg2}"
        
        # Fetch the updated record
        updated = db_instance.get_media_by_id(media_id)
        assert updated is not None, "Failed to retrieve the updated media item"
        
        # Verify core update properties
        assert updated['version'] == 2, f"Expected version to be incremented to 2, got {updated['version']}"
        assert updated['title'] == update_media['title'], "Title was not updated correctly"
        assert updated['content'] == update_content, "Content was not updated correctly"
        
        # Check that a second document version was created
        doc_versions = db_instance.get_all_document_versions(media_id, include_content=True)
        assert len(doc_versions) == 2, f"Expected 2 document versions, got {len(doc_versions)}"
        
        # Verify the content of both versions
        latest_version = max(doc_versions, key=lambda v: v['version_number'])
        assert latest_version['content'] == update_content, "Latest document version has incorrect content"

    @given(media_data=st_media_data())
    def test_soft_delete_makes_item_unfindable_by_default(self, db_instance: MediaDatabase, media_data: dict):
        unique_word = f"hypothesis_{uuid.uuid4().hex}"
        media_data["content"] = f"{media_data['content']} {unique_word}"
        media_id, _, _ = db_instance.add_media_with_keywords(**media_data)
        original = db_instance.get_media_by_id(media_id)
        assert original is not None
        db_instance.soft_delete_media(media_id)
        assert db_instance.get_media_by_id(media_id) is None
        results, total = db_instance.search_media_db(search_query=unique_word)
        assert total == 0
        raw_record = db_instance.get_media_by_id(media_id, include_deleted=True)
        assert raw_record is not None
        assert raw_record['deleted'] == 1
        assert raw_record['version'] == original['version'] + 1


class TestSearchProperties:
    @given(media_data=st_media_data())
    def test_search_finds_item_by_its_properties(self, db_instance: MediaDatabase, media_data: dict):
        unique_word = f"hypothesis_{uuid.uuid4().hex}"
        media_data["content"] = f"{media_data['content']} {unique_word}"
        media_id, _, _ = db_instance.add_media_with_keywords(**media_data)
        results, total = db_instance.search_media_db(search_query=unique_word, search_fields=['content'])
        assert total == 1
        assert results[0]['id'] == media_id
        keyword_to_find = media_data["keywords"][0]
        results, total = db_instance.search_media_db(search_query=None, must_have_keywords=[keyword_to_find],
                                                     media_ids_filter=[media_id])
        assert total == 1
        assert results[0]['id'] == media_id
        results, total = db_instance.search_media_db(search_query=None, media_types=[media_data['media_type']],
                                                     media_ids_filter=[media_id])
        assert total == 1
        assert results[0]['id'] == media_id

    @given(item1=st_media_data(), item2=st_media_data())
    def test_search_isolates_results_correctly(self, db_instance: MediaDatabase, item1: dict, item2: dict):
        item1_kws = set(kw.lower() for kw in item1['keywords'])
        item2_kws = set(kw.lower() for kw in item2['keywords'])
        assume(item1_kws.isdisjoint(item2_kws))
        item1["content"] += f" item1_{uuid.uuid4().hex}"
        item2["content"] += f" item2_{uuid.uuid4().hex}"
        id1, _, _ = db_instance.add_media_with_keywords(**item1)
        id2, _, _ = db_instance.add_media_with_keywords(**item2)
        keyword_to_find = item1['keywords'][0]
        results, total = db_instance.search_media_db(search_query=None, must_have_keywords=[keyword_to_find],
                                                     media_ids_filter=[id1, id2])
        assert total == 1
        assert results[0]['id'] == id1

    @given(media_data=st_media_data())
    def test_soft_deleted_item_is_not_in_fts_search(self, db_instance: MediaDatabase, media_data: dict):
        unique_term = f"fts_{uuid.uuid4().hex}"
        media_data['title'] = f"{media_data['title']} {unique_term}"
        media_data['content'] += f" {uuid.uuid4().hex}"
        media_id, _, _ = db_instance.add_media_with_keywords(**media_data)
        results, total = db_instance.search_media_db(search_query=unique_term)
        assert total == 1
        was_deleted = db_instance.soft_delete_media(media_id)
        assert was_deleted is True
        results, total = db_instance.search_media_db(search_query=unique_term)
        assert total == 0


class TestIdempotencyAndConstraints:
    """Tests for idempotency of operations and enforcement of DB constraints."""

    @settings(deadline=None)
    @given(media_data=st_media_data())
    def test_mark_as_trash_is_idempotent(self, db_instance: MediaDatabase, media_data: dict):
        """
        Property: Marking an item as trash multiple times has the same effect as
        marking it once. The version should only increment on the first call.
        """
        media_data["content"] += f" {uuid.uuid4().hex}"
        media_id, _, _ = db_instance.add_media_with_keywords(**media_data)

        assert db_instance.mark_as_trash(media_id) is True
        item_v2 = db_instance.get_media_by_id(media_id, include_trash=True)
        assert item_v2['version'] == 2

        assert db_instance.mark_as_trash(media_id) is False
        item_still_v2 = db_instance.get_media_by_id(media_id, include_trash=True)
        assert item_still_v2['version'] == 2

    @given(
        media1=st_media_data(),
        media2=st_media_data(),
    )
    def test_add_media_with_conflicting_hash_is_handled(self,
                                                        db_instance: MediaDatabase,
                                                        media1: dict,
                                                        media2: dict):
        """
        Property: The database should handle content hash conflicts gracefully.

        When two items have the same content (and thus the same hash):
        1. Adding the second with overwrite=False should fail gracefully
        2. Adding the second with overwrite=True should update the existing item
        """
        # Generate a unique, deterministic content for this test run
        unique_content = f"Identical content for hash collision test {uuid.uuid4().hex}"

        # Ensure titles are different to test a metadata-only update
        assume(media1['title'] != media2['title'])

        # Set identical content to force hash collision
        media1['content'] = unique_content
        media2['content'] = unique_content

        # Use the SAME URL for both to ensure the update works
        shared_url = f"http://example.com/test-{uuid.uuid4()}"
        media1['url'] = shared_url
        media2['url'] = shared_url

        # Step 1: Add the first item
        id1, uuid1, msg1 = db_instance.add_media_with_keywords(**media1)
        assert id1 is not None, "Failed to add first item"
        assert "added" in msg1, f"Expected 'added' in message, got: {msg1}"

        # Verify item was correctly saved
        item1 = db_instance.get_media_by_id(id1)
        assert item1 is not None
        assert item1['content'] == unique_content
        assert item1['title'] == media1['title']
        original_version = item1['version']

        # Step 2: Try to add the second item with overwrite=False
        id2, uuid2, msg2 = db_instance.add_media_with_keywords(**media2, overwrite=False)
        assert id2 is None or id2 == id1, "Should not add a new item when URL exists"
        assert "exists" in msg2.lower(), f"Expected 'exists' in message, got: {msg2}"

        # Verify first item wasn't changed
        unchanged_item = db_instance.get_media_by_id(id1)
        assert unchanged_item['title'] == media1['title']
        assert unchanged_item['version'] == original_version

        # Step 3: Add with overwrite=True to update the existing item
        id3, uuid3, msg3 = db_instance.add_media_with_keywords(**media2, overwrite=True)
        assert id3 == id1, f"Expected to update existing ID {id1}, got {id3}"
        assert uuid3 == uuid1, f"Expected same UUID {uuid1}, got {uuid3}"
        assert "updated" in msg3.lower() or "already" in msg3.lower(), f"Expected update confirmation in message, got: {msg3}"

        # Step 4: Verify the final state - title should be updated but content remains the same
        final_item = db_instance.get_media_by_id(id1)
        assert final_item is not None, "Failed to retrieve updated item"
        assert final_item['title'] == media2['title'], "Title should be updated"
        assert final_item['content'] == unique_content, "Content should remain the same"
        assert final_item['version'] == original_version + 1, "Version should be incremented"


class TestTimeBasedAndSearchQueries:
    # ... other tests in this class are correct ...

    @given(days=st.integers(min_value=1, max_value=365))
    def test_empty_trash_respects_time_threshold(self, db_instance: MediaDatabase, days: int):
        """
        Property: `empty_trash` should only soft-delete items whose `trash_date`
        is older than the specified threshold.
        """
        media_id, _, _ = db_instance.add_media_with_keywords(
            title="Trash Test", content=f"...{uuid.uuid4().hex}", media_type="article", keywords=["test"])

        # This call handles versioning correctly, bumping version to 2
        db_instance.mark_as_trash(media_id)
        item_v2 = db_instance.get_media_by_id(media_id, include_trash=True)

        past_date = datetime.now(timezone.utc) - timedelta(days=days + 1)

        # FIX: The manual update MUST comply with the database triggers.
        # This means we have to increment the version and supply a client_id.
        # This makes the test setup robust.
        with db_instance.transaction():
            db_instance.execute_query(
                "UPDATE Media SET trash_date = ?, version = ?, client_id = ?, last_modified = ? WHERE id = ?",
                (
                    past_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                    item_v2['version'] + 1,  # Manually increment version for this setup step
                    'test_setup_client',
                    db_instance._get_current_utc_timestamp_str(),
                    media_id
                )
            )

        # Now the item is at version 3.
        # `empty_trash` will find this item and call `soft_delete_media`,
        # which will correctly read version 3 and update to version 4.
        processed_count, _ = empty_trash(db_instance=db_instance, days_threshold=days)
        assert processed_count == 1

        final_item = db_instance.get_media_by_id(media_id, include_trash=True, include_deleted=True)
        assert final_item['deleted'] == 1
        assert final_item['version'] == 4  # Initial: 1, Trash: 2, Manual Date Change: 3, Delete: 4


#
# End of test_media_db_properties.py
#######################################################################################################################
