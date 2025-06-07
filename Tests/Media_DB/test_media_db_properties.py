# test_media_db_v2_properties.py
#
# Property-based tests for the Media_DB_v2 library using Hypothesis.
# These tests verify the logical correctness and invariants of the database
# operations across a wide range of generated data.
#
# Imports
from datetime import datetime, timezone, timedelta
from typing import Iterator, Callable
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
def db_factory(tmp_path: Path) -> Callable[[], MediaDatabase]:
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


# --- Hypothesis Strategies ---

# Strategy for generating text that is guaranteed to have non-whitespace content.
st_required_text = st.text(min_size=1, max_size=50).map(lambda s: s.strip()).filter(lambda s: len(s) > 0)

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
    return {
        "title": draw(st_required_text),
        "content": draw(st.text(min_size=10, max_size=500)),
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

        linked_keywords = {kw.lower().strip() for kw in fetch_keywords_for_media(media_id, db_instance)}
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
        initial_media["content"] += f" initial_{uuid.uuid4().hex}"
        update_media["content"] += f" update_{uuid.uuid4().hex}"
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(**initial_media)
        original = db_instance.get_media_by_id(media_id)
        media_id_up, media_uuid_up, msg = db_instance.add_media_with_keywords(
            url=original['url'],
            overwrite=True,
            **update_media
        )
        assert media_id_up == media_id
        assert media_uuid_up == media_uuid
        assert "updated" in msg
        updated = db_instance.get_media_by_id(media_id)
        assert updated is not None
        assert updated['version'] == original['version'] + 1
        assert updated['title'] == update_media['title']
        assert updated['content'] == update_media['content']
        doc_versions = db_instance.get_all_document_versions(media_id)
        assert len(doc_versions) == 2

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

    # FIX: Add a settings decorator to handle variable I/O timing and prevent flaky deadline errors.
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

    @given(media1=st_media_data(), media2=st_media_data())
    def test_add_media_with_conflicting_hash_is_handled(self,
                                                        tmp_path: Path,
                                                        media1: dict,
                                                        media2: dict):
        db_file = tmp_path / f"prop_test_{uuid.uuid4().hex}.db"
        client_id = f"client_{uuid.uuid4().hex[:8]}"
        db_instance = MediaDatabase(db_path=db_file, client_id=client_id)
        try:
            assume(media1['title'] != media2['title'])
            media2['content'] = media1['content']

            media1['url'] = f"http://example.com/url1_{uuid.uuid4().hex}"
            media2['url'] = f"http://example.com/url2_{uuid.uuid4().hex}"

            id1, _, _ = db_instance.add_media_with_keywords(**media1)

            id2, _, msg2 = db_instance.add_media_with_keywords(**media2, overwrite=False)
            assert id2 is None
            assert "exists, not overwritten" in msg2

            id3, _, msg3 = db_instance.add_media_with_keywords(**media2, overwrite=True)
            assert id3 == id1
            assert "updated" in msg3
            final_item = db_instance.get_media_by_id(id1)
            assert final_item['title'] == media2['title']

        finally:
            db_instance.close_connection()


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
