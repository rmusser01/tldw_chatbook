"""Tests for task-5: conversation-level RAG retrieval scope storage.

Covers ``Chat/rag_scope.py``'s storage layer -- ``read_conversation_scope``/
``write_conversation_scope`` over ``conversations.metadata["rag_scope"]``
(the same read-merge-write seam the chat-dictionaries attach mechanism uses)
-- and ``SessionScopeHolder``'s not-yet-persisted-session lifecycle. Uses a
real file-backed ``CharactersRAGDB`` (tmp_path), mirroring
``Tests/RAG/test_scope_pipeline_enforcement.py``'s fixture pattern.
"""

import json

import pytest

from tldw_chatbook.Chat.rag_scope import (
    CONVERSATION_METADATA_SCOPE_KEY,
    RagScope,
    ScopeItem,
    SessionScopeHolder,
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
    read_conversation_scope,
    write_conversation_scope,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


@pytest.fixture()
def cha_db(tmp_path):
    return CharactersRAGDB(tmp_path / "cha.db", client_id="task5-test")


@pytest.fixture()
def conversation_id(cha_db):
    return cha_db.add_conversation({"title": "Scoped conversation"})


def _raw_metadata(db, conversation_id) -> dict:
    record = db.get_conversation_by_id(conversation_id)
    return json.loads(record.get("metadata") or "{}")


class TestReadWriteRoundTrip:
    def test_write_then_read_round_trips(self, cha_db, conversation_id):
        scope = RagScope(
            items=(ScopeItem(SOURCE_TYPE_MEDIA, "42"), ScopeItem(SOURCE_TYPE_NOTE, "n1")),
            updated_at="2026-07-21T00:00:00+00:00",
        )

        write_conversation_scope(cha_db, conversation_id, scope)
        result = read_conversation_scope(cha_db, conversation_id)

        assert result == scope

    def test_no_stored_scope_reads_as_none(self, cha_db, conversation_id):
        assert read_conversation_scope(cha_db, conversation_id) is None

    def test_write_none_deletes_the_key(self, cha_db, conversation_id):
        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")
        write_conversation_scope(cha_db, conversation_id, scope)
        assert read_conversation_scope(cha_db, conversation_id) is not None

        write_conversation_scope(cha_db, conversation_id, None)

        assert read_conversation_scope(cha_db, conversation_id) is None
        assert CONVERSATION_METADATA_SCOPE_KEY not in _raw_metadata(
            cha_db, conversation_id
        )

    def test_write_preserves_other_metadata_keys(self, cha_db, conversation_id):
        """The read-merge-write must not clobber unrelated metadata (e.g.
        chat-dictionaries' active_dictionaries key living in the same JSON blob)."""
        record = cha_db.get_conversation_by_id(conversation_id)
        cha_db.update_conversation(
            conversation_id,
            {"metadata": json.dumps({"active_dictionaries": [1, 2]})},
            expected_version=record["version"],
        )

        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")
        write_conversation_scope(cha_db, conversation_id, scope)

        raw = _raw_metadata(cha_db, conversation_id)
        assert raw["active_dictionaries"] == [1, 2]
        assert raw[CONVERSATION_METADATA_SCOPE_KEY]["items"] == [
            {"source_type": SOURCE_TYPE_MEDIA, "source_id": "1"}
        ]

    def test_write_unknown_conversation_raises(self, cha_db):
        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")
        with pytest.raises(ValueError):
            write_conversation_scope(cha_db, "does-not-exist", scope)


class TestMalformedMetadataGuard:
    def test_malformed_metadata_json_reads_as_none(self, cha_db, conversation_id):
        record = cha_db.get_conversation_by_id(conversation_id)
        cha_db.update_conversation(
            conversation_id,
            {"metadata": "{not valid json"},
            expected_version=record["version"],
        )

        assert read_conversation_scope(cha_db, conversation_id) is None

    def test_non_dict_metadata_json_reads_as_none(self, cha_db, conversation_id):
        record = cha_db.get_conversation_by_id(conversation_id)
        cha_db.update_conversation(
            conversation_id,
            {"metadata": json.dumps(["not", "a", "dict"])},
            expected_version=record["version"],
        )

        assert read_conversation_scope(cha_db, conversation_id) is None

    def test_malformed_scope_payload_reads_as_none(self, cha_db, conversation_id):
        record = cha_db.get_conversation_by_id(conversation_id)
        cha_db.update_conversation(
            conversation_id,
            {
                "metadata": json.dumps(
                    {CONVERSATION_METADATA_SCOPE_KEY: {"version": 99, "items": []}}
                )
            },
            expected_version=record["version"],
        )

        assert read_conversation_scope(cha_db, conversation_id) is None

    def test_unknown_conversation_reads_as_none(self, cha_db):
        assert read_conversation_scope(cha_db, "does-not-exist") is None


class TestWriteConversationScopeCorruptMetadataGuard:
    """PR #734 review (id 3621197388): write_conversation_scope must not
    normalize a corrupt existing ``metadata`` payload to ``{}`` and silently
    write that back -- doing so would erase whatever else lives in the same
    JSON column (e.g. chat-dictionaries' ``active_dictionaries``). Absent/
    ``None``/empty metadata is NOT corrupt and proceeds exactly as before
    (``TestReadWriteRoundTrip`` covers that path); this covers the
    fail-closed refusal for a present-but-invalid payload -- the write is
    skipped entirely, not raised.
    """

    def test_malformed_json_metadata_is_untouched_by_a_write_attempt(
        self, cha_db, conversation_id
    ):
        record = cha_db.get_conversation_by_id(conversation_id)
        cha_db.update_conversation(
            conversation_id,
            {"metadata": "{not valid json"},
            expected_version=record["version"],
        )

        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")
        write_conversation_scope(cha_db, conversation_id, scope)

        raw = cha_db.get_conversation_by_id(conversation_id)
        assert raw["metadata"] == "{not valid json"

    def test_non_dict_json_metadata_is_untouched_by_a_write_attempt(
        self, cha_db, conversation_id
    ):
        record = cha_db.get_conversation_by_id(conversation_id)
        original_raw = json.dumps(["not", "a", "dict"])
        cha_db.update_conversation(
            conversation_id,
            {"metadata": original_raw},
            expected_version=record["version"],
        )

        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")
        write_conversation_scope(cha_db, conversation_id, scope)

        raw = cha_db.get_conversation_by_id(conversation_id)
        assert raw["metadata"] == original_raw

    def test_delete_path_also_refuses_on_corrupt_metadata(
        self, cha_db, conversation_id
    ):
        """``scope=None`` (the 'clear scope' path) refuses identically -- it
        still needs the read-merge-write seam, and a corrupt payload is
        just as unsafe to normalize-and-write for a delete as for a set."""
        record = cha_db.get_conversation_by_id(conversation_id)
        original_raw = "{not valid json"
        cha_db.update_conversation(
            conversation_id,
            {"metadata": original_raw},
            expected_version=record["version"],
        )

        write_conversation_scope(cha_db, conversation_id, None)

        raw = cha_db.get_conversation_by_id(conversation_id)
        assert raw["metadata"] == original_raw

    def test_write_returns_none_without_raising_on_corrupt_metadata(
        self, cha_db, conversation_id
    ):
        record = cha_db.get_conversation_by_id(conversation_id)
        cha_db.update_conversation(
            conversation_id,
            {"metadata": "{not valid json"},
            expected_version=record["version"],
        )
        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")

        result = write_conversation_scope(cha_db, conversation_id, scope)

        assert result is None

    def test_normal_write_over_absent_metadata_is_unaffected(
        self, cha_db, conversation_id
    ):
        """Sanity check: absent metadata (this fixture's default) is NOT
        corrupt -- writes still proceed exactly as before this fix."""
        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_NOTE, "n1"),), updated_at="t1")

        write_conversation_scope(cha_db, conversation_id, scope)

        assert read_conversation_scope(cha_db, conversation_id) == scope


class TestZeroItemNormalization:
    def test_stored_zero_item_scope_reads_as_none(self, cha_db, conversation_id):
        """A stored scope with an empty items list is unscoped on read
        (Task 2 review adjudication): EffectiveScope has no distinct
        'scoped with nothing selected' state, and Save-with-zero-selected
        already means 'clear scope' at the UI layer."""
        record = cha_db.get_conversation_by_id(conversation_id)
        cha_db.update_conversation(
            conversation_id,
            {
                "metadata": json.dumps(
                    {
                        CONVERSATION_METADATA_SCOPE_KEY: {
                            "version": 1,
                            "items": [],
                            "updated_at": "t1",
                        }
                    }
                )
            },
            expected_version=record["version"],
        )

        assert read_conversation_scope(cha_db, conversation_id) is None

    def test_writing_a_zero_item_scope_still_round_trips_the_raw_write(
        self, cha_db, conversation_id
    ):
        """write_conversation_scope itself does not second-guess the caller --
        only the read side normalizes; this documents that boundary."""
        empty_scope = RagScope(items=(), updated_at="t1")
        write_conversation_scope(cha_db, conversation_id, empty_scope)

        assert (
            _raw_metadata(cha_db, conversation_id)[CONVERSATION_METADATA_SCOPE_KEY][
                "items"
            ]
            == []
        )
        assert read_conversation_scope(cha_db, conversation_id) is None


class TestSessionScopeHolder:
    def test_starts_empty(self):
        holder = SessionScopeHolder()
        assert holder.scope is None

    def test_set_and_get(self):
        holder = SessionScopeHolder()
        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1")

        holder.set(scope)

        assert holder.scope == scope

    def test_set_none_clears(self):
        holder = SessionScopeHolder()
        holder.set(RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "1"),), updated_at="t1"))

        holder.set(None)

        assert holder.scope is None

    def test_set_zero_item_scope_normalizes_to_none(self):
        """A zero-item scope reports unscoped exactly like
        ``read_conversation_scope`` normalizes a stored zero-item scope on
        read (TestZeroItemNormalization above) -- no Phase-2 producer of a
        held scope can create the divergence."""
        holder = SessionScopeHolder()

        holder.set(RagScope(items=(), updated_at="t1"))

        assert holder.scope is None

    def test_flush_to_zero_item_scope_writes_nothing(self, cha_db, conversation_id):
        holder = SessionScopeHolder()
        holder.set(RagScope(items=(), updated_at="t1"))

        holder.flush_to(cha_db, conversation_id)

        assert holder.scope is None
        assert read_conversation_scope(cha_db, conversation_id) is None
        assert CONVERSATION_METADATA_SCOPE_KEY not in _raw_metadata(
            cha_db, conversation_id
        )

    def test_flush_to_writes_through_and_empties_holder(self, cha_db, conversation_id):
        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_NOTE, "n1"),), updated_at="t1")
        holder = SessionScopeHolder()
        holder.set(scope)

        holder.flush_to(cha_db, conversation_id)

        assert holder.scope is None
        assert read_conversation_scope(cha_db, conversation_id) == scope

    def test_flush_to_is_a_noop_when_nothing_held(self, cha_db, conversation_id):
        holder = SessionScopeHolder()

        holder.flush_to(cha_db, conversation_id)  # must not raise

        assert holder.scope is None
        assert read_conversation_scope(cha_db, conversation_id) is None

    def test_flush_to_is_noop_even_for_an_unknown_conversation(self):
        """Nothing held means flush_to never touches the DB at all -- proven
        by not raising even though 'does-not-exist' has no conversation row
        (write_conversation_scope would raise ValueError if it were called)."""
        holder = SessionScopeHolder()

        holder.flush_to(db=None, conversation_id="does-not-exist")

        assert holder.scope is None
