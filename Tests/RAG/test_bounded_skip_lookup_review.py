"""Incremental indexing reads only IDs in the incoming batch."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from Tests.RAG.test_embedding_content_hash_cache import FakeRAGService, _entry
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.ingestion_indexing import index_entries

pytestmark = pytest.mark.bootstrap_profile


async def test_unrelated_tracking_rows_do_not_force_unchanged_batch_to_reindex(
    tmp_path,
):
    db = RAGIndexingDB(tmp_path / "tracking.db")
    modified = datetime(2026, 10, 1, 12, tzinfo=UTC)
    try:
        db.mark_item_indexed("wanted", "media", modified)
        with db.connection() as connection:
            connection.execute(
                "INSERT INTO indexed_items "
                "(item_id, item_type, last_indexed, last_modified) VALUES (?, ?, ?, ?)",
                ("unrelated", "media", modified, "unreadable legacy timestamp"),
            )
        service = FakeRAGService()
        result = await index_entries(
            service, db, [_entry("wanted", content="same text", last_modified=modified)]
        )
        assert result == {"indexed": 0, "skipped": 1, "failed": 0, "errors": []}
        assert service.indexed_docs == []
    finally:
        db.close()


def test_lookup_chunks_requested_ids_and_keeps_timestamp_offsets(tmp_path):
    db = RAGIndexingDB(tmp_path / "bounded.db")
    modified = datetime(2026, 10, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    statements = []
    try:
        db.mark_items_indexed(
            [(str(index), "media", modified, 0, None) for index in range(601)]
        )
        with db.connection() as connection:
            connection.set_trace_callback(statements.append)
        requested = [str(index) for index in range(600)]
        actual = db.get_indexed_items_by_ids("media", requested + ["0"])
        assert set(actual) == set(requested)
        assert actual["599"] == modified
        queries = [
            statement
            for statement in statements
            if "SELECT item_id, last_modified FROM indexed_items" in statement
        ]
        assert len(queries) == 2
        assert all("item_id IN (" in query for query in queries)
        assert all("'600'" not in query for query in queries)
    finally:
        db.close()


async def test_unreadable_requested_timestamp_preserves_reindex_fallback(tmp_path):
    db = RAGIndexingDB(tmp_path / "fallback.db")
    modified = datetime(2026, 10, 1, 12, tzinfo=UTC)
    try:
        db.mark_item_indexed("wanted", "media", modified)
        with db.connection() as connection:
            connection.execute(
                "UPDATE indexed_items SET last_modified = ? WHERE item_id = ?",
                ("unreadable legacy timestamp", "wanted"),
            )
        service = FakeRAGService()
        result = await index_entries(
            service, db, [_entry("wanted", content="same text", last_modified=modified)]
        )
        assert result == {"indexed": 1, "skipped": 0, "failed": 0, "errors": []}
        assert [document["id"] for document in service.indexed_docs] == ["media_wanted"]
        assert db.get_indexed_items_by_ids("media", ["wanted"]) == {"wanted": modified}
    finally:
        db.close()
