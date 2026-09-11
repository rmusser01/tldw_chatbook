"""Pin archive query plans on the production schema without planner statistics."""

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.parametrize("archive_scope", ["active", "archived"])
def test_library_archive_count_and_page_use_archive_index_without_stats(
    tmp_path, archive_scope
):
    db = CharactersRAGDB(tmp_path / "archive-plan.db", client_id="archive-plan")
    try:
        ids = [db.add_conversation({"title": f"Conversation {i}"}) for i in range(128)]
        archived = ids[::4]
        db.set_conversations_archived(
            archived,
            archived=True,
            expected_versions={cid: 1 for cid in archived},
        )
        conn = db.get_connection()
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'sqlite_stat1'"
            ).fetchone()
            is None
        )
        statements = []
        conn.set_trace_callback(statements.append)
        try:
            page = db.list_library_conversations_page(
                archive_scope=archive_scope, limit=12, offset=3
            )
        finally:
            conn.set_trace_callback(None)
        assert page["total"] == (32 if archive_scope == "archived" else 96)
        queries = [
            sql
            for sql in statements
            if sql.lstrip().upper().startswith("SELECT")
            and "FROM conversations WHERE" in " ".join(sql.split())
        ]
        assert len(queries) == 2, statements
        for query in queries:
            plan = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + query)]
            assert any(
                "SEARCH conversations USING" in detail
                and "idx_conversations_archive" in detail
                for detail in plan
            ), plan
    finally:
        db.close_connection()
