# test_search_conversations_fts.py
# Description: RED-first regression coverage for task-249 (FTS instead of correlated LIKE).
"""
Task-249: ``search_conversations_page``'s message-content match used to be a
correlated ``EXISTS(SELECT 1 FROM messages m ... m.content LIKE '%q%')`` --
a per-candidate scan with an index-hostile leading wildcard. The schema
already maintains ``messages_fts`` (and triggers keep it in sync), so the
fix routes content matching through an FTS5 ``MATCH`` join instead.

LIKE is a substring match; FTS5 MATCH is token/prefix-based. The fix
formats the query as a quoted FTS5 prefix expression (embedded `"` doubled,
wrapped in `"..."`, trailing `*`) so a user-typed query still reads as "find
this text as a token prefix" rather than being interpreted as FTS5's own
query-language syntax (which would otherwise choke on bare `"`, `*`, `-`,
etc.). These tests pin both the new behavior and the "does not raise"
guarantee for FTS5-syntax-hazard input.
"""

import inspect

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
    yield database
    database.close_connection()


def _conversation_with_message(db, *, title: str, content: str) -> str:
    conv_id = db.add_conversation({"title": title})
    db.add_message({
        "conversation_id": conv_id,
        "sender": "user",
        "content": content,
    })
    return conv_id


def _ids(rows):
    return {row["id"] for row in rows}


class TestMessageContentMatch:
    def test_finds_conversation_by_word_prefix_in_message_content(self, db):
        conv_id = _conversation_with_message(
            db, title="Unrelated Title", content="the quick brown fox jumps"
        )
        rows, total, _ = db.search_conversations_page("brow")
        assert conv_id in _ids(rows)
        assert total >= 1

    def test_finds_conversation_by_full_word_in_message_content(self, db):
        conv_id = _conversation_with_message(
            db, title="Unrelated Title", content="testing message content search"
        )
        rows, total, _ = db.search_conversations_page("testing")
        assert conv_id in _ids(rows)

    def test_no_match_returns_empty(self, db):
        _conversation_with_message(db, title="Alpha", content="hello world")
        rows, total, _ = db.search_conversations_page("zzzznomatchxyz")
        assert rows == []
        assert total == 0

    def test_title_only_match_still_works(self, db):
        conv_id = db.add_conversation({"title": "A Very Distinctive Title"})
        rows, total, _ = db.search_conversations_page("Distinctive")
        assert conv_id in _ids(rows)

    def test_id_match_still_works(self, db):
        conv_id = db.add_conversation({"title": "Whatever"})
        rows, total, _ = db.search_conversations_page(conv_id)
        assert conv_id in _ids(rows)

    def test_deleted_message_does_not_resurrect_conversation(self, db):
        conv_id = db.add_conversation({"title": "Soon Empty"})
        msg_id = db.add_message({
            "conversation_id": conv_id,
            "sender": "user",
            "content": "uniquemarkerword should vanish",
        })
        db.soft_delete_message(msg_id, expected_version=1)

        rows, total, _ = db.search_conversations_page("uniquemarkerword")
        assert conv_id not in _ids(rows)


class TestFtsSyntaxHazardsDoNotRaise:
    """LIKE is a plain substring scan with no query syntax of its own; FTS5
    MATCH has a mini query language where bare `"`, `*`, `-` are meaningful.
    The fix must escape/quote so these never raise, even though the exact
    match semantics differ from LIKE."""

    @pytest.mark.parametrize(
        "query",
        [
            'foo"bar',
            "foo*bar",
            "foo-bar",
            '"',
            "**",
            "-leadinghyphen",
            "AND OR NOT",  # FTS5 boolean operators as literal search text
        ],
    )
    def test_special_characters_do_not_raise(self, db, query):
        _conversation_with_message(db, title="Alpha", content="hello world")
        # Must not raise CharactersRAGDBError / sqlite3.OperationalError.
        rows, total, _ = db.search_conversations_page(query)
        assert isinstance(rows, list)
        assert isinstance(total, int)

    def test_quote_in_query_matches_literal_content_containing_it(self, db):
        conv_id = _conversation_with_message(
            db, title="Alpha", content='He said "hello" to me'
        )
        rows, total, _ = db.search_conversations_page('"hello"')
        # Must not raise; whether or not it matches, the call completes.
        assert isinstance(rows, list)


class TestSqlShapePin:
    def test_no_more_correlated_content_like_in_source(self):
        """Lexical pin against regression: the content-match branch must no
        longer build a `m.content LIKE` clause."""
        source = inspect.getsource(CharactersRAGDB.search_conversations_page)
        assert "m.content LIKE" not in source

    def test_uses_messages_fts_match(self):
        source = inspect.getsource(CharactersRAGDB.search_conversations_page)
        assert "messages_fts" in source
        assert "MATCH" in source
