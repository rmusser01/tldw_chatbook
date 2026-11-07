# test_library_conversation_search.py
# Description: golden-baseline equivalence tests for the Library conversation
# search content branch (task-2, non-console efficiency remediation wave 1).
"""
``search_library_conversations_page`` ORs several match branches; the
message-content branch was a correlated
``EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id =
conversations.id AND ... m.content LIKE '%q%')`` -- a per-candidate-row scan
of message bodies (task-249's rule: never a leading-wildcard LIKE inside a
correlated EXISTS). The rewrite swaps the EXISTS text for the uncorrelated
``id IN (SELECT m.conversation_id FROM messages m WHERE ...)`` shape already
proven on the Console seam (``_conversation_search_filter``).

These tests were written and run against the UNMODIFIED correlated form to
capture the golden result tuples, then re-run unchanged after the rewrite:
they must pass byte-identically on both. They pin the six equivalence
classes -- title exact, title substring, message MID-WORD substring (the
LIKE semantics the rewrite must preserve), FTS token match, keyword match,
no-match -> empty page/total 0 -- plus the per-row branch projection
(``matched_fields``, the observable projection of the ``hit_N`` columns)
for a row matching several branches at once.
"""

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def mem_db_instance(client_id):
    """Creates an in-memory DB instance for tests that don't need file persistence."""
    db = CharactersRAGDB(":memory:", client_id)
    yield db
    db.close_connection()


@pytest.fixture
def client_id():
    """Provides a consistent client ID for tests."""
    return "test_client_001"


BULK_CONVERSATIONS = 30
BULK_MESSAGES = 40


def _pin_last_modified(db, conv_id, value):
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET last_modified = ?, version = version + 1 "
            "WHERE id = ?",
            (value, conv_id),
        )


def _seed_library_corpus(db):
    """Seed filler conversations plus deterministic marker conversations.

    Filler content avoids every search marker ("needle", "window", "idow",
    "needle-kw") so totals prove the markers -- and only the markers --
    match. Each marker gets a distinct pinned ``last_modified`` so the
    golden ordering (exact-title first, then last_modified DESC) is fully
    deterministic regardless of insertion rowids.
    """
    for index in range(BULK_CONVERSATIONS):
        conv_id = db.add_conversation({"title": f"Bulk chat {index:02d}"})
        for slot in range(BULK_MESSAGES):
            db.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": "assistant",
                    "content": f"filler {index:02d}/{slot:02d}: the quick brown fox",
                }
            )

    def marker(*, title, messages=(), keywords=(), last_modified):
        conv_id = db.add_conversation({"title": title})
        for index, (sender, content) in enumerate(messages):
            db.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": sender,
                    "content": content,
                    "timestamp": f"2026-08-01T10:{index:02d}:00.000Z",
                }
            )
        for keyword in keywords:
            existing = db.get_keyword_by_text(keyword)
            keyword_id = existing["id"] if existing else db.add_keyword(keyword)
            db.link_conversation_to_keyword(conv_id, keyword_id)
        _pin_last_modified(db, conv_id, last_modified)
        return conv_id

    return {
        # Matches every "needle" branch at once: exact title, title
        # substring, message LIKE, FTS title, FTS message, keyword.
        "multi": marker(
            title="needle",
            messages=[("user", "needle body")],
            keywords=("needle-kw",),
            last_modified="2026-08-02T00:00:00.000Z",
        ),
        # Exact-title only: plain body, no keywords.
        "exact": marker(
            title="needle",
            messages=[("user", "ordinary body text")],
            last_modified="2026-08-01T00:00:00.000Z",
        ),
        # Title substring + FTS title only; newest last_modified overall, so
        # it proves exact-title hits still rank above mere recency.
        "substr": marker(
            title="a needle in the title",
            last_modified="2026-09-01T00:00:00.000Z",
        ),
        # Message LIKE + FTS message only.
        "fts": marker(
            title="plain title",
            messages=[("assistant", "the needle was found here")],
            last_modified="2026-08-05T00:00:00.000Z",
        ),
        # Keyword branch only.
        "kw": marker(
            title="unrelated",
            keywords=("needle-kw",),
            last_modified="2026-08-04T00:00:00.000Z",
        ),
        # MID-WORD substring: "indo" occurs only inside "window" (a true
        # mid-word fragment -- note "idow" is NOT a substring of
        # "window"), so the FTS token branches cannot match -- this
        # isolates the LIKE semantics the uncorrelated rewrite must
        # preserve.
        "midword": marker(
            title="unrelated project",
            messages=[("user", "please open the window")],
            last_modified="2026-08-03T00:00:00.000Z",
        ),
    }


class TestSearchLibraryConversationsGolden:
    def test_needle_golden_page_order_and_total(self, mem_db_instance):
        ids = _seed_library_corpus(mem_db_instance)

        page = mem_db_instance.search_library_conversations_page(
            query="needle", limit=10, offset=0
        )

        # Golden: five marker matches (bulk never matches), exact-title
        # group first (multi then exact by recency), then the rest by
        # last_modified DESC -- substring outranks nothing despite being
        # the newest row.
        assert page["total"] == 5
        assert [item["id"] for item in page["items"]] == [
            ids["multi"],
            ids["exact"],
            ids["substr"],
            ids["fts"],
            ids["kw"],
        ]
        assert [item["title"] for item in page["items"]] == [
            "needle",
            "needle",
            "a needle in the title",
            "plain title",
            "unrelated",
        ]

    def test_needle_golden_matched_fields_projection(self, mem_db_instance):
        ids = _seed_library_corpus(mem_db_instance)

        page = mem_db_instance.search_library_conversations_page(
            query="needle", limit=10, offset=0
        )
        by_id = {item["id"]: item for item in page["items"]}

        # Per-row hit_N projection: a row matching several branches reports
        # each field independently; single-branch rows report only theirs.
        assert by_id[ids["multi"]]["matched_fields"] == [
            "keywords",
            "message",
            "title",
        ]
        assert by_id[ids["exact"]]["matched_fields"] == ["title"]
        assert by_id[ids["substr"]]["matched_fields"] == ["title"]
        assert by_id[ids["fts"]]["matched_fields"] == ["message"]
        assert by_id[ids["kw"]]["matched_fields"] == ["keywords"]

        assert by_id[ids["multi"]]["matched_keywords"] == ["needle-kw"]
        assert by_id[ids["kw"]]["matched_keywords"] == ["needle-kw"]
        assert by_id[ids["exact"]]["matched_keywords"] == []
        assert by_id[ids["substr"]]["matched_keywords"] == []
        assert by_id[ids["fts"]]["matched_keywords"] == []

    def test_needle_golden_pagination_window(self, mem_db_instance):
        ids = _seed_library_corpus(mem_db_instance)

        page = mem_db_instance.search_library_conversations_page(
            query="needle", limit=2, offset=2
        )

        assert page["total"] == 5
        assert [item["id"] for item in page["items"]] == [
            ids["substr"],
            ids["fts"],
        ]

    def test_midword_substring_matches_via_message_branch_only(self, mem_db_instance):
        ids = _seed_library_corpus(mem_db_instance)

        page = mem_db_instance.search_library_conversations_page(
            query="indo", limit=10, offset=0
        )

        # "indo" lives mid-word inside "window": LIKE must still find it
        # while the FTS token branches stay silent, so the message field is
        # reported alone.
        assert page["total"] == 1
        assert [item["id"] for item in page["items"]] == [ids["midword"]]
        assert page["items"][0]["matched_fields"] == ["message"]
        assert page["items"][0]["matched_keywords"] == []

    def test_full_word_message_match_reports_message_field(self, mem_db_instance):
        ids = _seed_library_corpus(mem_db_instance)

        page = mem_db_instance.search_library_conversations_page(
            query="window", limit=10, offset=0
        )

        # Whole-word hit: message LIKE and FTS token branches agree.
        assert page["total"] == 1
        assert [item["id"] for item in page["items"]] == [ids["midword"]]
        assert page["items"][0]["matched_fields"] == ["message"]

    def test_no_match_returns_empty_page_and_zero_total(self, mem_db_instance):
        _seed_library_corpus(mem_db_instance)

        page = mem_db_instance.search_library_conversations_page(
            query="zzzznomatchxyz", limit=10, offset=0
        )

        assert page["total"] == 0
        assert page["items"] == []
