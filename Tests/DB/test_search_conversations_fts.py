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
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
    yield database
    database.close_connection()


def _conversation_with_message(db, *, title: str, content: str) -> str:
    conv_id = db.add_conversation({"title": title})
    db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": content,
        }
    )
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
        msg_id = db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "user",
                "content": "uniquemarkerword should vanish",
            }
        )
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
        _conversation_with_message(db, title="Alpha", content='He said "hello" to me')
        rows, total, _ = db.search_conversations_page('"hello"')
        # Must not raise; whether or not it matches, the call completes.
        assert isinstance(rows, list)


class TestSqlShapePin:
    @staticmethod
    def _search_source():
        return "\n".join(
            (
                inspect.getsource(CharactersRAGDB._conversation_search_filter),
                inspect.getsource(CharactersRAGDB.search_conversations_page),
            )
        )

    def test_no_more_correlated_content_like_in_source(self):
        """Lexical pin against regression: the content-match branch must no
        longer build a `m.content LIKE` clause."""
        source = self._search_source()
        assert "m.content LIKE" not in source

    def test_uses_messages_fts_match(self):
        source = self._search_source()
        assert "messages_fts" in source
        assert "MATCH" in source


def _seed_coherent_conversation_population(db: CharactersRAGDB) -> list[str]:
    conversation_ids = []
    for index in range(45):
        scope = (
            {"scope_type": "workspace", "workspace_id": f"workspace-{index % 3}"}
            if index % 2
            else {"scope_type": "global"}
        )
        conversation_ids.append(
            db.add_conversation({"title": f"Conversation {index:02d}", **scope})
        )

    db.add_message(
        {
            "conversation_id": conversation_ids[17],
            "sender": "user",
            "content": "coherentlocatorneedle appears only in this message",
        }
    )
    deleted_id = db.add_conversation({"title": "Deleted conversation"})
    deleted = db.get_conversation_by_id(deleted_id)
    db.soft_delete_conversation(deleted_id, expected_version=deleted["version"])

    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET last_modified = ? WHERE deleted = 0",
            ("2026-08-14T12:00:00.000Z",),
        )
    return sorted(conversation_ids, reverse=True)


class TestCoherentConversationPages:
    def test_pages_are_exact_stable_partitions_under_all_supported_filters(self, db):
        expected_ids = _seed_coherent_conversation_population(db)

        pages = [
            db.search_conversations_page(
                None, scope_type="all", limit=20, offset=offset
            )
            for offset in (0, 20, 40)
        ]

        assert [total for _, total, _ in pages] == [45, 45, 45]
        assert [len(rows) for rows, _, _ in pages] == [20, 20, 5]
        actual_ids = [row["id"] for rows, _, _ in pages for row in rows]
        assert actual_ids == expected_ids
        assert len(actual_ids) == len(set(actual_ids))

        global_rows, global_total, _ = db.search_conversations_page(
            None, scope_type="global", limit=20, offset=0
        )
        assert global_total == 23
        assert all(row["scope_type"] == "global" for row in global_rows)

        workspace_rows, workspace_total, _ = db.search_conversations_page(
            None,
            scope_type="workspace",
            workspace_id="workspace-1",
            limit=20,
            offset=0,
        )
        assert workspace_total == 8
        assert all(row["workspace_id"] == "workspace-1" for row in workspace_rows)

        fts_rows, fts_total, _ = db.search_conversations_page(
            "coherentlocator", scope_type="all", limit=20, offset=0
        )
        assert fts_total == 1
        assert fts_rows[0]["title"] == "Conversation 17"

    @pytest.mark.parametrize("mutation", ["insert", "delete"])
    def test_count_and_rows_share_one_wal_snapshot(self, tmp_path, mutation):
        counted = threading.Event()
        release = threading.Event()

        class CoordinatedReaderDB(CharactersRAGDB):
            def _after_conversation_page_count(self) -> None:
                counted.set()
                assert release.wait(5), "writer did not release the coordinated reader"

        db_path = tmp_path / f"coherent-{mutation}.db"
        reader = CoordinatedReaderDB(db_path, "reader")
        writer = CharactersRAGDB(db_path, "writer")
        before_ids = _seed_coherent_conversation_population(reader)
        expected_before = (45, tuple(before_ids[:20]))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                reader.search_conversations_page,
                None,
                scope_type="all",
                limit=20,
                offset=0,
            )
            assert counted.wait(5), "reader never reached the count/page boundary"
            if mutation == "insert":
                changed_id = writer.add_conversation({"title": "Concurrent insert"})
                _set_conversation_last_modified(
                    writer, changed_id, "2026-08-15T00:00:00.000Z"
                )
                after_ids = [changed_id, *before_ids]
                expected_after = (46, tuple(after_ids[:20]))
            else:
                changed_id = before_ids[0]
                changed = writer.get_conversation_by_id(changed_id)
                writer.soft_delete_conversation(
                    changed_id, expected_version=changed["version"]
                )
                after_ids = before_ids[1:]
                expected_after = (44, tuple(after_ids[:20]))
            release.set()
            rows, total, _ = future.result(timeout=5)

        observed = (total, tuple(row["id"] for row in rows))
        assert observed in {expected_before, expected_after}
        reader.close_connection()
        writer.close_connection()


class TestLocateConversationPage:
    def test_returns_only_the_bounded_page_owning_the_target(self, db):
        expected_ids = _seed_coherent_conversation_population(db)
        target_id = expected_ids[24]

        located = db.locate_conversation_page(
            target_id, scope_type="all", limit=20
        )

        assert located["offset"] == 20
        assert located["target_index"] == 24
        assert located["total"] == 45
        assert target_id in {row["id"] for row in located["rows"]}
        assert located["rows"][located["target_index"] - located["offset"]][
            "id"
        ] == target_id
        assert len(located["rows"]) == 20

    def test_handles_first_final_and_exactly_aligned_pages(self, db):
        expected_ids = _seed_coherent_conversation_population(db)

        first = db.locate_conversation_page(
            expected_ids[0], scope_type="all", limit=20
        )
        aligned = db.locate_conversation_page(
            expected_ids[20], scope_type="all", limit=20
        )
        final = db.locate_conversation_page(
            expected_ids[-1], scope_type="all", limit=20
        )

        assert (first["target_index"], first["offset"], len(first["rows"])) == (
            0,
            0,
            20,
        )
        assert (
            aligned["target_index"],
            aligned["offset"],
            aligned["rows"][0]["id"],
        ) == (20, 20, expected_ids[20])
        assert (final["target_index"], final["offset"], len(final["rows"])) == (
            44,
            40,
            5,
        )

    def test_unavailable_or_invalid_target_fails_closed(self, db):
        expected_ids = _seed_coherent_conversation_population(db)
        target_id = expected_ids[0]
        target = db.get_conversation_by_id(target_id)
        db.soft_delete_conversation(target_id, expected_version=target["version"])

        assert (
            db.locate_conversation_page(target_id, scope_type="all", limit=20) is None
        )
        assert (
            db.locate_conversation_page(
                "00000000-0000-4000-8000-000000000000",
                scope_type="all",
                limit=20,
            )
            is None
        )
        with pytest.raises(InputError, match="conversation_id"):
            db.locate_conversation_page("  ", scope_type="all", limit=20)
        with pytest.raises(InputError, match="limit"):
            db.locate_conversation_page(expected_ids[1], scope_type="all", limit=0)


# ---------------------------------------------------------------------------
# task-1337 (plan Task 4): Library read seams for conversations and messages.
# Additive, agent-safe projections: bounded pages with exact totals, honest
# match evidence, text-only windowed message reads, and no RAG-context
# adjunct data. RAG context lives in a JSON sidecar store owned by
# ChatConversationService, so the SQL seams below must never surface it.
# ---------------------------------------------------------------------------


def _library_conversation(db, *, title, messages=(), keywords=()):
    """Seed one conversation with ordered messages and linked keywords."""
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
        keyword_id = db.add_keyword(keyword)
        db.link_conversation_to_keyword(conv_id, keyword_id)
    return conv_id


def _set_conversation_last_modified(db, conv_id, value):
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET last_modified = ?, version = version + 1 "
            "WHERE id = ?",
            (value, conv_id),
        )


class TestListLibraryConversationsPage:
    def test_exact_total_and_stable_recency_order(self, db):
        older = _library_conversation(db, title="Older")
        newer = _library_conversation(db, title="Newer")
        third = _library_conversation(db, title="Third")
        _set_conversation_last_modified(db, older, "2026-08-01T09:00:00.000Z")
        _set_conversation_last_modified(db, third, "2026-08-02T09:00:00.000Z")
        _set_conversation_last_modified(db, newer, "2026-08-03T09:00:00.000Z")

        page = db.list_library_conversations_page(limit=2, offset=0)
        assert page["total"] == 3
        assert [item["id"] for item in page["items"]] == [newer, third]
        assert all(
            set(item)
            == {
                "id",
                "title",
                "created_at",
                "last_modified",
                "version",
                "keywords",
                "keyword_total",
                "keywords_truncated",
            }
            for item in page["items"]
        )

        rest = db.list_library_conversations_page(limit=2, offset=2)
        assert rest["total"] == 3
        assert [item["id"] for item in rest["items"]] == [older]

    def test_soft_deleted_conversations_are_excluded(self, db):
        keep = _library_conversation(db, title="Keep")
        drop = _library_conversation(db, title="Drop")
        db.soft_delete_conversation(drop, expected_version=1)

        page = db.list_library_conversations_page(limit=10, offset=0)
        assert page["total"] == 1
        assert [item["id"] for item in page["items"]] == [keep]

    def test_keywords_are_capped_with_exact_total(self, db):
        conv_id = _library_conversation(
            db, title="Keyword heavy", keywords=[f"kw-{index:02d}" for index in range(25)]
        )

        page = db.list_library_conversations_page(limit=10, offset=0)
        item = next(entry for entry in page["items"] if entry["id"] == conv_id)
        assert len(item["keywords"]) == 20
        assert item["keyword_total"] == 25
        assert item["keywords_truncated"] is True


class TestSearchLibraryConversationsPage:
    def test_multiple_matching_messages_count_conversation_once(self, db):
        conv_id = _library_conversation(
            db,
            title="Plain title",
            messages=[("user", "needle first"), ("assistant", "needle second")],
        )

        page = db.search_library_conversations_page(query="needle", limit=10, offset=0)
        assert page["total"] == 1
        assert [item["id"] for item in page["items"]] == [conv_id]

    def test_matched_fields_cover_title_message_and_keywords(self, db):
        conv_id = _library_conversation(
            db,
            title="needle in title",
            messages=[("user", "needle in body")],
            keywords=["needle-kw"],
        )

        page = db.search_library_conversations_page(query="needle", limit=10, offset=0)
        item = next(entry for entry in page["items"] if entry["id"] == conv_id)
        assert item["matched_fields"] == ["keywords", "message", "title"]
        assert item["matched_keywords"] == ["needle-kw"]

    def test_keyword_only_match_reports_keywords_field(self, db):
        conv_id = _library_conversation(db, title="unrelated", keywords=["needle-kw"])

        page = db.search_library_conversations_page(query="needle", limit=10, offset=0)
        item = next(entry for entry in page["items"] if entry["id"] == conv_id)
        assert item["matched_fields"] == ["keywords"]

    def test_exact_title_ranks_first_despite_recency(self, db):
        partial = _library_conversation(db, title="needle and more words")
        exact = _library_conversation(db, title="needle")
        _set_conversation_last_modified(db, exact, "2026-08-01T00:00:00.000Z")
        _set_conversation_last_modified(db, partial, "2026-08-03T00:00:00.000Z")

        page = db.search_library_conversations_page(query="needle", limit=10, offset=0)
        assert [item["id"] for item in page["items"]][:2] == [exact, partial]

    def test_like_wildcards_match_literally(self, db):
        percent = _library_conversation(db, title="100% coverage")
        other = _library_conversation(db, title="1000 coverage")

        page = db.search_library_conversations_page(query="100%", limit=10, offset=0)
        ids = {item["id"] for item in page["items"]}
        assert percent in ids
        assert other not in ids

    def test_fts_operator_input_is_inert(self, db):
        _library_conversation(
            db, title="alpha", messages=[("user", "hello world")]
        )
        for query in ('foo"bar', "foo*", "OR AND NOT", "(unclosed"):
            page = db.search_library_conversations_page(
                query=query, limit=10, offset=0
            )
            assert isinstance(page["total"], int)
            assert isinstance(page["items"], list)

    def test_soft_deleted_message_does_not_match(self, db):
        conv_id = _library_conversation(
            db, title="plain", messages=[("user", "vanishing needle")]
        )
        row = db.execute_query(
            "SELECT id FROM messages WHERE conversation_id = ?", (conv_id,)
        ).fetchone()
        db.soft_delete_message(row["id"], expected_version=1)

        page = db.search_library_conversations_page(query="needle", limit=10, offset=0)
        assert page["total"] == 0

    def test_soft_deleted_conversation_does_not_match(self, db):
        _library_conversation(
            db, title="needle title", messages=[("user", "needle body")]
        )
        drop = _library_conversation(
            db, title="needle elsewhere", messages=[("user", "needle body")]
        )
        db.soft_delete_conversation(drop, expected_version=1)

        page = db.search_library_conversations_page(query="needle", limit=10, offset=0)
        assert page["total"] == 1


class TestGetLibraryConversationMessages:
    def test_message_page_has_exact_totals_and_stable_order(self, db):
        conv_id = _library_conversation(
            db,
            title="paged",
            messages=[("user", f"body-{index}") for index in range(5)],
        )

        first = db.get_library_conversation_messages(
            conv_id, message_limit=2, message_offset=0
        )
        assert first["id"] == conv_id
        assert first["title"] == "paged"
        assert first["message_total"] == 5
        assert first["message_offset"] == 0
        assert first["returned_message_count"] == 2
        assert first["has_more"] is True
        assert first["next_message_offset"] == 2
        assert [message["text"] for message in first["messages"]] == [
            "body-0",
            "body-1",
        ]

        last = db.get_library_conversation_messages(
            conv_id, message_limit=2, message_offset=4
        )
        assert last["message_total"] == 5
        assert last["returned_message_count"] == 1
        assert last["has_more"] is False
        assert last["next_message_offset"] is None
        assert [message["text"] for message in last["messages"]] == ["body-4"]

    def test_message_projection_is_text_only(self, db):
        conv_id = db.add_conversation({"title": "img"})
        db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "user",
                "content": "see attached",
                "image_data": b"\x89PNG-binary-blob",
                "image_mime_type": "image/png",
            }
        )

        detail = db.get_library_conversation_messages(conv_id)
        assert detail["message_total"] == 1
        message = detail["messages"][0]
        assert set(message) == {
            "id",
            "sender",
            "timestamp",
            "revision",
            "total_chars",
            "char_start",
            "returned_chars",
            "has_more",
            "text",
        }
        assert not any(isinstance(value, bytes) for value in message.values())

    def test_page_windows_each_message_to_max_chars(self, db):
        conv_id = _library_conversation(
            db, title="win", messages=[("user", "y" * 100)]
        )

        detail = db.get_library_conversation_messages(conv_id, max_chars=10)
        message = detail["messages"][0]
        assert message["text"] == "y" * 10
        assert message["total_chars"] == 100
        assert message["char_start"] == 0
        assert message["returned_chars"] == 10
        assert message["has_more"] is True

    def test_long_message_windows_continue_where_previous_ended(self, db):
        content = "x" * 40_000 + "TAIL-MARKER"
        conv_id = db.add_conversation({"title": "long"})
        msg_id = db.add_message(
            {"conversation_id": conv_id, "sender": "user", "content": content}
        )

        first = db.get_library_conversation_messages(
            conv_id, message_id=msg_id, char_start=0, max_chars=8000
        )
        assert first["message_total"] == 1
        assert first["returned_message_count"] == 1
        window = first["messages"][0]
        assert window["total_chars"] == len(content)
        assert window["char_start"] == 0
        assert window["returned_chars"] == 8000
        assert window["has_more"] is True
        assert window["text"] == content[:8000]

        second = db.get_library_conversation_messages(
            conv_id, message_id=msg_id, char_start=8000, max_chars=8000
        )
        continuation = second["messages"][0]
        assert continuation["char_start"] == 8000
        assert continuation["text"] == content[8000:16000]

        tail = db.get_library_conversation_messages(
            conv_id, message_id=msg_id, char_start=40_000, max_chars=8000
        )
        tail_window = tail["messages"][0]
        assert tail_window["text"] == "TAIL-MARKER"
        assert tail_window["has_more"] is False

    def test_revision_changes_when_content_changes(self, db):
        conv_id = db.add_conversation({"title": "rev"})
        msg_id = db.add_message(
            {"conversation_id": conv_id, "sender": "user", "content": "before"}
        )
        before = db.get_library_conversation_messages(conv_id)["messages"][0][
            "revision"
        ]

        assert db.update_message(msg_id, {"content": "after"}, expected_version=1)
        after = db.get_library_conversation_messages(conv_id)["messages"][0][
            "revision"
        ]
        assert before != after

    def test_include_rag_context_is_always_false(self, db):
        conv_id = _library_conversation(
            db, title="ctx", messages=[("user", "hello")]
        )

        detail = db.get_library_conversation_messages(conv_id)
        assert detail["include_rag_context"] is False
        assert all(
            "rag_context" not in message and "citations" not in message
            for message in detail["messages"]
        )

    def test_soft_deleted_messages_are_excluded(self, db):
        conv_id = _library_conversation(
            db, title="del", messages=[("user", "stays"), ("user", "goes")]
        )
        row = db.execute_query(
            "SELECT id FROM messages WHERE conversation_id = ? AND content = 'goes'",
            (conv_id,),
        ).fetchone()
        db.soft_delete_message(row["id"], expected_version=1)

        detail = db.get_library_conversation_messages(conv_id)
        assert detail["message_total"] == 1
        assert [message["text"] for message in detail["messages"]] == ["stays"]

    def test_missing_conversation_returns_none(self, db):
        assert db.get_library_conversation_messages("no-such-conversation") is None
