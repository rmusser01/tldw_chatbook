"""``CharactersRAGDB.get_root_message_rows_page``: one page of parentless rows.

TASK-33628.7 review (PR #3004). A Console Delete that removes chained legacy
flat roots also removes the never-shown roots after them, so it reads the
conversation's parentless rows in root order. It used to read every live row
of the conversation and keep the parentless ones. This reader returns only the
parentless rows, a bounded page at a time after the previous page's last id,
with the few columns the hidden-row test needs -- no message text.

The order must be the one resume reads roots in, or the delete would place
"after the first deleted root" somewhere else than the transcript did: resume
filters ``get_message_tree_rows_for_conversation``, which orders by timestamp
and leaves equal timestamps in index order -- insertion order.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError

#: The ``(conversation_id, timestamp)`` index both page statements walk.
CONVERSATION_TIME_INDEX = "idx_msgs_conv_ts"


def _insert(db: CharactersRAGDB, conversation_id: str, rows) -> None:
    """Insert ``(id, parent, timestamp, content)`` rows in the given order."""
    for message_id, parent, timestamp, content in rows:
        db.add_message(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": content or "placeholder",
                "parent_message_id": parent,
                "timestamp": timestamp,
            }
        )
        if not content:
            # add_message refuses an empty row; older builds saved them.
            db.update_message(message_id, {"content": ""}, 1, preserve_descendants=True)


def _all_pages(db: CharactersRAGDB, conversation_id: str, limit: int) -> list:
    pages: list[list[dict]] = []
    after = None
    while True:
        page = db.get_root_message_rows_page(
            conversation_id, after_message_id=after, limit=limit
        )
        assert len(page) <= limit
        pages.append(page)
        if len(page) < limit:
            return pages
        after = page[-1]["id"]


def _tied_conversation(db: CharactersRAGDB) -> str:
    """Roots whose timestamps tie, inserted out of id order, among other rows."""
    conversation_id = db.add_conversation({"title": "Roots"})
    early, late = "2026-09-30T00:00:00.000000+00:00", "2026-09-30T00:00:01.000000+00:00"
    _insert(
        db,
        conversation_id,
        [
            ("r5", None, late, "r5"),
            ("r2", None, early, "r2"),
            ("r9", None, early, "r9"),
            ("k1", "r2", early, "k1"),  # parent-linked: never a root
            ("r1", None, early, ""),  # an empty row is still a root
            ("r7", None, late, "r7"),
            ("r3", None, early, "r3"),
            ("gone", None, early, "gone"),
        ],
    )
    gone = db.get_message_by_id("gone")
    db.soft_delete_message("gone", gone["version"])
    other = db.add_conversation({"title": "Other"})
    _insert(db, other, [("o1", None, early, "o1"), ("o2", None, late, "o2")])
    return conversation_id


@pytest.mark.parametrize("limit", [1, 2, 3, 6, 7, 100])
def test_pages_hold_the_live_roots_in_the_order_resume_reads_them(limit):
    db = CharactersRAGDB(":memory:", "root-pages")
    conversation_id = _tied_conversation(db)
    resume_order = [
        row["id"]
        for row in db.get_message_tree_rows_for_conversation(conversation_id)
        if row["parent_message_id"] is None
    ]
    # Ties stay in insertion order, not id order.
    assert resume_order == ["r2", "r9", "r1", "r3", "r5", "r7"]

    pages = _all_pages(db, conversation_id, limit)

    assert [row["id"] for page in pages for row in page] == resume_order
    # A page shorter than the limit ends the read; at most one is empty.
    assert all(len(page) == limit for page in pages[:-1])


def test_rows_carry_what_decides_whether_resume_shows_them_and_no_text():
    db = CharactersRAGDB(":memory:", "root-pages")
    conversation_id = _tied_conversation(db)

    rows = {
        row["id"]: row
        for row in db.get_root_message_rows_page(conversation_id, limit=100)
    }

    assert set(rows["r1"]) == {
        "id",
        "sender",
        "role",
        "metadata_json",
        "has_content",
        "has_image",
        "has_generation_state",
        "has_provider_continuation",
    }
    assert (rows["r1"]["has_content"], rows["r3"]["has_content"]) == (0, 1)
    assert rows["r3"]["has_image"] == 0
    assert rows["r3"]["role"] == "user"


def test_a_deleted_conversation_and_an_unknown_cursor_read_nothing():
    db = CharactersRAGDB(":memory:", "root-pages")
    conversation_id = _tied_conversation(db)

    assert (
        db.get_root_message_rows_page(
            conversation_id, after_message_id="missing", limit=10
        )
        == []
    )
    conversation = db.get_conversation_by_id(conversation_id)
    db.soft_delete_conversation(conversation_id, conversation["version"])
    assert db.get_root_message_rows_page(conversation_id, limit=10) == []


@pytest.mark.parametrize("limit", [0, -1])
def test_a_page_must_hold_at_least_one_row(limit):
    db = CharactersRAGDB(":memory:", "root-pages")
    with pytest.raises(InputError):
        db.get_root_message_rows_page("any", limit=limit)


def test_both_page_statements_walk_the_conversation_index_without_statistics():
    """First and later pages search the conversation's timestamp range.

    No ChaChaNotes database runs ``ANALYZE``. With no ``sqlite_stat1`` the
    planner could take a parent index for ``parent_message_id IS NULL`` --
    every conversation's flat rows -- and sort them; the statements keep the
    parent column off any index, and the later pages start their range at the
    previous page's last row instead of rescanning from the first.
    """
    db = CharactersRAGDB(":memory:", "root-pages")
    conversation_id = _tied_conversation(db)
    conn = db.get_connection()
    assert (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'"
        ).fetchone()
        is None
    )
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        first = db.get_root_message_rows_page(conversation_id, limit=2)
        db.get_root_message_rows_page(
            conversation_id, after_message_id=first[-1]["id"], limit=2
        )
    finally:
        conn.set_trace_callback(None)

    pages = [s for s in statements if "FROM messages m" in s]
    assert len(pages) == 2, statements
    plans = [
        [str(row[3]) for row in conn.execute("EXPLAIN QUERY PLAN " + statement)]
        for statement in pages
    ]
    for plan, expected in zip(
        plans,
        ["(conversation_id=?)", "(conversation_id=? AND timestamp>?)"],
    ):
        steps = [detail for detail in plan if detail.startswith("SEARCH m ")]
        assert steps == [
            f"SEARCH m USING INDEX {CONVERSATION_TIME_INDEX} {expected}"
        ], plan
        assert not any("TEMP B-TREE" in detail for detail in plan), plan
