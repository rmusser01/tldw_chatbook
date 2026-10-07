"""Message subtree delete/restore: SQL variable bounds and the descent plan.

TASK-33628.11, two halves.

* Variable bounds. A subtree delete can reach every row of a conversation,
  but one statement may bind only ``SQLITE_LIMIT_VARIABLE_NUMBER`` variables.
  TASK-33628.6 (PR #2965) batched the three id-list helpers on the delete and
  Undo paths -- the delete-proof capture inside ``soft_delete_message_subtree``,
  ``get_message_tombstones`` and ``restore_message_subtree`` -- through
  ``CharactersRAGDB._bounded_id_batches``. The boundary test below pins them at
  exactly the limit and one past it.
* Descent plan. Editing a message's content without ``preserve_descendants``
  tombstones every live descendant through a recursive CTE in
  ``CharactersRAGDB._update_message_uncoordinated``. Its steps filtered on an
  indexable ``conversation_id = ?``. No ChaChaNotes database runs ``ANALYZE``,
  so the planner has no ``sqlite_stat1``; with that term indexable it chose
  the ``(conversation_id, id)`` index and scanned the whole conversation once
  per descendant instead of searching by parent: 13.0 s to edit the second
  message of a 3,000-message chain, 0.15 s after. TASK-33628.6 fixed the same
  shape in ``soft_delete_message_subtree``.
"""

from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

#: The parent-link index both CTE steps search. ``idx_msgs_parent`` leads with
#: the same column; with no statistics the planner picks this one, and a plan
#: change that moves off it should be noticed here.
PARENT_INDEX = "idx_messages_variants_by_parent"


def _chain(db: CharactersRAGDB, conversation_id: str, count: int) -> list[str]:
    ids: list[str] = []
    parent = None
    for index in range(count):
        parent = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user" if index % 2 == 0 else "assistant",
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"{conversation_id} message {index}",
                "parent_message_id": parent,
            }
        )
        ids.append(parent)
    return ids


def _live(db: CharactersRAGDB, ids: list[str]) -> list[bool]:
    return [db.get_message_by_id(message_id) is not None for message_id in ids]


def _edit_statements(db: CharactersRAGDB, message_id: str) -> set[str]:
    """Edit ``message_id`` and return the descendants CTE statements it ran."""
    conn = db.get_connection()
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        current = db.get_message_by_id(message_id)
        db.update_message(message_id, {"content": "edited"}, current["version"])
    finally:
        conn.set_trace_callback(None)
    # Trigger sub-programs re-report their outer statement; dedupe them.
    return {s for s in statements if s.lstrip().startswith("WITH RECURSIVE descendants")}


def test_edit_descendants_delete_searches_children_by_parent_without_statistics():
    db = CharactersRAGDB(":memory:", "descendants-plan")
    conversation_id = db.add_conversation({"title": "Descendants plan"})
    ids = _chain(db, conversation_id, 6)
    conn = db.get_connection()
    assert (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'"
        ).fetchone()
        is None
    )

    statements = _edit_statements(db, ids[1])

    assert len(statements) == 2  # the SELECT that captures proofs, and the UPDATE
    for statement in statements:
        plan = [str(row[3]) for row in conn.execute("EXPLAIN QUERY PLAN " + statement)]
        assert plan, statement
        steps = [
            detail
            for detail in plan
            if detail.split()[1:2] in (["messages"], ["child"])
            and "(id=?)" not in detail  # the UPDATE's own rowid lookup
        ]
        # Both the seed step and the recursive step reach rows by parent.
        assert len(steps) == 2, plan
        for detail in steps:
            assert detail.startswith("SEARCH "), plan
            assert f"USING INDEX {PARENT_INDEX} (parent_message_id=?)" in detail, plan
        assert "idx_messages_conversation_id_id" not in " ".join(steps), plan
    assert _live(db, ids) == [True, True, False, False, False, False]


def test_edit_descendants_delete_stays_inside_the_conversation():
    """Unary ``+`` changes the plan, not the predicate: scope still holds."""
    db = CharactersRAGDB(":memory:", "descendants-plan")
    conversation_id = db.add_conversation({"title": "Descendants plan"})
    ids = _chain(db, conversation_id, 4)
    other = db.add_conversation({"title": "Other"})
    # A malformed cross-conversation child of ids[1] must not be tombstoned.
    stray = db.add_message(
        {
            "conversation_id": other,
            "sender": "user",
            "role": "user",
            "content": "stray",
            "parent_message_id": ids[1],
        }
    )

    _edit_statements(db, ids[1])

    assert _live(db, ids) == [True, True, False, False]
    assert _live(db, [stray]) == [True]


def test_preserved_descendants_are_not_touched():
    db = CharactersRAGDB(":memory:", "descendants-plan")
    conversation_id = db.add_conversation({"title": "Descendants plan"})
    ids = _chain(db, conversation_id, 4)
    current = db.get_message_by_id(ids[1])

    db.update_message(
        ids[1], {"content": "edited"}, current["version"], preserve_descendants=True
    )

    assert _live(db, ids) == [True, True, True, True]


@pytest.mark.parametrize("extra", [0, 1], ids=["at-limit", "one-past"])
def test_delete_and_restore_bind_ids_within_the_variable_limit(extra):
    """Delete, tombstone read and Undo each commit whole at the limit edge."""
    limit = 16
    db = CharactersRAGDB(":memory:", "subtree-bounds")
    conversation_id = db.add_conversation({"title": "Bounds"})
    # Flat rows (every parent NULL): only the seeds reach them, one each.
    ids = [
        db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": f"flat {index}",
                "parent_message_id": None,
            }
        )
        for index in range(limit + extra)
    ]
    conn = db.get_connection()
    default_limit = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
    try:
        first = db.get_message_by_id(ids[0])
        rows = db.soft_delete_message_subtree(
            ids[0], first["version"], subtree_message_ids=ids
        )
        tombstones = db.get_message_tombstones(ids)
        restored = db.restore_message_subtree(
            [(row["message_id"], row["version"]) for row in tombstones]
        )
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, default_limit)

    assert len(rows) == len(tombstones) == len(restored) == len(ids)
    assert _live(db, ids) == [True] * len(ids)
