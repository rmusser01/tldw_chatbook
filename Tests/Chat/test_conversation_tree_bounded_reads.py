"""Task 7 (wave 4): the conversation tree read is bounded to the root page.

``ChatConversationService.get_conversation_tree`` asks for ONE page of
``root_limit`` root threads. Its DB read,
``get_message_tree_rows_for_conversation``, nevertheless fetched EVERY live
message row of the conversation (text + JSON sidecars, no LIMIT) and the
service sliced the requested root page out of that full fetch in Python.
The new paged DB read,
``get_message_tree_rows_for_conversation_page``, pushes the root
LIMIT/OFFSET down into SQL and fetches only the page's roots plus their
descendants (one recursive CTE), together with the live root count --
three bounded statements in one transaction.

These tests pin the equivalence:

* ``test_paged_tree_reads_match_legacy_full_fetch_*`` -- golden matrix:
  for every (order x limit x offset) combination, the paged API returns
  exactly the rows (ids, per-parent order, full column values) that the
  old full fetch + Python slicing would render, and the rendered service
  tree (``root_threads`` + ``pagination``) is identical to the legacy
  in-memory path, replica included below verbatim from today's service.
* ``test_paged_tree_row_fetch_is_bounded`` -- trace evidence: on the
  ~300-message fixture the old path fetches every live row while the new
  path fetches only the page subtree plus one COUNT row, in 3 statements.
* ``test_paged_tree_handles_empty_and_edge_coordinates`` -- LIMIT 0,
  negative LIMIT ("no limit"), negative OFFSET, offsets past the end,
  empty conversations, and conversations that don't exist.

The fixture is deliberately hostile: 60 roots, a 15-level deep chain, a
120-sibling wide branch, timestamps SHUFFLED against structure (children
routinely timestamped before parents), soft-deleted rows inside and at
the root, image-bearing nodes, and one row whose parent lives in ANOTHER
conversation (never rendered by either path).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    database = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test-client")
    try:
        yield database
    finally:
        database.close_connection()


def _ts(index: int) -> str:
    """Strictly increasing, lexicographically ordered timestamps."""
    return f"2026-01-01T00:00:00.{index:06d}Z"


class TreeFixture:
    """~300 live messages under 60 roots, structure hostile to reordering.

    * ``root-deep`` carries a 15-level chain (depth well past 10).
    * ``root-wide`` carries 120 direct siblings (past 100).
    * ``root-spread`` carries 10 children x 6 grandchildren.
    * roots 4..20 carry 3 children each; roots 21..60 are leaves.
    * one extra root and one wide child are soft-deleted (excluded).
    * one row's parent lives in ANOTHER conversation (never rendered).
    * timestamps are a fixed-seed shuffle of distinct values, so children
      are routinely timestamped BEFORE their parents and sibling order is
      unrelated to insertion order (rowid order differs from timestamp
      order throughout).
    """

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db
        self.conversation_id = db.add_conversation({"title": "tree fixture"})
        other_id = db.add_conversation({"title": "other conversation"})
        other_root = self._add(other_id, None, "other-root", timestamp=_ts(90_000))

        # (parent key or None, child key); listed parents-before-children
        # so insertion is FK-safe. "FOREIGN" maps to the other
        # conversation's root: the child row lives in THIS conversation
        # but is unreachable from this conversation's roots.
        edges: list[tuple[str | None, str]] = []
        edges.append((None, "root-deep"))
        deep = "root-deep"
        for level in range(14):
            child = f"deep-{level}"
            edges.append((deep, child))
            deep = child
        edges.append((None, "root-wide"))
        for index in range(120):
            edges.append(("root-wide", f"wide-{index}"))
        edges.append((None, "root-spread"))
        for index in range(10):
            spread = f"spread-{index}"
            edges.append(("root-spread", spread))
            for grand in range(6):
                edges.append((spread, f"spread-{index}-{grand}"))
        for index in range(4, 21):
            edges.append((None, f"root-{index:02d}"))
            for child in range(3):
                edges.append((f"root-{index:02d}", f"child-{index:02d}-{child}"))
        for index in range(21, 61):
            edges.append((None, f"root-{index:02d}"))
        # Soft-deleted rows: one root and one wide child -- both excluded.
        edges.append((None, "root-deleted"))
        edges.append(("root-wide", "wide-deleted"))
        # Cross-conversation parent (see above).
        edges.append(("FOREIGN", "foreign-parent-child"))

        rng_ts = random.Random(42)
        timestamps = [_ts(index) for index in range(len(edges))]
        rng_ts.shuffle(timestamps)

        self.ids: dict[str, str] = {"FOREIGN": other_root}
        image_at = {"root-05", "deep-7", "wide-11"}
        with db.transaction():  # one commit for the whole fixture
            for edge_index, (parent_key, key) in enumerate(edges):
                parent_id = self.ids[parent_key] if parent_key is not None else None
                self.ids[key] = self._add(
                    self.conversation_id,
                    parent_id,
                    key,
                    timestamp=timestamps[edge_index],
                    image=b"\x89PNG-raw" if key in image_at else None,
                )

        db.soft_delete_message(self.ids["root-deleted"], 1)
        db.soft_delete_message(self.ids["wide-deleted"], 1)

    def _add(
        self,
        conversation_id: str,
        parent_message_id: str | None,
        content: str,
        *,
        timestamp: str,
        image: bytes | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": content,
            "parent_message_id": parent_message_id,
            "timestamp": timestamp,
        }
        if image is not None:
            payload["image_data"] = image
            payload["image_mime_type"] = "image/png"
        message_id = self.db.add_message(payload)
        assert message_id is not None
        return message_id


@pytest.fixture()
def tree(db):
    return TreeFixture(db)


# ---------------------------------------------------------------------------
# legacy oracle (replica of today's full-fetch + Python-slice path)
# ---------------------------------------------------------------------------


def _legacy_partition(
    rows: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[Any, list[Mapping[str, Any]]]]:
    """Verbatim replica of get_conversation_tree's partition step."""
    children_by_parent: dict[Any, list[Mapping[str, Any]]] = {}
    root_rows: list[Mapping[str, Any]] = []
    for row in rows:
        parent_id = row.get("parent_message_id")
        if parent_id is None:
            root_rows.append(row)
        else:
            children_by_parent.setdefault(parent_id, []).append(row)
    return root_rows, children_by_parent


def _legacy_slice(
    root_rows: list[Mapping[str, Any]],
    root_offset: int,
    root_limit: int,
) -> list[Mapping[str, Any]]:
    """Verbatim replica of get_conversation_tree's LIMIT/OFFSET slicing."""
    effective_offset = max(0, root_offset)
    if root_limit < 0:
        return root_rows[effective_offset:]
    return root_rows[effective_offset : effective_offset + root_limit]


def _legacy_render(
    db: CharactersRAGDB,
    conversation_id: str,
    *,
    root_offset: int,
    root_limit: int,
    order_by_timestamp: str,
    depth_cap: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The full pre-change get_conversation_tree shape, from the old read."""
    rows = db.get_message_tree_rows_for_conversation(
        conversation_id, order_by_timestamp=order_by_timestamp
    )
    root_rows, children_by_parent = _legacy_partition(rows)
    total_root_threads = len(root_rows)
    paged_root_rows = _legacy_slice(root_rows, root_offset, root_limit)
    service = ChatConversationService(db)
    root_threads, image_pending = service._build_message_tree(
        paged_root_rows, children_by_parent, depth_cap=depth_cap
    )
    service._hydrate_tree_images(image_pending)
    pagination = {
        "limit": root_limit,
        "offset": root_offset,
        "total_root_threads": total_root_threads,
        "has_more": root_offset + len(paged_root_rows) < total_root_threads,
    }
    return root_threads, pagination


def _collect_tree_ids(nodes: list[Mapping[str, Any]], out: list[str]) -> None:
    for node in nodes:
        out.append(str(node["id"]))
        _collect_tree_ids(node.get("children", []), out)


# ---------------------------------------------------------------------------
# fixture sanity
# ---------------------------------------------------------------------------


def test_fixture_shape_is_hostile_as_required(db, tree):
    rows = db.get_message_tree_rows_for_conversation(tree.conversation_id)
    assert len(rows) >= 300  # ~300 live rows incl. the foreign-parent child

    root_rows, children_by_parent = _legacy_partition(rows)
    assert len(root_rows) == 60  # 60 live roots; root-deleted excluded
    assert tree.ids["root-deleted"] not in {r["id"] for r in rows}
    assert tree.ids["wide-deleted"] not in {r["id"] for r in rows}

    by_id = {r["id"]: r for r in rows}
    # Wide branch: 120 live siblings under root-wide (wide-deleted gone).
    assert len(children_by_parent[tree.ids["root-wide"]]) == 120
    # Deep chain: 15 levels including the root.
    depth = 1
    cursor = tree.ids["root-deep"]
    while children_by_parent.get(cursor):
        assert len(children_by_parent[cursor]) == 1
        cursor = children_by_parent[cursor][0]["id"]
        depth += 1
    assert depth == 15
    # Mixed timestamps: many children timestamped BEFORE their parents.
    # (The foreign-parent child's parent is not in this conversation's
    # rows, so it is skipped -- it is never rendered by either path.)
    inverted = sum(
        1
        for parent, kids in children_by_parent.items()
        for kid in kids
        if parent in by_id and str(kid["timestamp"]) < str(by_id[parent]["timestamp"])
    )
    assert inverted > 50
    # Timestamps are distinct (no tie ambiguity in either path).
    assert len({str(r["timestamp"]) for r in rows}) == len(rows)
    # Images present and flagged, not inlined.
    assert sum(1 for r in rows if r["has_image"]) == 3
    assert all("image_data" not in r for r in rows)


# ---------------------------------------------------------------------------
# golden equivalence matrix
# ---------------------------------------------------------------------------


ORDERS = ["ASC", "DESC"]
LIMITS = [10, 25, 50, 100]
OFFSETS = [0, 1, 7, 30, 50, 59, 60, 75]  # first/middle/last/overflow


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("offset", OFFSETS)
def test_paged_tree_reads_match_legacy_full_fetch(db, tree, offset, limit, order):
    conversation_id = tree.conversation_id
    all_rows = db.get_message_tree_rows_for_conversation(
        conversation_id, order_by_timestamp=order
    )
    root_rows, children_by_parent = _legacy_partition(all_rows)
    expected_page = _legacy_slice(root_rows, offset, limit)

    rows, total_roots = db.get_message_tree_rows_for_conversation_page(
        conversation_id,
        root_offset=offset,
        root_limit=limit,
        order_desc=(order == "DESC"),
    )

    # Total root count matches the legacy len(root_rows).
    assert total_roots == len(root_rows)
    # Rows-level equivalence: page roots, in page order.
    new_roots = [r for r in rows if r["parent_message_id"] is None]
    assert [r["id"] for r in new_roots] == [r["id"] for r in expected_page]
    # Every non-root row's parent is inside the fetched subtree.
    subtree_ids = {r["id"] for r in rows}
    assert all(
        r["parent_message_id"] in subtree_ids for r in rows if r["parent_message_id"]
    )
    # Per-parent buckets identical in order and in full column values.
    old_by_id = {r["id"]: r for r in all_rows}
    new_buckets: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["parent_message_id"]:
            new_buckets[row["parent_message_id"]].append(row)
    for parent_id, new_bucket in new_buckets.items():
        assert [r["id"] for r in new_bucket] == [
            r["id"] for r in children_by_parent[parent_id]
        ]
    for row in rows:
        assert row == old_by_id[row["id"]]


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(
    "offset,limit,depth_cap",
    [
        (0, 50, 50),  # service defaults
        (30, 25, 50),  # middle page
        (0, 10_000, 10_000),  # the resume/fork "full tree" caps
        (0, 50, 5),  # depth-cap truncation still identical
        (-7, 50, 50),  # negative offset clamps to 0
        (0, -1, 50),  # negative limit means "no limit"
        (0, 0, 50),  # LIMIT 0: empty page, counts still reported
        (59, 50, 50),  # last root exactly
        (60, 50, 50),  # one past the end: empty page
        (500, 50, 50),  # far overflow
    ],
)
def test_paged_service_render_matches_legacy_render(
    db, tree, offset, limit, depth_cap, order
):
    service = ChatConversationService(db)
    result = service.get_conversation_tree(
        tree.conversation_id,
        root_limit=limit,
        root_offset=offset,
        order_by_timestamp=order,
        depth_cap=depth_cap,
    )
    legacy_threads, legacy_pagination = _legacy_render(
        db,
        tree.conversation_id,
        root_offset=offset,
        root_limit=limit,
        order_by_timestamp=order,
        depth_cap=depth_cap,
    )
    assert result["conversation"]["id"] == tree.conversation_id
    assert result["pagination"] == legacy_pagination
    assert result["depth_cap"] == depth_cap
    assert result["root_threads"] == legacy_threads
    # The rendered node ids come only from the page's subtree.
    rendered_ids = _collect_tree_ids(result["root_threads"], [])
    legacy_ids = _collect_tree_ids(legacy_threads, [])
    assert rendered_ids == legacy_ids


# ---------------------------------------------------------------------------
# row-fetch bound (trace evidence)
# ---------------------------------------------------------------------------


def test_paged_tree_row_fetch_is_bounded(db, tree):
    conversation_id = tree.conversation_id
    old_rows = db.get_message_tree_rows_for_conversation(conversation_id)
    old_row_count = len(old_rows)

    statements: list[str] = []
    conn = db.get_connection()
    conn.set_trace_callback(statements.append)
    try:
        rows, total_roots = db.get_message_tree_rows_for_conversation_page(
            conversation_id, root_offset=0, root_limit=50
        )
    finally:
        conn.set_trace_callback(None)

    # Three bounded statements, one transaction (BEGIN/COMMIT filtered).
    work = [
        s
        for s in statements
        if not s.strip().upper().startswith(("BEGIN", "COMMIT", "ROLLBACK"))
    ]
    assert len(work) == 3
    assert total_roots == 60

    # The fetched rows are exactly the first page's subtree.
    root_rows, children_by_parent = _legacy_partition(old_rows)
    expected_page = _legacy_slice(root_rows, 0, 50)
    expected_subtree: list[str] = []
    stack = [r["id"] for r in reversed(expected_page)]
    while stack:
        node_id = stack.pop()
        expected_subtree.append(node_id)
        stack.extend(r["id"] for r in reversed(children_by_parent.get(node_id, ())))
    assert sorted(r["id"] for r in rows) == sorted(expected_subtree)

    # Evidence: old path reads every live row of the conversation; the new
    # path reads only the page subtree plus the single COUNT row.
    new_row_count = len(rows) + 1
    assert new_row_count < old_row_count, (
        f"new path must read fewer rows: old={old_row_count} new={new_row_count}"
    )


# ---------------------------------------------------------------------------
# empty / absent conversations
# ---------------------------------------------------------------------------


def test_paged_tree_handles_empty_and_absent_conversations(db, tree):
    empty_id = db.add_conversation({"title": "empty"})
    rows, total_roots = db.get_message_tree_rows_for_conversation_page(
        empty_id, root_offset=0, root_limit=50
    )
    assert rows == []
    assert total_roots == 0

    rows, total_roots = db.get_message_tree_rows_for_conversation_page(
        "no-such-conversation", root_offset=0, root_limit=50
    )
    assert rows == []
    assert total_roots == 0

    service = ChatConversationService(db)
    result = service.get_conversation_tree(empty_id)
    assert result["root_threads"] == []
    assert result["pagination"] == {
        "limit": 50,
        "offset": 0,
        "total_root_threads": 0,
        "has_more": False,
    }
