"""TASK-22206: the conversation-tree build must be O(N), iterative, BLOB-lazy.

The pre-22206 ``ChatConversationService._build_message_tree`` recursed once per
message and issued one ``get_messages_for_conversation_by_parent_ids`` call per
node; with ``sqlite_stat1`` absent every one of those calls post-filtered a full
``idx_msgs_conv_ts`` scan, hydrating every ``image_data`` BLOB N times, and the
recursion depth equalled the conversation length (RecursionError at ~980-message
linear chains -- the default recursion limit is 1000).

These tests pin the replacement:

* ``test_2000_message_linear_chain_resumes_without_recursion_error`` -- red
  before the fix (RecursionError in both the service build and the resume
  flattener) -- drives the full resume tree path at 2x the old breaking depth.
* ``test_new_build_matches_legacy_recursive_build_on_branched_fixture`` -- the
  legacy recursive algorithm is ported verbatim below as the oracle and both
  builds run against the same real SQLite file; output must be identical
  (sibling order, parenthood, roots, deep branch, image bytes, pagination,
  depth-cap truncation, DESC ordering).
* ``test_tree_build_query_count_is_independent_of_message_count`` -- red before
  the fix (the statement count grew linearly with N) -- a
  ``set_trace_callback`` probe asserts O(1) statements per
  ``get_conversation_tree`` and that image hydration is exactly one extra
  batched statement, present only when the conversation actually has images.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pytest

from tldw_chatbook.Chat.chat_conversation_service import (
    ChatConversationService,
    normalize_message_row,
)
from tldw_chatbook.Chat.console_conversation_hydration import (
    console_messages_from_conversation_tree,
)
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


def _add_message(
    db: CharactersRAGDB,
    conversation_id: str,
    *,
    index: int,
    parent_message_id: str | None,
    content: str | None = None,
    sender: str | None = None,
    image_data: bytes | None = None,
) -> str:
    payload: dict[str, Any] = {
        "conversation_id": conversation_id,
        "sender": sender or ("user" if index % 2 == 0 else "assistant"),
        "content": content if content is not None else f"message {index}",
        "parent_message_id": parent_message_id,
        "timestamp": _ts(index),
    }
    if image_data is not None:
        payload["image_data"] = image_data
        payload["image_mime_type"] = "image/png"
    message_id = db.add_message(payload)
    assert message_id is not None
    return message_id


def _add_linear_chain(
    db: CharactersRAGDB, conversation_id: str, count: int
) -> list[str]:
    ids: list[str] = []
    parent: str | None = None
    with db.transaction():  # one commit for the whole fixture
        for index in range(count):
            parent = _add_message(
                db, conversation_id, index=index, parent_message_id=parent
            )
            ids.append(parent)
    return ids


def _build_branched_fixture(db: CharactersRAGDB, conversation_id: str) -> dict[str, str]:
    """Two roots; root one carries siblings, an image, and a deep branch.

    root1                      root2
      +- a (image)
      |    +- a1
      |    +- a2
      |         +- deep0 .. deep9   (10-deep chain)
      +- b
    """
    ids: dict[str, str] = {}
    with db.transaction():
        ids["root1"] = _add_message(
            db, conversation_id, index=0, parent_message_id=None, content="root one"
        )
        ids["a"] = _add_message(
            db,
            conversation_id,
            index=1,
            parent_message_id=ids["root1"],
            content="branch a",
            image_data=b"\x89PNG-fake-bytes-a",
        )
        ids["b"] = _add_message(
            db,
            conversation_id,
            index=2,
            parent_message_id=ids["root1"],
            content="branch b",
        )
        ids["a1"] = _add_message(
            db, conversation_id, index=3, parent_message_id=ids["a"], content="a1"
        )
        ids["a2"] = _add_message(
            db, conversation_id, index=4, parent_message_id=ids["a"], content="a2"
        )
        parent = ids["a2"]
        for depth in range(10):
            parent = _add_message(
                db,
                conversation_id,
                index=5 + depth,
                parent_message_id=parent,
                content=f"deep {depth}",
            )
            ids[f"deep{depth}"] = parent
        ids["root2"] = _add_message(
            db, conversation_id, index=50, parent_message_id=None, content="root two"
        )
    return ids


# ---------------------------------------------------------------------------
# the legacy algorithm, ported verbatim as the equivalence oracle
# ---------------------------------------------------------------------------


def _legacy_build_message_tree(
    db: Any,
    conversation_id: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    order_by_timestamp: str,
    depth_cap: int,
    depth: int,
    seen_message_ids: set[str],
) -> list[dict[str, Any]]:
    """The pre-TASK-22206 recursive one-query-per-node build, verbatim."""
    nodes: list[dict[str, Any]] = []
    for row in rows:
        message_id = row.get("id")
        normalized_row = normalize_message_row(row)
        if normalized_row is None:
            continue
        if message_id is not None and message_id in seen_message_ids:
            normalized_row["children"] = []
            normalized_row["truncated"] = True
            nodes.append(normalized_row)
            continue

        next_seen = set(seen_message_ids)
        if message_id is not None:
            next_seen.add(message_id)

        if depth >= depth_cap:
            normalized_row["children"] = []
            normalized_row["truncated"] = True
            nodes.append(normalized_row)
            continue

        child_rows = db.get_messages_for_conversation_by_parent_ids(
            conversation_id,
            [message_id] if message_id is not None else [],
            order_by_timestamp=order_by_timestamp,
            include_deleted_conversation=False,
        )
        normalized_row["children"] = _legacy_build_message_tree(
            db,
            conversation_id,
            child_rows,
            order_by_timestamp=order_by_timestamp,
            depth_cap=depth_cap,
            depth=depth + 1,
            seen_message_ids=next_seen,
        )
        normalized_row["truncated"] = False
        nodes.append(normalized_row)
    return nodes


def _legacy_get_conversation_tree(
    service: ChatConversationService,
    conversation_id: str,
    *,
    root_limit: int = 50,
    root_offset: int = 0,
    order_by_timestamp: str = "ASC",
    depth_cap: int = 50,
) -> dict[str, Any]:
    """The pre-TASK-22206 ``get_conversation_tree``, verbatim."""
    conversation = service.get_conversation_metadata(conversation_id)
    if conversation is None:
        return {
            "conversation": None,
            "root_threads": [],
            "pagination": {
                "limit": root_limit,
                "offset": root_offset,
                "total_root_threads": 0,
                "has_more": False,
            },
            "depth_cap": depth_cap,
        }

    total_root_threads = service.db.count_root_messages_for_conversation(
        conversation_id,
        include_deleted_conversation=False,
    )
    root_rows = service.db.get_root_messages_for_conversation(
        conversation_id,
        limit=root_limit,
        offset=root_offset,
        order_by_timestamp=order_by_timestamp,
        include_deleted_conversation=False,
    )
    root_threads = _legacy_build_message_tree(
        service.db,
        conversation_id,
        root_rows,
        order_by_timestamp=order_by_timestamp,
        depth_cap=depth_cap,
        depth=1,
        seen_message_ids=set(),
    )

    return {
        "conversation": conversation,
        "root_threads": root_threads,
        "pagination": {
            "limit": root_limit,
            "offset": root_offset,
            "total_root_threads": total_root_threads,
            "has_more": root_offset + len(root_rows) < total_root_threads,
        },
        "depth_cap": depth_cap,
    }


# ---------------------------------------------------------------------------
# (a) 2000-message linear chain resumes without RecursionError
# ---------------------------------------------------------------------------


def test_2000_message_linear_chain_resumes_without_recursion_error(db):
    conversation_id = db.add_conversation({"title": "long chain"})
    assert conversation_id is not None
    message_ids = _add_linear_chain(db, conversation_id, 2000)
    service = ChatConversationService(db)

    # The exact caps the resume path uses (load_console_conversation_tree).
    tree = service.get_conversation_tree(
        conversation_id, root_limit=10_000, depth_cap=10_000
    )
    messages = console_messages_from_conversation_tree(tree, db=db)

    assert len(messages) == 2000
    assert messages[0].persisted_message_id == message_ids[0]
    assert messages[0].parent_message_id is None
    assert messages[-1].persisted_message_id == message_ids[-1]
    assert messages[-1].parent_message_id == message_ids[-2]
    assert messages[-1].content == "message 1999"
    assert tree["pagination"]["total_root_threads"] == 1


# ---------------------------------------------------------------------------
# (b) the new build is byte-identical to the legacy recursive build
# ---------------------------------------------------------------------------


def test_new_build_matches_legacy_recursive_build_on_branched_fixture(db):
    conversation_id = db.add_conversation({"title": "branched"})
    assert conversation_id is not None
    ids = _build_branched_fixture(db, conversation_id)
    service = ChatConversationService(db)

    new_tree = service.get_conversation_tree(conversation_id)
    legacy_tree = _legacy_get_conversation_tree(service, conversation_id)
    assert new_tree == legacy_tree

    # Structural spot-checks so the equality above cannot silently pass on
    # two identically-empty trees.
    roots = new_tree["root_threads"]
    assert [node["id"] for node in roots] == [ids["root1"], ids["root2"]]
    root1_children = roots[0]["children"]
    assert [node["id"] for node in root1_children] == [ids["a"], ids["b"]]
    assert root1_children[0]["image_data"] == b"\x89PNG-fake-bytes-a"
    assert root1_children[0]["image_mime_type"] == "image/png"
    assert root1_children[1]["image_data"] is None
    a_children = root1_children[0]["children"]
    assert [node["id"] for node in a_children] == [ids["a1"], ids["a2"]]
    node = a_children[1]
    for depth in range(10):
        assert len(node["children"]) == 1
        node = node["children"][0]
        assert node["id"] == ids[f"deep{depth}"]
        assert node["parent_message_id"] == (
            ids["a2"] if depth == 0 else ids[f"deep{depth - 1}"]
        )
    assert node["children"] == []
    assert node["truncated"] is False
    assert new_tree["pagination"]["total_root_threads"] == 2


def test_new_build_matches_legacy_for_desc_pagination_and_depth_cap(db):
    conversation_id = db.add_conversation({"title": "branched variants"})
    assert conversation_id is not None
    _build_branched_fixture(db, conversation_id)
    service = ChatConversationService(db)

    for kwargs in (
        {"order_by_timestamp": "DESC"},
        {"root_limit": 1, "root_offset": 1},
        {"root_limit": 1, "root_offset": 0},
        {"depth_cap": 3},
        {"depth_cap": 1},
        {"order_by_timestamp": "DESC", "depth_cap": 4, "root_limit": 1},
    ):
        new_tree = service.get_conversation_tree(conversation_id, **kwargs)
        legacy_tree = _legacy_get_conversation_tree(
            service, conversation_id, **kwargs
        )
        assert new_tree == legacy_tree, f"diverged for {kwargs}"

    # The depth-capped shape itself (not just parity): nodes at the cap are
    # truncated with no children.
    capped = service.get_conversation_tree(conversation_id, depth_cap=2)
    root1 = capped["root_threads"][0]
    assert root1["truncated"] is False
    for child in root1["children"]:
        assert child["truncated"] is True
        assert child["children"] == []


def test_missing_conversation_shape_is_unchanged(db):
    service = ChatConversationService(db)
    tree = service.get_conversation_tree("no-such-conversation")
    assert tree == {
        "conversation": None,
        "root_threads": [],
        "pagination": {
            "limit": 50,
            "offset": 0,
            "total_root_threads": 0,
            "has_more": False,
        },
        "depth_cap": 50,
    }


# ---------------------------------------------------------------------------
# (c) O(1) statements per tree build; image hydration lazy and batched
# ---------------------------------------------------------------------------


def _trace_statements(db: CharactersRAGDB, fn):
    connection = db.get_connection()
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    try:
        result = fn()
    finally:
        connection.set_trace_callback(None)
    return result, statements


def _message_selects(statements: list[str]) -> list[str]:
    return [
        statement
        for statement in statements
        if "FROM messages" in statement and statement.lstrip().upper().startswith(
            "SELECT"
        )
    ]


def test_tree_build_query_count_is_independent_of_message_count(db):
    service = ChatConversationService(db)

    small_id = db.add_conversation({"title": "small"})
    large_id = db.add_conversation({"title": "large"})
    assert small_id is not None and large_id is not None
    _add_linear_chain(db, small_id, 40)
    _add_linear_chain(db, large_id, 120)

    small_tree, small_statements = _trace_statements(
        db,
        lambda: service.get_conversation_tree(
            small_id, root_limit=10_000, depth_cap=10_000
        ),
    )
    large_tree, large_statements = _trace_statements(
        db,
        lambda: service.get_conversation_tree(
            large_id, root_limit=10_000, depth_cap=10_000
        ),
    )

    def _count_nodes(tree):
        count = 0
        stack = list(tree["root_threads"])
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node["children"])
        return count

    assert _count_nodes(small_tree) == 40
    assert _count_nodes(large_tree) == 120
    assert len(small_statements) == len(large_statements), (
        "statement count grew with conversation size: "
        f"{len(small_statements)} @40 msgs vs {len(large_statements)} @120 msgs"
    )
    assert len(_message_selects(large_statements)) <= 2, (
        "expected O(1) message reads per tree build, got: "
        + "\n".join(_message_selects(large_statements))
    )


def test_image_blobs_are_hydrated_lazily_in_one_batched_statement(db):
    service = ChatConversationService(db)

    plain_id = db.add_conversation({"title": "no images"})
    imaged_id = db.add_conversation({"title": "images"})
    assert plain_id is not None and imaged_id is not None
    _add_linear_chain(db, plain_id, 30)
    with db.transaction():
        parent = None
        for index in range(30):
            parent = _add_message(
                db,
                imaged_id,
                index=index,
                parent_message_id=parent,
                image_data=(b"\x00" * 64 if index in (3, 17) else None),
            )

    def _image_fetches(statements: list[str]) -> list[str]:
        # A statement that hydrates the BLOB selects the image_data COLUMN;
        # the tree read only derives the has_image flag from it.
        return [
            statement
            for statement in _message_selects(statements)
            if "image_data" in statement and "has_image" not in statement
        ]

    plain_tree, plain_statements = _trace_statements(
        db, lambda: service.get_conversation_tree(plain_id, root_limit=10_000)
    )
    imaged_tree, imaged_statements = _trace_statements(
        db, lambda: service.get_conversation_tree(imaged_id, root_limit=10_000)
    )

    assert _image_fetches(plain_statements) == [], (
        "an imageless conversation must not read BLOB columns at all"
    )
    assert len(_image_fetches(imaged_statements)) == 1, (
        "image hydration must be one batched statement, got: "
        + "\n".join(_image_fetches(imaged_statements))
    )

    def _nodes_with_images(tree):
        found = []
        stack = list(tree["root_threads"])
        while stack:
            node = stack.pop()
            if node["image_data"] is not None:
                found.append(node)
            stack.extend(node["children"])
        return found

    assert _nodes_with_images(plain_tree) == []
    imaged_nodes = _nodes_with_images(imaged_tree)
    assert len(imaged_nodes) == 2
    for node in imaged_nodes:
        assert node["image_data"] == b"\x00" * 64
        assert node["image_mime_type"] == "image/png"
