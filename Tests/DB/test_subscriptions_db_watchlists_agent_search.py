"""Storage contracts for bounded, authoritative Watchlists agent evidence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB


@pytest.fixture
def db(tmp_path: Path):
    owner = SubscriptionsDB(tmp_path / "subscriptions.db")
    try:
        yield owner
    finally:
        owner.close()


def _source(db: SubscriptionsDB, name: str, *, url: str | None = None) -> int:
    return db.add_subscription(
        name=name,
        type="rss",
        source=url or f"https://example.test/{name.casefold().replace(' ', '-')}",
    )


def _collection(db: SubscriptionsDB, name: str) -> int:
    with db.transaction() as conn:
        cursor = conn.execute("INSERT INTO watchlists (name) VALUES (?)", (name,))
        return int(cursor.lastrowid)


def _add_to_collection(
    db: SubscriptionsDB, collection_id: int, source_id: int
) -> None:
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO watchlist_sources (watchlist_id, subscription_id) VALUES (?, ?)",
            (collection_id, source_id),
        )


def _item(
    db: SubscriptionsDB,
    source_id: int,
    slug: str,
    *,
    title: str | None = None,
    content: str | None = "body",
    author: str | None = None,
    status: str = "new",
    published: str | None = "2026-08-14T12:00:00Z",
    created: str = "2026-08-14T12:05:00Z",
) -> int:
    with db.transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO subscription_items (
                subscription_id, url, title, content, author, status,
                published_date, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                f"https://items.test/{slug}",
                title or slug,
                content,
                author,
                status,
                published,
                created,
            ),
        )
        return int(cursor.lastrowid)


def _search(db: SubscriptionsDB, **kwargs):
    return db.search_items_for_agent(**kwargs)


def test_blank_search_returns_every_status_newest_effective_first(db: SubscriptionsDB) -> None:
    source_id = _source(db, "Status feed")
    expected = []
    for index, status in enumerate(("new", "reviewed", "ingested", "ignored", "error")):
        expected.append(
            _item(
                db,
                source_id,
                status,
                status=status,
                published=f"2026-08-{10 + index:02d}T12:00:00Z",
            )
        )

    page = _search(db, query="   ", limit=10)

    assert [row["id"] for row in page["items"]] == list(reversed(expected))
    assert {row["status"] for row in page["items"]} == {
        "new",
        "reviewed",
        "ingested",
        "ignored",
        "error",
    }
    assert all("effective_date" in row for row in page["items"])


def test_literal_and_terms_match_title_author_and_deep_body(db: SubscriptionsDB) -> None:
    source_id = _source(db, "Search feed")
    deep = "prefix " * 1_000 + "deep-body-token" + " suffix" * 1_000
    title_id = _item(db, source_id, "title", title="literal title-token")
    author_id = _item(db, source_id, "author", author="author-token analyst")
    body_id = _item(db, source_id, "body", content=deep)
    combined_id = _item(
        db,
        source_id,
        "combined",
        title="heading-token",
        content="prefix " * 1_000 + "second-deep-token" + " suffix" * 1_000,
    )
    operator_id = _item(
        db,
        source_id,
        "operator",
        title="retrieval OR rubric",
    )
    _item(db, source_id, "operator-decoy", title="retrieval rubric")

    assert [row["id"] for row in _search(db, query="title-token")["items"]] == [title_id]
    assert [row["id"] for row in _search(db, query="author-token")["items"]] == [author_id]
    body_row = _search(db, query="deep-body-token")["items"][0]
    assert body_row["id"] == body_id
    assert "content" not in body_row
    assert "extracted_data" not in body_row
    assert "deep-body-token" in body_row["content_match_context"]
    assert len(body_row["content_match_context"]) <= 2_000
    combined_row = _search(db, query="heading-token second-deep-token")["items"][0]
    assert combined_row["id"] == combined_id
    assert "second-deep-token" in combined_row["content_match_context"]
    assert [row["id"] for row in _search(db, query="retrieval OR rubric")["items"]] == [
        operator_id
    ]


@pytest.mark.parametrize(
    ("query", "matching_title"),
    (("100%", "literal 100% marker"), ("a_b", "literal a_b marker"), (r"a\b", r"literal a\b marker")),
)
def test_absent_fts_like_fallback_escapes_wildcards(
    db: SubscriptionsDB, query: str, matching_title: str
) -> None:
    source_id = _source(db, "Fallback feed")
    wanted = _item(db, source_id, f"wanted-{query}", title=matching_title)
    _item(db, source_id, f"decoy-{query}", title="literal 100x / axb / ab marker")
    with db.transaction() as conn:
        conn.execute("DROP TABLE subscription_items_fts")

    assert [row["id"] for row in _search(db, query=query)["items"]] == [wanted]


def test_available_fts_operational_failure_falls_back_in_the_same_call(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Damaged FTS feed")
    wanted = _item(db, source_id, "wanted", title="same-call-token")
    assert [row["id"] for row in _search(db, query="same-call-token")["items"]] == [
        wanted
    ]
    # Completeness is now monotonically cached. Removing a query-time shadow
    # table leaves the FTS virtual table present but makes MATCH raise an
    # OperationalError; this call must still return the literal LIKE answer.
    with db.transaction() as conn:
        conn.execute("DROP TABLE subscription_items_fts_idx")

    assert [row["id"] for row in _search(db, query="same-call-token")["items"]] == [
        wanted
    ]


def test_partial_fts_coverage_forces_like_even_when_fts_table_exists(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Partial FTS feed")
    wanted = _item(db, source_id, "deep", content="partial-only-token")
    with db.transaction() as conn:
        conn.execute("DELETE FROM subscription_items_fts WHERE rowid = ?", (wanted,))

    assert [row["id"] for row in _search(db, query="partial-only-token")["items"]] == [
        wanted
    ]


def test_equal_fts_cardinality_with_wrong_membership_forces_like(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Wrong membership feed")
    wanted = _item(db, source_id, "wanted", content="membership-token")
    _item(db, source_id, "other", content="other")
    with db.transaction() as conn:
        conn.execute("DELETE FROM subscription_items_fts_docsize WHERE id = ?", (wanted,))
        conn.execute(
            "INSERT INTO subscription_items_fts_docsize (id, sz) VALUES (?, ?)",
            (999_999, sqlite3.Binary(b"")),
        )
        assert conn.execute("SELECT COUNT(*) FROM subscription_items").fetchone()[0] == conn.execute(
            "SELECT COUNT(*) FROM subscription_items_fts_docsize"
        ).fetchone()[0]

    assert [row["id"] for row in _search(db, query="membership-token")["items"]] == [
        wanted
    ]


def test_incomplete_fts_is_rechecked_and_only_complete_state_is_cached(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Transition feed")
    wanted = _item(db, source_id, "wanted", content="transition-token")
    with db.transaction() as conn:
        conn.execute("DELETE FROM subscription_items_fts WHERE rowid = ?", (wanted,))

    assert [row["id"] for row in _search(db, query="transition-token")["items"]] == [
        wanted
    ], "the incomplete state must use LIKE"

    assert db.backfill_items_fts() == 1
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER subscription_items_fts_au")
        conn.execute(
            "UPDATE subscription_items SET content = 'no literal match remains' WHERE id = ?",
            (wanted,),
        )

    assert [row["id"] for row in _search(db, query="transition-token")["items"]] == [
        wanted
    ], "the same owner must recheck, prove complete, and switch to FTS"


def test_all_search_predicates_compose_as_an_intersection(db: SubscriptionsDB) -> None:
    selected_source = _source(db, "Selected")
    other_source = _source(db, "Other")
    collection_id = _collection(db, "Selected collection")
    _add_to_collection(db, collection_id, selected_source)
    wanted = _item(
        db,
        selected_source,
        "wanted",
        title="shared-token",
        status="reviewed",
        published="2026-08-14T00:00:00Z",
    )
    anchor = _item(
        db,
        selected_source,
        "cursor-anchor",
        title="shared-token",
        status="reviewed",
        published="2026-08-15T00:00:00Z",
    )
    _item(db, selected_source, "wrong-status", title="shared-token", status="new")
    _item(
        db,
        selected_source,
        "too-old",
        title="shared-token",
        status="reviewed",
        published="2026-08-13T23:59:59Z",
    )
    _item(db, other_source, "wrong-source", title="shared-token", status="reviewed")
    above_high_water = _item(
        db, selected_source, "above-high-water", title="shared-token", status="reviewed"
    )

    page = _search(
        db,
        query="shared-token",
        subscription_id=selected_source,
        watchlist_id=collection_id,
        statuses=["reviewed"],
        since="2026-08-14T00:00:00Z",
        snapshot_max_item_id=above_high_water - 1,
        after_effective_date="2026-08-15T00:00:00Z",
        after_item_id=anchor,
        limit=10,
    )

    assert [row["id"] for row in page["items"]] == [wanted]


def test_keyset_traversal_handles_ties_null_sink_deletion_and_later_inserts(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Paging feed")
    future = _item(db, source_id, "future", published="2030-01-01T00:00:00Z")
    tied_first = _item(db, source_id, "tie-1", published="2026-08-14T00:00:00Z")
    tied_second = _item(db, source_id, "tie-2", published="2026-08-14T00:00:00Z")
    older = _item(db, source_id, "older", published="2026-08-13T00:00:00Z")
    null_first = _item(db, source_id, "null-1", published=None, created="not-a-date")
    null_second = _item(db, source_id, "null-2", published=None, created="not-a-date")

    first = _search(db, limit=2)
    assert [row["id"] for row in first["items"]] == [future, tied_first]
    assert first["has_more"] is True
    assert first["snapshot_max_item_id"] == null_second

    later_future = _item(db, source_id, "later-future", published="2040-01-01T00:00:00Z")
    _item(db, source_id, "later-null", published=None, created="not-a-date")
    with db.transaction() as conn:
        conn.execute("DELETE FROM subscription_items WHERE id = ?", (tied_first,))

    second = _search(
        db,
        limit=2,
        snapshot_max_item_id=first["snapshot_max_item_id"],
        after_effective_date=first["items"][-1]["effective_date"],
        after_item_id=first["items"][-1]["id"],
    )
    third = _search(
        db,
        limit=2,
        snapshot_max_item_id=first["snapshot_max_item_id"],
        after_effective_date=second["items"][-1]["effective_date"],
        after_item_id=second["items"][-1]["id"],
    )

    assert [row["id"] for row in second["items"]] == [tied_second, older]
    assert [row["id"] for row in third["items"]] == [null_first, null_second]
    assert third["has_more"] is False
    traversed = [row["id"] for page in (first, second, third) for row in page["items"]]
    assert later_future not in traversed
    assert traversed == [future, tied_first, tied_second, older, null_first, null_second]


def test_one_lookahead_sets_has_more_and_is_not_returned(db: SubscriptionsDB) -> None:
    source_id = _source(db, "Lookahead feed")
    ids = [
        _item(db, source_id, str(index), published=f"2026-08-{14 - index:02d}T00:00:00Z")
        for index in range(3)
    ]

    statements: list[str] = []
    db.conn.set_trace_callback(statements.append)
    try:
        page = _search(db, limit=2)
    finally:
        db.conn.set_trace_callback(None)

    assert [row["id"] for row in page["items"]] == ids[:2]
    assert ids[2] not in {row["id"] for row in page["items"]}
    assert page["has_more"] is True
    item_page_selects = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT")
        and "FROM SUBSCRIPTION_ITEMS I" in statement.upper()
        and "JOIN SUBSCRIPTIONS S" in statement.upper()
    ]
    assert len(item_page_selects) == 1
    assert "LIMIT 3" in item_page_selects[0].upper()


def test_search_page_size_is_bounded_at_the_storage_seam(db: SubscriptionsDB) -> None:
    with pytest.raises(ValueError, match="at most"):
        _search(db, limit=51)


def test_source_and_collection_candidate_resolution_is_bounded_and_deterministic(
    db: SubscriptionsDB,
) -> None:
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO subscriptions (name, type, source) VALUES (?, 'rss', ?)",
            [(f"Source {index:04d}", f"https://bulk.test/{index}") for index in range(1_005)],
        )
        target = conn.execute(
            "INSERT INTO subscriptions (name, type, source) VALUES (?, 'rss', ?)",
            ("Needle beyond scan", "https://needle.test/feed"),
        ).lastrowid
        conn.executemany(
            "INSERT INTO watchlists (name) VALUES (?)",
            [("Alpha",), ("alpha",), ("Alphabet",), ("Alpine",)],
        )

    exact_source = db.resolve_source_candidates("https://needle.test/feed", limit=5)
    assert len(exact_source) == 1
    assert exact_source[0]["id"] == target
    assert exact_source[0]["name"] == "Needle beyond scan"
    assert exact_source[0]["source"] == "https://needle.test/feed"
    assert exact_source[0]["type"] == "rss"
    assert exact_source[0]["is_active"] == 1
    assert exact_source[0]["is_paused"] == 0
    assert exact_source[0]["last_checked"] is None
    assert exact_source[0]["last_successful_check"] is None
    assert exact_source[0]["created_at"]
    assert exact_source[0]["updated_at"]
    assert [row["id"] for row in db.resolve_source_candidates("needle", limit=5)] == [target]
    assert [row["id"] for row in db.resolve_source_candidates(target, limit=5)] == [target]

    exact = db.resolve_collection_candidates("ALPHA", limit=10)
    partial = db.resolve_collection_candidates("alp", limit=3)
    assert [row["name"] for row in exact] == ["Alpha", "alpha"]
    assert [row["name"] for row in partial] == ["Alpha", "alpha", "Alphabet"]
    assert db.resolve_collection_candidates(exact[0]["id"], limit=10) == [exact[0]]


def test_authoritative_joined_detail_distinguishes_missing_from_null_content(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Detail feed", url="https://detail.test/feed")
    item_id = _item(db, source_id, "null-content", content=None)

    detail = db.get_item_detail_for_agent(item_id)

    assert detail is not None
    assert detail["id"] == item_id
    assert detail["content"] is None
    assert detail["subscription_id"] == source_id
    assert detail["subscription_name"] == "Detail feed"
    assert detail["subscription_source"] == "https://detail.test/feed"
    assert db.get_item_detail_for_agent(999_999) is None


def test_source_collection_memberships_use_one_bounded_query(db: SubscriptionsDB) -> None:
    source_ids = [_source(db, f"Source {index}") for index in range(3)]
    collection_ids = [_collection(db, name) for name in ("Zulu", "Alpha")]
    _add_to_collection(db, collection_ids[0], source_ids[0])
    _add_to_collection(db, collection_ids[1], source_ids[0])
    _add_to_collection(db, collection_ids[1], source_ids[1])
    statements: list[str] = []
    db.conn.set_trace_callback(statements.append)
    try:
        memberships = db.get_source_collection_memberships(source_ids)
    finally:
        db.conn.set_trace_callback(None)

    membership_selects = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT")
        and "WATCHLIST_SOURCES" in statement.upper()
    ]
    assert len(membership_selects) == 1
    assert [row["name"] for row in memberships[source_ids[0]]] == ["Alpha", "Zulu"]
    assert [row["name"] for row in memberships[source_ids[1]]] == ["Alpha"]
    assert memberships[source_ids[2]] == []
    with pytest.raises(ValueError, match="at most"):
        db.get_source_collection_memberships(range(1, 52))
