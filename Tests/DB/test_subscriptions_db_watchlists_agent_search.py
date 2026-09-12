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


def _add_to_collection(db: SubscriptionsDB, collection_id: int, source_id: int) -> None:
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


def _briefing(
    db: SubscriptionsDB,
    collection_id: int,
    status: str,
    created_at: str,
    *,
    body: str | None = None,
) -> int:
    with db.transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO briefings (
                watchlist_id, status, body_markdown, selection_mode, model_used,
                item_count, featured_count, overflow_count, created_at, updated_at
            ) VALUES (?, ?, ?, 'auto_featured', 'provider/model', 1, 1, 0, ?, ?)
            """,
            (collection_id, status, body, created_at, created_at),
        )
        return int(cursor.lastrowid)


def _run(
    db: SubscriptionsDB, source_id: int, status: str, created_at: str
) -> int:
    with db.transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO local_watchlist_runs (
                source_id, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (source_id, status, created_at, created_at),
        )
        return int(cursor.lastrowid)


def test_agent_source_and_collection_pages_use_stable_filterable_keysets(
    db: SubscriptionsDB,
) -> None:
    alpha_upper = _source(db, "Alpha")
    alpha_lower = _source(db, "alpha")
    beta = _source(db, "Beta")
    selected = _collection(db, "Selected")
    other = _collection(db, "Other")
    _add_to_collection(db, selected, alpha_upper)
    _add_to_collection(db, selected, alpha_lower)
    _add_to_collection(db, other, beta)

    first = db.list_sources_for_agent(watchlist_id=selected, limit=1)
    second = db.list_sources_for_agent(
        watchlist_id=selected,
        limit=1,
        after_name_casefold_prefix=first["items"][0]["name_casefold_prefix"],
        after_name_prefix=first["items"][0]["name_prefix"],
        after_id=first["items"][0]["id"],
    )
    collections = db.list_collections_for_agent(name_query="select", limit=10)

    assert [row["id"] for row in first["items"] + second["items"]] == [
        alpha_upper,
        alpha_lower,
    ]
    assert first["has_more"] is True
    assert second["has_more"] is False
    assert collections["items"][0]["id"] == selected
    assert collections["items"][0]["source_count"] == 2
    assert collections["items"][0]["briefing_selection_mode"] == "auto_featured"
    assert collections["items"][0]["briefing_cadence_seconds"] is None


def test_agent_source_cursor_survives_deleted_anchor_without_skipping(
    db: SubscriptionsDB,
) -> None:
    first_id = _source(db, "Alpha")
    second_id = _source(db, "Bravo")
    third_id = _source(db, "Charlie")

    first = db.list_sources_for_agent(limit=1)
    with db.transaction() as conn:
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (first_id,))
    continued = db.list_sources_for_agent(
        limit=10,
        after_name_casefold_prefix=first["items"][0]["name_casefold_prefix"],
        after_name_prefix=first["items"][0]["name_prefix"],
        after_id=first["items"][0]["id"],
    )

    assert [row["id"] for row in continued["items"]] == [second_id, third_id]


def test_agent_collection_cursor_survives_renamed_anchor_without_duplication(
    db: SubscriptionsDB,
) -> None:
    first_id = _collection(db, "Alpha")
    second_id = _collection(db, "Bravo")
    third_id = _collection(db, "Charlie")

    first = db.list_collections_for_agent(limit=1)
    with db.transaction() as conn:
        conn.execute("UPDATE watchlists SET name = 'Zulu' WHERE id = ?", (first_id,))
    continued = db.list_collections_for_agent(
        limit=10,
        after_name_casefold_prefix=first["items"][0]["name_casefold_prefix"],
        after_name_prefix=first["items"][0]["name_prefix"],
        after_id=first["items"][0]["id"],
    )

    assert [row["id"] for row in continued["items"]] == [second_id, third_id]


def test_collection_latest_timestamps_follow_datetime_then_id_order(
    db: SubscriptionsDB,
) -> None:
    collection_id = _collection(db, "Mixed timestamps")
    _briefing(db, collection_id, "complete", "2026-08-13T10:00:00Z")
    expected_attempt = _briefing(
        db, collection_id, "failed", "2026-08-13 11:00:00"
    )
    expected_success = _briefing(
        db, collection_id, "complete", "2026-08-13 10:30:00"
    )

    row = db.list_collections_for_agent(limit=1)["items"][0]

    assert row["last_briefing_id"] == expected_attempt
    assert row["last_briefing_attempt_at"] == "2026-08-13 11:00:00"
    assert row["last_briefing_success_at"] == "2026-08-13 10:30:00"
    assert expected_success != expected_attempt


def test_agent_briefing_pages_latest_readable_and_operation_receipts(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Operations")
    collection_id = _collection(db, "Threats")
    _add_to_collection(db, collection_id, source_id)
    completed = _briefing(
        db, collection_id, "complete", "2026-08-20 10:00:00", body="# Readable"
    )
    failed = _briefing(db, collection_id, "failed", "2026-08-21 10:00:00")
    running = _run(db, source_id, "running", "2026-08-22 10:00:00")

    page = db.list_briefings_for_agent(watchlist_id=collection_id, limit=1)
    latest = db.get_latest_completed_briefing_for_agent(collection_id)
    operations = db.list_operations_for_agent(
        source_id=source_id, watchlist_id=collection_id, limit=10
    )

    assert page["items"][0]["id"] == failed
    assert page["has_more"] is True
    assert latest["briefing"]["id"] == completed
    assert [row["id"] for row in latest["newer_attempts"]] == [failed]
    assert [row["id"] for row in operations["source_runs"]] == [running]
    assert [row["id"] for row in operations["briefings"]] == [failed, completed]
    assert db.get_watchlist_run_for_agent(running)["source_id"] == source_id
    assert db.get_briefing_for_agent(failed)["watchlist_id"] == collection_id


def test_agent_briefing_provenance_is_immutable_ordered_and_marks_legacy(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Snapshot source")
    collection_id = _collection(db, "Snapshot collection")
    briefing_id = _briefing(
        db, collection_id, "complete", "2026-08-20 10:00:00", body="# Digest"
    )
    with db.transaction() as conn:
        conn.executemany(
            """
            INSERT INTO briefing_items (
                briefing_id, item_id, selection_position, citation_position,
                featured, cited, item_title, item_url, item_published_date,
                item_effective_date, source_id, source_name, source_type,
                source_url, provenance_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    briefing_id,
                    102,
                    1,
                    0,
                    0,
                    1,
                    "Second",
                    "https://user:pass@items.test/second?token=x#frag",
                    "2026-08-19T00:00:00Z",
                    "2026-08-19 00:00:00",
                    source_id,
                    "Snapshot source",
                    "rss",
                    "https://user:pass@sources.test/feed?token=x#frag",
                    2,
                ),
                (
                    briefing_id,
                    101,
                    0,
                    None,
                    1,
                    0,
                    "First",
                    "https://items.test/first",
                    "2026-08-18T00:00:00Z",
                    "2026-08-18 00:00:00",
                    source_id,
                    "Snapshot source",
                    "rss",
                    "https://sources.test/feed",
                    2,
                ),
                (
                    briefing_id,
                    103,
                    None,
                    None,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    1,
                ),
            ],
        )

    rows = db.get_briefing_provenance_for_agent(briefing_id, limit=10)

    assert [row["item_id"] for row in rows["selected"]] == [101, 102, 103]
    assert [row["item_id"] for row in rows["cited"]] == [102]
    assert rows["selected"][0]["item_effective_date"] == "2026-08-18 00:00:00"
    assert rows["selected"][-1]["provenance_version"] == 1


def test_blank_search_returns_every_status_newest_effective_first(
    db: SubscriptionsDB,
) -> None:
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


def test_agent_search_rows_have_an_exact_narrow_key_allowlist(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Narrow projection")
    item_id = _item(db, source_id, "narrow", content="bounded body")
    with db.transaction() as conn:
        conn.execute(
            """
            UPDATE subscription_items
            SET content_hash = 'hash-canary', categories = 'category-canary',
                enclosures = 'enclosure-canary', extracted_data = 'raw-canary',
                processing_error = 'error-canary', previous_hash = 'previous-canary',
                diff_summary = 'diff-canary', alert_matches = 'alert-canary',
                canonical_url = 'https://items.test/canonical',
                content_format = 'html', content_kind = 'article'
            WHERE id = ?
            """,
            (item_id,),
        )

    row = _search(db, limit=1)["items"][0]

    assert set(row) == {
        "id",
        "subscription_id",
        "url",
        "title",
        "published_date",
        "author",
        "status",
        "canonical_url",
        "created_at",
        "updated_at",
        "content_format",
        "content_kind",
        "effective_date",
        "subscription_name",
        "subscription_type",
        "subscription_source",
        "subscription_is_active",
        "subscription_is_paused",
        "subscription_created_at",
        "subscription_updated_at",
        "subscription_last_checked",
        "subscription_last_successful_check",
        "content_match_context",
    }


def test_literal_and_terms_match_title_author_and_deep_body(
    db: SubscriptionsDB,
) -> None:
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

    assert [row["id"] for row in _search(db, query="title-token")["items"]] == [
        title_id
    ]
    assert [row["id"] for row in _search(db, query="author-token")["items"]] == [
        author_id
    ]
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
    ("query", "matched_passage"),
    (
        ("evidence", "actual ÉVIDENCE passage"),
        ("NEAR/1", "actual NEAR 1 passage"),
    ),
)
def test_fts_match_context_uses_fts_token_normalization_for_deep_passages(
    db: SubscriptionsDB, query: str, matched_passage: str
) -> None:
    source_id = _source(db, "FTS context feed")
    item_id = _item(
        db,
        source_id,
        f"fts-context-{query}",
        content="leading noise " * 1_000 + matched_passage + " trailing noise" * 1_000,
    )

    rows = _search(db, query=query)["items"]

    assert [row["id"] for row in rows] == [item_id]
    assert matched_passage in rows[0]["content_match_context"]
    assert len(rows[0]["content_match_context"]) <= 2_000
    assert "content" not in rows[0]


@pytest.mark.parametrize(
    ("query", "matching_title"),
    (
        ("100%", "literal 100% marker"),
        ("a_b", "literal a_b marker"),
        (r"a\b", r"literal a\b marker"),
    ),
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
    wanted = _item(
        db,
        source_id,
        "deep",
        content="literal prefix " * 1_000 + "partial-only-token" + " suffix" * 1_000,
    )
    with db.transaction() as conn:
        conn.execute("DELETE FROM subscription_items_fts WHERE rowid = ?", (wanted,))

    rows = _search(db, query="partial-only-token")["items"]
    assert [row["id"] for row in rows] == [wanted]
    assert "partial-only-token" in rows[0]["content_match_context"]
    assert len(rows[0]["content_match_context"]) <= 2_000


def test_equal_fts_cardinality_with_wrong_membership_forces_like(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Wrong membership feed")
    wanted = _item(db, source_id, "wanted", content="membership-token")
    _item(db, source_id, "other", content="other")
    with db.transaction() as conn:
        conn.execute(
            "DELETE FROM subscription_items_fts_docsize WHERE id = ?", (wanted,)
        )
        conn.execute(
            "INSERT INTO subscription_items_fts_docsize (id, sz) VALUES (?, ?)",
            (999_999, sqlite3.Binary(b"")),
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM subscription_items").fetchone()[0]
            == conn.execute(
                "SELECT COUNT(*) FROM subscription_items_fts_docsize"
            ).fetchone()[0]
        )

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

    later_future = _item(
        db, source_id, "later-future", published="2040-01-01T00:00:00Z"
    )
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
    assert traversed == [
        future,
        tied_first,
        tied_second,
        older,
        null_first,
        null_second,
    ]


def test_one_lookahead_sets_has_more_and_is_not_returned(db: SubscriptionsDB) -> None:
    source_id = _source(db, "Lookahead feed")
    ids = [
        _item(
            db, source_id, str(index), published=f"2026-08-{14 - index:02d}T00:00:00Z"
        )
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
            [
                (f"Source {index:04d}", f"https://bulk.test/{index}")
                for index in range(1_005)
            ],
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
    assert [row["id"] for row in db.resolve_source_candidates("needle", limit=5)] == [
        target
    ]
    assert [row["id"] for row in db.resolve_source_candidates(target, limit=5)] == [
        target
    ]

    exact = db.resolve_collection_candidates("ALPHA", limit=10)
    partial = db.resolve_collection_candidates("alp", limit=3)
    assert [row["name"] for row in exact] == ["Alpha", "alpha"]
    assert [row["name"] for row in partial] == ["Alpha", "alpha", "Alphabet"]
    assert db.resolve_collection_candidates(exact[0]["id"], limit=10) == [exact[0]]


def test_source_resolution_prefers_exact_name_before_exact_url(
    db: SubscriptionsDB,
) -> None:
    collision = "https://collision.test/feed"
    exact_name_id = _source(db, collision, url="https://name-owner.test/feed")
    _source(db, "URL owner", url=collision)

    candidates = db.resolve_source_candidates(collision, limit=10)

    assert [row["id"] for row in candidates] == [exact_name_id]


def test_source_and_collection_resolution_casefolds_unicode_exact_and_partial_names(
    db: SubscriptionsDB,
) -> None:
    exact_source = _source(db, "Équipe CERT")
    partial_source = _source(db, "Nord Équipe Signal")
    exact_collection = _collection(db, "Équipe Watch")
    partial_collection = _collection(db, "Veille Équipe Nord")
    with db.transaction() as conn:
        # SQLite's dynamic typing permits malformed non-text values despite
        # the declared TEXT affinity. The connection-local UDF must not let
        # either row crash an otherwise valid resolution query.
        conn.execute(
            "INSERT INTO subscriptions (name, type, source) VALUES (?, 'rss', ?)",
            (sqlite3.Binary(b"\x80"), "https://malformed.test/feed"),
        )
        conn.execute(
            "INSERT INTO watchlists (name) VALUES (?)", (sqlite3.Binary(b"\x80"),)
        )

    assert [row["id"] for row in db.resolve_source_candidates("éQUIPE cert")] == [
        exact_source
    ]
    assert [row["id"] for row in db.resolve_source_candidates("équipe sig")] == [
        partial_source
    ]
    assert [row["id"] for row in db.resolve_collection_candidates("éQUIPE watch")] == [
        exact_collection
    ]
    assert [row["id"] for row in db.resolve_collection_candidates("équipe nord")] == [
        partial_collection
    ]


def test_unicode_scope_casefold_is_registered_on_read_only_connections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "read-only-unicode.db"
    mutable = SubscriptionsDB(path)
    source_id = _source(mutable, "Équipe CERT")
    collection_id = _collection(mutable, "Équipe Watch")
    mutable.close()

    reader = SubscriptionsDB(path, read_only=True)
    try:
        assert [
            row["id"] for row in reader.resolve_source_candidates("équipe cert")
        ] == [source_id]
        assert [
            row["id"] for row in reader.resolve_collection_candidates("équipe watch")
        ] == [collection_id]
    finally:
        reader.close()


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


def test_source_collection_memberships_use_one_bounded_query(
    db: SubscriptionsDB,
) -> None:
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
        if statement.lstrip().upper().startswith(("SELECT", "WITH"))
        and "WATCHLIST_SOURCES" in statement.upper()
    ]
    assert len(membership_selects) == 1
    assert [row["name"] for row in memberships[source_ids[0]]["collections"]] == [
        "Alpha",
        "Zulu",
    ]
    assert memberships[source_ids[0]]["has_more"] is False
    assert [row["name"] for row in memberships[source_ids[1]]["collections"]] == [
        "Alpha"
    ]
    assert memberships[source_ids[1]]["has_more"] is False
    assert memberships[source_ids[2]] == {"collections": [], "has_more": False}
    with pytest.raises(ValueError, match="at most"):
        db.get_source_collection_memberships(range(1, 52))


def test_source_collection_memberships_bound_each_source_with_lookahead(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "High-cardinality source")
    for index in range(25):
        _add_to_collection(db, _collection(db, f"Collection {index:03d}"), source_id)
    statements: list[str] = []
    db.conn.set_trace_callback(statements.append)
    try:
        memberships = db.get_source_collection_memberships([source_id])
    finally:
        db.conn.set_trace_callback(None)

    result = memberships[source_id]
    assert [row["name"] for row in result["collections"]] == [
        f"Collection {index:03d}" for index in range(20)
    ]
    assert result["has_more"] is True
    membership_selects = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("WITH")
        and "WATCHLIST_SOURCES" in statement.upper()
    ]
    assert len(membership_selects) == 1
    assert "MEMBERSHIP_RANK <= 21" in membership_selects[0].upper()
