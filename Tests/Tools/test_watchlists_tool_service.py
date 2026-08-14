"""Contracts for the shared synchronous Watchlists agent-tool service."""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Tools.watchlists_tool_service import WatchlistsToolService


@pytest.fixture
def db(tmp_path: Path):
    owner = SubscriptionsDB(tmp_path / "subscriptions.db")
    try:
        yield owner
    finally:
        owner.close()


def _source(
    db: SubscriptionsDB,
    name: str,
    *,
    url: str | None = None,
    active: bool = True,
    paused: bool = False,
) -> int:
    with db.transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO subscriptions (
                name, type, source, is_active, is_paused, created_at, updated_at,
                last_checked, last_successful_check
            ) VALUES (?, 'rss', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                url or f"https://sources.test/{name.casefold().replace(' ', '-')}",
                active,
                paused,
                "2026-08-01 09:00:00",
                "2026-08-02 10:00:00",
                "2026-08-13 11:00:00",
                "2026-08-13 10:55:00",
            ),
        )
        return int(cursor.lastrowid)


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
    content: str | None = "body evidence",
    author: str | None = "Example Author",
    status: str = "new",
    published: str | None = "2026-08-14T12:00:00Z",
    created: str = "2026-08-14T12:05:00Z",
) -> int:
    with db.transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO subscription_items (
                subscription_id, url, title, content, author, status,
                published_date, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                "2026-08-14T12:10:00Z",
            ),
        )
        return int(cursor.lastrowid)


def _service(
    db_or_resolver: SubscriptionsDB | Any,
    *,
    runtime_source_loader=lambda: "local",
) -> WatchlistsToolService:
    resolver = db_or_resolver if callable(db_or_resolver) else lambda: db_or_resolver
    return WatchlistsToolService(
        db_resolver=resolver,
        runtime_source_loader=runtime_source_loader,
        clock=lambda: datetime(2026, 8, 14, 21, 30, tzinfo=UTC),
    )


def _payload(text: str) -> dict[str, Any]:
    value = json.loads(text)
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    "arguments",
    (
        {"unknown": "value"},
        {"query": "needle", "extra": False},
        [],
        None,
    ),
)
def test_search_rejects_unknown_keys_and_non_objects_before_dependencies(
    arguments: object,
) -> None:
    calls: list[str] = []
    service = WatchlistsToolService(
        db_resolver=lambda: calls.append("db"),
        runtime_source_loader=lambda: calls.append("runtime"),
    )

    result = _payload(service.search_items(arguments))

    assert result["status"] == "invalid_argument"
    assert result["retryable"] is False
    assert calls == []


@pytest.mark.parametrize(
    ("arguments", "message_fragment"),
    (
        ({"query": None}, "query"),
        ({"query": 1}, "query"),
        ({"query": "x" * 513}, "512"),
        ({"query": " ".join(f"t{i}" for i in range(33))}, "32"),
        ({"cursor": 7}, "cursor"),
        ({"cursor": "not-yet-supported"}, "cursor support"),
    ),
)
def test_search_rejects_invalid_query_and_cursor_values_before_database(
    arguments: dict[str, object], message_fragment: str
) -> None:
    calls: list[str] = []
    service = _service(lambda: calls.append("db"))

    result = _payload(service.search_items(arguments))

    assert result["status"] == "invalid_argument"
    assert result["retryable"] is False
    assert message_fragment in result["message"]
    assert calls == []


def test_blank_query_browses_and_nonblank_query_collapses_whitespace(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Whitespace feed")
    newest = _item(db, source_id, "newest", title="Alpha Beta result")

    browse = _payload(_service(db).search_items({"query": " \t\n "}))
    search = _payload(_service(db).search_items({"query": "  Alpha \t  Beta  "}))

    assert browse["query_mode"] == "browse"
    assert [item["id"] for item in browse["items"]] == [
        f"local:watchlist_item:{newest}"
    ]
    assert search["query_mode"] == "literal_full_text"
    assert [item["id"] for item in search["items"]] == [
        f"local:watchlist_item:{newest}"
    ]


@pytest.mark.parametrize("field", ("source", "collection"))
@pytest.mark.parametrize(
    "value",
    (
        True,
        False,
        0,
        -1,
        "local:subscription:0",
        "local:watchlist:0",
        "server:subscription:1",
        "server:watchlist:1",
        "local:watchlist_item:1",
    ),
)
def test_scope_ids_reject_bools_nonpositive_wrong_and_foreign_ids(
    db: SubscriptionsDB, field: str, value: object
) -> None:
    result = _payload(_service(db).search_items({field: value}))

    assert result["status"] == "invalid_argument"
    assert result["retryable"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source", 2**63),
        ("collection", 2**63),
        ("source", f"local:subscription:{2**63}"),
        ("collection", f"local:watchlist:{2**63}"),
        ("source", "local:subscription:1٢"),
        ("collection", "local:watchlist:1٢"),
    ),
)
def test_scope_ids_reject_values_outside_sqlite_range_and_non_ascii_digits(
    db: SubscriptionsDB, field: str, value: object
) -> None:
    result = _payload(_service(db).search_items({field: value}))

    assert result["status"] == "invalid_argument"
    assert result["retryable"] is False


def test_scope_accepts_positive_integer_canonical_id_and_numeric_name(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "123")
    collection_id = _collection(db, "456")
    _add_to_collection(db, collection_id, source_id)

    by_ids = _payload(
        _service(db).search_items(
            {
                "source": f"local:subscription:{source_id}",
                "collection": collection_id,
            }
        )
    )
    by_names = _payload(
        _service(db).search_items({"source": "123", "collection": "456"})
    )

    assert by_ids["scope"] == by_names["scope"]
    assert by_ids["scope"]["source"]["id"] == f"local:subscription:{source_id}"
    assert by_ids["scope"]["collection"] == {
        "id": f"local:watchlist:{collection_id}",
        "name": "456",
    }


@pytest.mark.parametrize(
    "arguments",
    (
        {"collection": None},
        {"source": None},
        {"collection": "c" * 257},
        {"source": "s" * 2049},
        {"collection": "   "},
        {"source": "   "},
    ),
)
def test_scope_text_is_nonblank_and_bounded_before_database(
    arguments: dict[str, object],
) -> None:
    calls: list[str] = []
    result = _payload(_service(lambda: calls.append("db")).search_items(arguments))

    assert result["status"] == "invalid_argument"
    assert result["retryable"] is False
    assert calls == []


@pytest.mark.parametrize(
    "statuses",
    (
        [],
        "new",
        ["new", "new"],
        ["NEW"],
        ["new", 7],
        ["unknown"],
    ),
)
def test_supplied_statuses_must_be_nonempty_unique_allowlisted_strings(
    statuses: object,
) -> None:
    result = _payload(_service(lambda: None).search_items({"statuses": statuses}))

    assert result["status"] == "invalid_argument"
    assert result["retryable"] is False


def test_absent_statuses_searches_all_statuses(db: SubscriptionsDB) -> None:
    source_id = _source(db, "All statuses")
    for status in ("new", "reviewed", "ingested", "ignored", "error"):
        _item(db, source_id, status, status=status)

    result = _payload(_service(db).search_items({"limit": 10}))

    assert {item["status"] for item in result["items"]} == {
        "new",
        "reviewed",
        "ingested",
        "ignored",
        "error",
    }


@pytest.mark.parametrize(
    "since",
    (
        "",
        "2026-02-30",
        "2026-08-14 12:00:00",
        "2026-08-14T12:00:00",
        "yesterday",
        123,
    ),
)
def test_since_rejects_invalid_date_values(since: object) -> None:
    result = _payload(_service(lambda: None).search_items({"since": since}))

    assert result["status"] == "invalid_argument"
    assert result["retryable"] is False


def test_since_accepts_date_or_rfc3339_and_is_inclusive_after_utc_normalization(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Date floor")
    boundary = _item(db, source_id, "boundary", published="2026-08-14T08:30:00Z")
    _item(db, source_id, "before", published="2026-08-14T08:29:59Z")

    date_result = _payload(_service(db).search_items({"since": "2026-08-14"}))
    offset_result = _payload(
        _service(db).search_items({"since": "2026-08-14T01:30:00-07:00"})
    )

    assert f"local:watchlist_item:{boundary}" in {
        item["id"] for item in date_result["items"]
    }
    assert [item["id"] for item in offset_result["items"]] == [
        f"local:watchlist_item:{boundary}"
    ]


@pytest.mark.parametrize("limit", (True, False, 0, 51, 1.0, "10"))
def test_limit_accepts_only_integers_from_one_through_fifty(limit: object) -> None:
    result = _payload(_service(lambda: None).search_items({"limit": limit}))

    assert result["status"] == "invalid_argument"
    assert result["retryable"] is False


def test_limit_defaults_to_ten(db: SubscriptionsDB) -> None:
    source_id = _source(db, "Default limit")
    for index in range(11):
        _item(db, source_id, str(index))

    result = _payload(_service(db).search_items({}))

    assert result["returned_count"] == 10
    assert result["has_more"] is True


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"item_id": 1},
        {"item_id": True},
        {"item_id": "local:watchlist_item:0"},
        {"item_id": f"local:watchlist_item:{2**63}"},
        {"item_id": "local:watchlist_item:1٢"},
        {"item_id": "local:subscription:1"},
        {"item_id": "server:watchlist_item:1"},
        {"item_id": "local:watchlist_item:1", "extra": 1},
    ),
)
def test_detail_accepts_only_one_exact_canonical_positive_item_id(
    arguments: dict[str, object],
) -> None:
    calls: list[str] = []
    result = _payload(_service(lambda: calls.append("db")).get_item(arguments))

    assert result["status"] == "invalid_argument"
    assert result["retryable"] is False
    assert calls == []


def test_scope_resolution_precedence_trimming_disambiguation_and_round_trip(
    db: SubscriptionsDB,
) -> None:
    collision = "https://collision.test/feed"
    exact_name_id = _source(db, collision, url="https://name-owner.test/feed")
    _source(db, "URL owner", url=collision)
    unicode_id = _source(db, "Équipe CERT")
    partial_id = _source(db, "Unique Signal Feed")
    ambiguous_ids = [
        _source(db, "Shared Alpha"),
        _source(db, "Shared Beta"),
    ]
    exact_collection_id = _collection(db, "Équipe Watch")
    partial_collection_id = _collection(db, "Unique Collection Signal")
    ambiguous_collection_ids = [
        _collection(db, "Shared Collection A"),
        _collection(db, "Shared Collection B"),
    ]
    service = _service(db)

    assert (
        _payload(service.search_items({"source": f"  {collision}  "}))["scope"][
            "source"
        ]["id"]
        == f"local:subscription:{exact_name_id}"
    )
    assert (
        _payload(service.search_items({"source": "  équipe cert  "}))["scope"][
            "source"
        ]["id"]
        == f"local:subscription:{unicode_id}"
    )
    assert (
        _payload(service.search_items({"source": " signal feed "}))["scope"]["source"][
            "id"
        ]
        == f"local:subscription:{partial_id}"
    )
    assert (
        _payload(service.search_items({"collection": "  équipe watch "}))["scope"][
            "collection"
        ]["id"]
        == f"local:watchlist:{exact_collection_id}"
    )
    assert (
        _payload(service.search_items({"collection": "collection signal"}))["scope"][
            "collection"
        ]["id"]
        == f"local:watchlist:{partial_collection_id}"
    )

    ambiguous_source = _payload(service.search_items({"source": "shared"}))
    assert ambiguous_source == {
        "status": "needs_disambiguation",
        "retryable": False,
        "message": "source is ambiguous; retry with one candidate id",
        "candidates": [
            {
                "id": f"local:subscription:{source_id}",
                "name": name,
                "url": f"https://sources.test/{name.casefold().replace(' ', '-')}",
            }
            for source_id, name in zip(
                ambiguous_ids, ("Shared Alpha", "Shared Beta"), strict=True
            )
        ],
    }
    for candidate in ambiguous_source["candidates"]:
        retried = _payload(service.search_items({"source": candidate["id"]}))
        assert retried["status"] == "ok"
        assert retried["scope"]["source"]["id"] == candidate["id"]

    ambiguous_collection = _payload(
        service.search_items({"collection": "shared collection"})
    )
    assert [candidate["id"] for candidate in ambiguous_collection["candidates"]] == [
        f"local:watchlist:{collection_id}" for collection_id in ambiguous_collection_ids
    ]
    for candidate in ambiguous_collection["candidates"]:
        retried = _payload(service.search_items({"collection": candidate["id"]}))
        assert retried["status"] == "ok"
        assert retried["scope"]["collection"]["id"] == candidate["id"]


def test_scope_disambiguation_is_bounded_and_missing_is_nonretryable(
    db: SubscriptionsDB,
) -> None:
    for index in range(25):
        _source(db, f"Bounded candidate {index:02d}")
    service = _service(db)

    ambiguous = _payload(service.search_items({"source": "bounded candidate"}))
    missing = _payload(service.search_items({"collection": "does not exist"}))

    assert ambiguous["status"] == "needs_disambiguation"
    assert 1 < len(ambiguous["candidates"]) <= 10
    assert missing == {
        "status": "not_found",
        "retryable": False,
        "message": "collection was not found",
    }


def test_collection_and_source_scope_is_an_intersection_not_a_widening(
    db: SubscriptionsDB,
) -> None:
    selected = _source(db, "Selected source")
    member = _source(db, "Collection member")
    collection_id = _collection(db, "Selected collection")
    _add_to_collection(db, collection_id, member)
    _item(db, selected, "selected-item")
    _item(db, member, "member-item")

    result = _payload(
        _service(db).search_items({"source": selected, "collection": collection_id})
    )

    assert result["status"] == "ok"
    assert result["items"] == []
    assert result["returned_count"] == 0


def test_no_match_is_successful_empty_data(db: SubscriptionsDB) -> None:
    source_id = _source(db, "No match")
    _item(db, source_id, "present", title="present evidence")

    result = _payload(_service(db).search_items({"query": "absent-token"}))

    assert result["status"] == "ok"
    assert result["items"] == []
    assert result["returned_count"] == 0
    assert result["has_more"] is False
    assert result["next_cursor"] is None


def test_missing_and_transient_database_dependencies_are_structured_and_scrubbed(
    tmp_path: Path,
) -> None:
    missing = _payload(_service(lambda: None).search_items({}))

    legacy_path = tmp_path / "legacy.db"
    sqlite3.connect(legacy_path).close()
    legacy = SubscriptionsDB(legacy_path, read_only=True)
    try:
        pre_migration = _payload(_service(lambda: legacy).search_items({}))
    finally:
        legacy.close()

    def transient_resolver():
        raise TimeoutError("/secret/path subscriptions timed out with token=secret")

    transient = _payload(_service(transient_resolver).search_items({}))

    expected_permanent = {
        "status": "feature_unavailable",
        "retryable": False,
        "message": (
            "local Watchlists data is unavailable; open Watchlists in Local mode "
            "to initialize or migrate it, then retry"
        ),
    }
    assert missing == expected_permanent
    assert pre_migration == expected_permanent
    assert transient == {
        "status": "feature_unavailable",
        "retryable": True,
        "message": "local Watchlists data is temporarily unavailable; retry later",
    }
    assert "secret" not in json.dumps(transient)


def test_successful_search_has_exact_core_shape_and_membership_truth(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(
        db, "Example CERT", url="https://sources.test/example.xml", paused=True
    )
    collection_id = _collection(db, "Threat Intel")
    _add_to_collection(db, collection_id, source_id)
    item_id = _item(
        db,
        source_id,
        "advisory",
        title="Example advisory",
        author="Example CERT author",
        status="reviewed",
    )

    result = _payload(
        _service(db).search_items(
            {
                "query": "Example advisory",
                "source": source_id,
                "collection": collection_id,
                "statuses": ["reviewed"],
                "limit": 1,
            }
        )
    )

    assert result == {
        "status": "ok",
        "query_mode": "literal_full_text",
        "ordering": "effective_date_desc_item_id_asc",
        "as_of": "2026-08-14T21:30:00Z",
        "snapshot_max_item_id": item_id,
        "returned_count": 1,
        "has_more": False,
        "next_cursor": None,
        "scope": {
            "collection": {
                "id": f"local:watchlist:{collection_id}",
                "name": "Threat Intel",
            },
            "source": {
                "id": f"local:subscription:{source_id}",
                "name": "Example CERT",
                "type": "rss",
                "url": "https://sources.test/example.xml",
                "is_active": True,
                "is_paused": True,
                "created_at": "2026-08-01 09:00:00",
                "updated_at": "2026-08-02 10:00:00",
                "last_checked": "2026-08-13 11:00:00",
                "last_successful_check": "2026-08-13 10:55:00",
            },
        },
        "items": [
            {
                "id": f"local:watchlist_item:{item_id}",
                "title": "Example advisory",
                "url": "https://items.test/advisory",
                "author": "Example CERT author",
                "status": "reviewed",
                "effective_date": "2026-08-14 12:00:00",
                "published_date": "2026-08-14T12:00:00Z",
                "created_at": "2026-08-14T12:05:00Z",
                "updated_at": "2026-08-14T12:10:00Z",
                "source": {
                    "id": f"local:subscription:{source_id}",
                    "name": "Example CERT",
                    "type": "rss",
                    "url": "https://sources.test/example.xml",
                    "is_active": True,
                    "is_paused": True,
                },
                "collections": [
                    {
                        "id": f"local:watchlist:{collection_id}",
                        "name": "Threat Intel",
                    }
                ],
                "collections_truncated": False,
                "evidence": {
                    "content_is_untrusted": True,
                    "snippet": "body evidence",
                },
            }
        ],
    }
    assert "last_updated" not in json.dumps(result)


def test_search_reports_omitted_collection_memberships_honestly(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Many memberships")
    item_id = _item(db, source_id, "many")
    for index in range(25):
        _add_to_collection(db, _collection(db, f"Collection {index:02d}"), source_id)

    item = _payload(_service(db).search_items({"limit": 1}))["items"][0]

    assert item["id"] == f"local:watchlist_item:{item_id}"
    assert len(item["collections"]) == 20
    assert item["collections_truncated"] is True


def test_detail_has_search_metadata_parity_and_distinguishes_null_from_missing(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Detail source", url="https://sources.test/detail")
    collection_id = _collection(db, "Detail collection")
    _add_to_collection(db, collection_id, source_id)
    item_id = _item(db, source_id, "null-body", content=None)
    service = _service(db)

    search_item = _payload(service.search_items({"source": source_id}))["items"][0]
    detail = _payload(service.get_item({"item_id": f"local:watchlist_item:{item_id}"}))
    missing = _payload(service.get_item({"item_id": "local:watchlist_item:999999"}))

    assert detail["status"] == "ok"
    assert detail["item"] == {
        **search_item,
        "evidence": {"content_is_untrusted": True, "content": None},
    }
    assert missing == {
        "status": "not_found",
        "retryable": False,
        "message": "Watchlists item was not found",
    }


@pytest.mark.parametrize("handler_name", ("search_items", "get_item"))
def test_server_runtime_short_circuits_database_with_exact_outcome(
    handler_name: str,
) -> None:
    calls: list[str] = []
    service = _service(
        lambda: calls.append("db"), runtime_source_loader=lambda: "server"
    )
    arguments = (
        {} if handler_name == "search_items" else {"item_id": "local:watchlist_item:1"}
    )

    result = _payload(getattr(service, handler_name)(arguments))

    assert result == {
        "status": "unsupported",
        "retryable": False,
        "message": (
            "server Watchlists search is not supported; switch Watchlists to "
            "Local before retrying"
        ),
    }
    assert calls == []


def test_runtime_source_is_loaded_per_call_and_invalid_arguments_win(
    db: SubscriptionsDB,
) -> None:
    state = {"active_source": "server"}
    resolver_calls: list[str] = []

    def resolver():
        resolver_calls.append("db")
        return db

    service = _service(resolver, runtime_source_loader=lambda: dict(state))

    invalid = _payload(service.search_items({"bogus": True}))
    server = _payload(service.search_items({}))
    state["active_source"] = "local"
    local = _payload(service.search_items({}))

    assert invalid["status"] == "invalid_argument"
    assert server["status"] == "unsupported"
    assert local["status"] == "ok"
    assert resolver_calls == ["db"]


@pytest.mark.parametrize(
    "runtime_state", (None, {}, {"active_source": []}, "damaged", object())
)
def test_absent_or_malformed_runtime_state_uses_local_default(
    db: SubscriptionsDB, runtime_state: object
) -> None:
    result = _payload(
        _service(db, runtime_source_loader=lambda: runtime_state).search_items({})
    )

    assert result["status"] == "ok"


def _database_snapshot(db: SubscriptionsDB) -> dict[str, object]:
    tables = (
        "schema_version",
        "subscriptions",
        "subscription_items",
        "watchlists",
        "watchlist_sources",
    )
    with db.transaction() as conn:
        return {
            "tables": {
                table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
                for table in tables
            },
            "schema": [
                tuple(row)
                for row in conn.execute(
                    "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
                )
            ],
        }


def test_search_and_detail_are_read_only_against_mutable_database_and_runtime(
    db: SubscriptionsDB,
) -> None:
    source_id = _source(db, "Read only", active=False, paused=True)
    collection_id = _collection(db, "Read only collection")
    _add_to_collection(db, collection_id, source_id)
    item_id = _item(db, source_id, "read-only", status="ignored")
    runtime_state = {"active_source": "local", "policy_canary": ["unchanged"]}
    before_db = _database_snapshot(db)
    before_runtime = deepcopy(runtime_state)
    service = _service(db, runtime_source_loader=lambda: runtime_state)

    assert _payload(service.search_items({"statuses": ["ignored"]}))["status"] == "ok"
    assert (
        _payload(service.get_item({"item_id": f"local:watchlist_item:{item_id}"}))[
            "status"
        ]
        == "ok"
    )

    assert _database_snapshot(db) == before_db
    assert runtime_state == before_runtime
