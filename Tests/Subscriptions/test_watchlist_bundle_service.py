import sqlite3

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


@pytest.fixture
def service(db):
    return WatchlistBundleService(db)


def test_create_and_list(service):
    created = service.create("Morning AI Brief", tags=["ai", "daily"])
    assert created["name"] == "Morning AI Brief"
    assert created["tags"] == ["ai", "daily"]

    listed = service.list_watchlists()
    assert [row["name"] for row in listed] == ["Morning AI Brief"]


def test_name_collision_is_case_insensitive_and_suffixes(service):
    service.create("Unsorted")
    second = service.create("unsorted")
    third = service.create("UNSORTED")
    assert second["name"] == "unsorted (2)"
    assert third["name"] == "UNSORTED (3)"


@pytest.mark.parametrize(
    ("policy", "outcome", "expected_name"),
    [
        ("return_existing", "existing", "Security"),
        ("auto_suffix", "created", "security (2)"),
    ],
)
def test_create_with_sources_applies_explicit_collision_policy(
    service, db, policy, outcome, expected_name
):
    existing = service.create("Security", description="keep")
    original_source = db.add_subscription(
        name="Original", type="rss", source="https://feeds.example/original"
    )
    requested_source = db.add_subscription(
        name="Requested", type="rss", source="https://feeds.example/requested"
    )
    service.add_source(existing["id"], original_source)

    result = service.create_with_sources(
        "security",
        description="replace",
        tags=["new"],
        source_ids=[requested_source],
        if_exists=policy,
    )

    assert result["outcome"] == outcome
    assert result["watchlist"]["name"] == expected_name
    if policy == "return_existing":
        assert result["watchlist"]["description"] == "keep"
        assert service.list_sources(existing["id"]) == [original_source]
    else:
        assert service.list_sources(result["watchlist"]["id"]) == [requested_source]


def test_create_with_sources_conflict_leaves_existing_unchanged(service, db):
    existing = service.create("Security", description="keep")
    source_id = db.add_subscription(
        name="Requested", type="rss", source="https://feeds.example/requested"
    )

    with pytest.raises(ValueError, match="already exists"):
        service.create_with_sources(
            "security",
            description="replace",
            tags=["new"],
            source_ids=[source_id],
            if_exists="conflict",
        )

    assert service.list_watchlists() == [existing]
    assert service.list_sources(existing["id"]) == []


@pytest.mark.parametrize(
    ("existing_name", "requested_name"),
    [("ÄI", "äi"), ("Straße", "STRASSE")],
)
@pytest.mark.parametrize("policy", ["conflict", "return_existing", "auto_suffix"])
def test_create_with_sources_uses_unicode_casefold_for_every_collision_policy(
    service, existing_name, requested_name, policy
):
    existing = service.create(existing_name)

    if policy == "conflict":
        with pytest.raises(ValueError, match="already exists"):
            service.create_with_sources(
                requested_name,
                description=None,
                tags=None,
                source_ids=[],
                if_exists=policy,
            )
        assert service.list_watchlists() == [existing]
        return

    result = service.create_with_sources(
        requested_name,
        description=None,
        tags=None,
        source_ids=[],
        if_exists=policy,
    )

    if policy == "return_existing":
        assert result["outcome"] == "existing"
        assert result["watchlist"] == existing
    else:
        assert result["outcome"] == "created"
        assert result["watchlist"]["name"] == f"{requested_name} (2)"


def test_create_with_sources_rolls_back_collection_when_source_is_missing(service):
    with pytest.raises(KeyError, match="source"):
        service.create_with_sources(
            "Atomic",
            description=None,
            tags=None,
            source_ids=[999_999],
            if_exists="conflict",
        )

    assert service.list_watchlists() == []


def test_create_with_sources_rolls_back_collection_when_membership_insert_fails(
    service, db
):
    source_id = db.add_subscription(
        name="Blocked", type="rss", source="https://feeds.example/blocked"
    )
    db.conn.execute(
        "CREATE TRIGGER reject_watchlist_membership "
        "BEFORE INSERT ON watchlist_sources "
        "BEGIN SELECT RAISE(ABORT, 'blocked membership'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="blocked membership"):
        service.create_with_sources(
            "Atomic",
            description=None,
            tags=None,
            source_ids=[source_id],
            if_exists="conflict",
        )

    assert service.list_watchlists() == []


def test_create_with_sources_accepts_100_memberships_and_rejects_101(service, db):
    source_ids = [
        db.add_subscription(
            name=f"Source {index}",
            type="rss",
            source=f"https://feeds.example/{index}",
        )
        for index in range(101)
    ]

    result = service.create_with_sources(
        "Maximum",
        description=None,
        tags=None,
        source_ids=source_ids[:100],
        if_exists="conflict",
    )

    assert result["membership_count"] == 100
    with pytest.raises(ValueError, match="at most 100"):
        service.create_with_sources(
            "Too many",
            description=None,
            tags=None,
            source_ids=source_ids,
            if_exists="conflict",
        )
    assert [row["name"] for row in service.list_watchlists()] == ["Maximum"]


def test_update_sources_is_atomic_and_rejects_missing_sources(service, db):
    watchlist = service.create("Security")
    keep = db.add_subscription(name="Keep", type="rss", source="https://feeds.example/keep")
    remove = db.add_subscription(name="Remove", type="rss", source="https://feeds.example/remove")
    service.add_source(watchlist["id"], keep)
    service.add_source(watchlist["id"], remove)

    with pytest.raises(KeyError, match="source"):
        service.update_sources(
            watchlist["id"], add_ids=[999_999], remove_ids=[remove]
        )

    assert service.list_sources(watchlist["id"]) == [keep, remove]

    result = service.update_sources(watchlist["id"], add_ids=[], remove_ids=[remove])
    assert result["added"] == 0
    assert result["removed"] == 1
    assert service.list_sources(watchlist["id"]) == [keep]


def test_update_sources_rolls_back_add_when_later_remove_fails(service, db):
    watchlist = service.create("Security")
    remove = db.add_subscription(
        name="Remove", type="rss", source="https://feeds.example/remove"
    )
    add = db.add_subscription(
        name="Add", type="rss", source="https://feeds.example/add"
    )
    service.add_source(watchlist["id"], remove)
    db.conn.execute(
        "CREATE TRIGGER reject_watchlist_membership_delete "
        "BEFORE DELETE ON watchlist_sources "
        "BEGIN SELECT RAISE(ABORT, 'blocked removal'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="blocked removal"):
        service.update_sources(
            watchlist["id"], add_ids=[add], remove_ids=[remove]
        )

    assert service.list_sources(watchlist["id"]) == [remove]


def test_rename_also_avoids_collision(service):
    service.create("Security")
    other = service.create("Papers")
    renamed = service.rename(other["id"], "security")
    assert renamed["name"] == "security (2)"


def test_membership_add_remove_and_idempotent_add(service, db):
    watchlist = service.create("Morning")
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")

    service.add_source(watchlist["id"], source_id)
    service.add_source(watchlist["id"], source_id)  # idempotent
    assert service.list_sources(watchlist["id"]) == [source_id]

    service.remove_source(watchlist["id"], source_id)
    assert service.list_sources(watchlist["id"]) == []


def test_source_can_belong_to_multiple_watchlists(service, db):
    first = service.create("Morning")
    second = service.create("Security")
    source_id = db.add_subscription(name="HN", type="rss", source="https://b.example/f")

    service.add_source(first["id"], source_id)
    service.add_source(second["id"], source_id)

    assert service.list_sources(first["id"]) == [source_id]
    assert service.list_sources(second["id"]) == [source_id]


def test_delete_removes_membership_but_not_sources(service, db):
    watchlist = service.create("Morning")
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    service.add_source(watchlist["id"], source_id)

    service.delete(watchlist["id"])

    assert service.list_watchlists() == []
    assert db.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 1
    # Verify cascade delete: membership row should be gone
    assert service.list_sources(watchlist["id"]) == []
    assert db.conn.execute(
        "SELECT COUNT(*) FROM watchlist_sources WHERE watchlist_id = ?",
        (watchlist["id"],)
    ).fetchone()[0] == 0


def test_list_watchlists_limit_actually_limits(service):
    for index in range(5):
        service.create(f"List {index}")

    limited = service.list_watchlists(limit=2)
    assert len(limited) == 2

    all_of_them = service.list_watchlists()
    assert len(all_of_them) == 5

    offset_page = service.list_watchlists(limit=2, offset=2)
    assert [row["name"] for row in offset_page] == [row["name"] for row in all_of_them[2:4]]


def test_create_rejects_blank_name(service):
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        service.create("   ")

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        service.create("")


def test_rename_rejects_blank_name(service):
    watchlist = service.create("Security")
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        service.rename(watchlist["id"], "   ")

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        service.rename(watchlist["id"], "")


def test_rename_to_own_exact_name_has_no_suffix(service):
    watchlist = service.create("Security")
    renamed = service.rename(watchlist["id"], "Security")
    assert renamed["name"] == "Security"


def test_rename_to_own_case_variant_has_no_suffix(service):
    watchlist = service.create("Security")
    renamed = service.rename(watchlist["id"], "security")
    assert renamed["name"] == "security"


def test_list_source_rows_returns_names_and_types(service, db):
    watchlist = service.create("Morning")
    a = db.add_subscription(name="ArXiv: AI", type="rss", source="https://a.example/f")
    b = db.add_subscription(name="anthropic.com", type="url", source="https://b.example/")
    service.add_source(watchlist["id"], a)
    service.add_source(watchlist["id"], b)

    rows = service.list_source_rows(watchlist["id"])
    assert [r["name"] for r in rows] == ["ArXiv: AI", "anthropic.com"]
    assert {r["type"] for r in rows} == {"rss", "url"}
    assert {r["id"] for r in rows} == {a, b}


def test_list_source_rows_is_empty_for_a_watchlist_with_no_sources(service):
    watchlist = service.create("Empty")
    assert service.list_source_rows(watchlist["id"]) == []


def test_list_source_rows_uses_a_single_query(service, db, monkeypatch):
    watchlist = service.create("Morning")
    for index in range(6):
        service.add_source(
            watchlist["id"],
            db.add_subscription(name=f"S{index}", type="rss", source=f"https://s{index}.example/f"),
        )

    class _Counting:
        def __init__(self, inner):
            self._inner = inner
            self.execute_count = 0

        def execute(self, *args, **kwargs):
            self.execute_count += 1
            return self._inner.execute(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    counting = _Counting(db.conn)
    monkeypatch.setattr(type(db), "conn", property(lambda self: counting))
    service.list_source_rows(watchlist["id"])
    assert counting.execute_count == 1


class _CountingConnection:
    """Counts ``execute`` calls on the wrapped connection (Task 1's pattern),
    reused here so Task 7's two new resolvers pin the same "exactly one
    query, regardless of row count" invariant as ``list_source_rows`` above.
    """

    def __init__(self, inner):
        self._inner = inner
        self.execute_count = 0

    def execute(self, *args, **kwargs):
        self.execute_count += 1
        return self._inner.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_list_all_source_rows_returns_every_source_sorted_by_name(service, db):
    b = db.add_subscription(name="Bravo", type="rss", source="https://b.example/f")
    a = db.add_subscription(name="alpha", type="url", source="https://a.example/")

    rows = service.list_all_source_rows()
    assert [r["id"] for r in rows] == [a, b]
    assert [r["name"] for r in rows] == ["alpha", "Bravo"]


def test_list_all_source_rows_is_empty_with_no_sources(service):
    assert service.list_all_source_rows() == []


def test_list_all_source_rows_uses_a_single_query(service, db, monkeypatch):
    for index in range(6):
        db.add_subscription(name=f"S{index}", type="rss", source=f"https://s{index}.example/f")

    counting = _CountingConnection(db.conn)
    monkeypatch.setattr(type(db), "conn", property(lambda self: counting))
    service.list_all_source_rows()
    assert counting.execute_count == 1


def test_list_unassigned_source_rows_excludes_sources_in_any_watchlist(service, db):
    watchlist = service.create("Morning")
    assigned = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    unassigned = db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
    service.add_source(watchlist["id"], assigned)

    rows = service.list_unassigned_source_rows()
    assert [r["id"] for r in rows] == [unassigned]
    assert rows[0]["name"] == "Krebs"


def test_list_unassigned_source_rows_is_empty_when_everything_is_assigned(service, db):
    watchlist = service.create("Morning")
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    service.add_source(watchlist["id"], source_id)

    assert service.list_unassigned_source_rows() == []


def test_list_unassigned_source_rows_uses_a_single_query(service, db, monkeypatch):
    watchlist = service.create("Morning")
    for index in range(6):
        source_id = db.add_subscription(
            name=f"S{index}", type="rss", source=f"https://s{index}.example/f"
        )
        if index % 2 == 0:
            service.add_source(watchlist["id"], source_id)

    counting = _CountingConnection(db.conn)
    monkeypatch.setattr(type(db), "conn", property(lambda self: counting))
    service.list_unassigned_source_rows()
    assert counting.execute_count == 1


def test_list_watchlists_for_source_is_the_mirror_of_list_sources(service, db):
    """Review wave, M5: one query for "which watchlists is this in".

    The source-first assign flow used to answer this by calling
    `list_sources` once per watchlist. This is the same answer, and it has to
    agree with `list_sources` exactly -- a candidate list built from a
    disagreeing membership set would offer a watchlist the source is already
    in, or hide one it is not.
    """
    left = service.create("Left")
    right = service.create("Right")
    service.create("Neither")
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/f"
    )
    other_id = db.add_subscription(
        name="HN", type="rss", source="https://b.example/f"
    )

    service.add_source(left["id"], source_id)
    service.add_source(right["id"], source_id)
    service.add_source(right["id"], other_id)

    assert sorted(service.list_watchlists_for_source(source_id)) == sorted(
        [left["id"], right["id"]]
    )
    assert service.list_watchlists_for_source(other_id) == [right["id"]]
    # Agrees with the other direction, watchlist by watchlist.
    for watchlist in service.list_watchlists():
        expected = source_id in service.list_sources(watchlist["id"])
        assert (
            watchlist["id"] in service.list_watchlists_for_source(source_id)
        ) is expected


def test_list_watchlists_for_source_is_empty_for_an_unassigned_source(service, db):
    source_id = db.add_subscription(
        name="Loose", type="rss", source="https://c.example/f"
    )
    assert service.list_watchlists_for_source(source_id) == []
    assert service.list_watchlists_for_source(9999) == []
