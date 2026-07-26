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


MIGRATION_KEY = "folders_to_watchlists"


def test_migrate_folders_groups_by_folder(service, db):
    first = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f",
                                folder="Research")
    second = db.add_subscription(name="HN", type="rss", source="https://b.example/f",
                                 folder="Research")
    third = db.add_subscription(name="Krebs", type="rss", source="https://c.example/f",
                                folder="Security")

    assert service.migrate_folders() is True

    names = {row["name"] for row in service.list_watchlists()}
    assert names == {"Research", "Security"}

    by_name = {row["name"]: row["id"] for row in service.list_watchlists()}
    assert sorted(service.list_sources(by_name["Research"])) == sorted([first, second])
    assert service.list_sources(by_name["Security"]) == [third]


def test_migrate_folders_puts_folderless_sources_in_unsorted(service, db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")

    service.migrate_folders()

    by_name = {row["name"]: row["id"] for row in service.list_watchlists()}
    assert "Unsorted" in by_name
    assert service.list_sources(by_name["Unsorted"]) == [source_id]


def test_migrate_folders_runs_once(service, db):
    db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f", folder="Research")

    assert service.migrate_folders() is True
    assert service.migrate_folders() is False
    assert len(service.list_watchlists()) == 1

    marker = db.conn.execute(
        "SELECT COUNT(*) FROM watchlist_migration_state WHERE key = ?", (MIGRATION_KEY,)
    ).fetchone()[0]
    assert marker == 1


def test_migrate_folders_is_noop_with_no_sources(service):
    assert service.migrate_folders() is True
    assert service.list_watchlists() == []


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
