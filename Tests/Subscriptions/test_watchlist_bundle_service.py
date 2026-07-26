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
