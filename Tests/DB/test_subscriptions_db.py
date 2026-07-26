import pytest
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


def test_watchlists_columns_exist(db):
    cursor = db.conn.cursor()
    cols = {row[1] for row in cursor.execute("PRAGMA table_info(subscription_items)")}
    assert "queued_for_briefing" in cols
    assert "run_id" in cols
    assert "alert_matches" in cols

    cols = {row[1] for row in cursor.execute("PRAGMA table_info(subscription_filters)")}
    assert "priority" in cols
    assert "is_include_required" in cols


def test_subscription_filters_action_constraint_allows_include(db):
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )
    cursor = db.conn.cursor()
    cursor.execute(
        "INSERT INTO subscription_filters (subscription_id, name, conditions, action) VALUES (?, ?, ?, ?)",
        (source_id, "include ai", "{}", "include"),
    )
    db.conn.commit()


def test_foreign_keys_enforced_on_runtime_connection(db):
    # PRAGMA foreign_keys is per-connection and defaults to OFF. The pragma in
    # _initialize_schema runs on a connection that is closed immediately after,
    # so it does not cover the thread-local connection everything else uses.
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_deleting_subscription_cascades_to_its_items(db):
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscription_items (subscription_id, url, title) VALUES (?, ?, ?)",
            (source_id, "https://a.example/1", "An item"),
        )

    with db.transaction() as conn:
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (source_id,))

    orphans = db.conn.execute("SELECT COUNT(*) FROM subscription_items").fetchone()[0]
    assert orphans == 0
