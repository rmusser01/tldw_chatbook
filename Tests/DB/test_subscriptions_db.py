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
    cursor = db.conn.cursor()
    cursor.execute(
        "INSERT INTO subscription_filters (subscription_id, name, conditions, action) VALUES (?, ?, ?, ?)",
        (1, "include ai", "{}", "include"),
    )
    db.conn.commit()
