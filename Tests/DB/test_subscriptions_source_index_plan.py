"""task-34399 item 3: ``idx_subscriptions_source``.

The exact-source lookups that OPML import runs inside ``BEGIN IMMEDIATE``
(``_find_exact_source_id``, ``get_subscription_id_by_source``) share one
query shape -- ``WHERE source = ?`` -- which used to full-scan
``subscriptions``. The index follows this store's house migration pattern:
it lives in the idempotent ``CREATE INDEX IF NOT EXISTS`` block of
``_initialize_schema``, so every open (fresh or existing v2 database)
creates it with no schema-version bump and no data migration -- the same
way ``idx_subscriptions_last_checked`` and friends shipped.
"""

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

_SOURCE = "https://example.com/feed.xml"


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs-source.db"), client_id="test")


def test_index_exists_on_the_source_column(db):
    columns = [
        row[2]
        for row in db.conn.execute("PRAGMA index_info(idx_subscriptions_source)")
    ]

    assert columns == ["source"]


def test_exact_source_lookup_resolves_through_the_index(db):
    source_id = db.add_subscription(
        name="Example", type="rss", source=_SOURCE
    )

    assert db.get_subscription_id_by_source(_SOURCE) == source_id
    assert db.get_subscription_id_by_source("https://other.example/rss") is None


def test_reopening_an_existing_v2_database_restores_the_index(tmp_path):
    """The idempotent-schema route must reach databases that predate the
    index: no version bump is involved, so this simulates a v2 file that
    lost the index (i.e. one created before this change)."""
    path = str(tmp_path / "subs-preexisting.db")
    first = SubscriptionsDB(path, client_id="test")
    first.conn.execute("DROP INDEX idx_subscriptions_source")
    first.conn.commit()
    # Sanity: the fixture really removed it before reopening.
    assert (
        first.conn.execute(
            "PRAGMA index_info(idx_subscriptions_source)"
        ).fetchone()
        is None
    )

    second = SubscriptionsDB(path, client_id="test")

    columns = [
        row[2]
        for row in second.conn.execute("PRAGMA index_info(idx_subscriptions_source)")
    ]
    assert columns == ["source"]


def test_exact_source_lookup_does_not_scan_subscriptions(db):
    """The plan pin (scripts/check_index_plan_pins.py).

    A wall-clock budget would be a flake on shared CI; the query plan is
    the durable form of "does not read every row". The plan is only
    evidence when captured the way production runs it: Subscriptions_DB
    never runs ANALYZE, so no user's database has a sqlite_stat1 and the
    planner works from default row-count estimates.
    """
    db.add_subscription(name="Example", type="rss", source=_SOURCE)

    assert (
        db.conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='sqlite_stat1'"
        ).fetchone()
        is None
    ), (
        "this fixture must reproduce the no-stats production state; a plan "
        "captured with sqlite_stat1 present is not the plan users run"
    )

    lookup_sql = "SELECT id FROM subscriptions WHERE source = ?"
    plan = " | ".join(
        str(row[3])
        for row in db.conn.execute("EXPLAIN QUERY PLAN " + lookup_sql, (_SOURCE,))
    )

    assert "idx_subscriptions_source" in plan
    assert "SCAN subscriptions" not in plan, plan
