import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.fts_backfill import backfill_subscription_items_fts


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


def _drop_ai_trigger(db):
    """Simulate pre-existing (legacy) rows: with no `_ai` trigger, a row
    inserted afterwards is never written into the FTS index, exactly like a
    row that already existed in `subscription_items` before the FTS index
    was ever created on a real upgraded database. Matches the established
    pattern in Tests/DB/test_subscriptions_db_watchlists.py."""
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER subscription_items_fts_ai")


def _insert_legacy_item(db, subscription_id, url, title, content):
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscription_items "
            "(subscription_id, url, title, content, content_kind, content_format) "
            "VALUES (?, ?, ?, ?, 'article', 'text')",
            (subscription_id, url, title, content),
        )


def test_wired_backfill_makes_preexisting_items_searchable(db):
    """task-688: the upgrade path end to end. A database with items that
    predate the FTS index becomes fully searchable after the wired
    (looping-to-completion) path runs, not just after a single chunk."""
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _drop_ai_trigger(db)
    for index in range(12):
        _insert_legacy_item(
            db,
            source_id,
            f"https://a.example/{index}",
            f"Item {index}",
            "retrieval quality rubric",
        )

    # Confirm the rows really are unindexed first, or this test would pass
    # vacuously.
    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts_docsize"
    ).fetchone()[0] == 0

    total = backfill_subscription_items_fts(db, chunk_size=5)

    assert total == 12
    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("rubric",),
    ).fetchone()[0] == 12


def test_wired_backfill_is_idempotent_once_complete(db):
    """A second call after completion indexes nothing and does not corrupt
    the index (fts5 'integrity-check' stays clean)."""
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _drop_ai_trigger(db)
    _insert_legacy_item(db, source_id, "https://a.example/1", "Item", "alpha content")

    first_total = backfill_subscription_items_fts(db)
    assert first_total == 1

    second_total = backfill_subscription_items_fts(db)
    assert second_total == 0

    # Raises DatabaseError if the FTS index is actually corrupt.
    db.conn.execute(
        "INSERT INTO subscription_items_fts(subscription_items_fts) VALUES ('integrity-check')"
    )


def test_wired_backfill_on_already_fully_indexed_db_is_a_noop(db):
    """A database with no legacy backlog at all (the common case, since the
    `_ai` trigger indexes every item going forward) should not error and
    should report nothing to do."""
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _insert_legacy_item(db, source_id, "https://a.example/1", "Item", "alpha content")

    assert backfill_subscription_items_fts(db) == 0
