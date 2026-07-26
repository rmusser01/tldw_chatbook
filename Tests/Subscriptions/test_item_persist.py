import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


@pytest.fixture
def source_id(db):
    return db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")


def test_persists_full_column_set(db, source_id):
    item = {
        "url": "https://a.example/1",
        "title": "RAG Evaluation",
        "content": "retrieval quality rubric",
        "content_kind": "article",
        "content_format": "text",
        "content_hash": "hash-1",
        "author": "A. Author",
        "canonical_url": "https://a.example/1",
        "change_percentage": 12.5,
        "diff_summary": "+2 -1",
        "change_type": "content",
        "alert_matches": [7],
    }
    with db.transaction() as conn:
        persist_subscription_item(conn, source_id, item, run_id=42, now="2026-07-25T00:00:00Z")

    row = db.conn.execute(
        "SELECT content, content_kind, content_format, status, run_id, alert_matches, "
        "canonical_url, change_percentage, diff_summary, change_type "
        "FROM subscription_items WHERE url = ?",
        ("https://a.example/1",),
    ).fetchone()

    assert row[0] == "retrieval quality rubric"
    assert row[1] == "article"
    assert row[2] == "text"
    assert row[3] == "new"
    assert row[4] == 42
    assert row[5] == "[7]"
    assert row[6] == "https://a.example/1"
    assert row[7] == 12.5
    assert row[8] == "+2 -1"
    assert row[9] == "content"


def test_returns_the_real_row_id_on_upsert_not_lastrowid(db, source_id):
    """Regression for fix round 2.

    ``cursor.lastrowid`` reflects the last genuine INSERT on the
    *connection*, not the row an ``ON CONFLICT ... DO UPDATE`` actually
    touched. Insert A, then insert B (B is now the connection's
    lastrowid), then re-upsert A: the returned id must be A's, not B's.
    This is reachable in production because ``_upsert_subscription_items``
    never sets ``canonical_url``, so when the same item later round-trips
    through ``_add_subscription_item`` (whose guard compares against
    ``canonical_url``), the guard's NULL comparison never matches, falls
    through, and the upsert resolves as an UPDATE on a row that is not the
    most recently inserted one on that connection.
    """
    item_a = {"url": "https://a.example/1", "title": "A", "content_hash": "h1", "content": "a"}
    item_b = {"url": "https://a.example/2", "title": "B", "content_hash": "h2", "content": "b"}
    with db.transaction() as conn:
        id_a = persist_subscription_item(conn, source_id, item_a, run_id=1, now="2026-07-25T00:00:00Z")
        id_b = persist_subscription_item(conn, source_id, item_b, run_id=1, now="2026-07-25T00:00:00Z")
        assert id_b != id_a  # sanity: two distinct rows exist

        reupsert_id = persist_subscription_item(
            conn, source_id, item_a, run_id=2, now="2026-07-25T01:00:00Z"
        )

    assert reupsert_id == id_a
    assert reupsert_id != id_b

    row = db.conn.execute(
        "SELECT id FROM subscription_items WHERE url = ?", ("https://a.example/1",)
    ).fetchone()
    assert row[0] == id_a


@pytest.mark.parametrize("preserved_status", ["reviewed", "ignored", "ingested"])
def test_upsert_preserves_status_from_user_action(db, source_id, preserved_status):
    """Regression for fix round 1, Finding 2.

    ``reviewed``, ``ignored``, and ``ingested`` are all set by a deliberate
    user action on the item (see Event_Handlers/subscription_events.py for
    the accept/ignore/review actions). Because execute_run re-upserts every
    kept item on every poll, an unchanged item must not silently flip back
    to ``new`` and resurface as unread just because a scheduled re-fetch
    happened to touch the row.
    """
    item = {"url": "https://a.example/1", "title": "T", "content_hash": "h", "content": "body"}
    with db.transaction() as conn:
        persist_subscription_item(conn, source_id, item, run_id=1, now="2026-07-25T00:00:00Z")
        conn.execute(
            "UPDATE subscription_items SET status = ? WHERE url = ?",
            (preserved_status, "https://a.example/1"),
        )
        persist_subscription_item(conn, source_id, item, run_id=2, now="2026-07-25T01:00:00Z")

    row = db.conn.execute(
        "SELECT status, run_id FROM subscription_items WHERE url = ?", ("https://a.example/1",)
    ).fetchone()
    assert row[0] == preserved_status
    assert row[1] == 2


def test_upsert_resets_error_status_to_new(db, source_id):
    """``error`` is not a user action -- a successful re-fetch should clear it."""
    item = {"url": "https://a.example/1", "title": "T", "content_hash": "h", "content": "body"}
    with db.transaction() as conn:
        persist_subscription_item(conn, source_id, item, run_id=1, now="2026-07-25T00:00:00Z")
        conn.execute(
            "UPDATE subscription_items SET status = 'error' WHERE url = ?",
            ("https://a.example/1",),
        )
        persist_subscription_item(conn, source_id, item, run_id=2, now="2026-07-25T01:00:00Z")

    row = db.conn.execute(
        "SELECT status, run_id FROM subscription_items WHERE url = ?", ("https://a.example/1",)
    ).fetchone()
    assert row[0] == "new"
    assert row[1] == 2


def test_rejects_invalid_kind_format_pairing(db, source_id):
    item = {
        "url": "https://a.example/1",
        "title": "T",
        "content_hash": "h",
        "content_kind": "change",
        "content_format": "markdown",
    }
    with pytest.raises(ValueError, match="content_kind"):
        with db.transaction() as conn:
            persist_subscription_item(conn, source_id, item, run_id=1, now="2026-07-25T00:00:00Z")
