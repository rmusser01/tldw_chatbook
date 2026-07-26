import tempfile
from pathlib import Path

import pytest

from tldw_chatbook.DB import base_db as base_db_module
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB


class _TrackedConnection:
    def __init__(self, conn):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "closed", False)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        if name in {"_conn", "closed"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._conn, name, value)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._conn.__exit__(exc_type, exc, traceback)

    def close(self):
        object.__setattr__(self, "closed", True)
        return self._conn.close()


@pytest.mark.unit
def test_subscriptions_db_schema_initialization_reuses_and_closes_connection(
    tmp_path, monkeypatch
):
    """Updated for task-689.

    Before task-689, ``_initialize_schema`` opened a throwaway connection via
    ``with closing(self._get_connection()) as conn:`` and closed it
    immediately -- harmless for a file database (a later connection reaches
    the same file) but the exact reason a ``:memory:`` database ended up
    with an empty ``.conn``: the schema landed on a connection nothing else
    could ever see. This test used to assert that immediate close as the
    expected behavior; it now asserts the fixed behavior instead.

    ``_initialize_schema`` now runs on ``self.conn``, the thread-local
    connection every other method reuses, so schema init opens exactly one
    connection for the whole object, and that connection must still not be
    leaked -- the original motivation for wrapping schema init in
    ``closing(...)`` at all (commit ec489c052). It just needs to stay open
    until ``db.close()``, not close immediately after init.
    """
    connections = []
    original_connect = base_db_module.sqlite3.connect

    def tracked_connect(*args, **kwargs):
        conn = _TrackedConnection(original_connect(*args, **kwargs))
        connections.append(conn)
        return conn

    monkeypatch.setattr(base_db_module.sqlite3, "connect", tracked_connect)

    db = SubscriptionsDB(tmp_path / "subscriptions.db")
    try:
        assert len(connections) == 1
        assert connections[0].closed is False
    finally:
        db.close()

    assert connections[0].closed is True


@pytest.mark.unit
def test_subscriptions_db_basic_add_and_list():
    # Use a temporary sqlite file to avoid thread issues with ':memory:'
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "subscriptions.db"
        db = SubscriptionsDB(str(db_path))
        try:
            # Add a subscription
            sub_id = db.add_subscription(
                name="Test Feed",
                type="rss",
                source="https://example.com/feed.xml",
                tags=["news", "tech"],
                priority=3,
                folder="Smoke",
            )
            assert isinstance(sub_id, int) and sub_id > 0

            # Fetch it back
            sub = db.get_subscription(sub_id)
            assert sub is not None
            assert sub["name"] == "Test Feed"
            assert sub["type"] == "rss"

            # Record a successful check with one new item
            db.record_check_result(
                subscription_id=sub_id,
                items=[
                    {
                        "url": "https://example.com/article-1?utm=abc",
                        "title": "An Article",
                        "content_hash": "hash1",
                    }
                ],
                stats={
                    "response_time_ms": 120,
                    "bytes_transferred": 1024,
                    "new_items_found": 1,
                },
            )

            # New items should be present
            items = db.get_new_items()
            assert len(items) >= 1
            assert items[0]["title"]
        finally:
            db.close()
