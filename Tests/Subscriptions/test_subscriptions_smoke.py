import tempfile
from pathlib import Path

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB


@pytest.mark.unit
def test_subscriptions_db_basic_add_and_list():
    # Use a temporary sqlite file to avoid thread issues with ':memory:'
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "subscriptions.db"
        db = SubscriptionsDB(str(db_path))

        # Add a subscription
        sub_id = db.add_subscription(
            name="Test Feed",
            type="rss",
            source="https://example.com/feed.xml",
            tags=["news", "tech"],
            priority=3,
            folder="Smoke"
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
            items=[{
                "url": "https://example.com/article-1?utm=abc",
                "title": "An Article",
                "content_hash": "hash1"
            }],
            stats={"response_time_ms": 120, "bytes_transferred": 1024, "new_items_found": 1}
        )

        # New items should be present
        items = db.get_new_items()
        assert len(items) >= 1
        assert items[0]["title"]

