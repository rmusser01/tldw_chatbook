import pytest

pytestmark = pytest.mark.unit


def test_normalize_carries_the_reader_fields():
    """The reader cannot render what the read path drops.

    `get_new_items` is `SELECT i.*`, so the body is present in the row. This
    normalizer rebuilt an explicit dict and omitted it, which meant every
    downstream consumer saw a title-only item no matter what was persisted.
    """
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )

    row = {
        "id": 7,
        "subscription_id": 3,
        "title": "Claude Opus 4.5 is now available",
        "url": "https://example.test/a",
        "content": "body text that must survive",
        "content_kind": "article",
        "content_format": "markdown",
        "change_percentage": None,
        "change_type": None,
        "diff_summary": None,
    }

    item = normalize_watchlist_item("local", row)

    assert item["content"] == "body text that must survive"
    assert item["content_kind"] == "article"
    assert item["content_format"] == "markdown"


def test_normalize_carries_the_change_fields():
    """A `change` item renders from these three and nothing else."""
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )

    row = {
        "id": 8,
        "subscription_id": 3,
        "title": "anthropic.com/news",
        "url": "https://anthropic.test/news",
        "content": "+ added line\n- removed line",
        "content_kind": "change",
        "content_format": "diff",
        "change_percentage": 12.0,
        "change_type": "structural",
        "diff_summary": "2 lines changed",
    }

    item = normalize_watchlist_item("local", row)

    assert item["content_kind"] == "change"
    assert item["change_percentage"] == 12.0
    assert item["change_type"] == "structural"
    assert item["diff_summary"] == "2 lines changed"


def test_normalize_tolerates_a_row_with_no_body():
    """Every pre-existing item has `content` NULL; that must not raise."""
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )

    item = normalize_watchlist_item(
        "local", {"id": 9, "subscription_id": 3, "title": "Old item"}
    )

    assert item["content"] is None
    assert item["content_kind"] is None
