import pytest
from tldw_chatbook.Subscriptions.watchlist_filter_service import WatchlistFilterService


@pytest.fixture
def service():
    return WatchlistFilterService()


def test_keyword_include(service):
    items = [{"title": "AI news", "summary": "", "content": ""}]
    filters = [
        {"id": 1, "priority": 1, "action": "include", "conditions": {"type": "keyword", "mode": "contains", "pattern": "AI"}, "is_include_required": False}
    ]
    result = service.evaluate(items, filters)
    assert result[0]["filter_decision"] == "include"


def test_exclude_wins_over_include(service):
    items = [{"title": "AI news", "summary": "", "content": ""}]
    filters = [
        {"id": 1, "priority": 1, "action": "include", "conditions": {"type": "keyword", "pattern": "AI"}, "is_include_required": False},
        {"id": 2, "priority": 0, "action": "exclude", "conditions": {"type": "keyword", "pattern": "AI"}, "is_include_required": False},
    ]
    result = service.evaluate(items, filters)
    # Lower priority number evaluated first; exclude wins.
    assert result[0]["filter_decision"] == "exclude"
