import pytest
from tldw_chatbook.Subscriptions.watchlist_content_alert_service import WatchlistContentAlertService


@pytest.fixture
def service():
    return WatchlistContentAlertService()


def test_keyword_match(service):
    rules = [
        {"id": 1, "name": "AI alert", "severity": "warning", "conditions": {"type": "keyword", "pattern": "AI"}}
    ]
    matches = service.evaluate({"title": "AI news", "summary": "", "content": ""}, rules)
    assert len(matches) == 1
    assert matches[0]["rule_id"] == 1
    assert matches[0]["severity"] == "warning"
