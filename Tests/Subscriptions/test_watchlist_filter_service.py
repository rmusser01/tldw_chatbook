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


def test_exclude_filter_matches_whole_page_even_if_conditions_carry_a_scope_key(service):
    """TASK-1363, AC#2: a per-rule scope is a content-alert-rule concept only.
    `WatchlistFilterService._matches` never reads `conditions["scope"]` -- it
    calls `build_rule_haystack(item)` with no `scope` argument at all, so it
    always gets the "anywhere" default -- so even a filter whose conditions
    happen to carry a scope key (e.g. copied from a content-alert rule, or a
    future editor that shares one schema for both) must still match anywhere
    on the page. A narrowed exclude filter could otherwise admit an item the
    user told the app to drop, which is exactly the regression TASK-1343
    already fixed once.

    The pattern here is deliberately present ONLY in the page-wide text
    (`rule_match_text`), not in the "added" text a scope would narrow to, so
    this could not pass by accident if scope narrowing leaked into the
    filter path.
    """
    from tldw_chatbook.Subscriptions.watchlist_rule_matching import (
        RULE_MATCH_ADDED_TEXT_KEY,
        RULE_MATCH_TEXT_KEY,
    )

    items = [{
        "title": "",
        "content": "",
        RULE_MATCH_TEXT_KEY: "sponsored placement sits deep in the unchanged part of this page",
        RULE_MATCH_ADDED_TEXT_KEY: "totally unrelated new text",
    }]
    filters = [{
        "id": 1,
        "priority": 0,
        "action": "exclude",
        "conditions": {"type": "keyword", "pattern": "sponsored placement", "scope": "appeared"},
        "is_include_required": False,
    }]
    result = service.evaluate(items, filters)
    assert result[0]["filter_decision"] == "exclude", (
        "the filter must still match the whole page even though its "
        "conditions carry a scope key -- WatchlistFilterService never reads it"
    )
