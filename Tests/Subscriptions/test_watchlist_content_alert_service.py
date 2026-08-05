import pytest

from tldw_chatbook.Subscriptions.watchlist_content_alert_service import WatchlistContentAlertService
from tldw_chatbook.Subscriptions.watchlist_rule_matching import (
    RULE_MATCH_ADDED_TEXT_KEY,
    RULE_MATCH_REMOVED_TEXT_KEY,
    RULE_MATCH_TEXT_KEY,
    build_rule_haystack,
)


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


# --- TASK-1363: per-rule appeared/disappeared/anywhere scope ----------------


def _change_item(**overrides):
    """A site-change item carrying the page-wide text plus a diff's added and
    removed segments, the shape `URLMonitor.check_url` produces.
    """
    item = {
        "title": "Change detected: Test source",
        "content": "the diff body itself, never matched against directly",
        RULE_MATCH_TEXT_KEY: (
            "an old phrase that was always here. brand new phrase just landed."
        ),
        RULE_MATCH_ADDED_TEXT_KEY: "brand new phrase just landed.",
        RULE_MATCH_REMOVED_TEXT_KEY: "",
    }
    item.update(overrides)
    return item


def test_appeared_scope_ignores_the_synthetic_change_title_and_metadata(service):
    """task-1363 review: "appeared" matches ONLY the added page text, never the
    change item's own metadata. A site change's title is the synthetic
    "Change detected: <source name>", present on every check, so a pattern that
    sits in the source name (here "Test source") must NOT fire under "appeared"
    -- otherwise the scope that exists to cut page-wide noise would itself fire
    on every change. Reds if "appeared" folds title/summary/author back into
    the haystack.
    """
    item = _change_item()  # title = "Change detected: Test source"

    source_name_rule = [{
        "id": 9,
        "name": "Matches the source name in the synthetic title",
        "conditions": {"type": "keyword", "pattern": "Test source", "scope": "appeared"},
    }]
    assert service.evaluate(item, source_name_rule) == [], (
        "the synthetic change title is not text that appeared on the page; an "
        "'appeared' rule must not match it"
    )
    # Sanity: the same pattern DOES match under "anywhere" (it is in the title),
    # proving the item genuinely carries it and the appeared-scope miss is the
    # scoping, not a missing field.
    source_name_anywhere = [{
        "id": 10,
        "name": "Same pattern, anywhere scope",
        "conditions": {"type": "keyword", "pattern": "Test source", "scope": "anywhere"},
    }]
    assert [m["rule_id"] for m in service.evaluate(item, source_name_anywhere)] == [10]


def test_appeared_scope_matches_added_text_but_not_an_unchanged_old_phrase(service):
    """AC#1/#3: "appeared" matches text a change introduced, not text that was
    already on the page before the change -- even though the old phrase is
    still on the page (and would match under "anywhere").
    """
    item = _change_item()

    new_phrase_rule = [{
        "id": 1,
        "name": "New phrase",
        "conditions": {"type": "keyword", "pattern": "brand new phrase", "scope": "appeared"},
    }]
    assert [m["rule_id"] for m in service.evaluate(item, new_phrase_rule)] == [1]

    old_phrase_appeared = [{
        "id": 2,
        "name": "Old phrase, appeared scope",
        "conditions": {"type": "keyword", "pattern": "an old phrase", "scope": "appeared"},
    }]
    assert service.evaluate(item, old_phrase_appeared) == [], (
        "the old phrase did not appear in this change -- it must not match "
        "under scope=appeared even though it is still on the page"
    )

    old_phrase_anywhere = [{
        "id": 3,
        "name": "Old phrase, anywhere scope",
        "conditions": {"type": "keyword", "pattern": "an old phrase", "scope": "anywhere"},
    }]
    assert [m["rule_id"] for m in service.evaluate(item, old_phrase_anywhere)] == [3], (
        "the same phrase must still match under scope=anywhere -- it is a "
        "genuine part of the current page"
    )


def test_mixed_scopes_in_one_evaluate_do_not_cross_contaminate(service):
    """Qodo perf refactor: `evaluate` caches the haystack per scope so a
    page-wide join is not rebuilt for every rule. The cache must be keyed by
    scope -- a rule of one scope must never receive another scope's haystack.
    Drive three rules of three different scopes through ONE call and confirm
    each sees its own slice. Reds if the cache serves the wrong haystack (e.g.
    a single shared key).
    """
    item = _change_item(**{
        RULE_MATCH_ADDED_TEXT_KEY: "brand new phrase just landed.",
        RULE_MATCH_REMOVED_TEXT_KEY: "gone phrase used to be here.",
    })
    rules = [
        {  # appeared: matches added text, not removed, not the old page text
            "id": 1,
            "name": "appeared",
            "conditions": {"type": "keyword", "pattern": "brand new phrase", "scope": "appeared"},
        },
        {  # disappeared: matches removed text only
            "id": 2,
            "name": "disappeared",
            "conditions": {"type": "keyword", "pattern": "gone phrase", "scope": "disappeared"},
        },
        {  # anywhere: matches the old page phrase (present in RULE_MATCH_TEXT_KEY)
            "id": 3,
            "name": "anywhere",
            "conditions": {"type": "keyword", "pattern": "an old phrase", "scope": "anywhere"},
        },
        {  # appeared rule for the GONE phrase -> must NOT match (it's removed, not added)
            "id": 4,
            "name": "appeared-for-removed",
            "conditions": {"type": "keyword", "pattern": "gone phrase", "scope": "appeared"},
        },
        {  # disappeared rule for the NEW phrase -> must NOT match (it's added, not removed)
            "id": 5,
            "name": "disappeared-for-added",
            "conditions": {"type": "keyword", "pattern": "brand new phrase", "scope": "disappeared"},
        },
    ]
    matched_ids = sorted(m["rule_id"] for m in service.evaluate(item, rules))
    assert matched_ids == [1, 2, 3], (
        "each scope must see only its own text: appeared->added, "
        "disappeared->removed, anywhere->page; the cross rules (4, 5) must miss"
    )


def test_disappeared_scope_matches_removed_text_only(service):
    """AC#1/#3: "disappeared" matches text a change removed, and that rule
    must not also match under "appeared" -- the two scopes are disjoint.
    """
    item = _change_item(**{
        RULE_MATCH_ADDED_TEXT_KEY: "",
        RULE_MATCH_REMOVED_TEXT_KEY: "gone phrase used to be here.",
    })

    disappeared_rule = [{
        "id": 1,
        "name": "Gone phrase",
        "conditions": {"type": "keyword", "pattern": "gone phrase", "scope": "disappeared"},
    }]
    assert [m["rule_id"] for m in service.evaluate(item, disappeared_rule)] == [1]

    appeared_rule = [{
        "id": 2,
        "name": "Gone phrase, appeared scope",
        "conditions": {"type": "keyword", "pattern": "gone phrase", "scope": "appeared"},
    }]
    assert service.evaluate(item, appeared_rule) == [], (
        "removed text must not match under scope=appeared"
    )


def test_absent_or_anywhere_scope_keeps_page_wide_behaviour(service):
    """AC#3 regression: a rule with no `scope` key at all -- every rule that
    existed before this task -- keeps matching the whole page, identically to
    an explicit scope="anywhere".
    """
    item = _change_item()

    no_scope_rule = [{
        "id": 1,
        "name": "Old phrase, no scope key",
        "conditions": {"type": "keyword", "pattern": "an old phrase"},
    }]
    explicit_anywhere_rule = [{
        "id": 2,
        "name": "Old phrase, explicit anywhere",
        "conditions": {"type": "keyword", "pattern": "an old phrase", "scope": "anywhere"},
    }]

    no_scope_matches = service.evaluate(item, no_scope_rule)
    anywhere_matches = service.evaluate(item, explicit_anywhere_rule)
    assert [m["rule_id"] for m in no_scope_matches] == [1]
    assert [m["rule_id"] for m in anywhere_matches] == [2]


def test_unrecognized_scope_value_falls_back_to_anywhere(service):
    """`build_rule_haystack` treats an unknown scope as "anywhere" (a safe
    default), and the service must pass whatever string the rule holds
    straight through rather than validating it first.
    """
    item = _change_item()
    rules = [{
        "id": 1,
        "name": "Typo'd scope",
        "conditions": {"type": "keyword", "pattern": "an old phrase", "scope": "evrywhere"},
    }]
    assert [m["rule_id"] for m in service.evaluate(item, rules)] == [1]


def test_build_rule_haystack_no_scope_argument_matches_the_anywhere_default():
    """AC#2's free half, pinned directly: the default `scope` parameter value
    produces byte-identical output to calling with no `scope` argument at all
    -- which is what `WatchlistFilterService` does, since it never passes one.
    """
    item = _change_item()
    assert build_rule_haystack(item) == build_rule_haystack(item, scope="anywhere")
    assert "an old phrase" in build_rule_haystack(item)
    assert "brand new phrase" in build_rule_haystack(item)
