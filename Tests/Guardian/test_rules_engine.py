# Tests/Guardian/test_rules_engine.py
"""Rules engine: except patterns, span cap, version-keyed compilation cache."""
import json

import pytest

from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.Guardian.rules_engine import (
    MAX_SPAN_CHARS,
    RulesCache,
    compile_rules,
)


def _row(**over) -> dict:
    base = {
        "id": 1,
        "name": "crisis awareness",
        "topic": "crisis_awareness",
        "pattern": r"suicid\w*",
        "except_patterns": json.dumps(
            ["prevention", "research", "clinical", "study", "therapy"]
        ),
        "action": "notify",
        "severity": "critical",
        "is_crisis": 1,
        "notification_frequency": "once_per_conversation",
        "display_mode": "inline_banner",
        "escalate_session_threshold": None,
        "escalate_window_threshold": None,
        "escalate_window_days": None,
        "cooldown_minutes": None,
        "feeds_discovery": 0,
        "enabled": 1,
    }
    base.update(over)
    return base


def test_match_respects_except_patterns():
    compiled = compile_rules([_row()])
    hits = compiled.match("I have been feeling suicidal lately")
    assert len(hits) == 1
    assert hits[0].is_crisis is True
    assert hits[0].span_text == "suicidal"
    assert hits[0].rule_row["name"] == "crisis awareness"
    # The seeded vocabulary (research/treatment/clinical context) excepts
    # the whole draft for the rule.
    assert compiled.match("reading about suicide prevention research") == []
    assert compiled.match("clinical suicide study therapy notes") == []


def test_except_patterns_accept_plain_lists_and_disabled_rules_are_excluded():
    compiled = compile_rules(
        [
            _row(except_patterns=["hotline"]),
            _row(id=2, name="off", topic="off", pattern=r"off", enabled=0),
        ]
    )
    assert compiled.match("call the hotline about being suicidal") == []
    assert compiled.match("the office is closed") == [], (
        "disabled rules must not be compiled into matches"
    )


def test_span_capped_at_80_chars():
    compiled = compile_rules([_row(pattern=r"a{200}", except_patterns=[])])
    hits = compiled.match("a" * 200)
    assert len(hits) == 1
    assert len(hits[0].span_text) == MAX_SPAN_CHARS == 80


def test_match_reports_every_distinct_rule_once():
    compiled = compile_rules(
        [
            _row(),
            _row(id=2, name="second", topic="second", pattern=r"suicid\w*",
                 except_patterns=[], is_crisis=0),
        ]
    )
    hits = compiled.match("suicidal again")
    assert {hit.rule_row["id"] for hit in hits} == {1, 2}


@pytest.fixture()
def db(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    yield database
    database.close()


def test_compiled_cache_invalidates_on_rule_write(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    try:
        # Start from a store without seed rules to keep counts exact.
        for row in database.list_rules():
            database.delete_rule(row["id"])
        cache = RulesCache(database)
        cache.get()
        cache.get()
        assert cache.compile_count == 1, "warm gets must not recompile"
        database.upsert_rule(name="new", topic="new", pattern="zzz")
        cache.get()
        assert cache.compile_count == 2, (
            "a rule write bumps the rules version and forces a recompile"
        )
        cache.get()
        assert cache.compile_count == 2
    finally:
        database.close()


def test_compiled_cache_ttl_revalidates_without_recompile(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    try:
        cache = RulesCache(database, ttl_seconds=60.0)
        cache.get(now=0.0)
        cache.get(now=30.0)
        assert cache.compile_count == 1
        # TTL expiry re-validates the version; unchanged version means the
        # compiled set is reused, not rebuilt.
        cache.get(now=91.0)
        assert cache.compile_count == 1
        database.upsert_rule(name="late", topic="late", pattern="late")
        cache.get(now=92.0)
        assert cache.compile_count == 2
    finally:
        database.close()
