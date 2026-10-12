# Tests/Guardian/test_guardian_db.py
"""GuardianDB v1: crisis write-boundary caps, escalation ladder, digest-only storage, seeding."""
import hashlib
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Guardian_DB import GuardianDB, GuardianRuleConflict


@pytest.fixture()
def db(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    yield database
    database.close()


def _iso(days_offset: float = 0.0) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(days=days_offset)
    ).isoformat()


def _crisis_rule(**over) -> dict:
    base = {
        "name": "crisis awareness",
        "topic": "crisis_awareness",
        "pattern": r"suicid\w*",
        "action": "notify",
        "severity": "critical",
        "is_crisis": 1,
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Crisis write-boundary caps (ADR-204 contract 7)
# ---------------------------------------------------------------------------


def test_upsert_rule_rejects_crisis_with_block_action(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    try:
        with pytest.raises(GuardianRuleConflict) as excinfo:
            database.upsert_rule(**_crisis_rule(action="block"))
        assert excinfo.value.reason_code == "guardian_rule_conflict"
        assert database.list_rules(enabled_only=True) == [] or all(
            row["name"] != "crisis awareness"
            for row in database.list_rules(enabled_only=True)
        ), "a rejected rule must not be written"
    finally:
        database.close()


def test_upsert_rule_rejects_crisis_with_redact_action(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    try:
        with pytest.raises(GuardianRuleConflict):
            database.upsert_rule(**_crisis_rule(action="redact"))
    finally:
        database.close()


def test_upsert_rule_rejects_crisis_with_feeds_discovery(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    try:
        with pytest.raises(GuardianRuleConflict) as excinfo:
            database.upsert_rule(**_crisis_rule(feeds_discovery=1))
        assert excinfo.value.reason_code == "guardian_rule_conflict"
    finally:
        database.close()


def test_upsert_rule_rejects_crisis_flag_added_to_existing_block_rule(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    try:
        rule_id = database.upsert_rule(
            name="blocker", topic="offtopic", pattern="spam", action="block"
        )
        with pytest.raises(GuardianRuleConflict):
            # Merged row (block action + is_crisis=1) must be rejected too,
            # not just whole-row inserts.
            database.upsert_rule(id=rule_id, is_crisis=1)
    finally:
        database.close()


# ---------------------------------------------------------------------------
# Escalation ladder + crisis cap (ADR-204 contract 7/8)
# ---------------------------------------------------------------------------


def test_bump_escalation_applies_ladder_but_caps_crisis_at_notify(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    try:
        rule_id = database.upsert_rule(
            name="escalating", topic="escalation", pattern="x",
            action="notify",
            escalate_session_threshold=2,
            escalate_window_threshold=3,
            escalate_window_days=7,
        )
        first = database.bump_escalation(rule_id, now=_iso())
        assert first["session_count"] == 1
        assert first["current_action"] == "notify"
        second = database.bump_escalation(rule_id, now=_iso())
        assert second["session_count"] == 2
        assert second["current_action"] == "redact", (
            "session threshold reached must escalate one rung"
        )
        third = database.bump_escalation(rule_id, now=_iso())
        assert third["window_count"] == 3
        assert third["current_action"] == "block", (
            "window threshold reached must escalate to the ladder apex"
        )

        crisis_id = database.upsert_rule(**_crisis_rule(
            escalate_session_threshold=2,
            escalate_window_threshold=3,
            escalate_window_days=7,
        ))
        state = None
        for _ in range(3):
            state = database.bump_escalation(crisis_id, now=_iso())
        assert state is not None
        assert state["session_count"] == 3
        assert state["window_count"] == 3
        assert state["current_action"] == "notify", (
            "crisis rules may never hold or escalate to redact/block"
        )
    finally:
        database.close()


def test_bump_escalation_resets_window_beyond_window_days(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    try:
        rule_id = database.upsert_rule(
            name="w", topic="w", pattern="w", action="notify",
            escalate_window_threshold=2, escalate_window_days=7,
        )
        database.bump_escalation(rule_id, now=_iso(0))
        later = database.bump_escalation(rule_id, now=_iso(8))
        assert later["window_count"] == 1, "a window older than window_days resets"
        assert later["session_count"] == 2, "the session counter never resets"
    finally:
        database.close()


# ---------------------------------------------------------------------------
# Digest-only storage (ADR-204 contract 2)
# ---------------------------------------------------------------------------


def test_alert_roundtrip_stores_digest_not_text(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    try:
        draft = "the exact typed draft text must never persist in guardian storage"
        digest = hashlib.sha256("".join(draft.split()).encode()).hexdigest()
        alert_id = database.insert_alert(
            rule_id=None, session_id="sess-1", visit_id="visit-1",
            topic="doomscrolling", message_digest=digest,
        )
        with database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM guardian_alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        assert row is not None
        values = [str(value) for value in tuple(row)]
        assert digest in values
        for value in values:
            assert draft not in value, "no column may carry message text"
            assert "typed draft" not in value
        assert row["topic"] == "doomscrolling"
        assert row["session_id"] == "sess-1"
        assert row["visit_id"] == "visit-1"
    finally:
        database.close()


def test_count_recent_alerts_scopes_by_session_visit_and_time(db):
    db.insert_alert(rule_id=1, session_id="s1", visit_id="v1",
                    topic="t", message_digest="d1")
    db.insert_alert(rule_id=1, session_id="s1", visit_id="v2",
                    topic="t", message_digest="d2")
    db.insert_alert(rule_id=1, session_id="s2", visit_id="v3",
                    topic="t", message_digest="d3")
    epoch = "1970-01-01T00:00:00+00:00"
    assert db.count_recent_alerts(1, since_iso=epoch) == 3
    assert db.count_recent_alerts(1, session_id="s1", since_iso=epoch) == 2
    assert db.count_recent_alerts(1, visit_id="v1", since_iso=epoch) == 1
    future = "2999-01-01T00:00:00+00:00"
    assert db.count_recent_alerts(1, since_iso=future) == 0


def test_prune_alerts_cuts_old_only(db):
    keep_id = db.insert_alert(
        rule_id=1, session_id="s", visit_id="v", topic="t", message_digest="keep"
    )
    old_id = db.insert_alert(
        rule_id=1, session_id="s", visit_id="v", topic="t", message_digest="old"
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE guardian_alerts SET ts = ? WHERE id = ?",
            ("2020-01-01T00:00:00+00:00", old_id),
        )
    cutoff = "2026-01-01T00:00:00+00:00"
    pruned = db.prune_alerts(cutoff)
    assert pruned == 1
    with db.connection() as conn:
        remaining = {
            int(row["id"])
            for row in conn.execute("SELECT id FROM guardian_alerts")
        }
    assert remaining == {keep_id}


def test_cooldown_set_and_active_check(db):
    rule_id = db.upsert_rule(name="cd", topic="cd", pattern="cd", action="block")
    db.set_cooldown(rule_id, "2999-01-01T00:00:00+00:00")
    assert db.cooldown_active(rule_id, now=_iso()) is True
    db.set_cooldown(rule_id, "2020-01-01T00:00:00+00:00")
    assert db.cooldown_active(rule_id, now=_iso()) is False


def test_delete_rule_removes_rule_and_escalation_state(db):
    rule_id = db.upsert_rule(name="gone", topic="gone", pattern="gone")
    db.bump_escalation(rule_id, now=_iso())
    db.delete_rule(rule_id)
    assert db.get_rule(rule_id) is None
    with db.connection() as conn:
        state = conn.execute(
            "SELECT * FROM guardian_escalation_state WHERE rule_id = ?",
            (rule_id,),
        ).fetchone()
    assert state is None


# ---------------------------------------------------------------------------
# Seed rules + storage lifecycle
# ---------------------------------------------------------------------------


def test_fresh_build_seeds_exactly_three_rules(db):
    rules = db.list_rules()
    names = {row["name"] for row in rules}
    assert len(rules) == 3, f"expected the three seed rules, got {names}"
    crisis = next(row for row in rules if row["is_crisis"] == 1)
    assert crisis["action"] == "notify"
    assert crisis["severity"] == "critical"
    assert crisis["feeds_discovery"] == 0
    excepts = crisis["except_patterns"]  # already parsed by list_rules
    for token in ("prevention", "hotline", "awareness", "research", "study",
                  "clinical", "treatment", "therapy"):
        assert token in excepts
    doom = next(
        row for row in rules if row["topic"] == "doomscrolling"
    )
    assert doom["display_mode"] == "silent_log"
    assert doom["severity"] == "info"
    assert all(row["feeds_discovery"] == 0 for row in rules)


def test_second_build_does_not_reseed(tmp_path):
    path = tmp_path / "guardian.sqlite"
    first = GuardianDB(path, "test-client")
    try:
        assert len(first.list_rules()) == 3
    finally:
        first.close()
    second = GuardianDB(path, "test-client")
    try:
        assert len(second.list_rules()) == 3, "reopening must not re-seed"
    finally:
        second.close()


def test_schema_reports_version_1(db):
    with db.connection() as conn:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    assert int(row[0]) == 1


# ---------------------------------------------------------------------------
# App-owned builder (ADR-204 contract 1: zero footprint when off)
# ---------------------------------------------------------------------------


def _fake_app_self() -> SimpleNamespace:
    return SimpleNamespace(_guardian_db_lock=threading.Lock())


@pytest.mark.bootstrap_profile
def test_app_builder_returns_none_and_creates_no_file_when_disabled(
    tmp_path, monkeypatch
):
    from tldw_chatbook import app as app_module
    from tldw_chatbook.Guardian import settings as guardian_settings

    monkeypatch.setattr(
        guardian_settings,
        "guardian_setting",
        lambda key, default=None: False if key == "enabled" else default,
    )
    target = tmp_path / "guardian.sqlite"
    monkeypatch.setattr(guardian_settings, "guardian_db_path", lambda: target)

    fake_self = _fake_app_self()
    assert app_module.TldwCli.get_guardian_db(fake_self) is None
    assert not target.exists(), (
        "a disabled Guardian must never create its database file"
    )


@pytest.mark.bootstrap_profile
def test_app_builder_builds_seeded_store_when_enabled(tmp_path, monkeypatch):
    from tldw_chatbook import app as app_module
    from tldw_chatbook.Guardian import settings as guardian_settings

    monkeypatch.setattr(
        guardian_settings,
        "guardian_setting",
        lambda key, default=None: True if key == "enabled" else default,
    )
    target = tmp_path / "guardian.sqlite"
    monkeypatch.setattr(guardian_settings, "guardian_db_path", lambda: target)

    fake_self = _fake_app_self()
    built = app_module.TldwCli.get_guardian_db(fake_self)
    try:
        assert built is not None
        assert target.exists()
        assert len(built.list_rules()) == 3
        assert app_module.TldwCli.get_guardian_db(fake_self) is built, (
            "the builder caches its store"
        )
    finally:
        built.close()
