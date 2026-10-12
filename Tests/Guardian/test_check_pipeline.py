# Tests/Guardian/test_check_pipeline.py
"""GuardianChecker pipeline: gate, dedup-vs-recording, redact, fail-open, visits."""
import hashlib
import uuid
from datetime import datetime, timezone

import pytest

from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.Guardian import settings as guardian_settings
from tldw_chatbook.Guardian.check_pipeline import GuardianChecker

FIXED_NOW = "2026-09-29T12:00:00+00:00"
EPOCH = "1970-01-01T00:00:00+00:00"


@pytest.fixture()
def gate_off(monkeypatch):
    reads = {"n": 0}

    def fake_setting(key, default=None):
        if key == "enabled":
            reads["n"] += 1
            return False
        return default

    monkeypatch.setattr(guardian_settings, "guardian_setting", fake_setting)
    return reads


@pytest.fixture()
def gate_on(monkeypatch):
    reads = {"n": 0}

    def fake_setting(key, default=None):
        if key == "enabled":
            reads["n"] += 1
            return True
        return default

    monkeypatch.setattr(guardian_settings, "guardian_setting", fake_setting)
    return reads


def _fresh_db(tmp_path, rules):
    db = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    for row in db.list_rules():
        db.delete_rule(row["id"])
    for rule in rules:
        db.upsert_rule(**rule)
    return db


def _notes():
    return {"rows": [], "notified": []}


def _checker(db, notes, *, session_id="sess-1", db_getter=None, now=None):
    async def append_system_row(text: str) -> None:
        notes["rows"].append(text)

    def notify(text: str, severity: str) -> None:
        notes["notified"].append((text, severity))

    return GuardianChecker(
        db_getter=db_getter or (lambda: db),
        session_id_getter=lambda: session_id,
        notify=notify,
        append_system_row=append_system_row,
        now=now or (lambda: FIXED_NOW),
    )


def _digest(draft: str) -> str:
    return hashlib.sha256("".join(draft.split()).encode()).hexdigest()


async def test_gate_off_returns_allow_without_db_touch(gate_off):
    def _forbidden_getter():
        raise AssertionError("the disabled gate must never touch the store")

    notes = _notes()
    checker = _checker(None, notes, db_getter=_forbidden_getter)

    first = await checker.check("one more thing before bed")
    second = await checker.check("and another")

    assert first["action"] == "allow"
    assert second["action"] == "allow"
    assert first["recorded"] is False and second["recorded"] is False
    assert notes["rows"] == [] and notes["notified"] == []
    assert gate_off["n"] == 1, "the gate is a single cached config read"


async def test_command_kind_draft_skipped(gate_on):
    def _forbidden_getter():
        raise AssertionError("command drafts must be skipped before the store")

    notes = _notes()
    checker = _checker(None, notes, db_getter=_forbidden_getter)

    for draft in ("/model gpt-4o", "  /fix the indentation"):
        result = await checker.check(draft)
        assert result["action"] == "allow"
        assert result["recorded"] is False
    assert notes["rows"] == [] and notes["notified"] == []


async def test_every_match_records_even_when_notice_deduped(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "late night",
                "topic": "late_night_work",
                "pattern": "one more thing",
                "action": "notify",
                "notification_frequency": "once_per_day",
                "display_mode": "inline_banner",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    first = await checker.check("just one more thing before bed")
    second = await checker.check("surely one more thing will finish it")

    assert first["action"] == "notify"
    assert first["notice"] is not None
    assert second["notice"] is None, "once_per_day dedups notice surfacing"
    assert first["recorded"] is True and second["recorded"] is True
    assert len(notes["rows"]) == 1, "only the first hit surfaces a row"
    rules = db.list_rules()
    assert db.count_recent_alerts(rules[0]["id"], since_iso=EPOCH) == 2, (
        "every match records an alert row regardless of notice dedup"
    )
    assert gate_on["n"] == 1, "the gate stays a single cached read"


async def test_redact_rewrites_draft_in_memory(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "secret scrubber",
                "topic": "secrets",
                "pattern": "SECRET PHRASE",
                "action": "redact",
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)
    draft = "please keep the SECRET PHRASE hidden"

    result = await checker.check(draft)

    assert result["action"] == "redact"
    assert result["redacted_draft"] == (
        "please keep the [redacted: secret scrubber] hidden"
    )
    assert draft == "please keep the SECRET PHRASE hidden", (
        "the rewrite happens in memory; the caller's draft is untouched"
    )
    assert len(notes["rows"]) == 1
    assert "redacted" in notes["rows"][0]


async def test_escalated_block_returns_notice_and_records_hit(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "held topic",
                "topic": "held",
                "pattern": "forbidden tangent",
                "action": "redact",
                "escalate_session_threshold": 1,
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    result = await checker.check("a forbidden tangent appears")

    assert result["action"] == "block", (
        "the session threshold escalates redact one rung to block"
    )
    assert result["notice"] is not None
    assert "held topic" in result["notice"]["text"]
    assert result["recorded"] is True
    rules = db.list_rules()
    assert db.count_recent_alerts(rules[0]["id"], since_iso=EPOCH) == 1, (
        "a blocked send still counts the typed intent"
    )
    assert notes["rows"] == [], "block notices route through the seam refusal"


async def test_checker_error_fails_open_with_one_notification_per_signature(
    gate_on,
):
    errors = [RuntimeError("boom")]
    calls = {"n": 0}

    def db_getter():
        calls["n"] += 1
        raise errors[0]

    notes = _notes()
    checker = _checker(None, notes, db_getter=db_getter)

    first = await checker.check("hello")
    assert first["action"] == "allow" and first["recorded"] is False
    assert len(notes["notified"]) == 1
    assert "Guardian" in notes["notified"][0][0]

    second = await checker.check("hello again")
    assert second["action"] == "allow"
    assert len(notes["notified"]) == 1, "same signature stays silent this visit"

    errors[0] = ValueError("different failure")
    third = await checker.check("hello once more")
    assert third["action"] == "allow"
    assert len(notes["notified"]) == 2, "a new signature notifies once"

    errors[0] = ValueError("different failure")
    checker.begin_visit()
    fourth = await checker.check("new visit same failure")
    assert fourth["action"] == "allow"
    assert len(notes["notified"]) == 3, (
        "per-visit error dedup resets with the visit"
    )


async def test_visit_id_minted_on_first_check_and_reset_by_begin_visit(
    gate_on, tmp_path
):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "tracker",
                "topic": "tracked",
                "pattern": "tracked phrase",
                "notification_frequency": "every_message",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    assert checker.visit_id is None, "no visit id exists before the first check"
    await checker.check("a tracked phrase lands")
    first_visit = checker.visit_id
    uuid.UUID(first_visit)  # a valid uuid
    await checker.check("another tracked phrase")
    assert checker.visit_id == first_visit, "the visit id holds across checks"

    checker.begin_visit()
    assert checker.visit_id is None
    await checker.check("a tracked phrase again")
    second_visit = checker.visit_id
    assert second_visit != first_visit

    rules = db.list_rules()
    with db.connection() as conn:
        visit_ids = {
            row["visit_id"]
            for row in conn.execute(
                "SELECT visit_id FROM guardian_alerts WHERE rule_id = ?",
                (rules[0]["id"],),
            )
        }
    assert visit_ids == {first_visit, second_visit}


async def test_crisis_notice_carries_resources_and_disclaimer_inline(
    gate_on, tmp_path
):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "crisis awareness",
                "topic": "crisis_awareness",
                "pattern": r"suicid\w*",
                "action": "notify",
                "severity": "critical",
                "is_crisis": 1,
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    result = await checker.check("I have been feeling suicidal")

    assert result["action"] == "notify"
    notice = result["notice"]
    assert notice is not None and notice["is_crisis"] is True
    text = notice["text"]
    for token in (
        "988",
        "741741",
        "1-800-662-4357",
        "findahelpline.com",
        "tldw is not a mental health service",
    ):
        assert token in text, f"crisis notices must carry {token!r} inline"
    assert len(notes["rows"]) == 1
    assert "findahelpline.com" in notes["rows"][0]


async def test_silent_log_records_without_surfacing(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "quiet doom",
                "topic": "doomscrolling",
                "pattern": "doomscroll",
                "action": "notify",
                "severity": "info",
                "notification_frequency": "every_message",
                "display_mode": "silent_log",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    result = await checker.check("I doomscroll every night")

    assert result["action"] == "notify"
    assert result["notice"] is None, "silent_log surfaces nothing inline"
    assert result["recorded"] is True
    assert notes["rows"] == []
    rules = db.list_rules()
    assert db.count_recent_alerts(rules[0]["id"], since_iso=EPOCH) == 1


async def test_alert_rows_store_digest_not_draft_text(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "tracked",
                "topic": "tracked",
                "pattern": "tracked",
                "notification_frequency": "every_message",
            }
        ],
    )
    draft = "a tracked draft whose exact wording must never persist"
    notes = _notes()
    checker = _checker(db, notes)

    await checker.check(draft)

    with db.connection() as conn:
        rows = [
            tuple(row) for row in conn.execute("SELECT * FROM guardian_alerts")
        ]
    assert rows
    for row in rows:
        for value in row:
            assert draft not in str(value)
    assert any(_digest(draft) == str(value) for row in rows for value in row), (
        "the digest of the whitespace-stripped draft is what persists"
    )


async def test_check_uses_injected_clock_for_timestamps(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "clocked",
                "topic": "clocked",
                "pattern": "clocked",
                "notification_frequency": "every_message",
            }
        ],
    )
    notes = _notes()
    fixed = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc).isoformat()
    checker = _checker(db, notes, now=lambda: fixed)

    await checker.check("a clocked phrase")

    with db.connection() as conn:
        ts_values = [
            row["ts"]
            for row in conn.execute("SELECT ts FROM guardian_alerts")
        ]
    assert ts_values == [fixed]


# ---------------------------------------------------------------------------
# Escalated-block cooldown arming (ADR-204 contract 6, carried P4 ruling)
# ---------------------------------------------------------------------------


async def test_escalated_block_arms_the_rule_cooldown(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "held topic",
                "topic": "held",
                "pattern": "forbidden tangent",
                "action": "redact",
                "escalate_session_threshold": 1,
                "cooldown_minutes": 5,
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    result = await checker.check("a forbidden tangent appears")

    assert result["action"] == "block", (
        "the session threshold escalates redact one rung to block"
    )
    rules = db.list_rules()
    assert db.cooldown_active(rules[0]["id"], now=FIXED_NOW) is True, (
        "an ESCALATED block arms the rule's cooldown in the same record hop"
    )


async def test_base_action_block_does_not_arm_cooldown(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "hard stop",
                "topic": "stopped",
                "pattern": "hard stop phrase",
                "action": "block",
                "cooldown_minutes": 5,
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    result = await checker.check("a hard stop phrase here")

    assert result["action"] == "block"
    rules = db.list_rules()
    assert db.cooldown_active(rules[0]["id"], now=FIXED_NOW) is False, (
        "a base-action block is the user's configured intent, not an "
        "escalation event: no cooldown is armed"
    )


async def test_escalated_notify_or_redact_does_not_arm_cooldown(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "climbing",
                "topic": "climbing",
                "pattern": "climbing phrase",
                "action": "notify",
                "escalate_session_threshold": 1,
                "cooldown_minutes": 5,
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    result = await checker.check("a climbing phrase")

    assert result["action"] == "redact", "escalated one rung, below block"
    rules = db.list_rules()
    assert db.cooldown_active(rules[0]["id"], now=FIXED_NOW) is False, (
        "cooldowns arm only on escalated BLOCK, not on lesser escalations"
    )


async def test_block_short_circuits_before_hooks(dummy_hook_spy=None):
    # Seam-level pin (the checker alone cannot observe hook emission):
    # dispatching through the real ConsolePromptQueueUIController with a
    # blocking checker must never reach the send chain where ADR-148
    # UserPromptSubmit hooks fire. Implemented against the seam harness in
    # test_dispatch_seam, which owns the fakes.
    from Tests.Guardian.test_dispatch_seam import (
        run_block_short_circuit_case,
    )

    await run_block_short_circuit_case()
