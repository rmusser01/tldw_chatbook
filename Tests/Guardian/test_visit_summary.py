# Tests/Guardian/test_visit_summary.py
"""finalize_visit: payload shape, empty-visit no-row rule, retention, storage."""
import json
from datetime import datetime, timedelta

import pytest

from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.Guardian import settings as guardian_settings
from tldw_chatbook.Guardian.check_pipeline import GuardianChecker

FIXED_NOW = "2026-09-29T12:00:00+00:00"
EPOCH = "1970-01-01T00:00:00+00:00"


@pytest.fixture()
def gate_on(monkeypatch):
    def fake_setting(key, default=None):
        if key == "enabled":
            return True
        return default

    monkeypatch.setattr(guardian_settings, "guardian_setting", fake_setting)


@pytest.fixture()
def gate_off(monkeypatch):
    monkeypatch.setattr(
        guardian_settings, "guardian_setting",
        lambda key, default=None: False if key == "enabled" else default,
    )


def _fresh_db(tmp_path, rules):
    db = GuardianDB(tmp_path / "guardian.sqlite", "summary-client")
    for row in db.list_rules():
        db.delete_rule(row["id"])
    for rule in rules:
        db.upsert_rule(**rule)
    return db


def _notes():
    return {"rows": [], "notified": []}


def _checker(db, notes, *, session_id="sess-1", db_getter=None):
    async def append_system_row(text: str) -> None:
        notes["rows"].append(text)

    def notify(text: str, severity: str) -> None:
        notes["notified"].append((text, severity))

    return GuardianChecker(
        db_getter=db_getter or (lambda: db),
        session_id_getter=lambda: session_id,
        notify=notify,
        append_system_row=append_system_row,
        now=lambda: FIXED_NOW,
    )


def _summaries(db):
    with db.connection() as conn:
        return [
            dict(row)
            for row in conn.execute("SELECT * FROM guardian_visit_summaries")
        ]


# ---------------------------------------------------------------------------
# Storage: insert_visit_summary roundtrip (GuardianDB)
# ---------------------------------------------------------------------------


def test_insert_visit_summary_roundtrip(tmp_path):
    db = GuardianDB(tmp_path / "guardian.sqlite", "summary-client")
    try:
        payload = {"per_topic_counts": {"late_night_work": 2},
                   "escalated_rules": [], "trend_notices": []}
        row_id = db.insert_visit_summary(
            visit_id="v1", session_id="sess-1", payload=payload
        )
        assert isinstance(row_id, int) and row_id > 0

        stored = db.get_visit_summary("v1")
        assert stored is not None
        assert stored["session_id"] == "sess-1"
        assert json.loads(stored["payload"]) == payload

        # Same visit finalized twice: replaced, never duplicated.
        db.insert_visit_summary(
            visit_id="v1", session_id="sess-1", payload={"per_topic_counts": {}}
        )
        assert len(_summaries(db)) == 1
    finally:
        db.close()


def test_get_visit_summary_missing_returns_none(tmp_path):
    db = GuardianDB(tmp_path / "guardian.sqlite", "summary-client")
    try:
        assert db.get_visit_summary("nope") is None
    finally:
        db.close()


def test_count_topic_alerts_scopes_to_the_visit(tmp_path):
    db = GuardianDB(tmp_path / "guardian.sqlite", "summary-client")
    try:
        db.insert_alert(
            rule_id=None, session_id="s", visit_id="v1",
            topic="fixation:x", message_digest="d", ts=FIXED_NOW,
        )
        assert db.count_topic_alerts("fixation:x", visit_id="v1") == 1
        assert db.count_topic_alerts("fixation:x", visit_id="v2") == 0
    finally:
        db.close()


# ---------------------------------------------------------------------------
# finalize_visit: the empty-visit rule and the payload shape
# ---------------------------------------------------------------------------


async def test_finalize_empty_visit_mints_nothing(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [{"name": "tracker", "topic": "tracked", "pattern": "tracked phrase"}],
    )
    notes = _notes()
    checker = _checker(db, notes)

    # No check ever ran: no visit id, no alerts -- a nav bounce mints nothing.
    assert checker.finalize_visit() is None
    assert _summaries(db) == []
    with db.connection() as conn:
        trend_rows = conn.execute(
            "SELECT COUNT(*) FROM guardian_alerts WHERE rule_id IS NULL"
        ).fetchone()[0]
    assert trend_rows == 0


async def test_finalize_gate_off_never_touches_the_store(gate_off):
    def _forbidden_getter():
        raise AssertionError("a disabled Guardian must never touch the store")

    notes = _notes()
    checker = _checker(None, notes, db_getter=_forbidden_getter)
    assert checker.finalize_visit() is None


async def test_finalize_silent_only_visit_stores_no_row(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "quiet doom",
                "topic": "doomscrolling",
                "pattern": "doomscroll",
                "display_mode": "silent_log",
                "notification_frequency": "every_message",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    await checker.check("I doomscroll every night")
    assert checker.finalize_visit() is None, (
        "silent-log-only visits are invisible by design: no row, no payload"
    )
    assert _summaries(db) == []


async def test_finalize_silent_escalated_only_visit_stores_no_row(
    gate_on, tmp_path
):
    """Carried P5 ruling (Task 2 review): the storage gate is the brief's
    letter -- store ONLY when ``per_topic_counts or trend_notices`` is
    non-empty. A silent_log rule escalating to block leaves escalation
    bookkeeping in its own state row, but the visit itself is invisible by
    design: no summary row, no payload."""
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "quiet escalator",
                "topic": "doomscrolling",
                "pattern": "doomscroll",
                "display_mode": "silent_log",
                "action": "redact",
                "escalate_session_threshold": 1,
                "notification_frequency": "every_message",
                "cooldown_minutes": 5,
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    # One hit clears the session threshold: redact -> block (escalated,
    # cooldown armed) -- but display_mode is silent_log, so the visit's
    # surfaced artifacts are all empty.
    await checker.check("doomscroll again")

    assert checker.finalize_visit() is None
    assert _summaries(db) == [], (
        "silent-escalated-only visits store nothing (P5: escalation state "
        "is not a surfaced artifact)"
    )


async def test_finalize_non_empty_visit_stores_payload(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "late night",
                "topic": "late_night_work",
                "pattern": "one more thing",
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
            },
            {
                "name": "quiet doom",
                "topic": "doomscrolling",
                "pattern": "doomscroll",
                "display_mode": "silent_log",
                "notification_frequency": "every_message",
            },
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    await checker.check("just one more thing before bed")
    await checker.check("I doomscroll every night")

    payload = checker.finalize_visit()
    assert payload is not None
    # Non-silent topics only: the silent-log topic stays invisible.
    assert payload["per_topic_counts"] == {"late_night_work": 1}
    assert payload["escalated_rules"] == []
    assert payload["trend_notices"] == []

    rows = _summaries(db)
    assert len(rows) == 1
    assert rows[0]["visit_id"] == checker.visit_id
    assert rows[0]["session_id"] == "sess-1"
    assert json.loads(rows[0]["payload"]) == payload


async def test_finalize_lists_escalated_rules(gate_on, tmp_path):
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

    await checker.check("a forbidden tangent appears")  # escalates to block
    payload = checker.finalize_visit()
    assert payload is not None
    assert payload["escalated_rules"] == ["held topic"]


async def test_finalize_runs_the_trend_cadence_before_storing(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "obsession",
                "topic": "obsession",
                "pattern": "obsessing",
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)

    # Seed enough history for a fixation notice (share 1.0, >= 30 hits),
    # then one live non-silent check to make the visit non-empty.
    rules = db.list_rules()
    now = datetime.fromisoformat(FIXED_NOW)
    for i in range(30):
        db.insert_alert(
            rule_id=rules[0]["id"], session_id="sess-1", visit_id="older-visit",
            topic="obsession", message_digest="h" * 64,
            ts=(now - timedelta(days=i % 5)).isoformat(),
        )
    await checker.check("obsessing again")

    payload = checker.finalize_visit()
    assert payload is not None
    assert [n["label"] for n in payload["trend_notices"]] == [
        "fixation:obsession"
    ], "analyze() runs at visit end and its notices ride the summary"
    with db.connection() as conn:
        trend_rows = conn.execute(
            "SELECT visit_id FROM guardian_alerts WHERE rule_id IS NULL"
        ).fetchall()
    assert trend_rows, "the trend notice is itself an alert row"
    assert trend_rows[0]["visit_id"] == checker.visit_id


async def test_finalize_prunes_alerts_past_retention(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [
            {
                "name": "tracker",
                "topic": "tracked",
                "pattern": "tracked phrase",
                "notification_frequency": "every_message",
                "display_mode": "inline_banner",
            }
        ],
    )
    notes = _notes()
    checker = _checker(db, notes)
    rules = db.list_rules()
    now = datetime.fromisoformat(FIXED_NOW)
    db.insert_alert(
        rule_id=rules[0]["id"], session_id="sess-1", visit_id="ancient",
        topic="tracked", message_digest="old" * 21,
        ts=(now - timedelta(days=400)).isoformat(),
    )

    await checker.check("a tracked phrase lands")
    payload = checker.finalize_visit()
    assert payload is not None

    with db.connection() as conn:
        ancient_left = conn.execute(
            "SELECT COUNT(*) FROM guardian_alerts WHERE visit_id = 'ancient'"
        ).fetchone()[0]
    assert ancient_left == 0, (
        "finalize_visit runs the retention sweep (best-effort, logged)"
    )


async def test_finalize_degrades_to_none_when_the_store_breaks(gate_on, tmp_path):
    db = _fresh_db(
        tmp_path,
        [{"name": "tracker", "topic": "tracked", "pattern": "tracked"}],
    )
    notes = _notes()
    checker = _checker(db, notes)

    await checker.check("a tracked draft")

    class _ExplodingDB:
        def __getattr__(self, name):
            raise RuntimeError("store unavailable")

    original = checker._db_getter
    checker._db_getter = lambda: _ExplodingDB()
    try:
        assert checker.finalize_visit() is None, (
            "a visit summary never raises: degrade to None"
        )
    finally:
        checker._db_getter = original
