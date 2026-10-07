# Tests/Guardian/test_dreams_interop.py
"""Guardian x Dreams interop: the feeds_discovery aggregate feed (ADR-204).

The discovery gate is the load-bearing privacy boundary (contract 3): a
rule's hits may influence Dreams' outbound discovery ONLY when the rule
sets ``feeds_discovery=1``, and crisis-flagged rules can never set it (the
write boundary in ``GuardianDB.upsert_rule`` refuses). These tests pin the
reader, the Dreams wiring, and the whole-payload absence of crisis topic
text -- the goals-gate pattern from Tests/Dreams/test_query_synthesis.py.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.Dreams import profile_sources
from tldw_chatbook.Dreams.cycle_service import CycleDeps, _refresh_profile_signals
from tldw_chatbook.Dreams.interest_profile import snapshot
from tldw_chatbook.Dreams.query_synthesis import synthesize_queries

CRISIS_TOPIC = "crisis_awareness"
DISCOVERY_TOPIC = "late_night_work"
CLOSED_TOPIC = "doomscrolling"


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _guardian_db(tmp_path, *, discovery_alerts=0, crisis_alerts=0,
                 closed_alerts=0, old_alerts=0) -> GuardianDB:
    """A fresh store with one rule per gate posture and seeded alert rows."""
    db = GuardianDB(tmp_path / "guardian.sqlite", "interop-client")
    for row in db.list_rules():
        db.delete_rule(row["id"])
    discovery_rule = db.upsert_rule(
        name="late-night work",
        topic=DISCOVERY_TOPIC,
        pattern="one more thing",
        feeds_discovery=1,
    )
    closed_rule = db.upsert_rule(
        name="quiet doom",
        topic=CLOSED_TOPIC,
        pattern="doomscroll",
        feeds_discovery=0,
    )
    crisis_rule = db.upsert_rule(
        name="crisis awareness",
        topic=CRISIS_TOPIC,
        pattern="in crisis",
        is_crisis=1,
        action="notify",
    )
    now = datetime.now(timezone.utc)
    for rule_id, topic, count, offset_days in (
        (discovery_rule, DISCOVERY_TOPIC, discovery_alerts, 1),
        (crisis_rule, CRISIS_TOPIC, crisis_alerts, 1),
        (closed_rule, CLOSED_TOPIC, closed_alerts, 1),
        (discovery_rule, DISCOVERY_TOPIC, old_alerts, 20),
    ):
        for index in range(count):
            db.insert_alert(
                rule_id=rule_id,
                session_id="sess-1",
                visit_id="visit-1",
                topic=topic,
                message_digest=f"{topic}-{index}".ljust(64, "0"),
                ts=_iso(now - timedelta(days=offset_days, minutes=index)),
            )
    return db


def _dreams_db(tmp_path) -> DreamsDB:
    return DreamsDB(tmp_path / "dreams.sqlite", "interop-dreams")


# ---------------------------------------------------------------------------
# read_guardian_topics: counts per topic, feeds_discovery rules only
# ---------------------------------------------------------------------------


def test_read_guardian_topics_counts_only_feeds_discovery_alerts(tmp_path):
    db = _guardian_db(
        tmp_path,
        discovery_alerts=3,
        crisis_alerts=5,
        closed_alerts=4,
        old_alerts=2,
    )
    try:
        rows = profile_sources.read_guardian_topics(db)
        assert rows == [
            {
                "facet": "topic",
                "text": DISCOVERY_TOPIC,
                "weight": pytest.approx(0.75),  # 3 in-window touches x 0.25
                "searchable": 1,
                "source": "guardian",
            }
        ], "only the opted-in rule's in-window alerts count"
    finally:
        db.close()


def test_read_guardian_topics_empty_store_returns_no_rows(tmp_path):
    db = _guardian_db(tmp_path)
    try:
        assert profile_sources.read_guardian_topics(db) == []
    finally:
        db.close()


# ---------------------------------------------------------------------------
# The whole-payload absence test (spec-binding, ADR-204 contract 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardian_topics_never_carry_crisis_text_into_synthesis(
    tmp_path, monkeypatch
):
    """A snapshot built from guardian topics (with a crisis rule's alerts
    recorded in the same store) must contain the crisis topic text NOWHERE
    in the synthesized outbound payload -- the same whole-payload
    assertion the goals gate pins."""
    # The Dreams settings seam is pinned (the test_interest_profile
    # pattern): no live config read inside the test sandbox.
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default=None: default,
    )
    db = _guardian_db(tmp_path, discovery_alerts=2, crisis_alerts=4)
    dreams = _dreams_db(tmp_path)
    try:
        rows = profile_sources.read_guardian_topics(db)
        assert [row["text"] for row in rows] == [DISCOVERY_TOPIC]

        # The cycle's wiring shape: merged guardian rows land in the
        # profile under origin 'guardian' (the _preferred_sources output).
        from tldw_chatbook.Dreams.cycle_service import _upsert_profile_signals

        merged = [
            {"facet": "topic", "text": row["text"], "weight": row["weight"]}
            for row in rows
        ]
        await asyncio.to_thread(
            _upsert_profile_signals,
            dreams,
            merged,
            {(row["facet"], row["text"]): "guardian" for row in rows},
            now_iso=_iso(datetime.now(timezone.utc)),
        )

        snap = snapshot(dreams, now_epoch=datetime.now(timezone.utc).timestamp())
        calls: list[dict] = []

        def chat(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {"message": {"content": "late night work tips\nsecond query"}}
                ]
            }

        await synthesize_queries(
            chat, snapshot=snap, count=3, exploration_slots=1
        )
        payload = json.loads(calls[0]["messages_payload"][0]["content"])
        assert DISCOVERY_TOPIC in json.dumps(payload), (
            "the opted-in aggregate DOES feed discovery"
        )
        assert CRISIS_TOPIC not in json.dumps(payload), (
            "crisis topic text must appear NOWHERE in the outbound payload"
        )
    finally:
        db.close()
        dreams.close()


# ---------------------------------------------------------------------------
# CycleDeps wiring (per-source degradation, ruling R18 discipline)
# ---------------------------------------------------------------------------


def _no_chat():
    raise RuntimeError("no chat in this test")


def _deps(dreams: DreamsDB, *, guardian_getter) -> CycleDeps:
    return CycleDeps(
        dreams_db=dreams,
        chachanotes_db_getter=lambda: None,
        media_db_getter=lambda: None,
        subs_db_getter=lambda: None,
        pc_service_getter=lambda: None,
        chat_getter=_no_chat,
        guardian_db_getter=guardian_getter,
    )


@pytest.mark.asyncio
async def test_refresh_with_guardian_getter_none_degrades_silently(tmp_path):
    dreams = _dreams_db(tmp_path)
    try:
        notes = await _refresh_profile_signals(
            _deps(dreams, guardian_getter=lambda: None),
            datetime.now(timezone.utc),
        )
        assert notes == [], "a None getter skips the source silently"
    finally:
        dreams.close()


@pytest.mark.asyncio
async def test_refresh_reads_guardian_topics_under_the_guardian_origin(tmp_path):
    db = _guardian_db(tmp_path, discovery_alerts=2, crisis_alerts=3)
    dreams = _dreams_db(tmp_path)
    try:
        notes = await _refresh_profile_signals(
            _deps(dreams, guardian_getter=lambda: db),
            datetime.now(timezone.utc),
        )
        assert notes == []
        rows = {
            (row["facet"], row["text"]): row
            for row in dreams.list_profile()
        }
        assert rows[("topic", DISCOVERY_TOPIC)]["source"] == "guardian"
        assert ("topic", CRISIS_TOPIC) not in rows, (
            "crisis topic text never lands in the Dreams profile"
        )
    finally:
        db.close()
        dreams.close()


@pytest.mark.asyncio
async def test_refresh_degrades_with_a_note_when_the_guardian_read_fails(tmp_path):
    dreams = _dreams_db(tmp_path)

    class _BrokenStore:
        def connection(self):
            raise RuntimeError("store unavailable")

    try:
        notes = await _refresh_profile_signals(
            _deps(dreams, guardian_getter=lambda: _BrokenStore()),
            datetime.now(timezone.utc),
        )
        assert notes and notes[0].startswith("profile signals: guardian failed")
    finally:
        dreams.close()
