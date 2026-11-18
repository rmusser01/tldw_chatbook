# Tests/Guardian/test_trend_analyzer.py
"""Trend analyzer (fixation/doomloop), the daily scheduler task, and retention.

Pins, per the spec §Trend analyzer:
* inputs are rule-generated alert rows only (``rule_id IS NOT NULL``) --
  analyzer output never feeds analyzer input;
* definitions: fixation share >= 0.6 over 7 days with >= 30 hits on one
  topic; doomloop >= 20 hits/day on one topic for >= 3 consecutive days;
  thresholds read live from ``[guardian]`` settings;
* every trend notice is itself an alert row (``rule_id=None``, topic
  ``fixation:<t>`` / ``doomloop:<t>``), frequency-capped one per topic per
  visit;
* the projection emits ``guardian:trends`` daily while enabled, gated by a
  Guardian-store watermark (stable across queue reloads); the handler is
  the dream_track spawn pattern and never raises into the loop.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.Guardian import settings as guardian_settings
from tldw_chatbook.Guardian.trend_analyzer import analyze

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
NOW_ISO = NOW.isoformat()
EPOCH = "1970-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    database = GuardianDB(tmp_path / "guardian.sqlite", "trend-client")
    for row in database.list_rules():
        database.delete_rule(row["id"])
    yield database
    database.close()


@pytest.fixture()
def live_thresholds(monkeypatch):
    """Pass every [guardian] setting through to the real defaults."""

    def fake_setting(key, default=None):
        if key == "enabled":
            return True
        if default is not None:
            return default
        return guardian_settings.GUARDIAN_DEFAULTS.get(key)

    monkeypatch.setattr(guardian_settings, "guardian_setting", fake_setting)
    return fake_setting


def _rule(db, topic):
    return db.upsert_rule(
        name=f"rule {topic}", topic=topic, pattern=f"pat-{topic}",
        notification_frequency="every_message",
    )


def _hit(db, rule_id, topic, *, days_ago=0.0):
    ts = (NOW - timedelta(days=days_ago)).isoformat()
    return db.insert_alert(
        rule_id=rule_id, session_id="s1", visit_id="v1", topic=topic,
        message_digest="d" * 64, ts=ts,
    )


# ---------------------------------------------------------------------------
# Fixation (sustained-topic share)
# ---------------------------------------------------------------------------


def test_fixation_boundary_29_hits_no_30_yes(db, live_thresholds):
    topic_rule = _rule(db, "obsession")
    other_rule = _rule(db, "other")
    for i in range(29):
        _hit(db, topic_rule, "obsession", days_ago=i % 7)
    _hit(db, other_rule, "other", days_ago=1)

    notices = analyze(db, now=NOW_ISO)
    assert notices == [], "29 hits is below fixation_min_hits even at share 1.0"

    _hit(db, topic_rule, "obsession", days_ago=2)
    notices = analyze(db, now=NOW_ISO, visit_id="trends-a")
    assert [n.label for n in notices] == ["fixation:obsession"], (
        "30 of 31 hits in the window is a fixation notice"
    )


def test_fixation_share_059_is_not_enough(db, live_thresholds):
    topic_rule = _rule(db, "obsession")
    other_rule = _rule(db, "other")
    for i in range(59):
        _hit(db, topic_rule, "obsession", days_ago=i % 7)
    for i in range(41):
        _hit(db, other_rule, "other", days_ago=i % 7)

    notices = analyze(db, now=NOW_ISO)
    assert notices == [], "59/100 share 0.59 must not fire at threshold 0.6"


def test_fixation_ignores_hits_outside_the_window(db, live_thresholds):
    topic_rule = _rule(db, "obsession")
    for i in range(30):
        _hit(db, topic_rule, "obsession", days_ago=20)  # outside the 7d window

    assert analyze(db, now=NOW_ISO) == [], (
        "hits older than fixation_window_days never count toward the share"
    )


# ---------------------------------------------------------------------------
# Doomloop (repetitive volume)
#
# Doomloop reads ALL history while fixation reads only the 7-day window,
# so these fixtures place their days at 8-10 days ago -- inside doomloop's
# scope, outside fixation's -- keeping each definition isolated.
# ---------------------------------------------------------------------------


def test_doomloop_three_consecutive_days_fires(db, live_thresholds):
    topic_rule = _rule(db, "doomscrolling")
    for day in (10, 9, 8):
        for i in range(20):
            _hit(db, topic_rule, "doomscrolling", days_ago=day)

    notices = analyze(db, now=NOW_ISO, visit_id="trends-a")
    assert [n.label for n in notices] == ["doomloop:doomscrolling"]


def test_doomloop_broken_streak_does_not_fire(db, live_thresholds):
    topic_rule = _rule(db, "doomscrolling")
    for day in (10, 9, 7):  # day 8 missing -> no 3 consecutive days
        for i in range(20):
            _hit(db, topic_rule, "doomscrolling", days_ago=day)

    assert analyze(db, now=NOW_ISO) == [], "a gap breaks the streak"


def test_doomloop_below_daily_volume_does_not_fire(db, live_thresholds):
    topic_rule = _rule(db, "doomscrolling")
    for day in (10, 9, 8, 7):
        for i in range(19):  # 19 < 20 per day, even 4 days straight
            _hit(db, topic_rule, "doomscrolling", days_ago=day)

    assert analyze(db, now=NOW_ISO) == []


# ---------------------------------------------------------------------------
# Loop exclusion + row shape + frequency cap
# ---------------------------------------------------------------------------


def test_analyzer_excludes_its_own_rows(db, live_thresholds):
    """Trend rows (rule_id NULL) must never feed analyzer input."""
    topic_rule = _rule(db, "obsession")
    for i in range(29):
        _hit(db, topic_rule, "obsession", days_ago=i % 7)
    # Five NULL-rule rows on the same topic: counting them would reach 34
    # hits at share 1.0 and (wrongly) fire the fixation notice.
    for i in range(5):
        db.insert_alert(
            rule_id=None, session_id="s1", visit_id="v1", topic="obsession",
            message_digest="t" * 64, ts=(NOW - timedelta(days=1)).isoformat(),
        )

    assert analyze(db, now=NOW_ISO) == [], (
        "rule_id IS NULL rows are analyzer output, never analyzer input"
    )


def test_trend_notice_rows_are_capped_one_per_topic_per_visit(db, live_thresholds):
    topic_rule = _rule(db, "obsession")
    for i in range(30):
        _hit(db, topic_rule, "obsession", days_ago=i % 7)

    first = analyze(db, now=NOW_ISO, visit_id="visit-1")
    second = analyze(db, now=NOW_ISO, visit_id="visit-1")
    third = analyze(db, now=NOW_ISO, visit_id="visit-2")

    assert len(first) == 1
    assert second == [], "same topic, same visit: capped after the first notice"
    assert len(third) == 1, "a fresh visit may notice the same topic again"

    with db.connection() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT rule_id, topic, visit_id FROM guardian_alerts"
                " WHERE rule_id IS NULL"
            )
        ]
    assert [row["topic"] for row in rows] == [
        "fixation:obsession", "fixation:obsession",
    ]
    assert {row["visit_id"] for row in rows} == {"visit-1", "visit-2"}


def test_trend_notice_carries_kind_topic_and_message(db, live_thresholds):
    topic_rule = _rule(db, "doomscrolling")
    for day in (10, 9, 8):
        for i in range(22):
            _hit(db, topic_rule, "doomscrolling", days_ago=day)

    (notice,) = analyze(db, now=NOW_ISO)
    assert notice.kind == "doomloop"
    assert notice.topic == "doomscrolling"
    assert notice.label == "doomloop:doomscrolling"
    assert "doomscrolling" in notice.message
    assert notice.message  # a ready-to-surface sentence, not empty


def test_thresholds_read_live_from_settings(db, monkeypatch):
    def tuned(key, default=None):
        if key == "enabled":
            return True
        tuned_values = {
            "fixation_share_threshold": 0.5,
            "fixation_window_days": 7,
            "fixation_min_hits": 2,
            "doomloop_hits_per_day": 99,
        }
        return tuned_values.get(key, default)

    monkeypatch.setattr(guardian_settings, "guardian_setting", tuned)
    topic_rule = _rule(db, "obsession")
    other_rule = _rule(db, "other")
    _hit(db, topic_rule, "obsession", days_ago=1)
    _hit(db, topic_rule, "obsession", days_ago=2)
    _hit(db, other_rule, "other", days_ago=1)  # 2/3 share ~0.67 >= 0.5

    notices = analyze(db, now=NOW_ISO)
    assert [n.label for n in notices] == ["fixation:obsession"], (
        "a lowered fixation_min_hits must detect on 2 hits"
    )


# ---------------------------------------------------------------------------
# Projection: guardian:trends, one daily task, watermark-stable
# ---------------------------------------------------------------------------


def _projection(db, monkeypatch, *, enabled=True):
    from tldw_chatbook.Scheduling.services.guardian_projection import (
        GuardianProjection,
    )

    monkeypatch.setattr(
        "tldw_chatbook.Scheduling.services.guardian_projection.guardian_setting",
        lambda key, default=None: enabled if key == "enabled" else default,
    )
    return GuardianProjection(lambda: db)


def test_projection_id_shape_has_one_definition_point():
    from tldw_chatbook.Scheduling.services import guardian_projection as gp

    assert gp.GUARDIAN_TRENDS_TASK_ID == "guardian:trends"
    assert gp.GUARDIAN_TRENDS_TASK_TYPE == "guardian_trends"
    assert gp.parse_guardian_task_id("guardian:trends") == "trends"
    assert gp.parse_guardian_task_id("dreams:cycle") is None
    assert gp.parse_guardian_task_id(None) is None
    assert gp.parse_guardian_task_id("guardian:") is None


def test_projection_emits_one_daily_task_due_now_when_never_run(db, monkeypatch):
    proj = _projection(db, monkeypatch)
    tasks = proj.tasks(NOW)
    assert [task.id for task in tasks] == ["guardian:trends"]
    task = tasks[0]
    assert task.type == "guardian_trends"
    assert task.next_run_at == NOW, "never-run trends are due on the next tick"


def test_projection_watermark_pins_next_run_stable_across_reloads(db, monkeypatch):
    proj = _projection(db, monkeypatch)
    last_run = (NOW - timedelta(hours=2)).isoformat()
    db.set_meta("trends_last_run", last_run)

    first = {t.id: t for t in proj.tasks(NOW)}["guardian:trends"]
    second = {
        t.id: t for t in proj.tasks(NOW + timedelta(minutes=30))
    }["guardian:trends"]

    expected = NOW - timedelta(hours=2) + timedelta(hours=24)
    assert first.next_run_at == expected
    assert second.next_run_at == expected, (
        "a reload must not slide the daily slot (the watermark owns it)"
    )


def test_projection_emits_nothing_when_disabled(db, monkeypatch):
    proj = _projection(db, monkeypatch, enabled=False)
    assert proj.tasks(NOW) == []


def test_projection_emits_nothing_without_a_store(db, monkeypatch):
    from tldw_chatbook.Scheduling.services.guardian_projection import (
        GuardianProjection,
    )

    _projection(db, monkeypatch)  # enabled=True, but no store to read
    assert GuardianProjection(lambda: None).tasks(NOW) == []


def test_projection_list_jobs_stamps_the_queue_owner(db, monkeypatch):
    proj = _projection(db, monkeypatch)
    (task,) = proj.list_jobs(owner_id="local", now=NOW)
    assert task.owner_id == "local"


# ---------------------------------------------------------------------------
# Daily handler: dream_track spawn pattern, never raises, retention
# ---------------------------------------------------------------------------


class _DispatchRecorder:
    def __init__(self):
        self.calls = []

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        return {"persisted": True}


def _deps(db, recorder):
    from tldw_chatbook.Scheduling.scheduler.handlers.guardian_trend_handler import (
        GuardianTrendDeps,
    )

    return GuardianTrendDeps(guardian_db=db, dispatch_service=recorder)


async def test_handler_spawns_nothing_without_deps():
    from tldw_chatbook.Scheduling.scheduler.handlers import guardian_trend_handler
    from tldw_chatbook.Scheduling.scheduler.handlers.guardian_trend_handler import (
        GuardianTrendHandler,
    )

    guardian_trend_handler._SPAWNED_TREND_RUNS.clear()
    handler = GuardianTrendHandler(deps_getter=lambda: None)
    await handler.handle({"id": "guardian:trends", "type": "guardian_trends"})
    assert not guardian_trend_handler._SPAWNED_TREND_RUNS, (
        "no deps (disabled or store-less) must spawn nothing"
    )


async def test_handler_ignores_foreign_task_ids():
    from tldw_chatbook.Scheduling.scheduler.handlers.guardian_trend_handler import (
        GuardianTrendHandler,
    )

    handler = GuardianTrendHandler(deps_getter=lambda: object())
    await handler.handle({"id": "dreams:cycle", "type": "guardian_trends"})


async def test_handler_runs_analysis_dispatch_and_retention(
    db, live_thresholds, monkeypatch
):
    from tldw_chatbook.Scheduling.scheduler.handlers import guardian_trend_handler
    from tldw_chatbook.Scheduling.scheduler.handlers.guardian_trend_handler import (
        GuardianTrendHandler,
    )

    monkeypatch.setattr(
        "tldw_chatbook.Utils.timestamps.utc_now_iso", lambda: NOW_ISO
    )
    topic_rule = _rule(db, "obsession")
    for i in range(30):
        _hit(db, topic_rule, "obsession", days_ago=i % 7)
    # One stale row far past retention: the daily run must prune it.
    stale_ts = (NOW - timedelta(days=400)).isoformat()
    db.insert_alert(
        rule_id=topic_rule, session_id="s0", visit_id="v0",
        topic="obsession", message_digest="old" * 21, ts=stale_ts,
    )
    recorder = _DispatchRecorder()
    guardian_trend_handler._SPAWNED_TREND_RUNS.clear()
    handler = GuardianTrendHandler(deps_getter=lambda: _deps(db, recorder))

    await handler.handle({"id": "guardian:trends", "type": "guardian_trends"})
    spawned = list(guardian_trend_handler._SPAWNED_TREND_RUNS)
    await asyncio.gather(*spawned)

    assert len(recorder.calls) == 1, "one dispatch per NEW trend notice"
    call = recorder.calls[0]
    assert call["category"] == "guardian"
    assert "obsession" in call["message"]
    with db.connection() as conn:
        trend_rows = conn.execute(
            "SELECT COUNT(*) FROM guardian_alerts WHERE rule_id IS NULL"
        ).fetchone()[0]
        stale_left = conn.execute(
            "SELECT COUNT(*) FROM guardian_alerts WHERE ts = ?", (stale_ts,)
        ).fetchone()[0]
    assert trend_rows == 1
    assert stale_left == 0, "the daily run prunes past alert_retention_days"
    assert db.get_meta("trends_last_run") == NOW_ISO, (
        "the watermark advances so tomorrow's slot is projected"
    )


async def test_handler_never_raises_on_a_broken_store():
    from tldw_chatbook.Scheduling.scheduler.handlers import guardian_trend_handler
    from tldw_chatbook.Scheduling.scheduler.handlers.guardian_trend_handler import (
        GuardianTrendDeps,
        GuardianTrendHandler,
    )

    class _Broken:
        def __getattr__(self, name):
            raise RuntimeError("store unavailable")

    handler = GuardianTrendHandler(
        deps_getter=lambda: GuardianTrendDeps(
            guardian_db=_Broken(), dispatch_service=None
        )
    )
    guardian_trend_handler._SPAWNED_TREND_RUNS.clear()
    await handler.handle({"id": "guardian:trends", "type": "guardian_trends"})
    spawned = list(guardian_trend_handler._SPAWNED_TREND_RUNS)
    await asyncio.gather(*spawned)  # must settle without raising
