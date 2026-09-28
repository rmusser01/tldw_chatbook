"""Scheduler integration for [dreams]: projection, handler, and id contract.

Dreams Phase 1, Task 5. These tests pin the three seams the scheduler
integration adds:

* the id shape (``dreams:cycle``) has exactly one definition point
  (:data:`DREAMS_TASK_PREFIX` + :func:`parse_dreams_task_id`), per the
  two-copies-drift lesson ``briefing_projection`` documents;
* the projection's attempt-aware watermark rule -- never-attempted is due
  now, a latest failure retries one cadence later, and a ``complete``/
  ``partial`` collection pins the next run one cadence after its
  ``completed_at``;
* the handler's fire-and-forget contract -- it never raises into the loop,
  spawns nothing without deps, and dispatches ``trigger="scheduled"``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Dreams.settings import DREAMS_DEFAULTS
from tldw_chatbook.Scheduling.scheduler.handlers import (
    dream_track_handler,
    dreams_handler,
)
from tldw_chatbook.Scheduling.scheduler.handlers.dream_track_handler import (
    DreamTrackHandler,
)
from tldw_chatbook.Scheduling.scheduler.handlers.dreams_handler import (
    DreamsCycleHandler,
)
from tldw_chatbook.Scheduling.services.dreams_projection import (
    DREAMS_TASK_PREFIX,
    DREAMS_TRACK_PREFIX,
    DreamsProjection,
    parse_dream_track_task_id,
    parse_dreams_task_id,
)

_NOW = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)
_CADENCE = timedelta(hours=12)


def test_parse_roundtrip_and_rejects_foreign_ids():
    assert parse_dreams_task_id(f"{DREAMS_TASK_PREFIX}:cycle") == "cycle"
    assert parse_dreams_task_id("briefing:7") is None
    assert parse_dreams_task_id(None) is None


# --- dream_track task ids + per-item projection (Phase 2 Task 4) ---------------


def test_parse_dream_track_task_id_roundtrip_and_rejects_foreign_ids():
    assert parse_dream_track_task_id(f"{DREAMS_TRACK_PREFIX}:7") == 7
    assert parse_dream_track_task_id("dreams:cycle") is None
    assert parse_dream_track_task_id("dream_track:abc") is None
    assert parse_dream_track_task_id("dream_track:") is None
    assert parse_dream_track_task_id("dream_track:0") is None
    assert parse_dream_track_task_id(None) is None


def _tracked_item(db, *, last_checked=None, cadence_seconds=12 * 3600,
                  status=None, query_template="watch {region}"):
    item_id = db.create_tracked_item(
        mechanism="question", intent="topic", cadence_seconds=cadence_seconds,
        query_template=query_template)
    if last_checked is not None:
        db.touch_tracked_checked(item_id, last_checked.isoformat())
    if status is not None:
        db.set_tracked_status(item_id, status)
    return item_id


def test_projection_emits_track_checks_for_active_items_with_clamps(
        tmp_path, monkeypatch):
    proj, db = _projection(tmp_path, monkeypatch)
    never = _tracked_item(db)  # never checked -> due immediately
    recent = _tracked_item(db, last_checked=_NOW - timedelta(hours=6))
    overdue = _tracked_item(db, last_checked=_NOW - timedelta(hours=96))
    _tracked_item(db, last_checked=_NOW - timedelta(hours=6), status="paused")
    _tracked_item(db, last_checked=_NOW - timedelta(hours=6),
                  status="retired")

    tasks = {task.id: task for task in proj.tasks(_NOW)}

    assert set(tasks) == {
        "dreams:cycle",
        f"dream_track:{never}",
        f"dream_track:{recent}",
        f"dream_track:{overdue}",
    }, "paused and retired items emit nothing"
    assert tasks[f"dream_track:{never}"].next_run_at == _NOW
    assert tasks[f"dream_track:{recent}"].next_run_at == _NOW + \
        timedelta(hours=6)
    # 96h since the last check + 12h cadence = 84h overdue > 48h clamp:
    # skip the stale pileup, next run one cadence from now.
    assert tasks[f"dream_track:{overdue}"].next_run_at == _NOW + _CADENCE
    for task in tasks.values():
        if task.id.startswith(f"{DREAMS_TRACK_PREFIX}:"):
            assert task.type == "dream_track_check"
            assert task.title.startswith("Dreams tracked check")


def test_projection_no_track_tasks_when_disabled(tmp_path, monkeypatch):
    """A disabled Dreams emits nothing even with tracked items waiting."""
    proj, db = _projection(tmp_path, monkeypatch, enabled=False)
    _tracked_item(db)
    assert proj.tasks(_NOW) == []


def _projection(tmp_path, monkeypatch, enabled=True):
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda s, k, d: True if (k == "enabled" and enabled) else d,
    )
    db = DreamsDB(tmp_path / "d.sqlite", "t")
    return DreamsProjection(lambda: db), db


def test_projection_emits_nothing_when_disabled(tmp_path, monkeypatch):
    proj, db = _projection(tmp_path, monkeypatch, enabled=False)
    assert proj.tasks(datetime.now(timezone.utc)) == []


def test_projection_due_now_when_never_run_and_retries_after_failure(
    tmp_path, monkeypatch
):
    proj, db = _projection(tmp_path, monkeypatch)
    now = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)
    tasks = proj.tasks(now)
    assert [t.id for t in tasks] == ["dreams:cycle"]
    assert tasks[0].next_run_at <= now  # never attempted -> due immediately
    cid = db.create_collection("2026-09-21", "scheduled", "d")
    db.set_collection_status(cid, "failed", completed_at=now.isoformat())
    retry = proj.tasks(now)[0]
    assert retry.next_run_at > now  # failed -> one cadence later
    db.set_collection_status(cid, "complete", completed_at=now.isoformat())
    done = proj.tasks(now)[0]
    assert done.next_run_at == now + timedelta(
        hours=DREAMS_DEFAULTS["cadence_hours"]
    )


def test_projection_emits_nothing_without_db(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda s, k, d: True if k == "enabled" else d,
    )
    proj = DreamsProjection(lambda: None)
    assert proj.tasks(datetime.now(timezone.utc)) == []


async def test_handler_spawns_nothing_without_deps():
    dreams_handler._SPAWNED_CYCLES.clear()
    handler = DreamsCycleHandler(deps_getter=lambda: None)
    # Must neither raise nor spawn anything.
    await handler.handle({"id": "dreams:cycle"})
    assert not dreams_handler._SPAWNED_CYCLES


async def test_handler_ignores_foreign_task_ids(monkeypatch):
    dreams_handler._SPAWNED_CYCLES.clear()

    async def _boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("run_cycle must not be reached")

    monkeypatch.setattr(
        "tldw_chatbook.Dreams.cycle_service.run_cycle", _boom
    )
    handler = DreamsCycleHandler(deps_getter=lambda: object())
    await handler.handle({"id": "briefing:7"})
    await handler.handle({"id": "not-a-dreams-id"})
    assert not dreams_handler._SPAWNED_CYCLES


async def test_handler_dispatches_scheduled_trigger(monkeypatch):
    dreams_handler._SPAWNED_CYCLES.clear()
    calls: list[tuple[object, str]] = []

    async def fake_run_cycle(deps, *, trigger):
        calls.append((deps, trigger))
        return {"collection_id": 1, "status": "complete", "stories": 0}

    monkeypatch.setattr(
        "tldw_chatbook.Dreams.cycle_service.run_cycle", fake_run_cycle
    )
    deps = object()
    handler = DreamsCycleHandler(deps_getter=lambda: deps)
    await handler.handle({"id": "dreams:cycle"})
    # handle() returns before the spawned task runs; settle it.
    await asyncio.gather(*dreams_handler._SPAWNED_CYCLES)
    assert calls == [(deps, "scheduled")]
    dreams_handler._SPAWNED_CYCLES.clear()


# --- DreamTrackHandler (Phase 2 Task 4) -----------------------------------------


async def test_track_handler_spawns_nothing_without_deps():
    dream_track_handler._SPAWNED_TRACK_CHECKS.clear()
    handler = DreamTrackHandler(deps_getter=lambda: None)
    # Must neither raise nor spawn anything.
    await handler.handle({"id": "dream_track:7"})
    assert not dream_track_handler._SPAWNED_TRACK_CHECKS


async def test_track_handler_ignores_foreign_task_ids(monkeypatch):
    dream_track_handler._SPAWNED_TRACK_CHECKS.clear()

    async def _boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("run_track_check must not be reached")

    monkeypatch.setattr(
        "tldw_chatbook.Dreams.track_service.run_track_check", _boom
    )
    handler = DreamTrackHandler(deps_getter=lambda: object())
    await handler.handle({"id": "dreams:cycle"})
    await handler.handle({"id": "dream_track:abc"})
    await handler.handle({"id": "not-a-dreams-id"})
    assert not dream_track_handler._SPAWNED_TRACK_CHECKS


async def test_track_handler_dispatches_run_track_check(monkeypatch):
    dream_track_handler._SPAWNED_TRACK_CHECKS.clear()
    calls: list[tuple[object, int]] = []

    async def fake_run_track_check(deps, tracked_item_id):
        calls.append((deps, tracked_item_id))
        return {"status": "baseline", "notified": False}

    monkeypatch.setattr(
        "tldw_chatbook.Dreams.track_service.run_track_check",
        fake_run_track_check,
    )
    deps = object()
    handler = DreamTrackHandler(deps_getter=lambda: deps)
    await handler.handle({"id": "dream_track:9"})
    # handle() returns before the spawned task runs; settle it.
    await asyncio.gather(*dream_track_handler._SPAWNED_TRACK_CHECKS)
    assert calls == [(deps, 9)]
    dream_track_handler._SPAWNED_TRACK_CHECKS.clear()
