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
from tldw_chatbook.Scheduling.scheduler.handlers import dreams_handler
from tldw_chatbook.Scheduling.scheduler.handlers.dreams_handler import (
    DreamsCycleHandler,
)
from tldw_chatbook.Scheduling.services.dreams_projection import (
    DREAMS_TASK_PREFIX,
    DreamsProjection,
    parse_dreams_task_id,
)


def test_parse_roundtrip_and_rejects_foreign_ids():
    assert parse_dreams_task_id(f"{DREAMS_TASK_PREFIX}:cycle") == "cycle"
    assert parse_dreams_task_id("briefing:7") is None
    assert parse_dreams_task_id(None) is None


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
