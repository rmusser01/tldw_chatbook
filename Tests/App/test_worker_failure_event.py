"""TASK-1240: a worker that dies leaves a trace naming its exception type."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.worker import WorkerState

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_worker_error_records_worker_failed(monkeypatch):
    from textual.worker import Worker

    from Tests.UI.app_factory import _build_test_app

    recorded: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append(
            {"component": component, "event": event, **fields}
        ),
    )

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        recorded.clear()
        worker = MagicMock(spec=Worker)
        worker.name = "scheduler_worker"
        worker.group = "scheduling"
        worker.error = ValueError("boom")
        event = Worker.StateChanged(worker, WorkerState.ERROR)
        await app.on_worker_state_changed(event)
        await pilot.pause()

    # Select by identity, not position: real background workers started by
    # _build_test_app (scheduler_loop.run, model-catalog refresh, FTS backfill)
    # also route through this hook during pilot.pause() and could append their
    # own worker_failed entries after this one.
    failure = next(
        (f for f in recorded
         if f["event"] == "worker_failed" and f["operation"] == "scheduler_worker"),
        None,
    )
    assert failure is not None, f"no worker_failed for the injected worker, got {recorded}"
    assert failure["exception_type"] == "ValueError"
    # The message must not travel: "boom" is caller-supplied text.
    assert "boom" not in str(failure)


@pytest.mark.asyncio
async def test_successful_worker_records_nothing(monkeypatch):
    """Only failures persist. A start/success event per transition would emit a
    line per keystroke-triggered search across 500+ worker sites."""
    from textual.worker import Worker

    from Tests.UI.app_factory import _build_test_app

    recorded: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append(
            {"component": component, "event": event, **fields}
        ),
    )

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        recorded.clear()
        worker = MagicMock(spec=Worker)
        worker.name = "some_worker"
        worker.group = "misc"
        worker.error = None
        await app.on_worker_state_changed(
            Worker.StateChanged(worker, WorkerState.SUCCESS)
        )
        await pilot.pause()

    # Select by identity, exactly as the sibling above does, and for exactly
    # the same reason: real background workers started by _build_test_app
    # (scheduler_loop.run, model-catalog refresh, FTS backfill) route through
    # this same hook during pilot.pause() and may append their own entries. A
    # blanket "no worker_failed anywhere in the recorder" assertion passes only
    # while none of them happens to fail -- it asserts what its own sibling
    # documents as false. The recorder therefore keeps **fields, so `operation`
    # is available to discriminate on.
    assert not [
        r
        for r in recorded
        if r["event"] in {"worker_failed", "worker_started"}
        and r.get("operation") == "some_worker"
    ], f"the successful worker recorded an event: {recorded}"


@pytest.mark.asyncio
async def test_screen_navigation_worker_transition_does_not_warn_unhandled():
    """task-2726: every tab switch runs a 'screen-navigation' worker; its
    transitions must be acknowledged by the registry instead of warning
    'No handler found' once per navigation."""
    from loguru import logger as loguru_root_logger
    from textual.worker import Worker

    from Tests.UI.app_factory import _build_test_app

    warnings: list[str] = []
    sink_id = loguru_root_logger.add(
        lambda message: warnings.append(str(message)), level="WARNING"
    )
    try:
        app = _build_test_app()
        async with app.run_test(size=(120, 40)) as pilot:
            worker = MagicMock(spec=Worker)
            worker.name = "handle_screen_navigation"
            worker.group = "screen-navigation"
            worker.error = None
            # Real Workers always carry a description; the acknowledging
            # handler's get_worker_info reads it.
            worker.description = "handle_screen_navigation=<NavigateToScreen>"
            await app.on_worker_state_changed(
                Worker.StateChanged(worker, WorkerState.SUCCESS)
            )
            await pilot.pause()
    finally:
        loguru_root_logger.remove(sink_id)

    assert not [
        w
        for w in warnings
        if "No handler found" in w and "handle_screen_navigation" in w
    ], f"navigation worker warned unhandled: {warnings}"


@pytest.mark.asyncio
async def test_unknown_worker_group_still_warns_unhandled():
    """Guard for task-2726: acknowledging known fire-and-forget groups must not
    swallow the unhandled-worker warning for genuinely unknown workers."""
    from loguru import logger as loguru_root_logger
    from textual.worker import Worker

    from Tests.UI.app_factory import _build_test_app

    warnings: list[str] = []
    sink_id = loguru_root_logger.add(
        lambda message: warnings.append(str(message)), level="WARNING"
    )
    try:
        app = _build_test_app()
        async with app.run_test(size=(120, 40)) as pilot:
            worker = MagicMock(spec=Worker)
            worker.name = "mystery_worker"
            worker.group = "task-2726-unknown-group"
            worker.error = None
            await app.on_worker_state_changed(
                Worker.StateChanged(worker, WorkerState.SUCCESS)
            )
            await pilot.pause()
    finally:
        loguru_root_logger.remove(sink_id)

    assert [
        w
        for w in warnings
        if "No handler found" in w and "mystery_worker" in w
    ], f"unknown worker did not warn: {warnings}"
