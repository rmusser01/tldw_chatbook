"""TASK-1240: a worker that dies leaves a trace naming its exception type."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.worker import WorkerState

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_worker_error_records_worker_failed(monkeypatch):
    from textual.worker import Worker

    from Tests.UI.test_screen_navigation import _build_test_app

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

    failures = [r for r in recorded if r["event"] == "worker_failed"]
    assert failures, f"no worker_failed recorded, got {recorded}"
    assert failures[-1]["exception_type"] == "ValueError"
    assert failures[-1]["operation"] == "scheduler_worker"
    # The message must not travel: "boom" is caller-supplied text.
    assert "boom" not in str(failures[-1])


@pytest.mark.asyncio
async def test_successful_worker_records_nothing(monkeypatch):
    """Only failures persist. A start/success event per transition would emit a
    line per keystroke-triggered search across 500+ worker sites."""
    from textual.worker import Worker

    from Tests.UI.test_screen_navigation import _build_test_app

    recorded: list[dict] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.persist_event",
        lambda component, event, **fields: recorded.append({"event": event}),
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

    assert not [r for r in recorded if r["event"] in {"worker_failed", "worker_started"}]
