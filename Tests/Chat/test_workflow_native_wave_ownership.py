"""Wave authority follows registered resources rather than Python implementations."""

import functools
import threading

import pytest

from Tests.Chat.test_console_native_preparation import (
    native_host as native_host,  # noqa: PLC0414
)
from tldw_chatbook.Chat.console_workflow_wave import _WorkflowWave, run_workflow_wave

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.mark.parametrize("method", ["_launch", "_release", "_open", "run"])
async def test_transparent_wrapper_preserves_native_wave_ownership(
    native_host, monkeypatch, method
):
    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    original = getattr(_WorkflowWave, method)

    @functools.wraps(original)
    async def launch(self, *args, **kwargs):
        return await original(self, *args, **kwargs)

    @functools.wraps(original)
    def release(self, **kwargs):
        return original(self, **kwargs)

    monkeypatch.setattr(
        _WorkflowWave, method, release if method == "_release" else launch
    )
    await run_workflow_wave(h.phase, (prepared,))
    snapshot = h.service.get(h.workflow.id)
    assert len(h.calls) == 1
    assert len(snapshot.results) == 1
    assert snapshot.results[0].status == "completed"
    assert prepared._closed
    assert h.runtime._workflow_waves == {}
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_preflight_refusal_through_wrapped_reader_retires_without_work(
    native_host, monkeypatch
):
    from tldw_chatbook.Chat import console_workflow_preparation as preparation

    h = native_host
    cancelled = threading.Event()
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-1",
        binding=h.binding,
        resolution=h.selected,
        should_cancel=cancelled.is_set,
    )
    original = preparation.run_preparation_read

    @functools.wraps(original)
    async def read(*args, **kwargs):
        return await original(*args, **kwargs)

    monkeypatch.setattr(preparation, "run_preparation_read", read)
    cancelled.set()
    try:
        with pytest.raises(PermissionError, match="source changed"):
            await prepared._read(lambda: pytest.fail("revoked callback entered"))
    finally:
        cancelled.clear()
        await prepared.close()
    assert prepared not in h.runtime._workflow_preparations
    assert not h.runtime._preparation_reads
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert not h.calls
