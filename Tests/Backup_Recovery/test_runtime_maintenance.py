"""Live maintenance preserves actual producers across refused capture attempts."""

import asyncio
import time

import pytest

from Tests.Backup_Recovery.test_image_edit_maintenance import start
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.runtime_maintenance import (
    _bind,
    _resume_hooks,
    _settle_stage,
)
from tldw_chatbook.Chat.console_image_edit_operations import ImageEditOperationRegistry


def bind(registry):
    return _bind(
        registry, "Chat.console_image_edit_operations", "ImageEditOperationRegistry"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_wait", [False, True])
async def test_refused_capture_reopens_actual_registry_without_canceling_edit(
    cancel_wait,
):
    registry = ImageEditOperationRegistry()
    entered, release = asyncio.Event(), asyncio.Event()

    async def runner(_generation):
        entered.set()
        await release.wait()

    operation = start(registry, runner)
    await entered.wait()
    closed = []
    hook = bind(registry)
    try:
        if cancel_wait:
            waiting = asyncio.create_task(
                _settle_stage([hook], closed, time.monotonic() + 5)
            )
            await asyncio.sleep(0)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        else:
            with pytest.raises(RecoveryRequired, match="runtime_work_not_settled"):
                await _settle_stage([hook], closed, time.monotonic())
        assert start(registry, runner, "during") is None
        assert not operation.cancel_event.is_set()
        assert not operation.task.done()
        await _resume_hooks(closed)
        resumed = start(registry, runner, "after")
        assert resumed is not None
        release.set()
        await asyncio.gather(operation.task, resumed.task)
    finally:
        release.set()
        await operation.task


@pytest.mark.asyncio
async def test_binding_uses_original_owner_methods_and_rejects_custom_owner():
    registry = ImageEditOperationRegistry()

    def unexpected():
        pytest.fail("instance override was called")

    registry._maintenance_close_admission = unexpected
    hook = bind(registry)
    closed = []
    await _settle_stage([hook], closed, time.monotonic())
    assert registry._maintenance_closed
    await _resume_hooks(closed)
    assert not registry._maintenance_closed
    with pytest.raises(RecoveryRequired, match="runtime_owner_unqualified"):
        bind(object())


def test_runtime_cannot_retire_storage_before_producers_settle(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = r"""
import asyncio
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.runtime_maintenance import RuntimeMaintenance
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
async def main():
    app = object.__new__(TldwCli)
    runtime = RuntimeMaintenance(app)
    try:
        try:
            runtime.retire_local_caches()
        except RecoveryRequired as error:
            assert str(error) == 'runtime_producers_not_settled'
        else:
            raise AssertionError('storage retired without producer settlement')
    finally:
        if runtime.pause is not None:
            runtime.pause.resume()
asyncio.run(main())
print('retired and reopened')
"""
    _run(tmp_path, "runtime", "unsettled", script=script)
