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


@pytest.mark.parametrize("mode", ["lazy", "held", "cancel", "changed", "wrong"])
def test_listening_intake_quiesces_with_the_live_runtime(tmp_path, mode):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = r"""
import asyncio, sys, threading, time
from pathlib import Path
import pytest
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.runtime_maintenance import RuntimeMaintenance
from tldw_chatbook.Subscriptions import article_listening_lease as listening
from tldw_chatbook.config import get_user_data_dir

async def main():
    mode = sys.argv[2]
    app = TldwCli()
    runtime = RuntimeMaintenance(app)
    app._backup_runtime_maintenance = runtime
    producer = listening.listening_producer
    worker = None
    entered, finish = threading.Event(), threading.Event()
    try:
        root = get_user_data_dir() / "article_audio"
        if mode == "wrong":
            listening.listening_producer = object()
            with pytest.raises(RecoveryRequired, match="runtime_owner_unqualified"):
                await runtime.settle_producers(time.monotonic()+10)
            return
        if mode in {"held", "cancel"}:
            db = app.subscriptions_db
            def accepted():
                with listening.hold_listening(db.db_path, profile=None):
                    entered.set()
                    assert finish.wait(12)
            worker = asyncio.create_task(asyncio.to_thread(accepted))
            assert await asyncio.to_thread(entered.wait, 8)
            if mode == "cancel":
                owner = asyncio.current_task()
                async def cancel_after_fence():
                    async with asyncio.timeout(8):
                        while not producer._lifetime.closed:
                            await asyncio.sleep(.01)
                    owner.cancel()
                observer = asyncio.create_task(cancel_after_fence())
                with pytest.raises(asyncio.CancelledError):
                    await runtime.settle_producers(time.monotonic()+10)
                await observer
            else:
                with pytest.raises(RecoveryRequired, match="runtime_work_not_settled"):
                    await runtime.settle_producers(time.monotonic()+.2)
            assert producer._lifetime.closed and not worker.done()
            with pytest.raises(RecoveryRequired, match="runtime_producer_paused"):
                with listening.hold_listening(db.db_path, profile=None): pass
            finish.set()
            await worker
            return
        await runtime.settle_producers(time.monotonic()+10)
        assert not root.exists() and producer._lifetime.closed
        with pytest.raises(RecoveryRequired, match="runtime_producer_paused"):
            with listening.hold_listening(root / "unopened.db", profile=None): pass
        assert not root.exists()
        if mode == "changed":
            runtime.retire_local_caches()
            listening.listening_producer = listening.ListeningProducer()
            with pytest.raises(RecoveryRequired, match="participant_runtime_coverage_incomplete"):
                runtime._require_storage_coverage(runtime.pause)
    finally:
        finish.set()
        if worker is not None: await worker
        listening.listening_producer = producer
        if runtime.closed: await runtime.resume()
        assert not producer._lifetime.closed
        await app._shutdown_app_owned_lifecycles()
        await app.tts_service.close()
        await app.tts_service.wait_closed()
asyncio.run(main())
print('retired and reopened')
"""
    _run(tmp_path, "listening", mode, script=script)
