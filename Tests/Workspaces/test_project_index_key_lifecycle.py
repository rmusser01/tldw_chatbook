"""App closure retains accepted key work before native process shutdown."""

import asyncio
import sys
import threading

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.parametrize("phase", ["queue", "initialization", "active"])
@private_profile_test
async def test_app_close_retains_accepted_key_until_original_callback_retires(
    tmp_path, request, phase
):
    from Tests.Workspaces.test_project_index_key_process import (
        managed_index,
        prepare,
        trace_functions,
        wait_event,
    )
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Workspaces.project_index import runtime as index, key_process

    async with managed_index(tmp_path) as rig:
        entered, release = threading.Event(), threading.Event()

        def observe(function, event, result):
            if event != "call":
                return
            blocked = (
                phase == "initialization" and function is key_process.KeyProcess._start
            )
            if phase == "active" and function is key_process.write_message:
                blocked = sys._getframe(2).f_locals["value"]["type"] == "Command"
            if blocked:
                entered.set()
                assert release.wait(60)

        index._KEY_PROCESS_ENABLED = True
        pending = closing = None
        locked = False
        try:
            if phase == "queue":
                await prepare(rig)
                owner = rig.app._project_index_key_owner
                owner.lock.acquire()
                locked = True
            with trace_functions(
                (key_process.KeyProcess._start, key_process.write_message), observe
            ):
                pending = asyncio.create_task(prepare(rig))
                if phase == "queue":
                    async with asyncio.timeout(30):
                        while not owner.lifetime.calls:
                            await asyncio.sleep(0.01)
                else:
                    await wait_event(entered)
                    owner = rig.app._project_index_key_owner
                assert owner in key_process._owners and owner.lifetime.calls
                assert (
                    rig.controller._preparation_reads == rig.runtime._preparation_reads
                )
                assert len(rig.runtime._preparation_reads) == 1
                closing = asyncio.create_task(
                    index.close_project_index_service(rig.app)
                )
                async with asyncio.timeout(30):
                    while not getattr(rig.app, "_project_index_key_closed", False):
                        await asyncio.sleep(0.01)
                assert not closing.done() and not pending.done()
                closing.cancel()
                await asyncio.sleep(0)
                closing.cancel()
                await asyncio.sleep(0)
                assert not closing.done() and owner.lifetime.calls
                release.set()
                if locked:
                    owner.lock.release()
                    locked = False
                with pytest.raises(
                    RecoveryRequired, match="console_snapshot_owner_changed"
                ):
                    await pending
                with pytest.raises(asyncio.CancelledError):
                    await closing
                assert owner.retired and owner.closed and owner.unresolved is None
                assert owner.process.poll() == 0
                assert not owner.lifetime.calls and owner not in key_process._owners
                assert (
                    not rig.runtime._preparation_reads
                    and not rig.controller._preparation_reads
                )
                assert owner.selected == (0 if phase == "initialization" else 1)
        finally:
            release.set()
            if locked:
                owner.lock.release()
            await asyncio.gather(
                *(task for task in (pending, closing) if task is not None),
                return_exceptions=True,
            )
            index._KEY_PROCESS_ENABLED = False
