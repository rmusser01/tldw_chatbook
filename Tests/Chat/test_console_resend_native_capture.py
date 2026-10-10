"""Original Resend must retain the existing finite native capture boundary."""

import asyncio
import threading

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as snapshot_controls
from Tests.Chat import test_console_turn_resend as resend_controls
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Chat.console_turn_resend import resend_target_id, resend_turn

catalog_controls = snapshot_controls.catalog_controls
catalog_store = snapshot_controls.catalog_store
local_root = snapshot_controls.local_root
mcp_sources = snapshot_controls.mcp_sources
snapshot_case = snapshot_controls.snapshot_case
databases = resend_controls.databases

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_original_resend_keeps_checked_mcp_capture_off_loop_and_retired(
    snapshot_case, tmp_path, databases
):
    """Real broken rows and the original replay retain worker-owned source IO.

    The existing blocked gateway is the transport/selection substitute. All
    source readers, capture, hooks/admission, replay and row checks are original.
    Its refusal must occur before the replay clears any original row.
    """
    case = snapshot_case
    _, loop_inventory = snapshot_controls._loop_projection(case)
    console = resend_controls._console(tmp_path, databases, "error")
    await console.controller.submit_draft("original broken turn")
    user = resend_controls._user(console)
    assert resend_target_id(resend_controls._path(console)) == user.id
    before_rows = resend_controls._all_db_rows(console)
    before_path = resend_controls._path_snapshot(console)
    before_provider_calls = len(console.gateway.seen)
    # Install a real already-declared MCP service through the normal App seam;
    # the broken turn itself was prepared by the original file-backed fixture.
    console.controller.app = case.app
    console.gateway.mode = "blocked"
    loop_thread = threading.current_thread()
    probe = snapshot_controls._MaximumProbe(case.source, case.permissions, hold=True)
    task = None
    with probe.installed():
        try:
            task = asyncio.create_task(resend_turn(console.controller, user.id))
            # Existing 4s arrival and 8s held-worker limits. A Main-thread read
            # is never held by this existing probe, so RED cannot deadlock it.
            await catalog_controls._worker_entered(probe, task)
            assert probe.leases and all(
                lease in storage_admission._live_leases for lease in probe.leases
            )
            assert resend_controls._path_snapshot(console) == before_path
            assert len(console.gateway.seen) == before_provider_calls
            probe.release.set()
            result = await task
            assert (result.accepted, result.visible_copy) == (
                False,
                resend_controls.BLOCKED_COPY,
            )
            assert len(probe.read_threads) == len(probe.permission_threads) == 1
            assert all(
                thread is not loop_thread
                for thread in probe.read_threads + probe.permission_threads
            ), "original replay returned native MCP capture to its caller loop"
            assert loop_inventory == [True]
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
            # Preserve the original blocked-readiness ABI: its one SYSTEM copy
            # is appended, while every prior native row remains untouched.
            after_rows = dict(
                (row[0], row[1:]) for row in resend_controls._all_db_rows(console)
            )
            assert all(after_rows[row[0]] == row[1:] for row in before_rows)
            path = resend_controls._path(console)
            assert (
                resend_controls._path_snapshot(console)[: len(before_path)]
                == before_path
            )
            assert path[-1].role is resend_controls.SYSTEM
            assert path[-1].content == resend_controls.BLOCKED_COPY
            assert resend_target_id(path) == user.id
            assert len(console.gateway.seen) == before_provider_calls
        finally:
            probe.release.set()
            if task is not None:
                await catalog_controls._settle(task, probe)
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
            assert not console.controller._maintenance_calls
