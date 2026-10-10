"""Runtime shutdown joins the original finite Workspace availability reader."""

import asyncio
import contextlib
import inspect
import sys
import threading

import pytest

from Tests.Chat.test_console_configuration_worker_lifetime import (
    _OriginalConfigurationWorkspaceRead,
)
from Tests.UI.test_console_hook_review_send_freeze import _until
from Tests.UI.test_console_received_intent_feedback import _received_console_case
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class _OriginalAvailabilityRead(_OriginalConfigurationWorkspaceRead):
    """Hold after the real binding SELECT under its original native operation."""

    def __init__(self, workspace, registry):
        super().__init__(registry)
        self.workspace = workspace
        self.code = inspect.unwrap(type(registry).list_runtime_bindings).__code__
        self.caller_code = (
            ConsoleWorkspaceController._read_workspace_files_availability.__code__
        )
        self.operation = self.participant = None
        self.invocations = 0

    def _started(self, code, _offset):
        if (
            code is self.caller_code
            and sys._getframe(1).f_locals.get("self") is self.workspace
        ):
            self.invocations += 1

    def _line(self, code, _line):
        if code is not self.code or self.entered.is_set():
            return
        frame = sys._getframe(1)
        if (
            frame.f_locals.get("self") is not self.registry
            or "rows" not in frame.f_locals
        ):
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not self.caller_code:
            parent = parent.f_back
        if parent is None or parent.f_locals.get("self") is not self.workspace:
            return
        self.connection = frame.f_locals["conn"]
        self.thread = threading.current_thread()
        self.participant = self.registry.db._maintenance_participant
        self.operation = getattr(storage._operation_local, "operation", None)
        with storage._lock:
            self.lease = self.participant.connections.get(self.connection)
            self.live_at_entry = (
                self.lease in storage._live_leases
                and self.lease.resource_thread is self.thread
                and self.operation in storage._operations
                and self.operation.participant is self.participant
            )
        self.entered.set()
        self.release_timed_out = not self.release.wait(8)

    @contextlib.contextmanager
    def installed(self):
        monitoring = sys.monitoring
        tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
        monitoring.use_tool_id(tool, "workspace-original-availability-shutdown")
        try:
            monitoring.register_callback(tool, monitoring.events.LINE, self._line)
            monitoring.register_callback(
                tool, monitoring.events.PY_START, self._started
            )
            monitoring.set_local_events(tool, self.code, monitoring.events.LINE)
            monitoring.set_local_events(
                tool, self.caller_code, monitoring.events.PY_START
            )
            yield self
        finally:
            self.release.set()
            monitoring.set_local_events(tool, self.code, 0)
            monitoring.set_local_events(tool, self.caller_code, 0)
            monitoring.register_callback(tool, monitoring.events.LINE, None)
            monitoring.register_callback(tool, monitoring.events.PY_START, None)
            monitoring.free_tool_id(tool)


def _assert_original_reader_live(probe):
    assert probe.entered.is_set() and probe.live_at_entry
    assert probe.thread is not threading.current_thread()
    assert not probe.release_timed_out and not probe.release.is_set()
    with storage._lock:
        assert probe.operation in storage._operations
        assert probe.participant.connections.get(probe.connection) is probe.lease
        assert probe.lease in storage._live_leases
    assert not probe.retired()


async def test_dispose_retains_original_workspace_availability_through_repeated_cancel(
    monkeypatch,
):
    async with _received_console_case(monkeypatch, "availability-lifetime") as case:
        workspace = case.console._workspace
        registry = case.console.app_instance.workspace_registry_service
        # Let only the already-issued initial projection finish before selecting
        # the next real refresh; no producer or permission callback is replaced.
        assert await _until(
            lambda: not workspace._workspace_files_availability_refresh_in_flight, 10
        )
        probe = _OriginalAvailabilityRead(workspace, registry)
        disposal = None
        with probe.installed():
            try:
                workspace._request_workspace_files_availability_refresh(
                    (case.session.workspace_id,)
                )
                assert await _until(probe.entered.is_set, 10)
                _assert_original_reader_live(probe)
                disposal = asyncio.create_task(case.runtime.dispose())
                assert await _until(
                    lambda: case.runtime._disposed or disposal.done(), 5
                )
                for _ in range(2):
                    disposal.cancel()
                    # Deliver cancellation and its normal Future callbacks; this
                    # is a scheduling checkpoint, not a widened shutdown grace.
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                _assert_original_reader_live(probe)
                assert not disposal.done(), (
                    "runtime disposal returned while its original Workspace "
                    "availability operation and native lease were still live"
                )
                probe.release.set()
                await asyncio.wait_for(
                    asyncio.gather(disposal, return_exceptions=True), 15
                )
                assert await _until(probe.retired, 5)
                with storage._lock:
                    assert probe.operation not in storage._operations
                    assert probe.connection not in probe.participant.connections
                    assert probe.lease not in storage._live_leases
                assert not workspace._workspace_files_availability_refresh_in_flight
                completed_reads = probe.invocations
                # A disposed runtime may still have a mounted screen/timers.
                # A different requested tuple must not start a successor read.
                workspace._request_workspace_files_availability_refresh(
                    (case.session.workspace_id, "availability-after-dispose")
                )
                assert not workspace._workspace_files_availability_refresh_in_flight
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert probe.invocations == completed_reads
            finally:
                probe.release.set()
                if disposal is not None:
                    await asyncio.wait_for(
                        asyncio.gather(disposal, return_exceptions=True), 15
                    )
                if probe.entered.is_set():
                    assert await _until(probe.retired, 5)
                assert await _until(
                    lambda: not workspace._workspace_files_availability_refresh_in_flight,
                    5,
                )
