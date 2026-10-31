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
from tldw_chatbook.Chat.console_preparation_reads import (
    _source_binding,
    preparation_reads_for,
)
from tldw_chatbook.UI.Console_Modules.workspace import (
    ConsoleWorkspaceController,
    _WorkspaceAvailabilityObsolete,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class _OriginalAvailabilityRead(_OriginalConfigurationWorkspaceRead):
    """Hold after the real binding SELECT under its original native operation."""

    def __init__(self, workspace, registry):
        super().__init__(registry)
        self.workspace = workspace
        self.code = inspect.unwrap(type(registry).list_runtime_bindings).__code__
        self.caller_code = getattr(
            ConsoleWorkspaceController, "_read_workspace_files_availability",
            ConsoleWorkspaceController._capture_workspace_files_availability,
        ).__code__
        self.operation = self.participant = None
        self.invocations = 0
        self.read = None
        publisher = ConsoleWorkspaceController._refresh_workspace_files_availability_snapshot
        self.publisher_code = publisher.__code__
        lines, first = inspect.getsourcelines(publisher)
        self.publication_lines = frozenset(
            first + index
            for index, line in enumerate(lines)
            if line.strip().startswith((
                "self._workspace_files_availability_by_id =",
                "self._workspace_files_runtime_bindings_by_id =",
                "self._workspace_files_availability_cached_at =",
            ))
        )
        assert len(self.publication_lines) == 3
        self.publications = []
        self.publication_overflow = False

    def _started(self, code, _offset):
        if (
            code is self.caller_code
            and sys._getframe(1).f_locals.get("self") is self.workspace
        ):
            self.invocations += 1

    def _line(self, code, _line):
        if code is self.publisher_code:
            frame = sys._getframe(1)
            if _line in self.publication_lines and frame.f_locals.get("self") is self.workspace:
                if len(self.publications) < 64:
                    self.publications.append((asyncio.current_task(), _line))
                else:
                    self.publication_overflow = True
            return
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
        binding = _source_binding.get()
        assert binding is not None and binding.active
        self.read = binding.read
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
        tool = next(value for value in range(1, 6) if monitoring.get_tool(value) is None)
        monitoring.use_tool_id(tool, "workspace-original-availability-shutdown")
        try:
            monitoring.register_callback(tool, monitoring.events.LINE, self._line)
            monitoring.register_callback(
                tool, monitoring.events.PY_START, self._started
            )
            monitoring.set_local_events(tool, self.code, monitoring.events.LINE)
            monitoring.set_local_events(tool, self.publisher_code, monitoring.events.LINE)
            monitoring.set_local_events(
                tool, self.caller_code, monitoring.events.PY_START
            )
            yield self
        finally:
            self.release.set()
            monitoring.set_local_events(tool, self.code, 0)
            monitoring.set_local_events(tool, self.publisher_code, 0)
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


@pytest.mark.parametrize("changed", ["outer", "folder"])
async def test_held_availability_source_change_cannot_publish(monkeypatch, changed):
    async with _received_console_case(monkeypatch, f"availability-source-{changed}") as case:
        workspace = case.console._workspace
        registry = case.console.app_instance.workspace_registry_service
        assert await _until(
            lambda: not workspace._workspace_files_availability_refresh_in_flight, 10
        )
        probe = _OriginalAvailabilityRead(workspace, registry)
        with probe.installed():
            try:
                workspace._request_workspace_files_availability_refresh(
                    (case.session.workspace_id,)
                )
                assert await _until(probe.entered.is_set, 10)
                _assert_original_reader_live(probe)
                read, producer = _held_preparation_read(probe)
                if changed == "outer":
                    monkeypatch.setattr(workspace, "_capture_workspace_files_availability", lambda _ids: ({}, {}))
                else:
                    monkeypatch.setattr(registry, "list_folder_bindings", lambda _id: ())
            finally:
                probe.release.set()
                assert await _until(probe.retired, 5)
                assert await _until(
                    lambda: not workspace._workspace_files_availability_refresh_in_flight, 5
                )
        _assert_obsolete_read_did_not_publish(probe, read, producer)


def _held_preparation_read(probe):
    read = probe.read
    assert read is not None and read.creator is probe.workspace
    assert read.task is probe.workspace._workspace_files_availability_task
    native_code = next(
        code
        for code in ConsoleWorkspaceController._read_owned_workspace_availability.__code__.co_consts
        if inspect.iscode(code) and code.co_name == "read_captured_availability"
    )
    assert read.callback.__code__ is native_code
    assert tuple(
        item
        for item in preparation_reads_for(probe.workspace._preparation_reads)
        if item.creator is probe.workspace
        and item.task is read.task
        and getattr(item.callback, "__code__", None) is native_code
    ) == (read,)
    assert not read.retired.done()
    producer = read._producer
    assert producer is not None
    assert not producer.done() and not producer.cancelled()
    return read, producer


def _assert_obsolete_read_did_not_publish(probe, read, producer):
    assert read._producer is producer
    assert read.retired.done() and not read.retired.cancelled()
    assert read.retired.result() is None
    assert producer.done() and not producer.cancelled()
    error = producer.exception()
    assert type(error) is _WorkspaceAvailabilityObsolete
    assert error.args == ("workspace_availability_source_changed",)
    assert not probe.publication_overflow
    assert all(task is not read.task for task, _line in probe.publications)
    with storage._lock:
        assert probe.operation not in storage._operations
        assert probe.connection not in probe.participant.connections
        assert probe.lease not in storage._live_leases


async def test_fresh_custom_availability_can_publish_after_source_refusal(
    tmp_path,
    monkeypatch,
):
    """A rejected old read does not freeze a separate current custom refresh."""
    from types import SimpleNamespace

    from Tests.UI.test_console_workspace_controller import (
        _AsyncWorkerScreen,
        _workspace_controller,
    )
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.models import (
        RuntimeBindingKind,
        RuntimeBindingStatus,
        WorkspaceRuntimeBinding,
    )
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    database = WorkspaceDB(
        tmp_path / "source-publication.sqlite", client_id="source-publication"
    )
    registry = LocalWorkspaceRegistryService(database)
    for workspace_id in ("old", "fresh"):
        registry.create_workspace(workspace_id=workspace_id, name=workspace_id)
    database.close()
    screen = _AsyncWorkerScreen()
    workspace = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )
    probe = _OriginalAvailabilityRead(workspace, registry)
    calls = []

    def folders(workspace_id):
        calls.append(workspace_id)
        return (
            WorkspaceRuntimeBinding(
                workspace_id=workspace_id,
                binding_id="fresh-binding",
                binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM,
                label="Fresh source",
                locator=str(tmp_path),
                status=RuntimeBindingStatus.READY,
            ),
        )

    with probe.installed():
        try:
            before = workspace._workspace_files_availability_cached_at
            workspace._request_workspace_files_availability_refresh(("old",))
            assert await _until(probe.entered.is_set, 10)
            _assert_original_reader_live(probe)
            read, producer = _held_preparation_read(probe)
            monkeypatch.setattr(registry, "list_folder_bindings", folders)
            probe.release.set()
            await asyncio.wait_for(asyncio.shield(read.task), 5)
            assert await _until(probe.retired, 5)
            _assert_obsolete_read_did_not_publish(probe, read, producer)
            assert workspace._workspace_files_availability_cached_at == before
            workspace._request_workspace_files_availability_refresh(("fresh",))
            fresh_task = screen.workers[-1][0]
            assert fresh_task is not read.task
            await asyncio.wait_for(asyncio.shield(fresh_task), 5)
            assert calls == ["fresh"]
            assert workspace._workspace_files_availability_by_id == {"fresh": True}
            assert workspace._workspace_files_runtime_bindings_by_id == {"fresh": ()}
            assert {
                line for task, line in probe.publications if task is fresh_task
            } == probe.publication_lines
            _assert_obsolete_read_did_not_publish(probe, read, producer)
            assert workspace._workspace_files_availability_cached_at > before
        finally:
            probe.release.set()
            await asyncio.gather(
                *(row[0] for row in screen.workers), return_exceptions=True
            )
            database.close()
