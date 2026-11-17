"""Complete key command through the stock retained Console preparation."""

import os
import asyncio
import threading
import time
import io
import sys
from dataclasses import replace
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

import pytest


from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    capture_run_admitted_workspace_roots,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Workspaces.project_index import runtime as index


@contextmanager
def trace_functions(functions, callback):
    """Observe exact original code, including already-created executor threads."""
    monitor = sys.monitoring
    tool = next(number for number in range(6) if monitor.get_tool(number) is None)
    monitor.use_tool_id(tool, "project-index-key-control")
    codes = {function.__code__: function for function in functions}

    def started(code, offset):
        callback(codes[code], "call", None)

    def returned(code, offset, value):
        callback(codes[code], "return", value)

    try:
        monitor.register_callback(tool, monitor.events.PY_START, started)
        monitor.register_callback(tool, monitor.events.PY_RETURN, returned)
        for code in codes:
            monitor.set_local_events(
                tool, code, monitor.events.PY_START | monitor.events.PY_RETURN
            )
        yield
    finally:
        for code in codes:
            monitor.set_local_events(tool, code, 0)
        monitor.register_callback(tool, monitor.events.PY_START, None)
        monitor.register_callback(tool, monitor.events.PY_RETURN, None)
        monitor.free_tool_id(tool)


@asynccontextmanager
async def managed_index(tmp_path):
    from tldw_chatbook import config
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    db = WorkspaceDB(config.get_workspaces_db_path())
    registry = LocalWorkspaceRegistryService(db)
    registry.create_workspace(workspace_id="key-process", name="Project")
    project = tmp_path / "project"
    project.mkdir()
    registry.add_folder_binding("key-process", project)
    app = SimpleNamespace(workspace_registry_service=registry)
    try:
        runtime = ConsoleRuntime(app)
        app.console_runtime = runtime
        store = ConsoleChatStore()
        controller = ConsoleChatController(store=store, provider_gateway=object())
        runtime.set_chat_store(store)
        runtime.set_chat_controller(controller)
        session = store.create_session(workspace_id="key-process")
        roots = capture_run_admitted_workspace_roots(session=session, registry=registry)
        cache = config.get_user_data_dir() / "project_index"
        try:
            yield SimpleNamespace(
                app=app,
                controller=controller,
                runtime=runtime,
                store=store,
                session=session,
                roots=roots,
                cache=cache,
            )
        finally:
            await runtime.dispose()
    finally:
        try:
            await index.close_project_index_service(app)
        finally:
            db.close()


async def prepare(rig):
    from tldw_chatbook.Chat.console_project_index_preparation import (
        prepare_project_index_for_composition,
    )

    return await prepare_project_index_for_composition(
        rig.controller, rig.session.id, rig.roots, None, None, None
    )


@private_profile_test
async def test_selected_nested_key_runs_in_child_and_retires(tmp_path, request):
    from tldw_chatbook.Workspaces.project_index import key_process

    async with managed_index(tmp_path) as rig:
        old = getattr(index, "_KEY_PROCESS_ENABLED", False)
        index._KEY_PROCESS_ENABLED = True
        counts = Counter()
        expected = await asyncio.to_thread(
            index.project_index_key,
            rig.app,
            rig.roots[0].workspace_id,
            rig.roots[0].binding_id,
            rig.roots[0].root,
        )
        keys = []

        def observe(function, event, arg):
            if event == "call" and function is index._project_index_key_owned:
                counts["parent_key"] += 1
            if (
                event == "return"
                and function is key_process._decode_key
                and arg is not None
            ):
                keys.append(arg)

        try:
            with trace_functions(
                (index._project_index_key_owned, key_process._decode_key), observe
            ):
                service, scopes = await prepare(rig)
        finally:
            index._KEY_PROCESS_ENABLED = old
        assert service is rig.app._project_index_service
        assert not scopes and not rig.cache.exists()
        assert counts["parent_key"] == 0
        assert keys == [expected]
        owner = rig.app._project_index_key_owner
        assert owner.process.pid != os.getpid()
        assert owner.selected == 1
        assert owner.receipts[-1]["connections"] >= 1
        assert owner.receipts[-1]["pins"] == 1
        assert not rig.controller._preparation_reads
        assert not rig.runtime._preparation_reads
    assert owner.retired and owner.unresolved is None
    assert owner.process.poll() == 0
    assert owner.process.stdin.closed and owner.process.stdout.closed
    assert not owner.monitor.is_alive()


@private_profile_test
async def test_default_route_keeps_parent_key_and_has_no_owner(tmp_path, request):
    assert not getattr(index, "_KEY_PROCESS_ENABLED", False)
    async with managed_index(tmp_path) as rig:
        service, scopes = await prepare(rig)
        assert service is rig.app._project_index_service
        assert not scopes and not rig.cache.exists()
        assert not hasattr(rig.app, "_project_index_key_owner")


@private_profile_test
async def test_ready_source_revocation_has_positive_idle_retirement(tmp_path, request):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Workspaces.project_index import key_process

    async with managed_index(tmp_path) as rig:
        index._KEY_PROCESS_ENABLED = True
        previous = threading.getprofile()

        def observe(frame, event, arg):
            if (
                event == "return"
                and frame.f_code is key_process.KeyProcess._start.__code__
            ):
                rig.session.incarnation_id = "replaced"

        threading.setprofile(observe)
        try:
            with pytest.raises(
                RecoveryRequired, match="console_snapshot_owner_changed"
            ):
                await prepare(rig)
        finally:
            threading.setprofile(previous)
            index._KEY_PROCESS_ENABLED = False
        owner = rig.app._project_index_key_owner
        assert owner.selected == 0
        assert owner.receipts and owner.receipts[0]["retired"]
        assert owner.unresolved is None


async def wait_event(event):
    async with asyncio.timeout(60):
        while not event.is_set():
            await asyncio.sleep(0.01)


@private_profile_test
async def test_custom_and_ambient_sources_fall_back_before_launch(tmp_path, request):
    from tldw_chatbook.Backup_Recovery.participants import core_operation
    from tldw_chatbook.Workspaces.project_index import key_process

    async with managed_index(tmp_path) as rig:
        service = await asyncio.to_thread(index.get_project_index_service, rig.app)
        original = service._reader
        service._reader = lambda key: original(key)
        index._KEY_PROCESS_ENABLED = True
        try:
            result, _ = await prepare(rig)
            assert result is service
            assert not hasattr(rig.app, "_project_index_key_owner")
        finally:
            service._reader = original
            index._KEY_PROCESS_ENABLED = False
        database = rig.app.workspace_registry_service.db

        def ambient():
            with service.registry_connection_scope():
                assert key_process._ambient_eligible(service, database)
                with database.connection():
                    assert not key_process._ambient_eligible(service, database)
                # The ordinary outer owner may retain an idle cached connection.
                assert key_process._ambient_eligible(service, database)
                with core_operation(database):
                    assert not key_process._ambient_eligible(service, database)
                with database.transaction(immediate=True):
                    assert not key_process._ambient_eligible(service, database)
                assert key_process._ambient_eligible(service, database)

        await asyncio.to_thread(ambient)


@private_profile_test
async def test_key_refusals_and_present_cache_do_not_replay(tmp_path, request):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path
    from tldw_chatbook.Workspaces.project_index import key_process

    async with managed_index(tmp_path) as rig:
        root = rig.roots[0]
        expected = await asyncio.to_thread(
            index.project_index_key,
            rig.app,
            root.workspace_id,
            root.binding_id,
            root.root,
        )
        service = rig.app._project_index_service

        def create_cache():
            with service._factory.retain(expected) as pin, pin.bind():
                store = service._factory(expected)
                store.close_native()
            return index_store_path(service._factory.data_root, expected)

        cache = await asyncio.to_thread(create_cache)
        before = []
        unchanged = []

        def observe(function, event, result):
            if function is key_process.KeyProcess.derive_key and event == "call":
                before[:] = [cache.read_bytes()]
            elif function is key_process.KeyProcess._receipt and event == "return":
                unchanged.append(cache.read_bytes() == before[0])

        index._KEY_PROCESS_ENABLED = True
        try:
            with trace_functions(
                (key_process.KeyProcess.derive_key, key_process.KeyProcess._receipt),
                observe,
            ):
                await prepare(rig)
            owner = rig.app._project_index_key_owner
            assert owner.selected == 1
            assert unchanged == [True, True]
            # Keep the original authority guard; only command facts differ.
            for roots in (
                (replace(root, binding_id="missing-binding"),),
                (replace(root, workspace_id="wrong-workspace"),),
            ):
                rig.roots = roots
                with trace_functions(
                    (
                        key_process.KeyProcess.derive_key,
                        key_process.KeyProcess._receipt,
                    ),
                    observe,
                ):
                    result, scopes = await prepare(rig)
                assert result is None and not scopes
                assert owner.unresolved is None
            assert owner.selected == 3
            assert len(owner.receipts) == 4
            assert unchanged == [True] * 4
        finally:
            index._KEY_PROCESS_ENABLED = False


@private_profile_test
async def test_queued_session_revocation_rechecks_before_command(tmp_path, request):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    async with managed_index(tmp_path) as rig:
        index._KEY_PROCESS_ENABLED = True
        try:
            await prepare(rig)
            owner = rig.app._project_index_key_owner
            owner.lock.acquire()
            waiting = asyncio.create_task(prepare(rig))
            try:
                async with asyncio.timeout(30):
                    while not owner.lifetime.calls:
                        await asyncio.sleep(0.01)
                assert (
                    rig.controller._preparation_reads == rig.runtime._preparation_reads
                )
                assert len(rig.controller._preparation_reads) == 1
                rig.store._sessions.pop(rig.session.id)
            finally:
                owner.lock.release()
            with pytest.raises(
                RecoveryRequired, match="console_snapshot_owner_changed"
            ):
                await waiting
            assert owner.selected == 1 and owner.unresolved is None
        finally:
            index._KEY_PROCESS_ENABLED = False


class CorruptFrame:
    """Alter an actual child reply after its native work, not the codec function."""

    def __init__(self, original, corruption, process):
        self.original, self.corruption, self.process = original, corruption, process
        self.buffer = io.BytesIO()
        self.changed = False

    def read(self, count):
        from tldw_chatbook.MCP.permission_process import read_message, write_message

        result = self.buffer.read(count)
        if result:
            return result
        frame = read_message(self.original)
        if not self.changed and (
            frame["type"] == "Settled" or self.corruption == "exit"
        ):
            self.changed = True
            if self.corruption == "ordinal":
                frame["ordinal"] += 1
            elif self.corruption == "receipt":
                frame["data"]["retirement"]["retired"] = False
            elif self.corruption == "value":
                frame["data"]["value"]["binding_id"] = "another-binding"
            else:
                self.process.kill()
                self.process.wait(timeout=10)
                return b""
        self.buffer = io.BytesIO()
        write_message(self.buffer, frame)
        self.buffer.seek(0)
        return self.buffer.read(count)

    def close(self):
        self.original.close()

    @property
    def closed(self):
        return self.original.closed


@pytest.mark.parametrize("corruption", ["ordinal", "receipt", "value", "exit"])
@private_profile_test
async def test_uncertain_reply_stays_registered_and_blocks_maintenance(
    tmp_path, request, corruption
):
    from tldw_chatbook.Workspaces.project_index import key_process
    from tldw_chatbook.MCP.permission_process import ProtocolError

    assertion = None
    with pytest.raises(ProtocolError, match="key_outcome_unresolved"):
        async with managed_index(tmp_path) as rig:
            index._KEY_PROCESS_ENABLED = True
            try:
                await prepare(rig)
                owner = rig.app._project_index_key_owner
                owner.process.stdout = CorruptFrame(
                    owner.process.stdout, corruption, owner.process
                )
                service, scopes = await prepare(rig)
                assert service is None and not scopes
                assert owner.unresolved is not None and owner in key_process._owners
                assert owner.retired and owner.process.poll() is not None
                assert owner.selected == 2
                key_process.participant._maintenance_close_admission()
                assert not await key_process.participant._maintenance_drain(
                    time.monotonic() + 10
                )
                assert not key_process.participant.drained()
                with pytest.raises(ProtocolError, match="key_process_not_drained"):
                    key_process.participant._maintenance_resume()
                await prepare(rig)
                assert owner.selected == 2
                assert not rig.controller._preparation_reads
            except BaseException as error:
                assertion = error
            finally:
                index._KEY_PROCESS_ENABLED = False
    if assertion is not None:
        raise assertion


@private_profile_test
async def test_healthy_maintenance_restarts_with_fresh_epoch(tmp_path, request):
    from tldw_chatbook.Workspaces.project_index import key_process
    from tldw_chatbook.Backup_Recovery.runtime_maintenance import (
        _bind,
        _settle_stage,
        _resume_hooks,
    )

    async with managed_index(tmp_path) as rig:
        index._KEY_PROCESS_ENABLED = True
        try:
            await prepare(rig)
            owner = rig.app._project_index_key_owner
            epoch, process = owner.epoch, owner.process
            hook = _bind(
                key_process.participant,
                "Workspaces.project_index.key_process",
                "KeyProcessLifetime",
            )
            closed = []
            await _settle_stage([hook], closed, time.monotonic() + 60)
            assert key_process.participant.drained() and owner.retired
            assert process.poll() == 0
            await _resume_hooks(closed)
            assert owner.epoch != epoch and owner.process is None
            await prepare(rig)
            assert owner.selected == 2 and owner.process.pid != process.pid
        finally:
            index._KEY_PROCESS_ENABLED = False


@pytest.mark.parametrize("route", ["dispatch", "next-send", "personal"])
@private_profile_test
async def test_shared_composition_cancellation_retains_selected_child_parent_tail(
    tmp_path, request, monkeypatch, route
):
    from tldw_chatbook.Chat import console_chat_controller as host

    async with managed_index(tmp_path) as rig:
        entered, release = threading.Event(), threading.Event()
        consumed = []

        async def mcp(*args, **kwargs):
            return None

        def status(service, key):
            entered.set()
            assert release.wait(60)
            return None

        rig.controller._compose_mcp_provider = mcp
        rig.controller._compose_local_provider = lambda **kwargs: (
            consumed.append(kwargs),
            None,
        )
        monkeypatch.setattr(
            host, "capture_run_admitted_workspace_roots", lambda **kwargs: rig.roots
        )
        monkeypatch.setattr(index, "project_index_status_for_console", status)
        index._KEY_PROCESS_ENABLED = True
        pending = asyncio.create_task(
            rig.controller._compose_agent_request_providers(
                session_id=rig.session.id,
                project_selection=None,
                project_authority_guard=None,
                publish_mcp_counts=route == "dispatch",
                **({} if route == "personal" else {"admitted_roots": rig.roots}),
            )
        )
        try:
            await wait_event(entered)
            owner = rig.app._project_index_key_owner
            assert owner.selected == 1 and owner.state == "ready"
            assert owner.receipts[-1]["retired"]
            assert rig.controller._preparation_reads == rig.runtime._preparation_reads
            assert len(rig.controller._preparation_reads) == 1
            pending.cancel()
            await asyncio.sleep(0)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done() and not consumed
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
            index._KEY_PROCESS_ENABLED = False
        assert (
            not rig.controller._preparation_reads and not rig.runtime._preparation_reads
        )
        assert not consumed and owner.selected == 1
