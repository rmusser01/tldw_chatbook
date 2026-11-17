"""Fresh-import source qualification for the disabled complete key command."""

import sys
import asyncio
import threading
import time
import inspect
import subprocess
import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


def observe_child_begin(monkeypatch, marker):
    """Signal original BEGIN-line dispatch; never claim an observed native busy wait."""
    from tldw_chatbook.DB.store_kernel import SQLiteStoreKernel

    code = inspect.unwrap(SQLiteStoreKernel.transaction).__code__
    source = Path(code.co_filename)
    line = next(
        number
        for number, text in enumerate(source.read_text().splitlines(), 1)
        if 'conn.execute("BEGIN IMMEDIATE" if immediate' in text
    )
    script = f"""
import sys
def _key_begin_profile(frame, event, arg):
    if event != "call" or frame.f_code.co_filename != {str(source)!r} or frame.f_code.co_name != "transaction":
        return
    import inspect
    module = sys.modules["tldw_chatbook.DB.store_kernel"]
    if frame.f_code is not inspect.unwrap(module.SQLiteStoreKernel.transaction).__code__:
        return
    monitor = sys.monitoring
    tool = next(number for number in range(6) if monitor.get_tool(number) is None)
    monitor.use_tool_id(tool, "key-native-BEGIN-witness")
    def line_event(code, lineno):
        if lineno == {line}:
            with open({str(marker)!r}, "x", encoding="ascii") as stream:
                stream.write("original-BEGIN-line-dispatched")
            monitor.set_local_events(tool, code, 0)
    monitor.register_callback(tool, monitor.events.LINE, line_event)
    monitor.set_local_events(tool, frame.f_code, monitor.events.LINE)
    sys.setprofile(None)
sys.setprofile(_key_begin_profile)
"""
    original = subprocess.Popen

    def launch(arguments, *args, **kwargs):
        if (
            isinstance(arguments, list)
            and "tldw_chatbook.Workspaces.project_index.key_worker" in arguments[-1]
        ):
            arguments = list(arguments)
            arguments[-1] = arguments[-1].replace(
                ";runpy.run_module(", ";exec(" + repr(script) + ");runpy.run_module("
            )
        return original(arguments, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)


@pytest.mark.parametrize("changed", ["service", "key-model"])
@private_profile_test
async def test_preimport_definition_change_does_not_become_stock(
    tmp_path, request, monkeypatch, changed
):
    from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexKey

    assert "tldw_chatbook.Workspaces.project_index.runtime" not in sys.modules
    function = (
        ProjectIndexService.__init__
        if changed == "service"
        else ProjectIndexKey.__post_init__
    )
    monkeypatch.setattr(function, "__code__", function.__code__.replace())
    from Tests.Workspaces.test_project_index_key_process import managed_index, prepare
    from tldw_chatbook.Workspaces.project_index import runtime as index

    async with managed_index(tmp_path) as rig:
        index._KEY_PROCESS_ENABLED = True
        try:
            service, _ = await prepare(rig)
            assert service is rig.app._project_index_service
            assert not hasattr(rig.app, "_project_index_key_owner")
        finally:
            index._KEY_PROCESS_ENABLED = False


@pytest.mark.parametrize(
    "selector", ["get_workspaces_db_path", "_default_base_data_dir"]
)
@private_profile_test
async def test_preimport_same_path_config_selector_never_selects_child(
    tmp_path, request, monkeypatch, selector
):
    from tldw_chatbook import config

    assert "tldw_chatbook.Chat.console_project_index_preparation" not in sys.modules
    original = getattr(config, selector)
    before = original()
    monkeypatch.setattr(original, "__code__", original.__code__.replace())
    assert original() == before
    from Tests.Workspaces.test_project_index_key_process import managed_index, prepare
    from tldw_chatbook.Workspaces.project_index import runtime as index

    async with managed_index(tmp_path) as rig:
        index._KEY_PROCESS_ENABLED = True
        try:
            service, _scopes = await prepare(rig)
            assert service is rig.app._project_index_service
            assert not hasattr(rig.app, "_project_index_key_owner")
        finally:
            index._KEY_PROCESS_ENABLED = False


@private_profile_test
async def test_missing_child_registry_never_creates_database(tmp_path, request):
    from tldw_chatbook import config
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.project_index import runtime as index, key_process
    from tldw_chatbook.MCP.permission_process import ProtocolError

    path = config.get_workspaces_db_path()
    assert not path.exists()
    database = WorkspaceDB(path, initialize_schema=False, must_exist=True)
    factory = index._AdmittedStoreFactory()
    owner = key_process.KeyProcess(
        SimpleNamespace(), key_process._facts(database, factory)
    )
    key_process._owners.add(owner)
    owner.deadline = time.monotonic() + 120
    try:
        with pytest.raises(ProtocolError) as caught:
            owner._start()
        owner._unresolve(("missing-registry-initialization", caught.value))
        assert not path.exists()
        assert owner.retired and owner.process.poll() is not None
        assert owner in key_process._owners and owner.unresolved is not None
        assert owner.selected == 0
        with pytest.raises(ProtocolError, match="key_outcome_unresolved"):
            owner.close()
    finally:
        if owner.process is not None and not owner.retired:
            owner._unresolve(("initialization-test-cleanup",))
        database.close()


@private_profile_test
async def test_child_rejects_foreign_data_identity_without_replay(tmp_path, request):
    from Tests.Workspaces.test_project_index_key_custody import native_service
    from tldw_chatbook.Workspaces.project_index import key_process
    from tldw_chatbook.MCP.permission_process import ProtocolError

    database, service, _arguments = native_service(tmp_path)
    facts = key_process._facts(database, service._factory)
    facts["expected_data_authority_id"] = "0" * 64
    owner = key_process.KeyProcess(SimpleNamespace(), facts)
    key_process._owners.add(owner)
    owner.deadline = time.monotonic() + 120
    try:
        with pytest.raises(ProtocolError) as caught:
            owner._start()
        owner._unresolve(("foreign-data-initialization", caught.value))
        assert owner.retired and owner.process.poll() is not None
        assert owner in key_process._owners and owner.unresolved is not None
        assert owner.selected == 0 and not owner.receipts
        assert database.get_schema_version() == database._CURRENT_SCHEMA_VERSION
        assert not (service._factory.data_root / "project_index").exists()
        with pytest.raises(ProtocolError, match="key_outcome_unresolved"):
            owner.close()
    finally:
        if owner.process is not None and not owner.retired:
            owner._unresolve(("foreign-data-test-cleanup",))
        database.close()


@private_profile_test
async def test_wrong_root_refuses_after_child_native_read_without_local_replay(
    tmp_path, request
):
    from Tests.Workspaces.test_project_index_key_process import managed_index, prepare
    from tldw_chatbook.Workspaces.project_index import runtime as index

    async with managed_index(tmp_path) as rig:
        other = tmp_path / "other"
        other.mkdir()
        rig.roots = (replace(rig.roots[0], root=other),)
        index._KEY_PROCESS_ENABLED = True
        try:
            service, scopes = await prepare(rig)
            owner = rig.app._project_index_key_owner
            assert service is None and not scopes
            assert owner.selected == 1 and owner.unresolved is None
            assert owner.receipts[-1]["connections"] == 1
            assert not rig.cache.exists()
        finally:
            index._KEY_PROCESS_ENABLED = False


@pytest.mark.parametrize(
    "change,route",
    [
        ("detach", "dispatch"),
        ("session", "dispatch"),
        ("cancel", "dispatch"),
        ("cancel", "next-send"),
        ("cancel", "personal"),
    ],
)
@private_profile_test
async def test_registry_fence_dispatch_observes_fresh_binding_and_parent_session(
    tmp_path, request, monkeypatch, change, route
):
    from Tests.Workspaces.test_project_index_key_process import (
        managed_index,
        wait_event,
        trace_functions,
    )
    from tldw_chatbook.Workspaces.project_index import runtime as index, key_process
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat import console_chat_controller as host

    async with managed_index(tmp_path) as rig:
        marker = tmp_path / "child-original-BEGIN"
        observe_child_begin(monkeypatch, marker)
        poised, continued, locked, release = (threading.Event() for _ in range(4))
        root = rig.roots[0]
        guards = []
        consumed = []
        diagnostic = tmp_path / "key-parent-events.jsonl"
        started = time.monotonic()

        async def mcp(*args, **kwargs):
            return None

        rig.controller._compose_mcp_provider = mcp
        rig.controller._compose_local_provider = lambda **kwargs: (
            consumed.append(kwargs),
            None,
        )
        monkeypatch.setattr(
            host, "capture_run_admitted_workspace_roots", lambda **kwargs: rig.roots
        )

        def observe(function, event, result):
            if function is not root.guard:
                frame = sys._getframe(2)
                owner = getattr(rig.app, "_project_index_key_owner", None)
                message = frame.f_locals.get("value") if event == "call" else result
                record = {
                    "elapsed": time.monotonic() - started,
                    "function": function.__qualname__,
                    "event": event,
                    "frame": message.get("type") if type(message) is dict else None,
                    "state": getattr(owner, "state", None),
                    "selected": getattr(owner, "selected", None),
                }
                if function is key_process.KeyProcess._unresolve and event == "call":
                    failure = frame.f_locals["record"][-1]
                    record.update(error_type=type(failure).__name__, error=str(failure))
                with diagnostic.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            if function is root.guard and event == "call":
                guards.append(True)
            if function is key_process.write_message and event == "call":
                frame = sys._getframe(2)
                if frame.f_locals["value"]["type"] == "Continue":
                    poised.set()
                    assert continued.wait(60)

        def writer():
            database = WorkspaceDB(
                rig.app.workspace_registry_service.db.db_path,
                initialize_schema=False,
                must_exist=True,
            )
            try:
                with database.transaction(immediate=True) as connection:
                    locked.set()
                    assert release.wait(60)
                    if change == "detach":
                        connection.execute(
                            "DELETE FROM workspace_runtime_bindings WHERE binding_id=?",
                            (root.binding_id,),
                        )
            finally:
                database.close()

        index._KEY_PROCESS_ENABLED = True
        writing = None
        with trace_functions(
            (
                key_process.write_message,
                key_process.read_message,
                key_process.KeyProcess._start,
                key_process.KeyProcess._unresolve,
                root.guard,
            ),
            observe,
        ):
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
                await wait_event(poised)
                writing = asyncio.create_task(asyncio.to_thread(writer))
                await wait_event(locked)
                continued.set()
                async with asyncio.timeout(60):
                    while not marker.exists():
                        await asyncio.sleep(0.01)
                assert not pending.done()
                assert guards == [True]
                if change == "session":
                    rig.session.incarnation_id = "replaced-during-sqlite-wait"
                elif change == "cancel":
                    pending.cancel()
                    await asyncio.sleep(0)
                    pending.cancel()
                    await asyncio.sleep(0)
                    assert not pending.done()
                    assert len(rig.controller._preparation_reads) == 1
                    assert (
                        rig.controller._preparation_reads
                        == rig.runtime._preparation_reads
                    )
                release.set()
                await writing
                if change == "session":
                    with pytest.raises(
                        RecoveryRequired, match="console_snapshot_owner_changed"
                    ):
                        await pending
                    assert not consumed
                elif change == "cancel":
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert not consumed
                else:
                    await pending
                    assert len(consumed) == 1
                    assert "project_index_scopes" not in consumed[0]
                owner = rig.app._project_index_key_owner
                assert owner.selected == 1 and owner.unresolved is None
                assert owner.receipts[-1]["connections"] == 1
                assert guards == [True]
            finally:
                continued.set()
                release.set()
                await asyncio.gather(
                    pending, *([writing] if writing else []), return_exceptions=True
                )
                index._KEY_PROCESS_ENABLED = False
