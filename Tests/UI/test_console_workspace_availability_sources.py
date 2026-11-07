"""Custom Workspace display callbacks keep their existing caller contract."""

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

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

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("custom", ["outer", "folder"])
async def test_custom_availability_reader_is_not_bypassed(tmp_path, monkeypatch, custom):
    database = WorkspaceDB(tmp_path / "custom.sqlite", client_id="custom-availability")
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(workspace_id="custom", name="Custom", assistant_defaults=None)
    screen = _AsyncWorkerScreen()
    controller = _workspace_controller(
        screen=screen, app_instance=SimpleNamespace(workspace_registry_service=registry),
    )
    calls = []
    if custom == "outer":
        def capture(ids):
            calls.append(tuple(ids))
            return {"custom": True}, {"custom": ()}
        monkeypatch.setattr(controller, "_capture_workspace_files_availability", capture)
    else:
        def folders(workspace_id):
            calls.append(workspace_id)
            return (WorkspaceRuntimeBinding(
                workspace_id=workspace_id, binding_id="custom-binding",
                binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM,
                label="Custom status", locator=str(tmp_path),
                status=RuntimeBindingStatus.READY,
            ),)
        monkeypatch.setattr(registry, "list_folder_bindings", folders)
    try:
        controller._request_workspace_files_availability_refresh(("custom",))
        await screen.workers[0][0]
        assert calls, "stock optimization bypassed an existing custom callback"
        assert controller._workspace_files_availability_by_id == {"custom": True}
    finally:
        await asyncio.gather(*(row[0] for row in screen.workers), return_exceptions=True)
        database.close()


@pytest.mark.parametrize("route", ["view", "availability"])
async def test_stock_default_availability_opens_no_sqlite_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    """Default's immutable empty display must not open a private SQLite child."""
    from tldw_chatbook.DB.private_sqlite_process import HelperLease
    from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID

    database = WorkspaceDB(tmp_path / "default.sqlite", client_id="default-display")
    registry = LocalWorkspaceRegistryService(database)
    registry.ensure_default_workspace()
    from Tests.Workspaces.test_workspace_registry_service import (
        _insert_stale_default_binding,
    )

    _insert_stale_default_binding(database, "legacy-default")
    database.close()
    screen = _AsyncWorkerScreen()
    controller = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )
    helper_spawns = 0
    real_start = HelperLease.start

    def counted_start(cls, *args, **kwargs):
        nonlocal helper_spawns
        helper_spawns += 1
        return real_start(*args, **kwargs)

    monkeypatch.setattr(HelperLease, "start", classmethod(counted_start))
    try:
        if route == "view":
            from tldw_chatbook.UI.Console_Modules.workspace import (
                _ConsoleRegistryDisplayReads,
            )

            assert (
                _ConsoleRegistryDisplayReads(registry).list_runtime_bindings(
                    DEFAULT_WORKSPACE_ID
                )
                == ()
            )
        else:
            controller._request_workspace_files_availability_refresh(
                (DEFAULT_WORKSPACE_ID,)
            )
            await screen.workers[0][0]
            assert controller._workspace_files_availability_by_id == {
                DEFAULT_WORKSPACE_ID: False
            }
            assert controller._workspace_files_runtime_bindings_by_id == {
                DEFAULT_WORKSPACE_ID: ()
            }
        assert helper_spawns == 0, "Default-only display opened a private SQLite helper"
        assert registry.list_runtime_bindings(DEFAULT_WORKSPACE_ID) == ()
        assert helper_spawns == 1, "public Default cleanup did not reach real SQLite"
        with database.connection() as connection:
            assert (
                connection.execute(
                    "SELECT binding_id FROM workspace_runtime_bindings WHERE binding_id = ?",
                    ("legacy-default",),
                ).fetchone()
                is None
            )
    finally:
        await asyncio.gather(
            *(row[0] for row in screen.workers), return_exceptions=True
        )
        database.close()


@pytest.mark.parametrize("route", ["view", "availability"])
@pytest.mark.parametrize(
    "drift",
    [
        "ui-check",
        "producer-check",
        "producer-policy",
        "producer-source",
        "replacement-checker",
        "selector-binding",
        "selector-code",
        "checker-code",
        "normalizer",
        "default-identity",
        "reader-code",
        "reader-defaults",
        "reader-closure",
        "cleanup-shadow",
        "lookup",
    ],
)
async def test_changed_default_policy_reaches_original_native_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    drift: str,
) -> None:
    """Source/customization refusals must positively reach the prior live route."""
    from tldw_chatbook.DB.private_sqlite_process import HelperLease
    from tldw_chatbook.UI.Console_Modules import workspace as display
    from tldw_chatbook.Workspaces import (
        DEFAULT_WORKSPACE_ID,
    )
    from tldw_chatbook.Workspaces import (
        registry_service as source,
    )

    database = WorkspaceDB(tmp_path / "changed.sqlite", client_id="changed-default")
    registry = LocalWorkspaceRegistryService(database)
    registry.ensure_default_workspace()
    database.close()
    screen = _AsyncWorkerScreen()
    controller = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )

    def unselected_checker(*args, **kwargs):
        raise AssertionError("replacement policy checker was invoked")

    reader = inspect.unwrap(LocalWorkspaceRegistryService.list_runtime_bindings)
    if drift == "ui-check":
        monkeypatch.setattr(display, "_DEFAULT_BINDING_DISPLAY_CHECK", ())
    elif drift == "producer-check":
        monkeypatch.setattr(source, "_DEFAULT_BINDING_DISPLAY_CHECK", ())
    elif drift == "producer-policy":
        monkeypatch.setattr(source, "_DEFAULT_BINDING_DISPLAY_POLICY", ())
    elif drift == "producer-source":
        monkeypatch.setattr(source, "_DEFAULT_BINDING_DISPLAY_SOURCE", ())
    elif drift == "replacement-checker":
        monkeypatch.setattr(
            source,
            "_DEFAULT_BINDING_DISPLAY_POLICY",
            (
                unselected_checker,
                unselected_checker.__code__,
                source._DEFAULT_BINDING_DISPLAY_SOURCE,
            ),
        )
    elif drift == "selector-binding":
        monkeypatch.setattr(
            source, "_default_binding_display_current", unselected_checker
        )
    elif drift == "selector-code":
        monkeypatch.setattr(
            source._default_binding_display_current,
            "__code__",
            unselected_checker.__code__,
        )
    elif drift == "checker-code":
        monkeypatch.setattr(
            source._stock_default_bindings_are_empty,
            "__code__",
            unselected_checker.__code__,
        )
    elif drift == "normalizer":
        original = source._normalize_required_text
        monkeypatch.setattr(
            source, "_normalize_required_text", lambda *args: original(*args)
        )
    elif drift == "default-identity":
        replacement = "".join((DEFAULT_WORKSPACE_ID[:-1], DEFAULT_WORKSPACE_ID[-1]))
        assert (
            replacement == DEFAULT_WORKSPACE_ID
            and replacement is not DEFAULT_WORKSPACE_ID
        )
        monkeypatch.setattr(source, "DEFAULT_WORKSPACE_ID", replacement)
    elif drift == "reader-code":
        monkeypatch.setattr(
            reader, "__code__", reader.__code__.replace(co_name="custom_default_reader")
        )
    elif drift == "reader-defaults":
        monkeypatch.setattr(reader, "__defaults__", (DEFAULT_WORKSPACE_ID,))
    elif drift == "reader-closure":

        def with_closure(receiver, workspace_id):
            return reader(receiver, workspace_id)

        assert with_closure.__closure__ is not None
        monkeypatch.setattr(
            LocalWorkspaceRegistryService, "list_runtime_bindings", with_closure
        )
    elif drift == "cleanup-shadow":
        original_cleanup = registry._delete_default_runtime_bindings
        monkeypatch.setattr(
            registry, "_delete_default_runtime_bindings", lambda: original_cleanup()
        )
    elif drift == "lookup":
        monkeypatch.setattr(
            LocalWorkspaceRegistryService,
            "__getattribute__",
            lambda receiver, name: object.__getattribute__(receiver, name),
        )
    else:
        raise AssertionError(drift)
    helper_spawns = 0
    real_start = HelperLease.start

    def counted_start(cls, *args, **kwargs):
        nonlocal helper_spawns
        helper_spawns += 1
        return real_start(*args, **kwargs)

    monkeypatch.setattr(HelperLease, "start", classmethod(counted_start))
    try:
        if route == "view":
            assert (
                display._ConsoleRegistryDisplayReads(registry).list_runtime_bindings(
                    DEFAULT_WORKSPACE_ID
                )
                == ()
            )
        else:
            controller._request_workspace_files_availability_refresh(
                (DEFAULT_WORKSPACE_ID,)
            )
            await screen.workers[0][0]
            assert controller._workspace_files_availability_by_id == {
                DEFAULT_WORKSPACE_ID: False
            }
            assert controller._workspace_files_runtime_bindings_by_id == {
                DEFAULT_WORKSPACE_ID: ()
            }
        assert helper_spawns == 1, (
            "declined policy bypassed the original native Default read"
        )
    finally:
        await asyncio.gather(
            *(row[0] for row in screen.workers), return_exceptions=True
        )
        database.close()


@pytest.mark.parametrize("route", ["view", "availability"])
async def test_memory_default_display_retains_original_lister(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    """Observe the unchanged memory callback without replacing a policy source."""
    import sys

    from tldw_chatbook.UI.Console_Modules import workspace as display
    from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID

    database = WorkspaceDB(":memory:", client_id="memory-default")
    registry = LocalWorkspaceRegistryService(database)
    registry.ensure_default_workspace()
    screen = _AsyncWorkerScreen()
    controller = _workspace_controller(
        screen=screen,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )
    monitor = sys.monitoring
    tool = next(value for value in range(1, 6) if monitor.get_tool(value) is None)
    code = inspect.unwrap(LocalWorkspaceRegistryService.list_runtime_bindings).__code__
    invocations = []
    monitor.use_tool_id(tool, "memory-default-display-callback")
    monitor.register_callback(
        tool, monitor.events.PY_START, lambda code, offset: invocations.append(code)
    )
    monitor.set_local_events(tool, code, monitor.events.PY_START)
    try:
        if route == "view":
            assert (
                display._ConsoleRegistryDisplayReads(registry).list_runtime_bindings(
                    DEFAULT_WORKSPACE_ID
                )
                == ()
            )
        else:
            controller._request_workspace_files_availability_refresh(
                (DEFAULT_WORKSPACE_ID,)
            )
            await screen.workers[0][0]
            assert controller._workspace_files_availability_by_id == {
                DEFAULT_WORKSPACE_ID: False
            }
        assert invocations, (
            "memory display bypassed its original runtime-binding callback"
        )
    finally:
        await asyncio.gather(
            *(row[0] for row in screen.workers), return_exceptions=True
        )
        monitor.set_local_events(tool, code, 0)
        monitor.register_callback(tool, monitor.events.PY_START, None)
        monitor.free_tool_id(tool)
        database.close()


_PREIMPORT_POLICY_SCRIPT = r"""
import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from Tests import network_guard, real_profile_guard
network_guard.install()
real_profile_guard.install()
from Tests.windows_private_fixture_runner import user_fixture_default_owner
route, drift = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
root = Path(os.environ['XDG_DATA_HOME'])
selector.write_text('[paths]\ndata_dir="' + root.as_posix() + '"\n', encoding='utf-8')
selector.chmod(0o600)

def main():
    from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID, registry_service as source
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.DB.private_sqlite_process import HelperLease
    assert 'tldw_chatbook.UI.Console_Modules.workspace' not in sys.modules
    def unselected(*args, **kwargs):
        raise AssertionError('unselected replacement checker was invoked')
    if drift == 'policy':
        source._DEFAULT_BINDING_DISPLAY_POLICY = (unselected, unselected.__code__, ())
    elif drift == 'check':
        source._DEFAULT_BINDING_DISPLAY_CHECK = ()
    elif drift == 'source':
        source._DEFAULT_BINDING_DISPLAY_SOURCE = ()
    else:
        raise AssertionError(drift)
    from tldw_chatbook.UI.Console_Modules.workspace import _ConsoleRegistryDisplayReads
    from Tests.UI.test_console_workspace_controller import _AsyncWorkerScreen, _workspace_controller
    database = WorkspaceDB(root / 'preimport.sqlite', client_id='preimport-default')
    registry = source.LocalWorkspaceRegistryService(database)
    registry.ensure_default_workspace()
    database.close()
    starts = []
    real_start = HelperLease.start
    def counted_start(cls, *args, **kwargs):
        starts.append(True)
        return real_start(*args, **kwargs)
    HelperLease.start = classmethod(counted_start)
    async def exercise():
        screen = _AsyncWorkerScreen()
        controller = _workspace_controller(screen=screen,
            app_instance=SimpleNamespace(workspace_registry_service=registry))
        try:
            if route == 'view':
                assert _ConsoleRegistryDisplayReads(registry).list_runtime_bindings(DEFAULT_WORKSPACE_ID) == ()
            else:
                controller._request_workspace_files_availability_refresh((DEFAULT_WORKSPACE_ID,))
                await screen.workers[0][0]
                assert controller._workspace_files_availability_by_id == {DEFAULT_WORKSPACE_ID: False}
            assert len(starts) == 1, 'pre-import mutation bypassed original native read'
        finally:
            await asyncio.gather(*(row[0] for row in screen.workers), return_exceptions=True)
            database.close()
    try:
        asyncio.run(exercise())
    finally:
        HelperLease.start = classmethod(real_start.__func__)
        database.close()
    print('retired and reopened')
with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize("route", ["view", "availability"])
@pytest.mark.parametrize("drift", ["policy", "check", "source"])
async def test_preimport_default_policy_changes_keep_original_native_read(
    tmp_path: Path,
    route: str,
    drift: str,
) -> None:
    """A fresh UI import cannot bless preinstalled malformed or custom records."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, route, drift, script=_PREIMPORT_POLICY_SCRIPT)
