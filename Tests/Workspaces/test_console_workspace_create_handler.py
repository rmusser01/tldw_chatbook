from types import SimpleNamespace

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateResult
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
)


class _Stub:
    def __init__(self, registry):
        self.notifications = []
        self.calls = []
        self.app_instance = SimpleNamespace(
            workspace_registry_service=registry,
            notify=lambda message, **kw: self.notifications.append(message),
        )

    def _sync_console_chat_core_state(self):
        self.calls.append("core")

    def _activate_console_session_for_workspace(self, workspace_id):
        self.calls.append(f"activate:{workspace_id}")

    def _sync_console_workspace_context(self):
        self.calls.append("context")

    def _sync_native_console_chat_ui(self):
        return "ui-sync-sentinel"

    def run_worker(self, work, **kw):
        self.calls.append(f"worker:{work}")


def _registry(tmp_path):
    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="console-handler-tests")
    service = LocalWorkspaceRegistryService(db)
    service.create_workspace(workspace_id="workspace-local-1", name="Workspace 1")
    return service


def test_make_active_runs_full_console_sequence(tmp_path):
    stub = _Stub(_registry(tmp_path))
    result = WorkspaceCreateResult(
        workspace_id="workspace-local-1", name="Workspace 1", make_active=True
    )
    ConsoleWorkspaceController._handle_workspace_create_result(stub, result)
    assert stub.calls == [
        "core",
        "activate:workspace-local-1",
        "context",
        "worker:ui-sync-sentinel",
    ]
    assert any("switched Console" in n for n in stub.notifications)


def test_not_active_only_resyncs_context(tmp_path):
    stub = _Stub(_registry(tmp_path))
    result = WorkspaceCreateResult(
        workspace_id="workspace-local-1", name="Workspace 1", make_active=False
    )
    ConsoleWorkspaceController._handle_workspace_create_result(stub, result)
    assert stub.calls == ["context"]
    assert any("Created Workspace 1." in n for n in stub.notifications)


def test_none_result_is_a_noop(tmp_path):
    stub = _Stub(_registry(tmp_path))
    ConsoleWorkspaceController._handle_workspace_create_result(stub, None)
    assert stub.calls == [] and stub.notifications == []


def test_failed_folders_surface_as_warnings(tmp_path):
    stub = _Stub(_registry(tmp_path))
    result = WorkspaceCreateResult(
        workspace_id="workspace-local-1",
        name="Workspace 1",
        failed_folders=(("/gone", "Folder does not exist"),),
        make_active=False,
    )
    ConsoleWorkspaceController._handle_workspace_create_result(stub, result)
    assert any("Folder does not exist" in n for n in stub.notifications)


class _RaisingActivateRegistry:
    """Wraps a real registry; ``set_active_workspace`` always raises
    (TASK-17962: the Console activation-failure seam)."""

    def __init__(self, inner):
        self._inner = inner

    def set_active_workspace(self, workspace_id):
        raise WorkspaceRegistryServiceError("could not activate")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_activation_failure_notifies_error_and_skips_sync(tmp_path):
    """TASK-17962: ``set_active_workspace`` raising after a successful
    create must not crash the handler -- it notifies an error and must not
    run the post-activation sync sequence (core state / session activation
    / context sync / UI worker), mirroring the Library seam test
    (``test_create_workspace_recomposes_after_activation_failure``).
    """
    stub = _Stub(_RaisingActivateRegistry(_registry(tmp_path)))
    result = WorkspaceCreateResult(
        workspace_id="workspace-local-1", name="Workspace 1", make_active=True
    )
    ConsoleWorkspaceController._handle_workspace_create_result(stub, result)
    assert stub.calls == []
    assert any("could not be activated" in n for n in stub.notifications)


def test_result_with_project_skills_offers_import(tmp_path, monkeypatch):
    offered = []
    import tldw_chatbook.UI.Console_Modules.workspace as ws_module

    monkeypatch.setattr(
        ws_module,
        "maybe_offer_project_skills_import",
        lambda app, discoveries: offered.append(discoveries),
    )
    stub = _Stub(_registry(tmp_path))
    result = WorkspaceCreateResult(
        workspace_id="workspace-local-1",
        name="Workspace 1",
        make_active=False,
        project_skills=("sentinel-discovery",),
    )
    ConsoleWorkspaceController._handle_workspace_create_result(stub, result)
    assert offered == [("sentinel-discovery",)]
