"""Custom Workspace display callbacks keep their existing caller contract."""

import asyncio
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
