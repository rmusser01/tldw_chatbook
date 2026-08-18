from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Checkbox, Input, Static

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Widgets.workspace_create_modal import (
    WorkspaceCreateModal,
    WorkspaceCreateResult,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


def _registry(tmp_path):
    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="create-modal-tests")
    return LocalWorkspaceRegistryService(db)


class _HarnessApp(App[None]):
    def __init__(self, registry):
        super().__init__()
        # NOTE: `App.__init__` already defines `self._registry` (a WeakSet of
        # mounted DOM nodes, consulted at shutdown in `_close_all`). Naming
        # this `_registry` clobbers that internal attribute and breaks
        # teardown with `TypeError: '...' object is not iterable`, so the
        # harness's own workspace registry gets a non-colliding name.
        self._workspace_registry = registry
        self.result = "unset"

    def on_mount(self) -> None:
        def _done(result):
            self.result = result

        self.push_screen(
            WorkspaceCreateModal(registry_service=self._workspace_registry), _done
        )


@pytest.mark.asyncio
async def test_browse_pushes_picker_and_fills_path(tmp_path):
    app = _HarnessApp(_registry(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, WorkspaceCreateModal)
        await pilot.click("#workspace-create-browse")
        await pilot.pause()
        assert isinstance(app.screen, SelectDirectory)
        project = tmp_path / "project"
        project.mkdir()
        app.screen.dismiss(project)
        await pilot.pause()
        assert app.screen is modal  # focus/stack returned to the modal
        assert (
            modal.query_one("#workspace-create-folder-path", Input).value
            == str(project)
        )


@pytest.mark.asyncio
async def test_escape_dismisses_with_none(tmp_path):
    app = _HarnessApp(_registry(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.result is None


@pytest.mark.asyncio
async def test_invalid_folder_shows_inline_error(tmp_path):
    app = _HarnessApp(_registry(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-folder-path", Input).value = str(
            tmp_path / "missing"
        )
        await pilot.click("#workspace-create-folder-add")
        await pilot.pause()
        error = modal.query_one("#workspace-create-error", Static)
        assert "does not exist" in str(error.renderable)
        assert app.result == "unset"  # still open


@pytest.mark.asyncio
async def test_create_with_folder_returns_result_and_binds(tmp_path):
    registry = _registry(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    app = _HarnessApp(registry)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-name", Input).value = "Video Tool"
        modal.query_one("#workspace-create-folder-path", Input).value = str(project)
        await pilot.click("#workspace-create-folder-add")
        await pilot.pause()
        await pilot.click("#workspace-create-confirm")
        await pilot.pause()
    result = app.result
    assert isinstance(result, WorkspaceCreateResult)
    assert result.name == "Video Tool"
    assert result.bound_folders == (str(project.resolve()),)
    assert result.failed_folders == ()
    assert result.make_active is True
    stored = registry.get_workspace(result.workspace_id)
    assert stored is not None and stored.name == "Video Tool"
    locators = [b.locator for b in registry.list_folder_bindings(result.workspace_id)]
    assert locators == [str(project.resolve())]


@pytest.mark.asyncio
async def test_duplicate_name_error_keeps_modal_open(tmp_path):
    registry = _registry(tmp_path)
    registry.create_workspace(workspace_id="workspace-local-9", name="Video Tool")
    app = _HarnessApp(registry)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-name", Input).value = "Video Tool"
        await pilot.click("#workspace-create-confirm")
        await pilot.pause()
        error = modal.query_one("#workspace-create-error", Static)
        assert "already exists" in str(error.renderable)
        assert app.result == "unset"


@pytest.mark.asyncio
async def test_make_active_checkbox_carried_on_result(tmp_path):
    app = _HarnessApp(_registry(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-make-active", Checkbox).value = False
        await pilot.click("#workspace-create-confirm")
        await pilot.pause()
    assert app.result.make_active is False
