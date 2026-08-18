from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, Input, Static

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Widgets.workspace_create_modal import (
    WorkspaceCreateModal,
    WorkspaceCreateResult,
)
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
)


def _registry(tmp_path):
    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="create-modal-tests")
    return LocalWorkspaceRegistryService(db)


class _RaisingListWorkspacesRegistry:
    """Wraps a real registry but fails ``list_workspaces()`` (Qodo finding
    6a): simulates a registry read failure surfacing during modal
    construction, before the modal is even pushed/mounted."""

    def __init__(self, inner: LocalWorkspaceRegistryService) -> None:
        self._inner = inner

    def list_workspaces(self, *args, **kwargs):
        raise WorkspaceRegistryServiceError("registry read failed")

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FlakyBindRegistry:
    """Wraps a real registry; ``add_folder_binding()`` fails until armed to
    succeed (Qodo finding 7's retry-capable binding-failure behavior)."""

    def __init__(self, inner: LocalWorkspaceRegistryService) -> None:
        self._inner = inner
        self.should_fail = True

    def add_folder_binding(self, workspace_id, path, **kwargs):
        if self.should_fail:
            raise WorkspaceRegistryServiceError("Folder does not exist")
        return self._inner.add_folder_binding(workspace_id, path, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


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


@pytest.mark.asyncio
async def test_double_submit_creates_exactly_one_workspace_no_crash(tmp_path):
    """Finding 1: rapid Enter-Enter (or a double-click race on Create) must
    not create a duplicate workspace, and must not raise ScreenStackError
    from a second dismiss landing after the modal has already popped.

    Leaving the name Input blank exercises the real race: each call falls
    back to a freshly-computed auto-generated name ("Workspace 1", then
    "Workspace 2"), so the registry's duplicate-name rejection does not
    incidentally save the unguarded code from creating two workspaces.
    """
    registry = _registry(tmp_path)
    app = _HarnessApp(registry)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, WorkspaceCreateModal)
        modal.query_one("#workspace-create-name", Input).value = ""
        # Simulates the second queued Enter/click landing before the screen
        # has actually popped off the stack.
        modal._create()
        modal._create()
        await pilot.pause()
    assert isinstance(app.result, WorkspaceCreateResult)
    workspaces = registry.list_workspaces()
    assert len(workspaces) == 1


@pytest.mark.asyncio
async def test_add_folder_empty_input_clears_stale_error(tmp_path):
    """Finding 5: an invalid-folder error must not linger once the field is
    cleared and Add is pressed again on blank input.

    Uses Button.press() rather than pilot.click() for the second press:
    Textual's own Button._on_click() skips press() while the button still
    carries its ~0.2s "-active" press-effect class, so two pilot.click()s on
    the same button in quick succession silently drop the second one -- an
    unrelated Textual debounce, not the seam this test is pinning.
    """
    app = _HarnessApp(_registry(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-folder-path", Input).value = str(
            tmp_path / "missing"
        )
        modal.query_one("#workspace-create-folder-add", Button).press()
        await pilot.pause()
        error = modal.query_one("#workspace-create-error", Static)
        assert "does not exist" in str(error.renderable)

        modal.query_one("#workspace-create-folder-path", Input).value = ""
        modal.query_one("#workspace-create-folder-add", Button).press()
        await pilot.pause()
        error = modal.query_one("#workspace-create-error", Static)
        assert str(error.renderable) == ""


@pytest.mark.asyncio
async def test_remove_folder_clears_stale_error(tmp_path):
    """Finding 5: removing a bound folder must not leave a prior invalid-
    folder error rendered."""
    registry = _registry(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    app = _HarnessApp(registry)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-folder-path", Input).value = str(project)
        modal.query_one("#workspace-create-folder-add", Button).press()
        await pilot.pause()

        modal.query_one("#workspace-create-folder-path", Input).value = str(
            tmp_path / "missing"
        )
        modal.query_one("#workspace-create-folder-add", Button).press()
        await pilot.pause()
        error = modal.query_one("#workspace-create-error", Static)
        assert "does not exist" in str(error.renderable)

        modal.query_one("#workspace-create-folder-remove-0", Button).press()
        await pilot.pause()
        error = modal.query_one("#workspace-create-error", Static)
        assert str(error.renderable) == ""


@pytest.mark.asyncio
async def test_overlong_name_shows_inline_error_and_creates_nothing(tmp_path):
    """Finding 2: a name sanitize_string(raw, 100) would truncate/strip must
    be rejected inline at the boundary rather than silently mangled into the
    registry."""
    registry = _registry(tmp_path)
    app = _HarnessApp(registry)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-name", Input).value = "x" * 300
        await pilot.click("#workspace-create-confirm")
        await pilot.pause()
        assert app.screen is modal
        error = modal.query_one("#workspace-create-error", Static)
        assert "too long" in str(error.renderable)
        assert app.result == "unset"
    assert registry.list_workspaces() == ()


@pytest.mark.asyncio
async def test_registry_read_failure_still_mounts_usable_modal(tmp_path):
    """Finding 6a: next_local_workspace_identity() raising during __init__
    (list_workspaces() failing) must not crash the modal before it can even
    mount -- it falls back to an empty suggested name and shows an inline
    error so the user can still type a name and retry."""
    stub = _RaisingListWorkspacesRegistry(_registry(tmp_path))
    app = _HarnessApp(stub)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, WorkspaceCreateModal)
        name_input = modal.query_one("#workspace-create-name", Input)
        assert name_input.value == ""
        assert name_input.placeholder == "Workspace name"
        error = modal.query_one("#workspace-create-error", Static)
        assert "registry could not be read" in str(error.renderable)


@pytest.mark.asyncio
async def test_identity_failure_at_create_resets_committed_for_retry(
    tmp_path, monkeypatch
):
    """Finding 6b: next_local_workspace_identity() raising inside _create()
    used to run AFTER _committed was set True and OUTSIDE the try/except
    that resets it, permanently locking the Create button for the session.
    It must now reset _committed so a subsequent Create succeeds."""
    import tldw_chatbook.Widgets.workspace_create_modal as wcm_module

    registry = _registry(tmp_path)
    app = _HarnessApp(registry)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, WorkspaceCreateModal)

        def _raise(_registry_service):
            raise WorkspaceRegistryServiceError("boom")

        monkeypatch.setattr(wcm_module, "next_local_workspace_identity", _raise)
        await pilot.click("#workspace-create-confirm")
        await pilot.pause()
        assert app.screen is modal
        assert app.result == "unset"
        error = modal.query_one("#workspace-create-error", Static)
        assert "boom" in str(error.renderable)

        monkeypatch.undo()
        # NOTE: a second pilot.click() on the *same* button in quick
        # succession is silently dropped by Textual's own ~0.2s "-active"
        # press-effect debounce (see test_add_folder_empty_input_clears_
        # stale_error above) -- use Button.press() to bypass it.
        modal.query_one("#workspace-create-confirm", Button).press()
        await pilot.pause()
    assert isinstance(app.result, WorkspaceCreateResult)
    assert len(registry.list_workspaces()) == 1


@pytest.mark.asyncio
async def test_folder_binding_failure_keeps_modal_open_and_retries(tmp_path):
    """Finding 7: a per-folder add_folder_binding() failure must not
    dismiss the modal or lose the already-created workspace -- it renders
    the failure inline, and a subsequent Create press retries only the
    remaining folder(s) without creating a second workspace."""
    inner = _registry(tmp_path)
    stub = _FlakyBindRegistry(inner)
    project = tmp_path / "project"
    project.mkdir()
    app = _HarnessApp(stub)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-name", Input).value = "Video Tool"
        modal.query_one("#workspace-create-folder-path", Input).value = str(project)
        await pilot.click("#workspace-create-folder-add")
        await pilot.pause()
        await pilot.click("#workspace-create-confirm")
        await pilot.pause()

        assert app.screen is modal
        assert app.result == "unset"
        error = modal.query_one("#workspace-create-error", Static)
        assert "Folder does not exist" in str(error.renderable)
        assert len(inner.list_workspaces()) == 1

        stub.should_fail = False
        # NOTE: a second pilot.click() on the *same* button in quick
        # succession is silently dropped by Textual's own ~0.2s "-active"
        # press-effect debounce (see test_add_folder_empty_input_clears_
        # stale_error above) -- use Button.press() to bypass it.
        modal.query_one("#workspace-create-confirm", Button).press()
        await pilot.pause()

    result = app.result
    assert isinstance(result, WorkspaceCreateResult)
    created = inner.list_workspaces()
    assert len(created) == 1
    assert result.workspace_id == created[0].workspace_id
    assert result.bound_folders == (str(project.resolve()),)
    assert result.failed_folders == ()


@pytest.mark.asyncio
async def test_escape_after_partial_create_returns_partial_result(tmp_path):
    """Finding 7: cancel after a partial create is a fact, not something
    Cancel/Escape can undo -- it must deliver the partial result instead of
    None so callers still sync their workspace list."""
    inner = _registry(tmp_path)
    stub = _FlakyBindRegistry(inner)
    project = tmp_path / "project"
    project.mkdir()
    app = _HarnessApp(stub)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-name", Input).value = "Video Tool"
        modal.query_one("#workspace-create-folder-path", Input).value = str(project)
        await pilot.click("#workspace-create-folder-add")
        await pilot.pause()
        await pilot.click("#workspace-create-confirm")
        await pilot.pause()
        assert app.result == "unset"

        await pilot.press("escape")
        await pilot.pause()

    result = app.result
    assert isinstance(result, WorkspaceCreateResult)
    assert result.name == "Video Tool"
    created = inner.list_workspaces()
    assert len(created) == 1
    assert result.workspace_id == created[0].workspace_id


@pytest.mark.asyncio
async def test_folder_with_skills_annotated_and_carried_on_result(tmp_path):
    registry = _registry(tmp_path)
    project = tmp_path / "project"
    skill = project / ".SKILLS" / "alpha-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\ndescription: x\n---\nB\n", encoding="utf-8")
    app = _HarnessApp(registry)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-folder-path", Input).value = str(project)
        await pilot.click("#workspace-create-folder-add")
        await pilot.pause()
        rows = [str(s.renderable) for s in modal.query(".workspace-create-folder-locator")]
        assert any("1 project skill" in row for row in rows)
        await pilot.click("#workspace-create-confirm")
        await pilot.pause()
    assert len(app.result.project_skills) == 1
    assert app.result.project_skills[0].entries[0].name == "alpha-skill"


@pytest.mark.asyncio
async def test_kill_switch_suppresses_folder_discovery_scan(tmp_path, monkeypatch):
    """Finding 1: with the ``[skills] project_skills_prompt_enabled``
    kill-switch off, ``_add_folder`` must not scan the bound folder for
    project skills AT ALL -- "no scanning" must be literally true, not
    merely "no offer" later. Monkeypatches ``discover_project_skills`` with
    a recorder to prove the call never happens.
    """
    calls: list = []

    def _recorder(root):
        calls.append(root)
        return None

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *a, **k: False)
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.workspace_create_modal.discover_project_skills",
        _recorder,
    )

    registry = _registry(tmp_path)
    project = tmp_path / "project"
    skill = project / ".SKILLS" / "alpha-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\ndescription: x\n---\nB\n", encoding="utf-8")

    app = _HarnessApp(registry)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#workspace-create-folder-path", Input).value = str(project)
        await pilot.click("#workspace-create-folder-add")
        await pilot.pause()

    assert calls == []
