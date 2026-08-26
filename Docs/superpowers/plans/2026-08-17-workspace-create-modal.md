# Workspace Create Modal (PR A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace zero-input workspace creation on all three surfaces (Console rail, Settings ▸ Workspaces, Library) with one shared modal that collects a name + optional folder bindings and explains what a workspace does.

**Architecture:** A `ModalScreen[WorkspaceCreateResult | None]` owns the service calls (`create_workspace` + `add_folder_binding`) so errors render inline; each surface keeps its own dismissal callback for post-create UI sync. A pure path validator is extracted from `add_folder_binding` so folders are vetted as they are added, before Create.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite (`WorkspaceDB`), pytest + Textual pilot.

**Spec:** `Docs/superpowers/specs/2026-08-17-workspace-create-modal-and-project-skills-design.md` (§4, §6, §7, §9 PR A)

## Global Constraints

- Work in a worktree under `<repo>/.worktrees/` branched off `origin/dev` (NEVER `/tmp`; see memory rules). Push after every task.
- pytest is venv-only: run `.venv/bin/pytest …` (if pytest is missing: `VIRTUAL_ENV=.venv uv pip install -e ".[dev]"`). `timeout` command is unavailable in this environment.
- Modal styling via `DEFAULT_CSS` on the widget (house pattern: `ConsoleWorkspaceSwitcherModal`); never hand-edit the CSS bundle.
- All user-visible strings that echo user/filesystem input render with `markup=False`.
- Folder bindings are added read-only (`allow_write=False`) — ADR-028; rw stays a Settings action.
- Escape always dismisses with `None`; nothing is created on cancel.
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- File one backlog task for PR A before starting (sweep IDs against origin/dev + all remotes/worktrees per `backlog/docs/lessons-backlog-hygiene.md`); set In Progress with this plan as the Implementation Plan.

---

### Task 1: Extract pure folder-path validator

**Files:**
- Modify: `tldw_chatbook/Workspaces/registry_service.py:659-726` (`add_folder_binding`)
- Test: `Tests/Workspaces/test_folder_binding_validator.py` (new)

**Interfaces:**
- Consumes: existing `find_root_binding_conflict`, `WorkspaceRegistryServiceError` in the same module.
- Produces: module-level `validate_folder_binding_path(path: str | Path, existing_locators: Sequence[str] = ()) -> Path` — raises `WorkspaceRegistryServiceError` with the exact messages `add_folder_binding` raises today; returns the resolved path. Task 2/3 import it.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Workspaces/test_folder_binding_validator.py
from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
    validate_folder_binding_path,
)


def test_valid_directory_resolves(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    assert validate_folder_binding_path(project) == project.resolve()


def test_missing_directory_rejected(tmp_path):
    with pytest.raises(WorkspaceRegistryServiceError, match="does not exist"):
        validate_folder_binding_path(tmp_path / "nope")


def test_filesystem_root_rejected():
    with pytest.raises(WorkspaceRegistryServiceError, match="filesystem root"):
        validate_folder_binding_path(Path("/"))


def test_home_directory_rejected():
    with pytest.raises(WorkspaceRegistryServiceError, match="home directory"):
        validate_folder_binding_path(Path.home())


def test_duplicate_locator_rejected(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(WorkspaceRegistryServiceError, match="already bound"):
        validate_folder_binding_path(project, [str(project.resolve())])


def test_nested_inside_existing_rejected(tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    with pytest.raises(WorkspaceRegistryServiceError, match="inside the already-bound"):
        validate_folder_binding_path(child, [str(parent.resolve())])


def test_existing_inside_candidate_rejected(tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    with pytest.raises(WorkspaceRegistryServiceError, match="remove it first"):
        validate_folder_binding_path(parent, [str(child.resolve())])


def test_sensitive_conflict_rejected(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    import tldw_chatbook.Workspaces.registry_service as rs

    monkeypatch.setattr(
        rs, "find_root_binding_conflict", lambda p: Path("/protected")
    )
    with pytest.raises(WorkspaceRegistryServiceError, match="protected path"):
        validate_folder_binding_path(project)


def test_add_folder_binding_still_enforces(tmp_path):
    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="validator-tests")
    service = LocalWorkspaceRegistryService(db)
    service.create_workspace(workspace_id="workspace-local-1", name="Workspace 1")
    project = tmp_path / "project"
    project.mkdir()
    binding = service.add_folder_binding("workspace-local-1", project)
    assert binding.locator == str(project.resolve())
    with pytest.raises(WorkspaceRegistryServiceError, match="already bound"):
        service.add_folder_binding("workspace-local-1", project)
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest Tests/Workspaces/test_folder_binding_validator.py -v` — expect ImportError: `validate_folder_binding_path` not defined.

- [ ] **Step 3: Implement.** In `registry_service.py`, add `Sequence` to the module's existing typing/collections.abc imports. Directly above `class LocalWorkspaceRegistryService`, add the function by MOVING the checks from `add_folder_binding:675-726` verbatim (keep the TASK-857 comment with them):

```python
def validate_folder_binding_path(
    path: str | Path,
    existing_locators: Sequence[str] = (),
) -> Path:
    """Resolve and vet a candidate folder-binding path (spec 2026-08-17 §4.2).

    Pure with respect to the registry: consults only the filesystem and the
    sensitive-path denylist, so creation UIs can vet folders before any
    workspace exists. Raises WorkspaceRegistryServiceError with the same
    user-facing messages ``add_folder_binding`` raised before extraction.
    """
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise WorkspaceRegistryServiceError(
            f"Folder path could not be resolved: {candidate}"
        ) from exc
    if not resolved.is_dir():
        raise WorkspaceRegistryServiceError(
            f"Folder does not exist or is not a directory: {resolved}"
        )
    if resolved == Path(resolved.anchor):
        raise WorkspaceRegistryServiceError(
            "The filesystem root cannot be bound to a workspace."
        )
    if resolved == Path.home().resolve():
        raise WorkspaceRegistryServiceError(
            "Your home directory itself cannot be bound; choose a "
            "project folder inside it."
        )
    conflict = find_root_binding_conflict(resolved)
    if conflict is not None:
        raise WorkspaceRegistryServiceError(
            f"'{resolved}' cannot be bound: it is, or contains, the "
            f"protected path '{conflict}'. Choose a folder that does "
            f"not overlap this application's own data, configuration, "
            f"or credential directories."
        )
    for locator in existing_locators:
        existing_path = Path(locator)
        if resolved == existing_path:
            raise WorkspaceRegistryServiceError(
                f"{resolved} is already bound to this workspace."
            )
        if existing_path in resolved.parents:
            raise WorkspaceRegistryServiceError(
                f"{resolved} is inside the already-bound folder "
                f"{existing_path}."
            )
        if resolved in existing_path.parents:
            raise WorkspaceRegistryServiceError(
                f"The already-bound folder {existing_path} is inside "
                f"{resolved}; remove it first."
            )
    return resolved
```

Then `add_folder_binding` replaces lines 675-726 with:

```python
        resolved = validate_folder_binding_path(
            path,
            [b.locator for b in self.list_folder_bindings(workspace_id)],
        )
```

- [ ] **Step 4: Run new tests + existing regressions** — `.venv/bin/pytest Tests/Workspaces/test_folder_binding_validator.py Tests/Workspaces/test_workspace_folder_bindings.py Tests/Workspaces/test_workspace_registry_service.py -v` — all PASS (read the passed count; "no tests ran" is a FAILED gate).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "refactor(workspaces): extract pure folder-binding path validator"` (+ trailer).

---

### Task 2: Modal skeleton + Browse-from-modal spike

The spec (§4.4) requires proving fspicker-over-modal FIRST — no site has ever pushed `SelectDirectory` from inside a `ModalScreen`.

**Files:**
- Create: `tldw_chatbook/Widgets/workspace_create_modal.py`
- Test: `Tests/Workspaces/test_workspace_create_modal.py` (new)

**Interfaces:**
- Consumes: `SelectDirectory` from `tldw_chatbook.Third_Party.textual_fspicker`; `LocalWorkspaceRegistryService`, `next_local_workspace_identity` from `tldw_chatbook.Workspaces.registry_service`.
- Produces: `WorkspaceCreateResult` dataclass and `WorkspaceCreateModal(ModalScreen[WorkspaceCreateResult | None])` with `__init__(self, *, registry_service)`. Widget ids Task 3 extends: `#workspace-create-name`, `#workspace-create-folder-path`, `#workspace-create-browse`, `#workspace-create-cancel`.

- [ ] **Step 1: Write the failing spike test**

```python
# Tests/Workspaces/test_workspace_create_modal.py
from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Input

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
        self._registry = registry
        self.result = "unset"

    def on_mount(self) -> None:
        def _done(result):
            self.result = result

        self.push_screen(
            WorkspaceCreateModal(registry_service=self._registry), _done
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
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest Tests/Workspaces/test_workspace_create_modal.py -v` — expect ImportError (module missing).

- [ ] **Step 3: Implement the skeleton**

```python
# tldw_chatbook/Widgets/workspace_create_modal.py
"""Shared workspace creation modal (spec 2026-08-17 §4).

Used by the Console rail, Settings ▸ Workspaces, and Library. The modal
owns the create/bind service calls so failures render inline; surfaces
own post-create UI sync via their dismissal callbacks (§4.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Static

from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
    next_local_workspace_identity,
    validate_folder_binding_path,
)

_WORKSPACE_EXPLAINER = (
    "A workspace scopes the Console to one project. Conversations started "
    "in it are grouped together, agents' project file access comes only "
    "from the folders you bind here (read-only unless you grant write in "
    "Settings), and retrieval can be narrowed to the workspace's items via "
    "its RAG Scope. Binding your project's folder is what makes a "
    "workspace more than a label — without one, agents have no file "
    "access. You can add or change folders later in Settings ▸ Workspaces."
)


@dataclass(frozen=True)
class WorkspaceCreateResult:
    """Outcome of a completed create dialog (spec §4.3)."""

    workspace_id: str
    name: str
    bound_folders: tuple[str, ...] = ()
    failed_folders: tuple[tuple[str, str], ...] = ()  # (path, error message)
    make_active: bool = True
    #: ProjectSkillsDiscovery entries for bound folders containing .SKILLS/.
    #: Stays empty until project-skills discovery ships (spec §5.5 / PR B).
    project_skills: tuple = ()


class WorkspaceCreateModal(ModalScreen["WorkspaceCreateResult | None"]):
    """Collect a workspace name + optional folder bindings."""

    DEFAULT_CSS = """
    WorkspaceCreateModal {
        align: center middle;
    }

    #workspace-create-modal {
        width: 72;
        height: auto;
        max-height: 32;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #workspace-create-explainer {
        color: $text-muted;
        margin: 0 0 1 0;
    }

    #workspace-create-error {
        color: $error;
        height: auto;
    }

    #workspace-create-folder-row {
        height: 3;
        min-height: 3;
    }

    #workspace-create-folder-path {
        width: 1fr;
    }

    #workspace-create-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }
    """

    BINDINGS = [("escape", "dismiss", "Cancel")]
    AUTO_FOCUS = "#workspace-create-name"

    def __init__(self, *, registry_service: LocalWorkspaceRegistryService) -> None:
        super().__init__()
        self._registry = registry_service
        self._folders: list[str] = []
        self._error = ""

    def compose(self) -> ComposeResult:
        _, suggested_name = next_local_workspace_identity(self._registry)
        with Vertical(id="workspace-create-modal"):
            yield Static("New Workspace", classes="console-modal-header")
            yield Static(
                _WORKSPACE_EXPLAINER, id="workspace-create-explainer", markup=False
            )
            yield Input(
                value=suggested_name,
                id="workspace-create-name",
                placeholder="Workspace name",
            )
            with Horizontal(id="workspace-create-folder-row"):
                yield Input(
                    id="workspace-create-folder-path",
                    placeholder="~/path/to/project (optional)",
                )
                yield Button("Browse…", id="workspace-create-browse", compact=True)
            with Horizontal(id="workspace-create-actions"):
                yield Button("Cancel", id="workspace-create-cancel", compact=True)

    def action_dismiss(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#workspace-create-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#workspace-create-browse")
    def _browse(self, event: Button.Pressed) -> None:
        event.stop()

        def _picked(selected: Path | None) -> None:
            if selected is not None:
                self.query_one(
                    "#workspace-create-folder-path", Input
                ).value = str(selected)

        self.app.push_screen(
            SelectDirectory(title="Bind Project Folder"), _picked
        )
```

- [ ] **Step 4: Run** — `.venv/bin/pytest Tests/Workspaces/test_workspace_create_modal.py -v` — both PASS. **If the browse round-trip fails structurally** (stack/focus does not return), STOP: implement the fallback from spec §4.4 (modal dismisses to the picker and is re-pushed prefilled, following `tldw_chatbook/Widgets/document_generation_modal.py:301`'s chaining shape) and record the deviation in the task file.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(workspaces): create-modal skeleton with browse-from-modal spike"` (+ trailer).

---

### Task 3: Full modal — folder list, validation, create + bind, result

**Files:**
- Modify: `tldw_chatbook/Widgets/workspace_create_modal.py`
- Test: `Tests/Workspaces/test_workspace_create_modal.py`

**Interfaces:**
- Produces (used by Tasks 4-6): completed dialog dismisses with `WorkspaceCreateResult`; new ids `#workspace-create-folder-add`, `#workspace-create-folder-remove-{i}`, `#workspace-create-make-active`, `#workspace-create-confirm`, `#workspace-create-error`.

- [ ] **Step 1: Write the failing tests** (append)

```python
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
```

(add `from textual.widgets import Checkbox, Static` to the test imports)

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest Tests/Workspaces/test_workspace_create_modal.py -v` — new tests FAIL (missing ids).

- [ ] **Step 3: Implement.** Extend `compose()` between the folder row and actions:

```python
            with Vertical(id="workspace-create-folder-list"):
                for index, folder in enumerate(self._folders):
                    with Horizontal(classes="workspace-create-folder-item"):
                        yield Static(folder, classes="workspace-create-folder-locator", markup=False)
                        yield Button(
                            "Remove",
                            id=f"workspace-create-folder-remove-{index}",
                            compact=True,
                        )
            yield Static(self._error, id="workspace-create-error", markup=False)
            yield Checkbox(
                "Switch to this workspace", True, id="workspace-create-make-active"
            )
```

and add `Button("Add folder", id="workspace-create-folder-add", compact=True)` inside `#workspace-create-folder-row` after Browse, plus `Button("Create", id="workspace-create-confirm", compact=True)` in the actions row. Add handlers:

```python
    def _set_error(self, message: str) -> None:
        self._error = message
        self.query_one("#workspace-create-error", Static).update(message)

    @on(Button.Pressed, "#workspace-create-folder-add")
    def _add_folder(self, event: Button.Pressed) -> None:
        event.stop()
        raw = self.query_one("#workspace-create-folder-path", Input).value.strip()
        if not raw:
            return
        try:
            resolved = validate_folder_binding_path(raw, self._folders)
        except WorkspaceRegistryServiceError as exc:
            self._set_error(str(exc))
            return
        self._folders.append(str(resolved))
        self._error = ""
        self.query_one("#workspace-create-folder-path", Input).value = ""
        self.refresh(recompose=True)

    @on(Button.Pressed, "#workspace-create-folder-list Button")
    def _remove_folder(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        try:
            index = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        if 0 <= index < len(self._folders):
            del self._folders[index]
            self.refresh(recompose=True)

    @on(Input.Submitted, "#workspace-create-name")
    def _name_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._create()

    @on(Button.Pressed, "#workspace-create-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self._create()

    def _create(self) -> None:
        name = self.query_one("#workspace-create-name", Input).value.strip()
        workspace_id, generated_name = next_local_workspace_identity(self._registry)
        try:
            self._registry.create_workspace(
                workspace_id=workspace_id,
                name=name or generated_name,
                description="Created from the workspace setup dialog.",
            )
        except WorkspaceRegistryServiceError as exc:
            self._set_error(str(exc))
            return
        bound: list[str] = []
        failed: list[tuple[str, str]] = []
        for folder in self._folders:
            try:
                self._registry.add_folder_binding(workspace_id, folder)
                bound.append(folder)
            except WorkspaceRegistryServiceError as exc:
                failed.append((folder, str(exc)))
        self.dismiss(
            WorkspaceCreateResult(
                workspace_id=workspace_id,
                name=name or generated_name,
                bound_folders=tuple(bound),
                failed_folders=tuple(failed),
                make_active=self.query_one(
                    "#workspace-create-make-active", Checkbox
                ).value,
            )
        )
```

Note: `compose()` re-reads `self._error`/`self._folders` on recompose, so `refresh(recompose=True)` keeps rows and error in sync. The prefilled name must be captured once in `__init__` (`self._suggested_name`) and reused by `compose()` — otherwise recompose after add-folder would clobber a user-edited name. Store the name Input's current value into an attribute in `_add_folder`/`_remove_folder` before `refresh(recompose=True)` and restore it in `compose()`.

- [ ] **Step 4: Run** — `.venv/bin/pytest Tests/Workspaces/test_workspace_create_modal.py -v` — all PASS (including the Task 2 tests).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(workspaces): full create modal with folder validation and result"` (+ trailer).

---

### Task 4: Console wiring

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py:837-883` (`ConsoleWorkspaceController._create_console_workspace`)
- Test: `Tests/Workspaces/test_console_workspace_create_handler.py` (new)

**Interfaces:**
- Consumes: `WorkspaceCreateModal`, `WorkspaceCreateResult` from Task 3.
- Produces: `ConsoleWorkspaceController._handle_workspace_create_result(result: WorkspaceCreateResult | None) -> None` (PR B appends one line to it).

- [ ] **Step 1: Write the failing test.** The controller methods are testable unbound against a stub exposing only the seams they touch:

```python
# Tests/Workspaces/test_console_workspace_create_handler.py
from types import SimpleNamespace

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateResult
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


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
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest Tests/Workspaces/test_console_workspace_create_handler.py -v` — AttributeError, method missing.

- [ ] **Step 3: Implement.** Replace the body of `_create_console_workspace` (keep the registry-None guard verbatim) and add the handler:

```python
    def _create_console_workspace(self) -> None:
        """Open the shared create dialog (spec 2026-08-17 §4.3)."""
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            self.app_instance.notify(
                "Workspace service is not ready.", severity="warning"
            )
            return
        from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateModal

        self.push_screen(
            WorkspaceCreateModal(registry_service=registry_service),
            self._handle_workspace_create_result,
        )

    def _handle_workspace_create_result(self, result) -> None:
        """Console-side post-create sync; the modal already created/bound."""
        if result is None:
            return
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            return
        for _folder, message in result.failed_folders:
            self.app_instance.notify(message, severity="warning")
        if not result.make_active:
            self._sync_console_workspace_context()
            self.app_instance.notify(
                f"Created {result.name}.", severity="information"
            )
            return
        try:
            registry_service.set_active_workspace(result.workspace_id)
        except WorkspaceRegistryServiceError:
            logger.opt(exception=True).warning(
                "Unable to activate new Console workspace"
            )
            self.app_instance.notify(
                "Workspace created but could not be activated.", severity="error"
            )
            return
        self._sync_console_chat_core_state()
        self._activate_console_session_for_workspace(result.workspace_id)
        self._sync_console_workspace_context()
        self.run_worker(
            self._sync_native_console_chat_ui(), exclusive=True, group="console-sync"
        )
        # TASK-713: the whole sequence is invisible when the Workspace status
        # row is scrolled out of view -- keep the toast even with the modal.
        self.app_instance.notify(
            f"Created {result.name} and switched Console to it.",
            severity="information",
        )
```

`next_local_workspace_identity` may now be unused in this file — remove the import if so.

- [ ] **Step 4: Run** — `.venv/bin/pytest Tests/Workspaces/test_console_workspace_create_handler.py -v` plus `.venv/bin/pytest Tests/Workspaces/ Tests/UI/test_console_native_chat_flow.py -x -q` for regressions (any test asserting the old zero-input flow must be updated to drive the modal).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(console): route New workspace through the shared create modal"` (+ trailer).

---

### Task 5: Settings wiring

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:11886-11892` (render) and `:14414-14437` (`handle_workspace_create`)
- Test: existing Settings workspaces tests (locate: `grep -rln "settings-workspace-create" Tests/`)

- [ ] **Step 1: Find and update affected tests first.** `grep -rn "settings-workspace-create-name" tldw_chatbook Tests` — every hit outside the two regions above must be updated: tests that typed a name into the inline input now drive the modal (push via button, fill `#workspace-create-name`, click `#workspace-create-confirm`). Write the updated tests before changing the screen; run them; they FAIL against the old inline flow only where behavior genuinely changed (e.g. asserting the input exists).

- [ ] **Step 2: Implement.** In `_render_workspaces_detail`, replace the `Horizontal` create row (lines 11886-11892) with:

```python
        yield Button(
            "Create workspace…", id="settings-workspace-create", compact=True
        )
```

Replace `handle_workspace_create` with:

```python
    @on(Button.Pressed, "#settings-workspace-create")
    def handle_workspace_create(self, event: Button.Pressed) -> None:
        """Open the shared create dialog (spec 2026-08-17 §4.3)."""
        event.stop()
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None:
            return
        from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateModal

        def _done(result) -> None:
            if result is None:
                return
            status_parts = [message for _folder, message in result.failed_folders]
            if result.make_active:
                try:
                    registry.set_active_workspace(result.workspace_id)
                except WorkspaceRegistryServiceError as exc:
                    status_parts.append(str(exc))
            self._settings_workspaces_result = "; ".join(status_parts)
            self._refresh_settings_workspaces_pane()

        self.app.push_screen(WorkspaceCreateModal(registry_service=registry), _done)
```

- [ ] **Step 3: Run** — the updated Settings tests from Step 1 plus `.venv/bin/pytest Tests/Workspaces/ -q`. All PASS.

- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat(settings): workspace creation opens the shared create modal"` (+ trailer).

---

### Task 6: Library wiring

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:19853-19901` (`create_local_workspace`)
- Test: existing Library workspace tests (locate: `grep -rln "library-create-local-workspace" Tests/`), updated the same way as Task 5 Step 1.

- [ ] **Step 1: Update affected tests first** (same discipline as Task 5 Step 1, id `#library-create-local-workspace`).

- [ ] **Step 2: Implement.** Replace the handler body after the registry-None guard (keep the guard verbatim) with:

```python
        from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateModal

        def _done(result) -> None:
            if result is None:
                return
            notify = getattr(self.app_instance, "notify", None)
            for _folder, message in result.failed_folders:
                if callable(notify):
                    notify(message, severity="warning")
            if result.make_active:
                try:
                    registry_service.set_active_workspace(result.workspace_id)
                except Exception:
                    logger.opt(exception=True).warning(
                        "Failed to activate new Library workspace"
                    )
                    if callable(notify):
                        notify(
                            "Workspace created but could not be activated.",
                            severity="error",
                        )
                    return
            self._invalidate_library_workspace_depth_state()
            # TASK-716: preserve the rail's scroll offset across the rebuild.
            self._preserve_library_rail_scroll()
            self.refresh(recompose=True)
            if callable(notify):
                if result.make_active:
                    # TASK-713: activation retargets Console from another screen.
                    notify(
                        f"Created local workspace {result.name} and made it "
                        "active; Console now targets it.",
                        severity="information",
                    )
                else:
                    notify(
                        f"Created local workspace {result.name}.",
                        severity="information",
                    )

        self.app.push_screen(
            WorkspaceCreateModal(registry_service=registry_service), _done
        )
```

- [ ] **Step 3: Run** — updated Library tests + `.venv/bin/pytest Tests/Workspaces/ -q`. All PASS.

- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat(library): workspace creation opens the shared create modal"` (+ trailer).

---

### Task 7: Docs, supersession note, live verification, close-out

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-26-settings-workspaces-category-design.md` (§1)
- Modify: matching `Docs/User_Guide/` pages (locate: `grep -rln -i "workspace" Docs/User_Guide/`)
- Commit: `Docs/superpowers/specs/2026-08-17-workspace-create-modal-and-project-skills-design.md` (still uncommitted)

- [ ] **Step 1: Supersession note.** At the top of the 2026-07-26 spec's §1, add: *"Superseded in part (2026-08-17): workspace creation on every surface now runs through the shared creation modal — see `2026-08-17-workspace-create-modal-and-project-skills-design.md` §4. The management-home split below otherwise stands."*
- [ ] **Step 2: User Guide.** Update every page found by the grep that documents workspace creation (Console, Settings, Library) to describe the modal (name prefill, optional folders, read-only default, Switch checkbox) and refresh each page's "Verified against" stamp.
- [ ] **Step 3: Live verification** per the `verify` project skill: launch the TUI, create a workspace from all three surfaces (one with a folder, one cancelled with escape, one with "Switch" unchecked), confirm the Console rail/session behavior and Settings folder list. Evidence per `backlog/docs/lessons-live-verification.md`.
- [ ] **Step 4: Full targeted gate** — `.venv/bin/pytest Tests/Workspaces/ -q` plus every test file touched in Tasks 4-6; then a `--collect-only` sweep over `Tests/` to catch import breakage.
- [ ] **Step 5: Close out** — check off the backlog task's ACs, add Implementation Notes, `backlog task edit <id> -s Done`; commit docs + spec: `git add -A && git commit -m "docs(workspaces): create-modal user guide + spec + supersession note"` (+ trailer); push and open the PR A branch PR.
