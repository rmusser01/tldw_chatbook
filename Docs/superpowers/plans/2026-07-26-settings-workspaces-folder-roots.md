# Settings ▸ Workspaces + Folder Access Roots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the two-PR train from
`Docs/superpowers/specs/2026-07-26-settings-workspaces-category-design.md`:
PR1 makes workspace folder bindings real (service + file-tool enforcement),
PR2 adds the Settings ▸ Workspaces management category.

**Architecture:** PR1 extends `LocalWorkspaceRegistryService` with lifecycle +
folder-binding methods, adds a multi-root path validator, and routes the
agent file tools through a run-bound allowed-roots seam (ContextVar set by
`BuiltinToolProvider.invoke`, workspace id threaded through
`_compose_run_registry_and_allowed`). PR2 registers a new immediate-apply
Settings category rendering a list+detail management pane on top of the PR1
service. Zero bindings ⇒ byte-identical behavior to today.

**Tech Stack:** Python 3.11+, Textual, sqlite (WorkspaceDB), pytest
(venv-only: run everything with `.venv/bin/python -m pytest` from the repo
root).

## Global Constraints

- Branches: PR1 `feat/workspace-folder-bindings`, PR2
  `feat/settings-workspaces-category` stacked on PR1; both off `origin/dev`;
  merged as one train.
- Folder access default is read-only: `metadata={"access": "ro"}`; write is
  per-folder opt-in (`"rw"`).
- Deny binding the filesystem root and `Path.home()` itself.
- Stored binding status is display-only; enforcement re-checks
  `Path.is_dir()` at call time.
- Reads are confined to sandbox+roots as well as writes (deliberate Codex
  divergence — ADR-028 §c; do not widen reads).
- Existing tool gates (`[tools] read_file_enabled` etc.) are untouched.
- The Default workspace can never hold bindings
  (`save_runtime_binding` already enforces this — keep that error path).
- Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss`; if a component
  sheet changes, regenerate via `cd tldw_chatbook/css && python build_css.py`.
- Blocked/invalid states must explain themselves inline, never
  tooltip-only, and never via a disabled Button that is expected to emit
  Pressed (disabled Textual Buttons never emit Pressed).
- All copy strings in this plan are exact — use them verbatim.

---

## PR1 — service + enforcement

### Task 1: `unarchive_workspace` + duplicate-name guard

**Files:**
- Modify: `tldw_chatbook/Workspaces/registry_service.py` (near
  `archive_workspace`, added by TASK-714)
- Test: `Tests/Workspaces/test_workspace_registry_service.py` (append)

**Interfaces:**
- Consumes: existing `archive_workspace`, `get_workspace`,
  `list_workspaces(include_archived=...)`, `WorkspaceNotFound`,
  `WorkspaceRegistryServiceError`, `_normalize_required_text`,
  `self._now_factory()`, `self.db.transaction()`.
- Produces: `unarchive_workspace(workspace_id: str) -> WorkspaceRecord`;
  case-insensitive duplicate-name rejection inside `rename_workspace` and
  `create_workspace` raising
  `WorkspaceRegistryServiceError(f"A workspace named {name} already exists.")`.

- [ ] **Step 1: Write the failing tests** (append to the test file; it
  already imports `WorkspaceNotFound`, `WorkspaceRegistryServiceError`,
  `build_test_registry`, `DEFAULT_WORKSPACE_ID`)

```python
def test_unarchive_workspace_restores_listing_without_activating(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)
    service.ensure_default_workspace()
    service.create_workspace(workspace_id="ws-a", name="Workspace 1")
    service.set_active_workspace("ws-a")
    service.archive_workspace("ws-a")

    restored = service.unarchive_workspace("ws-a")

    assert restored.archived is False
    assert restored.active is False  # never auto-activates
    listed = {record.workspace_id for record in service.list_workspaces()}
    assert "ws-a" in listed
    active = service.get_active_workspace()
    assert active is not None and active.workspace_id == DEFAULT_WORKSPACE_ID


def test_unarchive_workspace_rejects_unknown_and_unarchived(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Workspace 1")

    with pytest.raises(WorkspaceNotFound):
        service.unarchive_workspace("ws-missing")
    with pytest.raises(WorkspaceNotFound):
        service.unarchive_workspace("ws-a")  # not archived


def test_duplicate_names_rejected_case_insensitively(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Client A")
    service.create_workspace(workspace_id="ws-b", name="Workspace 2")

    with pytest.raises(WorkspaceRegistryServiceError):
        service.rename_workspace("ws-b", "client a")
    with pytest.raises(WorkspaceRegistryServiceError):
        service.create_workspace(workspace_id="ws-c", name="CLIENT A")
    # Renaming to its own current name (case-changed) is allowed.
    renamed = service.rename_workspace("ws-a", "client A")
    assert renamed.name == "client A"


def test_archived_names_do_not_block_reuse(tmp_path: Path) -> None:
    service = build_test_registry(tmp_path)
    service.create_workspace(workspace_id="ws-a", name="Client A")
    service.archive_workspace("ws-a")

    created = service.create_workspace(workspace_id="ws-b", name="Client A")
    assert created.name == "Client A"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "Tests/Workspaces/test_workspace_registry_service.py" -q -k "unarchive or duplicate_names or archived_names"`
Expected: FAIL — `AttributeError: ... has no attribute 'unarchive_workspace'` and missing duplicate-name errors.

- [ ] **Step 3: Implement.** In `registry_service.py`, add below
  `archive_workspace`:

```python
    def unarchive_workspace(self, workspace_id: str) -> WorkspaceRecord:
        """Restore an archived workspace to listings (spec §2).

        Never auto-activates: the user chooses when to switch.

        Raises:
            WorkspaceNotFound: Unknown or not-archived workspace.
            WorkspaceRegistryServiceError: Storage failure.
        """
        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        record = self.get_workspace(safe_workspace_id)
        if record is None or not record.archived:
            raise WorkspaceNotFound(safe_workspace_id)
        now = self._now_factory()
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    UPDATE workspace_records
                    SET archived = 0, updated_at = ?
                    WHERE workspace_id = ?
                    """,
                    (now, safe_workspace_id),
                )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        restored = self.get_workspace(safe_workspace_id)
        if restored is None:
            raise WorkspaceRegistryServiceError("Workspace unarchive failed.")
        return restored

    def _reject_duplicate_name(
        self, name: str, *, exclude_workspace_id: str | None = None
    ) -> None:
        """Raise when a non-archived workspace already uses ``name``.

        Case-insensitive; archived workspaces do not block reuse (spec §2).
        """
        needle = name.strip().casefold()
        for record in self.list_workspaces():
            if exclude_workspace_id and record.workspace_id == exclude_workspace_id:
                continue
            if str(record.name or "").strip().casefold() == needle:
                raise WorkspaceRegistryServiceError(
                    f"A workspace named {name} already exists."
                )
```

  Then wire the guard in: in `create_workspace`, immediately after the
  `record = WorkspaceRecord(...)` construction line, insert
  `self._reject_duplicate_name(record.name)`. In `rename_workspace`
  (TASK-714), after the existing `safe_name` blank-check, insert
  `self._reject_duplicate_name(safe_name, exclude_workspace_id=safe_workspace_id)`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest "Tests/Workspaces/test_workspace_registry_service.py" -q`
Expected: PASS (all — including the pre-existing 27+ cases; `ensure_default_workspace` and `next_local_workspace_identity` generate unique names so nothing else trips the guard).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Workspaces/registry_service.py Tests/Workspaces/test_workspace_registry_service.py
git commit -m "feat(workspaces): unarchive + case-insensitive duplicate-name guard"
```

### Task 2: folder-binding service methods

**Files:**
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Modify: `tldw_chatbook/Workspaces/__init__.py` (export `BindingNotFound`)
- Test: `Tests/Workspaces/test_workspace_folder_bindings.py` (new file)

**Interfaces:**
- Consumes: existing `save_runtime_binding(binding) -> WorkspaceRuntimeBinding`
  (already rejects Default + unknown workspaces),
  `list_runtime_bindings(workspace_id, binding_kind=None)`,
  `WorkspaceRuntimeBinding(workspace_id, binding_id, binding_kind, label,
  locator, status, metadata, ...)`, `RuntimeBindingKind.LOCAL_FILESYSTEM`,
  `RuntimeBindingStatus.READY / .MISSING`.
- Produces (used by Tasks 3-6 and PR2):
  - `class BindingNotFound(WorkspaceRegistryServiceError)` with
    `binding_id` attribute.
  - `add_folder_binding(workspace_id: str, path: str | Path, *,
    allow_write: bool = False) -> WorkspaceRuntimeBinding`
  - `remove_runtime_binding(binding_id: str) -> None`
  - `list_folder_bindings(workspace_id: str) -> tuple[WorkspaceRuntimeBinding, ...]`
    (status recomputed from the filesystem; metadata carries
    `{"access": "ro"|"rw"}`).
  - `set_folder_binding_access(binding_id: str, *, allow_write: bool) ->
    WorkspaceRuntimeBinding` (PR2's rw toggle).

- [ ] **Step 1: Write the failing tests** (new file)

```python
"""Folder-binding service methods (spec 2026-07-26 settings-workspaces §2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID, LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.registry_service import (
    BindingNotFound,
    WorkspaceNotFound,
    WorkspaceRegistryServiceError,
)


def build_registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    return LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="folder-tests")
    )


@pytest.fixture()
def service(tmp_path: Path) -> LocalWorkspaceRegistryService:
    registry = build_registry(tmp_path)
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="Client A")
    return registry


def test_add_folder_binding_stores_canonical_ro_binding(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    folder = tmp_path / "project"
    folder.mkdir()

    binding = service.add_folder_binding("ws-a", folder)

    assert binding.locator == str(folder.resolve())
    assert binding.label == "project"
    assert str(binding.binding_kind) in ("local-filesystem", "RuntimeBindingKind.LOCAL_FILESYSTEM")
    assert binding.metadata["access"] == "ro"
    assert str(binding.status) in ("ready", "RuntimeBindingStatus.READY")


def test_add_folder_binding_allow_write(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    folder = tmp_path / "writable"
    folder.mkdir()
    binding = service.add_folder_binding("ws-a", folder, allow_write=True)
    assert binding.metadata["access"] == "rw"


def test_add_folder_binding_validation_matrix(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    missing = tmp_path / "nope"
    a_file = tmp_path / "file.txt"
    a_file.write_text("x")
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", missing)
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", a_file)
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", Path("/"))
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", Path.home())
    with pytest.raises(WorkspaceNotFound):
        service.add_folder_binding("ws-missing", tmp_path)


def test_add_folder_binding_rejects_default_workspace(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    folder = tmp_path / "any"
    folder.mkdir()
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding(DEFAULT_WORKSPACE_ID, folder)


def test_add_folder_binding_rejects_duplicates_and_nesting(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    service.add_folder_binding("ws-a", parent)

    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", parent)  # duplicate
    with pytest.raises(WorkspaceRegistryServiceError):
        service.add_folder_binding("ws-a", child)  # nested under existing

    # And the reverse direction: existing child blocks a new parent root.
    other = build_registry(tmp_path / "second-db")
    other.create_workspace(workspace_id="ws-b", name="Client B")
    other.add_folder_binding("ws-b", child)
    with pytest.raises(WorkspaceRegistryServiceError):
        other.add_folder_binding("ws-b", parent)


def test_list_folder_bindings_recomputes_status(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    folder = tmp_path / "ephemeral"
    folder.mkdir()
    service.add_folder_binding("ws-a", folder)
    folder.rmdir()

    bindings = service.list_folder_bindings("ws-a")
    assert len(bindings) == 1
    assert str(bindings[0].status) in ("missing", "RuntimeBindingStatus.MISSING")


def test_remove_and_toggle_access(
    service: LocalWorkspaceRegistryService, tmp_path: Path
) -> None:
    folder = tmp_path / "toggled"
    folder.mkdir()
    binding = service.add_folder_binding("ws-a", folder)

    updated = service.set_folder_binding_access(binding.binding_id, allow_write=True)
    assert updated.metadata["access"] == "rw"

    service.remove_runtime_binding(binding.binding_id)
    assert service.list_folder_bindings("ws-a") == ()
    with pytest.raises(BindingNotFound):
        service.remove_runtime_binding(binding.binding_id)
    with pytest.raises(BindingNotFound):
        service.set_folder_binding_access(binding.binding_id, allow_write=False)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "Tests/Workspaces/test_workspace_folder_bindings.py" -q`
Expected: FAIL — `ImportError: cannot import name 'BindingNotFound'`.

- [ ] **Step 3: Implement.** In `registry_service.py`:

```python
class BindingNotFound(WorkspaceRegistryServiceError):
    """Raised when a runtime binding id does not exist."""

    def __init__(self, binding_id: str) -> None:
        super().__init__(f"Runtime binding not found: {binding_id}")
        self.binding_id = binding_id
```

  Methods on `LocalWorkspaceRegistryService` (place after
  `save_runtime_binding`; `uuid4` from the module's existing imports or add
  `from uuid import uuid4`):

```python
    def add_folder_binding(
        self,
        workspace_id: str,
        path: str | Path,
        *,
        allow_write: bool = False,
    ) -> WorkspaceRuntimeBinding:
        """Bind a folder as a file-tool access root (spec §2).

        Read-only by default; canonical (resolved) locator; denies the
        filesystem root, the home directory itself, non-directories, and
        duplicate/nested roots within the same workspace. Default-workspace
        and unknown-workspace rejection is delegated to
        ``save_runtime_binding``.
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
        for existing in self.list_folder_bindings(workspace_id):
            existing_path = Path(existing.locator)
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
        binding = WorkspaceRuntimeBinding(
            workspace_id=workspace_id,
            binding_id=f"folder-{uuid4().hex[:12]}",
            binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM,
            label=resolved.name or str(resolved),
            locator=str(resolved),
            status=RuntimeBindingStatus.READY,
            metadata={"access": "rw" if allow_write else "ro"},
        )
        return self.save_runtime_binding(binding)

    def list_folder_bindings(
        self, workspace_id: str
    ) -> tuple[WorkspaceRuntimeBinding, ...]:
        """Local-filesystem bindings with status recomputed from disk."""
        bindings = self.list_runtime_bindings(
            workspace_id, binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM
        )
        refreshed: list[WorkspaceRuntimeBinding] = []
        for binding in bindings:
            actual = (
                RuntimeBindingStatus.READY
                if Path(binding.locator).is_dir()
                else RuntimeBindingStatus.MISSING
            )
            refreshed.append(
                WorkspaceRuntimeBinding(
                    workspace_id=binding.workspace_id,
                    binding_id=binding.binding_id,
                    binding_kind=binding.binding_kind,
                    label=binding.label,
                    locator=binding.locator,
                    status=actual,
                    metadata=binding.metadata,
                    created_at=binding.created_at,
                    updated_at=binding.updated_at,
                )
            )
        return tuple(refreshed)

    def remove_runtime_binding(self, binding_id: str) -> None:
        """Delete a runtime binding row (spec §2)."""
        safe_binding_id = _normalize_required_text(binding_id, "binding_id")
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM workspace_runtime_bindings WHERE binding_id = ?",
                    (safe_binding_id,),
                )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        if cursor.rowcount == 0:
            raise BindingNotFound(safe_binding_id)

    def set_folder_binding_access(
        self, binding_id: str, *, allow_write: bool
    ) -> WorkspaceRuntimeBinding:
        """Flip a folder binding's ro/rw access flag (spec §4 toggle)."""
        existing = self.get_runtime_binding(binding_id)
        if existing is None:
            raise BindingNotFound(binding_id)
        metadata = dict(existing.metadata)
        metadata["access"] = "rw" if allow_write else "ro"
        return self.save_runtime_binding(
            WorkspaceRuntimeBinding(
                workspace_id=existing.workspace_id,
                binding_id=existing.binding_id,
                binding_kind=existing.binding_kind,
                label=existing.label,
                locator=existing.locator,
                status=existing.status,
                metadata=metadata,
                created_at=existing.created_at,
            )
        )
```

  Check `list_runtime_bindings`'s actual signature first (it exists at
  `registry_service.py` ~line 584): if it does not accept a
  `binding_kind=` filter, filter in `list_folder_bindings` instead:
  `[b for b in self.list_runtime_bindings(workspace_id) if
  str(b.binding_kind) in ("local-filesystem",
  str(RuntimeBindingKind.LOCAL_FILESYSTEM))]`.
  Export `BindingNotFound` from `tldw_chatbook/Workspaces/__init__.py`
  alongside the existing service exports.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest "Tests/Workspaces/" -q`
Expected: PASS (new file + all pre-existing Workspaces suites).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Workspaces/registry_service.py tldw_chatbook/Workspaces/__init__.py Tests/Workspaces/test_workspace_folder_bindings.py
git commit -m "feat(workspaces): folder-binding service methods (add/list/remove/access)"
```

### Task 3: `validate_path_multi`

**Files:**
- Modify: `tldw_chatbook/Utils/path_validation.py`
- Test: `Tests/Utils/test_path_validation_multi.py` (new; if
  `Tests/Utils/` lacks an `__init__.py` convention check a sibling test
  file — `ls Tests/Utils/` — and mirror it)

**Interfaces:**
- Consumes: existing `validate_path(user_path, base_directory) -> Path`
  (raises `ValueError` on escape; relative paths resolve against base).
- Produces: `validate_path_multi(user_path: Union[str, Path], roots:
  Sequence[Union[str, Path]]) -> Path` — tries roots in order; relative
  paths resolve against `roots[0]`; raises `ValueError` whose message names
  every consulted root when none matches.

- [ ] **Step 1: Write the failing test**

```python
"""Multi-root path validation (spec 2026-07-26 settings-workspaces §3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Utils.path_validation import validate_path_multi


def test_accepts_path_inside_any_root(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    (root_b / "sub").mkdir(parents=True)
    root_a.mkdir()
    target = root_b / "sub" / "file.txt"
    target.write_text("x")

    assert validate_path_multi(target, [root_a, root_b]) == target.resolve()


def test_relative_paths_resolve_against_first_root(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "note.txt").write_text("x")

    resolved = validate_path_multi("note.txt", [sandbox, tmp_path / "other"])
    assert resolved == (sandbox / "note.txt").resolve()


def test_rejection_names_all_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")

    with pytest.raises(ValueError) as excinfo:
        validate_path_multi(outside, [root_a, root_b])
    message = str(excinfo.value)
    assert str(root_a) in message and str(root_b) in message


def test_empty_roots_rejected() -> None:
    with pytest.raises(ValueError):
        validate_path_multi("anything", [])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "Tests/Utils/test_path_validation_multi.py" -q`
Expected: FAIL — `ImportError: cannot import name 'validate_path_multi'`.

- [ ] **Step 3: Implement** (append to `path_validation.py`, reusing
  `validate_path` so every per-root security check — traversal, hidden-file
  rules — stays identical):

```python
def validate_path_multi(
    user_path: Union[str, Path], roots: Sequence[Union[str, Path]]
) -> Path:
    """Validate ``user_path`` against several allowed roots (first match wins).

    Relative paths resolve against ``roots[0]`` (the primary root — callers
    pass the tool sandbox first so legacy relative-path behavior is
    unchanged). The rejection message names every consulted root so a
    denial is actionable.

    Args:
        user_path: The path provided by the user or model.
        roots: Allowed base directories, in priority order.

    Returns:
        The validated absolute path.

    Raises:
        ValueError: No roots given, or the path escapes all of them.
    """
    root_list = [Path(root) for root in roots]
    if not root_list:
        raise ValueError("No allowed roots configured for path validation.")
    candidate = Path(user_path)
    for index, root in enumerate(root_list):
        if index > 0 and not candidate.is_absolute():
            continue  # relative paths anchor to the primary root only
        try:
            return validate_path(user_path, root)
        except ValueError:
            continue
    consulted = ", ".join(str(root.resolve()) for root in root_list)
    raise ValueError(
        f"Path '{user_path}' is outside every allowed root ({consulted})."
    )
```

  Add `Sequence` to the module's `typing` import if absent.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest "Tests/Utils/test_path_validation_multi.py" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Utils/path_validation.py Tests/Utils/test_path_validation_multi.py
git commit -m "feat(utils): validate_path_multi for multi-root file-tool validation"
```

### Task 4: run-bound allowed-roots seam

**Files:**
- Create: `tldw_chatbook/Tools/workspace_file_roots.py`
- Test: `Tests/Tools/test_workspace_file_roots.py` (new; mirror
  `Tests/Tools/` conventions — `ls Tests/Tools/` first)

**Interfaces:**
- Consumes: Task 2's `list_folder_bindings`;
  `tldw_chatbook.config.get_workspaces_db_path`;
  `LocalWorkspaceRegistryService`, `WorkspaceDB`.
- Produces (used by Tasks 5-6):
  - `run_workspace(workspace_id: str | None)` — context manager binding the
    current run's workspace id (ContextVar-backed, safe across
    `asyncio.run` inside worker threads).
  - `current_run_workspace_id() -> str | None`.
  - `allowed_file_roots(*, write: bool, sandbox_root: Path) ->
    tuple[Path, ...]` — `(sandbox_root, *existing folder roots)` for the
    run's workspace (fallback: active workspace); `write=True` keeps only
    `metadata["access"] == "rw"` folders; folders that are not directories
    at call time are dropped.
  - `_registry_factory` module attribute (callable returning a registry) —
    the test seam.

- [ ] **Step 1: Write the failing test**

```python
"""Run-bound allowed file roots (spec 2026-07-26 settings-workspaces §3)."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Tools import workspace_file_roots as wfr


def _registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="roots-tests")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="Client A")
    registry.create_workspace(workspace_id="ws-b", name="Client B")
    return registry


def test_roots_follow_run_workspace_not_active(tmp_path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    folder_a = tmp_path / "a"
    folder_a.mkdir()
    folder_b = tmp_path / "b"
    folder_b.mkdir()
    registry.add_folder_binding("ws-a", folder_a)
    registry.add_folder_binding("ws-b", folder_b)
    registry.set_active_workspace("ws-b")
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    with wfr.run_workspace("ws-a"):
        roots = wfr.allowed_file_roots(write=False, sandbox_root=sandbox)
    assert roots == (sandbox, folder_a.resolve())

    # Outside a run: falls back to the ACTIVE workspace (ws-b).
    roots = wfr.allowed_file_roots(write=False, sandbox_root=sandbox)
    assert roots == (sandbox, folder_b.resolve())


def test_write_roots_require_rw_and_existing_dirs(tmp_path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    ro_folder = tmp_path / "ro"
    ro_folder.mkdir()
    rw_folder = tmp_path / "rw"
    rw_folder.mkdir()
    gone = tmp_path / "gone"
    gone.mkdir()
    registry.add_folder_binding("ws-a", ro_folder)
    registry.add_folder_binding("ws-a", rw_folder, allow_write=True)
    registry.add_folder_binding("ws-a", gone, allow_write=True)
    gone.rmdir()  # deleted after binding: must drop out at call time
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    with wfr.run_workspace("ws-a"):
        read_roots = wfr.allowed_file_roots(write=False, sandbox_root=sandbox)
        write_roots = wfr.allowed_file_roots(write=True, sandbox_root=sandbox)

    assert ro_folder.resolve() in read_roots
    assert write_roots == (sandbox, rw_folder.resolve())


def test_registry_failure_degrades_to_sandbox_only(tmp_path, monkeypatch) -> None:
    def _boom():
        raise RuntimeError("registry down")

    monkeypatch.setattr(wfr, "_registry_factory", _boom)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    with wfr.run_workspace("ws-a"):
        assert wfr.allowed_file_roots(write=True, sandbox_root=sandbox) == (sandbox,)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "Tests/Tools/test_workspace_file_roots.py" -q`
Expected: FAIL — `ModuleNotFoundError: ... workspace_file_roots`.

- [ ] **Step 3: Implement** (new module):

```python
"""Run-bound workspace folder roots for the agent file tools.

Spec: Docs/superpowers/specs/2026-07-26-settings-workspaces-category-design.md §3.
The provider (`BuiltinToolProvider.invoke`) binds the run's workspace via
``run_workspace``; the file tools ask ``allowed_file_roots`` at call time.
Reads and writes are both confined to sandbox+roots (deliberate Codex
divergence, ADR-028) and stored binding status is never trusted — existence
is re-checked here on every call.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from loguru import logger

_RUN_WORKSPACE_ID: ContextVar[str | None] = ContextVar(
    "tldw_run_workspace_id", default=None
)


def _default_registry_factory():
    from tldw_chatbook.config import get_workspaces_db_path
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    return LocalWorkspaceRegistryService(
        WorkspaceDB(get_workspaces_db_path(), client_id="file-tools")
    )


#: Test seam: monkeypatch with a factory returning a prepared registry.
_registry_factory = _default_registry_factory


@contextmanager
def run_workspace(workspace_id: str | None) -> Iterator[None]:
    """Bind the current run's workspace for the duration of a tool call."""
    token = _RUN_WORKSPACE_ID.set(workspace_id)
    try:
        yield
    finally:
        _RUN_WORKSPACE_ID.reset(token)


def current_run_workspace_id() -> str | None:
    return _RUN_WORKSPACE_ID.get()


def allowed_file_roots(*, write: bool, sandbox_root: Path) -> tuple[Path, ...]:
    """Sandbox root plus the run's workspace folder roots, existing-only.

    Fail-safe: any registry failure degrades to sandbox-only rather than
    widening access.
    """
    roots: list[Path] = [sandbox_root]
    try:
        registry = _registry_factory()
        workspace_id = current_run_workspace_id()
        if workspace_id is None:
            active = registry.get_active_workspace()
            workspace_id = active.workspace_id if active is not None else None
        if workspace_id is None:
            return tuple(roots)
        for binding in registry.list_folder_bindings(workspace_id):
            if write and str(binding.metadata.get("access", "ro")) != "rw":
                continue
            folder = Path(binding.locator)
            if folder.is_dir():
                roots.append(folder)
    except Exception:
        logger.opt(exception=True).warning(
            "Workspace folder roots unavailable; file tools confined to sandbox"
        )
        return (sandbox_root,)
    return tuple(roots)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest "Tests/Tools/test_workspace_file_roots.py" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/workspace_file_roots.py Tests/Tools/test_workspace_file_roots.py
git commit -m "feat(tools): run-bound workspace file-roots seam (sandbox-plus-folders)"
```

### Task 5: route file tools through multi-root validation

**Files:**
- Modify: `tldw_chatbook/Tools/file_operation_tools.py` (three `execute`
  methods; `validate_path` call sites at ~line 106 (`ReadFileTool`), ~216
  (`ListDirectoryTool`), and inside `WriteFileTool.execute` ~line 368+)
- Test: `Tests/Tools/test_file_tools_workspace_roots.py` (new)

**Interfaces:**
- Consumes: Task 3 `validate_path_multi`, Task 4 `allowed_file_roots` /
  `run_workspace`; existing `_tool_sandbox_root()`.
- Produces: unchanged tool APIs; new behavior — reads/lists validate
  against `allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root())`,
  writes against `write=True`.

- [ ] **Step 1: Write the failing test**

```python
"""File tools honor workspace folder roots (spec §3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Tools import file_operation_tools as fot
from tldw_chatbook.Tools import workspace_file_roots as wfr
from tldw_chatbook.Tools.file_operation_tools import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)


@pytest.fixture()
def bound_workspace(tmp_path, monkeypatch):
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="tool-tests")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="Client A")
    ro_folder = tmp_path / "ro-project"
    rw_folder = tmp_path / "rw-project"
    ro_folder.mkdir()
    rw_folder.mkdir()
    registry.add_folder_binding("ws-a", ro_folder)
    registry.add_folder_binding("ws-a", rw_folder, allow_write=True)
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(sandbox))
    return {"ro": ro_folder, "rw": rw_folder, "sandbox": sandbox}


@pytest.mark.asyncio
async def test_read_allowed_in_bound_folder(bound_workspace) -> None:
    target = bound_workspace["ro"] / "notes.md"
    target.write_text("hello")
    with wfr.run_workspace("ws-a"):
        result = await ReadFileTool().execute(file_path=str(target))
    assert result.get("error") is None
    assert result["content"] == "hello"


@pytest.mark.asyncio
async def test_write_denied_in_ro_folder_allowed_in_rw(bound_workspace) -> None:
    ro_target = bound_workspace["ro"] / "out.txt"
    rw_target = bound_workspace["rw"] / "out.txt"
    with wfr.run_workspace("ws-a"):
        denied = await WriteFileTool().execute(
            file_path=str(ro_target), content="x"
        )
        allowed = await WriteFileTool().execute(
            file_path=str(rw_target), content="x"
        )
    assert denied.get("error")
    assert allowed.get("error") is None
    assert rw_target.read_text() == "x"


@pytest.mark.asyncio
async def test_denial_names_roots_and_other_workspace_is_denied(
    bound_workspace, tmp_path
) -> None:
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("x")
    with wfr.run_workspace("ws-a"):
        result = await ReadFileTool().execute(file_path=str(outside))
    assert result.get("error")
    assert str(bound_workspace["sandbox"]) in result["error"]

    with wfr.run_workspace("workspace-default"):
        default_denied = await ReadFileTool().execute(
            file_path=str(bound_workspace["ro"] / "notes.md")
        )
    assert default_denied.get("error")


@pytest.mark.asyncio
async def test_zero_bindings_parity_with_sandbox(bound_workspace) -> None:
    inside = bound_workspace["sandbox"] / "kept.txt"
    inside.write_text("sandboxed")
    with wfr.run_workspace("workspace-default"):
        listed = await ListDirectoryTool().execute(
            directory_path=str(bound_workspace["sandbox"])
        )
        read = await ReadFileTool().execute(file_path=str(inside))
    assert listed.get("error") is None
    assert read.get("error") is None
```

  Check `ListDirectoryTool.parameters` for its actual argument name
  (`directory_path` assumed — read the class; if it is `path`, adjust the
  test, not the tool).

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "Tests/Tools/test_file_tools_workspace_roots.py" -q`
Expected: FAIL — reads/writes in bound folders rejected ("outside the allowed directory").

- [ ] **Step 3: Implement.** In `file_operation_tools.py` add imports:

```python
from tldw_chatbook.Utils.path_validation import validate_path, validate_path_multi
from tldw_chatbook.Tools.workspace_file_roots import allowed_file_roots
```

  and replace each per-tool validation call:

```python
# ReadFileTool.execute and ListDirectoryTool.execute:
validated_path = validate_path_multi(
    file_path,
    allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root()),
)
# WriteFileTool.execute:
validated_path = validate_path_multi(
    file_path,
    allowed_file_roots(write=True, sandbox_root=_tool_sandbox_root()),
)
```

  (`ListDirectoryTool` validates its directory argument the same way; keep
  every surrounding error-dict shape untouched — the `ValueError` message
  from `validate_path_multi` flows into the existing `{"error": ...}`
  handling.)

- [ ] **Step 4: Run to verify pass + regression pin**

Run: `.venv/bin/python -m pytest "Tests/Tools/" -q`
Expected: PASS — new file AND every pre-existing file-tool test (those are
the zero-binding parity pin: they run without any workspace bindings and
must behave exactly as before).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/file_operation_tools.py Tests/Tools/test_file_tools_workspace_roots.py
git commit -m "feat(tools): file tools honor workspace folder roots (ro/rw, call-time)"
```

### Task 6: bind the run's workspace through the provider

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py`
  (`BuiltinToolProvider.__init__` ~line 273, `invoke` ~line 360)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
  (`_compose_run_registry_and_allowed` ~line 778 and its call site(s) —
  find with `grep -n "_compose_run_registry_and_allowed(" tldw_chatbook/Chat/console_agent_bridge.py`)
- Test: `Tests/Agents/test_builtin_provider_workspace_binding.py` (new;
  mirror `Tests/Agents/` conventions)

**Interfaces:**
- Consumes: Task 4 `run_workspace`.
- Produces: `BuiltinToolProvider(gate=None, workspace_id: str | None = None)`;
  `_compose_run_registry_and_allowed(..., workspace_id: str | None = None)`
  threading it; the bridge's caller passes the running session's
  `workspace_id` (the Console session object's `workspace_id` attribute —
  the caller that invokes `_compose_run_registry_and_allowed` has the
  session/store in scope; trace it at implementation time and pass
  `session.workspace_id`).

- [ ] **Step 1: Write the failing test**

```python
"""BuiltinToolProvider binds the run's workspace around tool execution."""

from __future__ import annotations

from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
from tldw_chatbook.Tools import workspace_file_roots as wfr


class _ProbeTool:
    name = "probe_workspace"
    description = "records the bound run workspace"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return {"workspace": wfr.current_run_workspace_id()}


class _OpenGate:
    def check(self, tool):
        return None


def test_invoke_binds_and_clears_run_workspace() -> None:
    provider = BuiltinToolProvider(gate=_OpenGate(), workspace_id="ws-a")
    provider._tools["probe_workspace"] = _ProbeTool()

    result = provider.invoke("builtin:probe_workspace", {})

    assert result.ok, result.error
    assert '"workspace": "ws-a"' in result.content
    assert wfr.current_run_workspace_id() is None  # cleared after invoke


def test_invoke_without_workspace_leaves_context_unset() -> None:
    provider = BuiltinToolProvider(gate=_OpenGate())
    provider._tools["probe_workspace"] = _ProbeTool()

    result = provider.invoke("builtin:probe_workspace", {})

    assert result.ok, result.error
    assert '"workspace": null' in result.content
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "Tests/Agents/test_builtin_provider_workspace_binding.py" -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'workspace_id'`.

- [ ] **Step 3: Implement.** In `tool_catalog.py`:

```python
    def __init__(
        self, gate: Any | None = None, workspace_id: str | None = None
    ) -> None:
        # spec §3: the run's workspace, bound around every tool execution so
        # file tools resolve THIS run's folder roots — not whatever
        # workspace the user is looking at when the tool fires.
        self._workspace_id = workspace_id
        ...  # existing body unchanged
```

  and in `invoke`, wrap the execution line:

```python
        from tldw_chatbook.Tools.workspace_file_roots import run_workspace

        try:
            with run_workspace(self._workspace_id):
                raw = asyncio.run(tool.execute(**args))
```

  (When `self._workspace_id` is `None`, `run_workspace(None)` keeps the
  ContextVar at `None`, so `allowed_file_roots` falls back to the active
  workspace — the spec's documented fallback.)
  In `console_agent_bridge.py`, add `workspace_id: str | None = None` to
  `_compose_run_registry_and_allowed`'s keyword parameters, pass it into
  `BuiltinToolProvider(gate=builtin_gate, workspace_id=workspace_id)`, and
  thread the session's `workspace_id` from the call site (grep as noted in
  **Files**; the surrounding function has the active run's session in
  scope).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest "Tests/Agents/" -q`
Expected: PASS (new + existing agent suites — the added keyword defaults to
`None` everywhere it is not passed).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Agents/test_builtin_provider_workspace_binding.py
git commit -m "feat(agents): bind run workspace through BuiltinToolProvider for file roots"
```

### Task 7: ADR-028 + PR1 assembly

**Files:**
- Create: `backlog/decisions/028-settings-workspaces-category-and-folder-roots.md`
- Test: none (docs) — full-suite verification instead.

**Interfaces:** none produced; records spec §8 items (a)-(d) verbatim.

- [ ] **Step 1: Write the ADR.** Content: title
  `# 028 - Settings owns workspace management; folders are file-tool access roots`;
  Date/Status/Relates-to header (relates to spec
  `2026-07-26-settings-workspaces-category-design.md`, supersedes the
  Stage-5 read-only boundary in
  `Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md`); then
  four decision sections copied from spec §8: (a) ownership supersession
  with Console/Library quick actions retained, (b) folder semantics
  (access roots, `ro` default, call-time existence validation, run-bound
  resolution, Default stays tool-less), (c) the deliberate Codex
  divergence — reads confined too, "do not 'fix' this to match Codex", (d)
  canonical locators as the `runtimeWorkspaceRoots`-shaped export surface
  for external runtimes.

- [ ] **Step 2: Full PR1 verification**

Run: `.venv/bin/python -m pytest "Tests/Workspaces/" "Tests/Tools/" "Tests/Agents/" "Tests/Utils/test_path_validation_multi.py" "Tests/UI/test_console_workspace_lifecycle.py" "Tests/UI/test_console_workspace_keyboard.py" -q`
Expected: PASS. Then the two big pins:
`.venv/bin/python -m pytest "Tests/UI/test_console_native_chat_flow.py" -q`
Expected: 201 passed (known-flaky exceptions documented in the workspace
program memory do not include this file).

- [ ] **Step 3: Commit + push + PR**

```bash
git add backlog/decisions/028-settings-workspaces-category-and-folder-roots.md
git commit -m "docs: ADR-028 Settings-owned workspace management + folder access roots"
git push -u origin feat/workspace-folder-bindings
gh pr create --base dev --title "feat(workspaces): folder access roots — service + file-tool enforcement (PR1 of 2)" --body "Implements spec 2026-07-26-settings-workspaces-category-design.md §§2-3,5,8. Zero bindings = behavior identical to today. PR2 (Settings category) stacks on this."
```

---

## PR2 — Settings ▸ Workspaces category
*(branch `feat/settings-workspaces-category` off PR1's branch)*

### Task 8: register the category

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_config_models.py`
  (`SettingsCategoryId` enum)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` —
  `_category_summaries` (~line 926), `_category_groups` (~line 1044), the
  Save/Revert suppression sites (comment at ~415 explains the pattern;
  actual suppression membership at ~1870-1876 where THEME and
  SPLASH_SCREEN are listed — add WORKSPACES beside them at BOTH sites
  found by `grep -n "SettingsCategoryId.THEME" tldw_chatbook/UI/Screens/settings_screen.py`),
  ownership records (~974-988 region), and `_render_detail_pane` dispatch
  (~line 7034) routing to a stub `_render_workspaces_detail`.
- Test: `Tests/UI/test_settings_workspaces_category.py` (new; harness
  pattern from `Tests/UI/test_settings_configuration_hub.py`:
  `DestinationHarness(app, "settings")`, `_open_settings_category(pilot,
  "#settings-category-workspaces")`, `_visible_text`)

**Interfaces:**
- Produces: `SettingsCategoryId.WORKSPACES = "workspaces"`; category button
  id `settings-category-workspaces`; stub pane rendering a Static
  `"Workspace management"` that Task 9 replaces.
- Category summary copy (verbatim):
  title `"Workspaces"`, description
  `"Create, rename, archive, and bind folders for agent file tools."`,
  status `"Immediate actions"`.

- [ ] **Step 1: Write the failing test**

```python
"""Settings ▸ Workspaces category registration (spec §4)."""

from __future__ import annotations

import pytest

from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
    _visible_text,
)


@pytest.mark.asyncio
async def test_workspaces_category_registered_and_immediate() -> None:
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        text = _visible_text(screen)

        assert "Workspace management" in text
        # Immediate-apply category: the guided Save/Revert buttons are
        # suppressed exactly like Theme/Splash.
        assert not screen.query("#settings-save-category")
```

  (Before writing, check the actual Save button id used by the suppression
  assertion in existing Theme tests — `grep -n "settings-save-category\|save.*suppress" Tests/UI/test_settings_configuration_hub.py`
  — and mirror whatever those tests assert. If the helpers above are named
  differently in that file, import the real names.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "Tests/UI/test_settings_workspaces_category.py" -q`
Expected: FAIL — no `#settings-category-workspaces` button.

- [ ] **Step 3: Implement.** Enum:
  `WORKSPACES = "workspaces"` (place after STORAGE). Summary entry in
  `_category_summaries` with the verbatim copy above; group it in
  `_category_groups` under the `"Data & Privacy"` tuple after Storage;
  add `SettingsCategoryId.WORKSPACES` to BOTH Save/Revert suppression
  sites beside THEME/SPLASH_SCREEN; ownership record mirroring the
  THEME record's shape (~974) with
  `owns_config_sections=()`,
  `runtime_owner="Workspace registry (immediate actions)"`,
  `boundary_copy="Lifecycle and folder bindings apply immediately; no draft state."`,
  `recovery_copy="Quick actions: switch/rename/archive in Console (Alt+W); create in Library."`;
  dispatch branch in `_render_detail_pane`:

```python
        if category is SettingsCategoryId.WORKSPACES:
            yield from self._render_workspaces_detail()
            return
```

  stub:

```python
    def _render_workspaces_detail(self) -> ComposeResult:
        yield Static("Workspace management", classes="destination-section")
```

  (Match the surrounding dispatch style — if `_render_detail_pane` uses
  `return self._render_x()` lists instead of `yield from`, mirror it.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest "Tests/UI/test_settings_workspaces_category.py" "Tests/UI/test_settings_configuration_hub.py" -q`
Expected: PASS (hub suite guards the category list/grouping invariants).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_config_models.py tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_workspaces_category.py
git commit -m "feat(settings): register Workspaces category (immediate-apply, stub pane)"
```

### Task 9: workspace list + create + lifecycle card

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
  (`_render_workspaces_detail` + handlers)
- Test: `Tests/UI/test_settings_workspaces_category.py` (append)

**Interfaces:**
- Consumes: `app_instance.workspace_registry_service` (
  `list_workspaces(include_archived=...)`, `get_active_workspace`,
  `create_workspace`, `rename_workspace`, `set_active_workspace`,
  `archive_workspace`, `unarchive_workspace`,
  `next_local_workspace_identity` for generated ids),
  `ConfirmationDialog` (`tldw_chatbook.Widgets.confirmation_dialog`;
  its confirm callback MUST be a coroutine function).
- Produces (ids Task 10/11 and tests rely on):
  `#settings-workspaces-list` (Vertical of row Buttons
  `#settings-workspace-row-{workspace_id}`),
  `#settings-workspaces-show-archived` (Checkbox),
  `#settings-workspace-create-name` (Input) +
  `#settings-workspace-create` (Button),
  detail card `#settings-workspace-card` with
  `#settings-workspace-rename-input`, `#settings-workspace-rename-apply`,
  `#settings-workspace-set-active`, `#settings-workspace-archive`,
  `#settings-workspace-unarchive`,
  result line `#settings-workspaces-result` (inline errors/confirmations —
  never tooltip-only). Selection state:
  `self._settings_selected_workspace_id: str | None`.
- Copy (verbatim): archive confirm message
  `f"Archive {name}? Its conversations stay saved and remain visible in Library; the workspace disappears from the switcher and the Console browser."`;
  Default card explanation
  `"The built-in Default workspace keeps its identity and stays tool-less; create a workspace to bind folders."`.

- [ ] **Step 1: Write the failing tests** (append)

```python
@pytest.mark.asyncio
async def test_create_rename_archive_unarchive_flow() -> None:
    from textual.widgets import Button, Checkbox, Input

    app = _build_test_app()
    registry = app.workspace_registry_service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")

        # Create with a free-form name.
        screen.query_one("#settings-workspace-create-name", Input).value = "Client X"
        screen.query_one("#settings-workspace-create", Button).press()
        await pilot.pause(0.3)
        created = [w for w in registry.list_workspaces() if w.name == "Client X"]
        assert len(created) == 1
        workspace_id = created[0].workspace_id

        # Select, rename.
        screen.query_one(f"#settings-workspace-row-{workspace_id}", Button).press()
        await pilot.pause(0.2)
        screen.query_one("#settings-workspace-rename-input", Input).value = "Client Y"
        screen.query_one("#settings-workspace-rename-apply", Button).press()
        await pilot.pause(0.3)
        assert registry.get_workspace(workspace_id).name == "Client Y"

        # Duplicate rename surfaces inline, not as a crash.
        screen.query_one("#settings-workspace-rename-input", Input).value = "Default"
        screen.query_one("#settings-workspace-rename-apply", Button).press()
        await pilot.pause(0.3)
        assert "already exists" in _visible_text(screen)

        # Set active.
        screen.query_one("#settings-workspace-set-active", Button).press()
        await pilot.pause(0.3)
        assert registry.get_active_workspace().workspace_id == workspace_id

        # Archive (confirm dialog), falls back to Default.
        screen.query_one("#settings-workspace-archive", Button).press()
        await pilot.pause(0.3)
        confirm = host.screen_stack[-1]
        confirm.query_one("#confirm-button", Button).press()
        await pilot.pause(0.4)
        assert registry.get_workspace(workspace_id).archived is True
        assert registry.get_active_workspace().workspace_id == "workspace-default"

        # Hidden until Show archived; then unarchive (no auto-activate).
        assert not screen.query(f"#settings-workspace-row-{workspace_id}")
        screen.query_one("#settings-workspaces-show-archived", Checkbox).value = True
        await pilot.pause(0.3)
        screen.query_one(f"#settings-workspace-row-{workspace_id}", Button).press()
        await pilot.pause(0.2)
        screen.query_one("#settings-workspace-unarchive", Button).press()
        await pilot.pause(0.3)
        record = registry.get_workspace(workspace_id)
        assert record.archived is False and record.active is False


@pytest.mark.asyncio
async def test_default_workspace_card_is_protected() -> None:
    from textual.widgets import Button

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-workspace-default", Button).press()
        await pilot.pause(0.2)

        assert not screen.query("#settings-workspace-rename-apply")
        assert not screen.query("#settings-workspace-archive")
        assert "stays tool-less" in _visible_text(screen)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "Tests/UI/test_settings_workspaces_category.py" -q`
Expected: FAIL — stub pane has none of the queried ids.

- [ ] **Step 3: Implement `_render_workspaces_detail` + handlers.**
  Rendering (shape; follow the settings screen's existing
  `settings-input-row` / `destination-section` classes):

```python
    def _render_workspaces_detail(self) -> ComposeResult:
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        yield Static("Workspace management", classes="destination-section")
        if registry is None:
            yield Static(
                "Workspace service is not ready. Restart Chatbook and retry.",
                id="settings-workspaces-result",
                classes="settings-status-row",
            )
            return
        show_archived = bool(getattr(self, "_settings_show_archived_workspaces", False))
        active = registry.get_active_workspace()
        active_id = active.workspace_id if active is not None else None
        with Horizontal(classes="settings-input-row"):
            yield Input(
                placeholder="New workspace name",
                id="settings-workspace-create-name",
                classes="settings-compact-input",
            )
            yield Button("Create", id="settings-workspace-create", compact=True)
        yield Checkbox(
            "Show archived", show_archived, id="settings-workspaces-show-archived"
        )
        with Vertical(id="settings-workspaces-list"):
            for record in registry.list_workspaces(include_archived=show_archived):
                marker = " (active)" if record.workspace_id == active_id else ""
                archived = " [archived]" if record.archived else ""
                folders = len(registry.list_folder_bindings(record.workspace_id)) if record.workspace_id != "workspace-default" else 0
                row = Button(
                    f"{record.name}{marker}{archived} - {folders} folders",
                    id=f"settings-workspace-row-{record.workspace_id}",
                    classes="settings-workspace-row",
                    compact=True,
                )
                yield row
        yield Static("", id="settings-workspaces-result", classes="settings-status-row")
        yield from self._render_workspace_card(registry, active_id)
```

  `_render_workspace_card` renders nothing when no selection; for the
  Default workspace it renders ONLY the protection Static (verbatim copy
  above); otherwise rename input+apply, Set active (omit when already
  active and render Static `"This workspace is active."` instead — never a
  disabled Button expected to explain itself), Archive OR Unarchive per
  `record.archived`, and (Task 10) the folder section. Handlers use
  `@on(Button.Pressed, "#settings-workspace-create")` etc.; every mutation
  is wrapped:

```python
        try:
            ...registry call...
        except WorkspaceRegistryServiceError as exc:
            self._set_settings_workspaces_result(str(exc))
            return
        self._refresh_settings_workspaces_pane()
```

  where `_set_settings_workspaces_result(text)` updates
  `#settings-workspaces-result` in place and
  `_refresh_settings_workspaces_pane()` triggers the category re-render via
  the screen's existing recompose path (mirror how Theme applies changes —
  find with `grep -n "def _render_theme" -A4 tldw_chatbook/UI/Screens/settings_screen.py`
  and follow whatever refresh idiom its action handlers use). Row press
  sets `self._settings_selected_workspace_id` then refreshes. Archive goes
  through `ConfirmationDialog` with an **async** confirm callback (it is
  awaited). Import `WorkspaceRegistryServiceError` at the settings module's
  workspace-import site.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest "Tests/UI/test_settings_workspaces_category.py" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_workspaces_category.py
git commit -m "feat(settings): workspace list, create, rename, active, archive/unarchive"
```

### Task 10: folder-bindings editor

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
  (extend `_render_workspace_card` + handlers)
- Test: `Tests/UI/test_settings_workspaces_category.py` (append)

**Interfaces:**
- Consumes: Task 2 service methods; Task 9 card scaffolding + result line.
- Produces ids: folder rows `#settings-workspace-folder-{binding_id}`
  (Static: `f"{locator} [{access}] {status}"`), per-row
  `#settings-workspace-folder-toggle-{binding_id}` (Button, label
  `"Allow write"` when ro / `"Read-only"` when rw) and
  `#settings-workspace-folder-remove-{binding_id}` (Button `"Remove"`);
  add row `#settings-workspace-folder-path` (Input, placeholder
  `"~/path/to/folder"`) + `#settings-workspace-folder-add` (Button `"Add folder"`).

- [ ] **Step 1: Write the failing test** (append)

```python
@pytest.mark.asyncio
async def test_folder_bindings_add_toggle_remove_and_inline_errors(tmp_path) -> None:
    from textual.widgets import Button, Input

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-folders", name="Folder WS")
    project = tmp_path / "project"
    project.mkdir()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-folders", Button).press()
        await pilot.pause(0.2)

        # Add (defaults to read-only).
        screen.query_one("#settings-workspace-folder-path", Input).value = str(project)
        screen.query_one("#settings-workspace-folder-add", Button).press()
        await pilot.pause(0.3)
        bindings = registry.list_folder_bindings("ws-folders")
        assert len(bindings) == 1
        binding = bindings[0]
        assert binding.metadata["access"] == "ro"
        assert "[ro]" in _visible_text(screen)

        # Toggle to rw.
        screen.query_one(
            f"#settings-workspace-folder-toggle-{binding.binding_id}", Button
        ).press()
        await pilot.pause(0.3)
        assert registry.list_folder_bindings("ws-folders")[0].metadata["access"] == "rw"

        # Invalid add explains inline.
        screen.query_one("#settings-workspace-folder-path", Input).value = str(
            tmp_path / "missing"
        )
        screen.query_one("#settings-workspace-folder-add", Button).press()
        await pilot.pause(0.3)
        assert "not a directory" in _visible_text(screen)

        # Remove.
        screen.query_one(
            f"#settings-workspace-folder-remove-{binding.binding_id}", Button
        ).press()
        await pilot.pause(0.3)
        assert registry.list_folder_bindings("ws-folders") == ()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "Tests/UI/test_settings_workspaces_category.py" -q -k folder`
Expected: FAIL — folder ids absent from the card.

- [ ] **Step 3: Implement.** Extend `_render_workspace_card` (non-Default
  branch) with a `"Folders (agent file-tool access)"` section Static, one
  row per `registry.list_folder_bindings(workspace_id)` (Static text
  `f"{binding.locator} [{binding.metadata.get('access', 'ro')}] {'ready' if str(binding.status).endswith('ready') else 'missing'}"`
  — compute from the recomputed status), the toggle/remove Buttons with the
  ids above, then the add row. Handlers:

```python
    @on(Button.Pressed, "#settings-workspace-folder-add")
    def _settings_workspace_add_folder(self, event: Button.Pressed) -> None:
        event.stop()
        registry = self.app_instance.workspace_registry_service
        raw = self.query_one("#settings-workspace-folder-path", Input).value
        try:
            registry.add_folder_binding(
                self._settings_selected_workspace_id, raw
            )
        except WorkspaceRegistryServiceError as exc:
            self._set_settings_workspaces_result(str(exc))
            return
        self._set_settings_workspaces_result("Folder added (read-only).")
        self._refresh_settings_workspaces_pane()
```

  Toggle and Remove handlers use `@on(Button.Pressed,
  ".settings-workspace-folder-toggle")` / `".settings-workspace-folder-remove"`
  class selectors with the binding id parsed from `event.button.id`
  (`event.button.id.rsplit("-", 1)` will split uuids — instead stash
  `event.button.binding_id = binding.binding_id` at compose time, exactly
  like the conversation browser stashes `conversation_id` on row buttons,
  and read `getattr(event.button, "binding_id")` in the handler). Toggle
  computes `allow_write = binding.metadata.get("access") != "rw"` from a
  fresh `list_folder_bindings` read, then
  `set_folder_binding_access(binding_id, allow_write=allow_write)`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest "Tests/UI/test_settings_workspaces_category.py" -q`
Expected: PASS (all category tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_workspaces_category.py
git commit -m "feat(settings): folder-bindings editor (add ro/rw toggle, remove, inline errors)"
```

### Task 11: freshness + cross-surface copy + PR2 assembly

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (refresh-on-resume;
  Overview recovery copy)
- Modify: `tldw_chatbook/Workspaces/display_state.py` +
  `tldw_chatbook/UI/Screens/chat_screen.py` (repoint the TASK-719 copy)
- Test: `Tests/UI/test_settings_workspaces_category.py` +
  `Tests/UI/test_settings_configuration_hub.py` (adjust pinned copy)

**Interfaces:**
- Consumes: everything prior.
- Produces: final copy strings (verbatim):
  - Overview recovery copy becomes:
    `"Sync status here is read-only - manage workspaces in Settings > Workspaces; switch in Console (Alt+W); run sync from the owning sync surfaces."`
  - `display_state.py` single-workspace recovery (currently
    `"Create one with the rail's New button or in Library > Details > Workspace."`)
    becomes:
    `"Create one with the rail's New button or in Settings > Workspaces."`
  - `chat_screen.py` zero-workspace toast gets the same new string.

- [ ] **Step 1: Write the failing tests.** Append to the category test
  file:

```python
@pytest.mark.asyncio
async def test_pane_refreshes_after_external_registry_change() -> None:
    from textual.widgets import Button

    app = _build_test_app()
    registry = app.workspace_registry_service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        assert not screen.query("#settings-workspace-row-ws-late")

        registry.create_workspace(workspace_id="ws-late", name="Late Arrival")
        screen._refresh_settings_workspaces_pane()
        await pilot.pause(0.3)
        assert screen.query_one("#settings-workspace-row-ws-late", Button)
```

  In the hub test file, update the assertions pinned by TASK-719 (find with
  `grep -n "Library > Details > Workspace" Tests/UI/test_settings_configuration_hub.py`)
  to the new Overview copy above.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "Tests/UI/test_settings_workspaces_category.py" -q -k refresh`
Expected: FAIL until the refresh helper is public-stable and the copy sites
are updated (hub tests fail on old copy).

- [ ] **Step 3: Implement.** (1) Call
  `self._refresh_settings_workspaces_pane()` from the screen's
  resume/refresh hook when the active category is WORKSPACES — find the
  hook with `grep -n "on_screen_resume\|def on_resume" tldw_chatbook/UI/Screens/settings_screen.py`
  and mirror how sync rows refresh there. (2) Apply the three verbatim copy
  replacements listed in **Interfaces** (grep each old string; they are
  unique). (3) Update the hub-test pins.

- [ ] **Step 4: Full PR2 verification**

Run: `.venv/bin/python -m pytest "Tests/UI/test_settings_workspaces_category.py" "Tests/UI/test_settings_configuration_hub.py" "Tests/Workspaces/" "Tests/UI/test_console_workspace_lifecycle.py" "Tests/UI/test_console_workspace_context_rail.py" -q`
Expected: PASS.

- [ ] **Step 5: Commit + push + PR**

```bash
git add -A tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/Workspaces/display_state.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/
git commit -m "feat(settings): workspaces pane freshness + management-home copy repoint"
git push -u origin feat/settings-workspaces-category
gh pr create --base dev --title "feat(settings): Workspaces management category (PR2 of 2)" --body "Implements spec 2026-07-26-settings-workspaces-category-design.md §4. Stacked on PR1 (folder roots); merge PR1 first, then this — one train."
```

### Task 12: live smoke (non-negotiable before merge)

**Files:** none (verification only).

- [ ] **Step 1:** Launch per `.claude/skills/verify/SKILL.md` on a scratch
  profile (`TLDW_CONFIG_PATH`, fresh `users_name`), from the PR2 worktree.
- [ ] **Step 2:** Drive: Settings ▸ Workspaces → create "Smoke WS" → bind a
  real temp folder → toggle rw → confirm the Console Details tray shows
  "File tools: 1 ready" for that workspace after switching to it (Alt+W).
- [ ] **Step 3:** With `[tools] read_file_enabled = true` in the scratch
  config and the local llama server, run an agent turn that reads a file
  inside the bound folder (allowed) and one outside (denied, error names
  the roots). Capture both transcripts into the session scratchpad.
- [ ] **Step 4:** Kill tmux server, delete the scratch profile dir.

## Self-review notes (already applied)

- Spec §2/§3/§4/§5/§8 each map to Tasks 1-2 / 3-6 / 8-11 / (5+2) / 7;
  §7 phasing = the PR boundaries; §6's suites appear in Tasks 4-6, 9-11
  steps; live proof = Task 12.
- Names used across tasks are consistent: `add_folder_binding`,
  `list_folder_bindings`, `remove_runtime_binding`,
  `set_folder_binding_access`, `BindingNotFound`, `validate_path_multi`,
  `allowed_file_roots`, `run_workspace`, `current_run_workspace_id`,
  `_registry_factory`, `BuiltinToolProvider(workspace_id=...)`,
  `_refresh_settings_workspaces_pane`, `_set_settings_workspaces_result`,
  `_settings_selected_workspace_id`.
- Known verify-at-implementation points are explicit: `list_runtime_bindings`
  filter signature (Task 2), `ListDirectoryTool` argument name (Task 5),
  the bridge call-site trace (Task 6), hub-test helper names (Task 8),
  Theme refresh idiom (Task 9), resume hook name (Task 11).
