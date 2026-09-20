# Workspace Binding Exclusions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users mark exact files/folders under a workspace folder binding as fully invisible to the agent (tools refuse access; listings omit them), managed from Settings and the Console files modal.

**Architecture:** Exclusions are per-binding data in `workspace_runtime_bindings.metadata_json`, managed by `LocalWorkspaceRegistryService`. Enforcement rides the existing sensitive-path pipeline via ONE injection point: a pure `merge_sensitive_context()` helper folds the effective exclusion set into the per-call `SensitivePathContext`, so `is_sensitive_path`, `sensitive_exclusions_under` (worker serialization + Git pathspecs), and `refuses_new_directory_chain` enforce user exclusions without touching their call sites. Run admission (ADR-102) attaches a per-root exclusion provider closure with snapshot ∪ high-water-mark semantics: mid-run additions apply on the next tool call, removals next run.

**Tech Stack:** Python ≥3.12, SQLite (existing tables, no migration), Textual 8.x for UI.

**Spec:** `Docs/superpowers/specs/2026-09-20-workspace-binding-exclusions-design.md`

## Global Constraints

- No new dependencies; no DB schema migration (exclusions live in `metadata_json`).
- Refusal copy is byte-identical to the system denylist's (`"Refused: '<path>' is a protected path and cannot be <verb>"`) — never say "excluded" in model-facing text (opacity rule).
- Casefold comparisons only on the deny side (`_compare_key`); confinement stays case-sensitive.
- User-facing word is "Excluded". Internal names use `binding exclusion` / `user_exclusion`. Never "denylist", "SensitiveExclusion" (existing class), or "deny" for this feature.
- Cap: 200 exclusions per binding. Whole-root exclusion rejected. Non-existent paths allowed.
- UI: `$ds-*` token classes only (ADR-150); single-letter htop-style keybindings per ADR-031; reuse existing settings CSS classes; footer hints only for implemented actions.
- Tests: targeted runs only (repo policy) — never a full suite sweep unless the user asks.
- Every registry write goes through `save_runtime_binding` (single serialized writer); SQL stays parameterized (existing methods).
- ADR-172 must exist before code lands (Task 10 creates it; if tasks land before it, create the ADR first — see task ordering note).

---

### Task 1: Registry exclusion CRUD + `BindingExclusion` model

**Files:**
- Modify: `tldw_chatbook/Workspaces/models.py` (add `BindingExclusion`)
- Modify: `tldw_chatbook/Workspaces/registry_service.py` (add 3 methods + validation helpers, near `add_folder_binding` ~line 2447)
- Test: `Tests/Workspaces/test_folder_binding_exclusions.py` (new)

**Interfaces:**
- Consumes: `WorkspaceRuntimeBinding` (frozen dataclass, `metadata: Mapping`), `save_runtime_binding`, `get_runtime_binding`, `RuntimeBindingKind.LOCAL_FILESYSTEM`, `WorkspaceRegistryServiceError`.
- Produces:
  - `BindingExclusion(path: str, kind: str, added_at: str)` — frozen NamedTuple; `path` is binding-relative POSIX, `kind` ∈ `{"file", "directory"}`.
  - `add_binding_exclusion(self, workspace_id: str, binding_id: str, path: str) -> WorkspaceRuntimeBinding`
  - `remove_binding_exclusion(self, workspace_id: str, binding_id: str, path: str) -> WorkspaceRuntimeBinding`
  - `list_binding_exclusions(self, binding_id: str) -> tuple[BindingExclusion, ...]`
  - Module-level `binding_exclusion_entries(binding) -> tuple[BindingExclusion, ...]` (defensive metadata reader; other modules import THIS to read exclusions).

- [ ] **Step 1: Write failing tests**

```python
"""Registry CRUD for per-binding exclusions (spec §1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.models import BindingExclusion
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    binding_exclusion_entries,
)
from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError


@pytest.fixture()
def registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    db = WorkspaceDB(tmp_path / "workspaces.db")
    try:
        service = LocalWorkspaceRegistryService(db)
        service.create_workspace("ws-excl", name="Exclusions WS")
        yield service
    finally:
        db.close()


@pytest.fixture()
def binding(registry: LocalWorkspaceRegistryService, tmp_path: Path) -> str:
    root = tmp_path / "repo"
    (root / "secrets").mkdir(parents=True)
    (root / "secrets" / "key.pem").write_text("k")
    (root / ".env.prod").write_text("v")
    binding = registry.add_folder_binding("ws-excl", root, allow_write=True)
    return binding.binding_id


def test_add_and_list_roundtrip(registry, binding):
    updated = registry.add_binding_exclusion("ws-excl", binding, "secrets")
    entries = registry.list_binding_exclusions(binding)
    assert entries == (BindingExclusion(path="secrets", kind="directory", added_at=entries[0].added_at),)
    assert binding_exclusion_entries(updated)[0].path == "secrets"


def test_kind_inferred_from_disk(registry, binding):
    registry.add_binding_exclusion("ws-excl", binding, ".env.prod")
    assert registry.list_binding_exclusions(binding)[0].kind == "file"


def test_nonexistent_path_defaults_to_directory(registry, binding):
    registry.add_binding_exclusion("ws-excl", binding, "build/output")
    assert registry.list_binding_exclusions(binding)[0].kind == "directory"
    assert registry.list_binding_exclusions(binding)[0].path == "build/output"


def test_remove(registry, binding):
    registry.add_binding_exclusion("ws-excl", binding, "secrets")
    registry.remove_binding_exclusion("ws-excl", binding, "secrets")
    assert registry.list_binding_exclusions(binding) == ()


@pytest.mark.parametrize("bad", ["", ".", "./", "/etc/passwd", "~/.ssh", "../outside", "a/../../outside"])
def test_invalid_paths_rejected(registry, binding, bad):
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.add_binding_exclusion("ws-excl", binding, bad)


def test_symlink_escape_rejected(registry, binding, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "repo" / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.add_binding_exclusion("ws-excl", binding, "link/inner")


def test_duplicate_casefold_rejected(registry, binding):
    registry.add_binding_exclusion("ws-excl", binding, "secrets")
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.add_binding_exclusion("ws-excl", binding, "Secrets/")


def test_cap_enforced(registry, binding):
    for i in range(200):
        registry.add_binding_exclusion("ws-excl", binding, f"e{i}")
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.add_binding_exclusion("ws-excl", binding, "e200")


def test_wrong_workspace_rejected(registry, binding):
    registry.create_workspace("ws-other", name="Other")
    with pytest.raises(WorkspaceRegistryServiceError):
        registry.add_binding_exclusion("ws-other", binding, "secrets")
```

Note: check how `LocalWorkspaceRegistryService` is constructed in an existing test (e.g. `Tests/Workspaces/test_console_workspace_reconcile.py`) and copy that fixture pattern exactly — if it takes extra args, mirror them.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Workspaces/test_folder_binding_exclusions.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'BindingExclusion'`.

- [ ] **Step 3: Implement model**

In `tldw_chatbook/Workspaces/models.py`, after `WorkspaceRuntimeBinding`:

```python
class BindingExclusion(NamedTuple):
    """One user-managed path excluded from agent access under a binding.

    ``path`` is binding-relative POSIX text; ``kind`` is display metadata
    only — enforcement always matches the path itself and everything under
    it (a file has nothing under it).
    """

    path: str
    kind: str  # "file" | "directory"
    added_at: str
```

- [ ] **Step 4: Implement registry methods**

In `registry_service.py`, near `add_folder_binding`:

```python
_MAX_BINDING_EXCLUSIONS = 200


def binding_exclusion_entries(binding: Any) -> tuple[BindingExclusion, ...]:
    """Read a binding's exclusions defensively (unknown shapes dropped)."""
    raw = (binding.metadata or {}).get("exclusions")
    if not isinstance(raw, list):
        return ()
    entries: list[BindingExclusion] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        kind = item.get("kind", "directory")
        added_at = item.get("added_at", "")
        if isinstance(path, str) and path and kind in {"file", "directory"} and isinstance(added_at, str):
            entries.append(BindingExclusion(path=path, kind=kind, added_at=added_at))
    return tuple(entries)


def _validated_exclusion_relative(binding: Any, path: str) -> PurePosixPath:
    raw = str(path).strip()
    candidate = PurePosixPath(raw)
    if not raw or raw.startswith("~") or candidate.is_absolute() or ".." in candidate.parts:
        raise WorkspaceRegistryServiceError(
            "Exclusion path must be relative to the binding root."
        )
    root = Path(str(binding.locator))
    try:
        resolved_root = root.resolve(strict=True)
        target = (root / candidate).resolve(strict=False)
    except OSError as exc:
        raise WorkspaceRegistryServiceError(
            "Binding root is not resolvable."
        ) from exc
    if target != resolved_root and resolved_root not in target.parents:
        raise WorkspaceRegistryServiceError(
            "Exclusion path escapes the binding root."
        )
    if target == resolved_root:
        raise WorkspaceRegistryServiceError(
            "Exclude specific paths; remove the binding to exclude the whole root."
        )
    return candidate
```

Then the three methods on `LocalWorkspaceRegistryService` (import `BindingExclusion`, `binding_exclusion_entries` is module-level; `from dataclasses import replace` at top if absent):

```python
    def add_binding_exclusion(
        self, workspace_id: str, binding_id: str, path: str
    ) -> WorkspaceRuntimeBinding:
        """Mark one binding-relative path invisible to agent file tools."""
        binding = self._binding_for_exclusion_edit(workspace_id, binding_id)
        relative = _validated_exclusion_relative(binding, path)
        key = relative.as_posix().casefold()
        entries = list(binding_exclusion_entries(binding))
        if any(entry.path.casefold() == key for entry in entries):
            raise WorkspaceRegistryServiceError("Path is already excluded.")
        if len(entries) >= _MAX_BINDING_EXCLUSIONS:
            raise WorkspaceRegistryServiceError(
                f"Exclusion limit reached ({_MAX_BINDING_EXCLUSIONS})."
            )
        absolute = Path(str(binding.locator)) / relative
        kind = "directory" if not absolute.exists() or absolute.is_dir() else "file"
        entries.append(
            BindingExclusion(path=relative.as_posix(), kind=kind, added_at=self._now_factory())
        )
        return self._save_binding_exclusions(binding, entries)

    def remove_binding_exclusion(
        self, workspace_id: str, binding_id: str, path: str
    ) -> WorkspaceRuntimeBinding:
        """Remove one exclusion (effective for new runs, not the live one)."""
        binding = self._binding_for_exclusion_edit(workspace_id, binding_id)
        relative = PurePosixPath(str(path).strip())
        key = relative.as_posix().casefold()
        entries = [e for e in binding_exclusion_entries(binding) if e.path.casefold() != key]
        return self._save_binding_exclusions(binding, entries)

    def list_binding_exclusions(self, binding_id: str) -> tuple[BindingExclusion, ...]:
        binding = self.get_runtime_binding(binding_id)
        if binding is None:
            return ()
        return binding_exclusion_entries(binding)

    def _binding_for_exclusion_edit(
        self, workspace_id: str, binding_id: str
    ) -> WorkspaceRuntimeBinding:
        binding = self.get_runtime_binding(binding_id)
        if (
            binding is None
            or str(binding.workspace_id) != str(workspace_id)
            or str(binding.binding_kind) != str(RuntimeBindingKind.LOCAL_FILESYSTEM)
        ):
            raise WorkspaceRegistryServiceError(
                "Folder binding not found in this workspace."
            )
        return binding

    def _save_binding_exclusions(
        self, binding: WorkspaceRuntimeBinding, entries: list[BindingExclusion]
    ) -> WorkspaceRuntimeBinding:
        metadata = dict(binding.metadata)
        metadata["exclusions"] = [
            {"path": e.path, "kind": e.kind, "added_at": e.added_at} for e in entries
        ]
        return self.save_runtime_binding(replace(binding, metadata=metadata))
```

Check `PurePosixPath` is imported in registry_service (add `from pathlib import PurePosixPath` if missing).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest Tests/Workspaces/test_folder_binding_exclusions.py -q`
Expected: all PASS.

- [ ] **Step 6: Run neighbor registry tests**

Run: `python -m pytest Tests/Workspaces/test_folder_binding_validator.py Tests/Workspaces/test_console_workspace_reconcile.py -q`
Expected: PASS (no behavior change to existing paths).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Workspaces/models.py tldw_chatbook/Workspaces/registry_service.py Tests/Workspaces/test_folder_binding_exclusions.py
git commit -m "feat: per-binding exclusion CRUD in workspace registry"
```

---

### Task 2: `merge_sensitive_context` helper in sensitive_paths

**Files:**
- Modify: `tldw_chatbook/Utils/sensitive_paths.py` (add helper after `resolve_sensitive_context`, ~line 620)
- Test: `Tests/Utils/test_sensitive_paths_user_exclusions.py` (new; check whether `Tests/Utils/` exists and has a sensitive-paths test file — if `Tests/Utils/test_sensitive_paths.py` exists, add there instead)

**Interfaces:**
- Consumes: `SensitivePathContext` (NamedTuple: `files`, `dirs`, `db_paths`, `user_data_dir`, `direct_child_denied_dirs`), `_resolved`, `_compare_key`.
- Produces: `merge_sensitive_context(base: SensitivePathContext, *, extra_files: Iterable[Path] = (), extra_dirs: Iterable[Path] = ()) -> SensitivePathContext` — same fields with extras resolved, deduped by `_compare_key`, order preserved (base entries first).

- [ ] **Step 1: Write failing tests**

```python
"""User-exclusion folding into SensitivePathContext (spec §2 injection point)."""
from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Utils.sensitive_paths import (
    SensitiveExclusion,
    is_sensitive_path,
    merge_sensitive_context,
    resolve_sensitive_context,
    sensitive_exclusions_under,
)


def test_merged_dir_refuses_path_and_children(tmp_path: Path):
    excluded = tmp_path / "secrets"
    base = resolve_sensitive_context()
    merged = merge_sensitive_context(base, extra_dirs=(excluded,))
    assert is_sensitive_path(excluded / "key.pem", context=merged)
    assert is_sensitive_path(excluded, context=merged)


def test_merged_file_refuses_exact_path_only(tmp_path: Path):
    excluded = tmp_path / "notes.txt"
    merged = merge_sensitive_context(resolve_sensitive_context(), extra_files=(excluded,))
    assert is_sensitive_path(excluded, context=merged)
    assert not is_sensitive_path(tmp_path / "other.txt", context=merged)


def test_merge_feeds_exclusions_under_root(tmp_path: Path):
    root = tmp_path
    excluded_dir = root / "build"
    excluded_file = root / ".env.local"
    merged = merge_sensitive_context(
        resolve_sensitive_context(),
        extra_files=(excluded_file,),
        extra_dirs=(excluded_dir,),
    )
    entries = sensitive_exclusions_under(root, context=merged)
    assert SensitiveExclusion("subtree", "build") in entries
    assert SensitiveExclusion("file", ".env.local") in entries


def test_merge_dedupes_casefold(tmp_path: Path):
    excluded = tmp_path / "Secrets"
    merged = merge_sensitive_context(
        merge_sensitive_context(resolve_sensitive_context(), extra_dirs=(excluded,)),
        extra_dirs=(tmp_path / "secrets",),
    )
    assert merged.dirs.count(excluded.resolve()) == 1


def test_merge_preserves_base_entries():
    base = resolve_sensitive_context()
    merged = merge_sensitive_context(base)
    assert merged == base
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Utils/test_sensitive_paths_user_exclusions.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'merge_sensitive_context'`.

- [ ] **Step 3: Implement**

In `sensitive_paths.py` after `resolve_sensitive_context`:

```python
def merge_sensitive_context(
    base: SensitivePathContext,
    *,
    extra_files: Iterable[Path] = (),
    extra_dirs: Iterable[Path] = (),
) -> SensitivePathContext:
    """Fold per-workspace user exclusions into a per-call context snapshot.

    The workspace-exclusions injection point (spec 2026-09-20): extras join
    ``files``/``dirs`` resolved and deduped by the denylist's own
    ``_compare_key`` discipline, so ``is_sensitive_path``,
    ``sensitive_exclusions_under``, and ``refuses_new_directory_chain``
    enforce them with no further call-site changes. Unresolvable extras are
    kept only if they resolve; entries that fail resolution are dropped
    exactly like the base set's own unresolved entries.
    """
    files = list(base.files)
    dirs = list(base.dirs)
    seen_files = {_compare_key(p) for p in files}
    seen_dirs = {_compare_key(p) for p in dirs}
    for raw in extra_files:
        resolved = _resolved(str(raw))
        if resolved is None:
            continue
        key = _compare_key(resolved)
        if key not in seen_files:
            seen_files.add(key)
            files.append(resolved)
    for raw in extra_dirs:
        resolved = _resolved(str(raw))
        if resolved is None:
            continue
        key = _compare_key(resolved)
        if key not in seen_dirs and key not in seen_files:
            seen_dirs.add(key)
            dirs.append(resolved)
    return base._replace(files=tuple(files), dirs=tuple(dirs))
```

(`_resolved` on a non-existent path returns the lexically-resolved absolute path or `None` — mirror how the module treats unresolved entries; if `_resolved` uses `strict=True`, use `Path(raw).expanduser().absolute()` instead so non-existent exclusions survive — check `_resolved`'s definition first and pick the branch that keeps non-existent paths.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Utils/test_sensitive_paths_user_exclusions.py -q`
Expected: all PASS.

- [ ] **Step 5: Run sensitive-path neighbor tests**

Run: `python -m pytest Tests/Tools/test_local_tool_sensitive_paths.py Tests/Tools/test_git_tool_sensitive_paths.py -q`
Expected: PASS (helper is additive).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Utils/sensitive_paths.py Tests/Utils/test_sensitive_paths_user_exclusions.py
git commit -m "feat: merge_sensitive_context folds user exclusions into deny context"
```

---

### Task 3: Executor merged context (`user_exclusion_paths`)

**Files:**
- Modify: `tldw_chatbook/Tools/workspace_tool_executor.py` (`__init__` ~line 100 area; `_build_request` branches at lines ~365–460)
- Test: `Tests/Tools/test_workspace_tool_executor.py` (extend; follow its existing fixture style for tmp roots)

**Interfaces:**
- Consumes: `merge_sensitive_context` (Task 2), `resolve_sensitive_context`, `SensitivePathContext`.
- Produces: `WorkspaceToolExecutor(workspace_root, *, user_exclusion_paths: Callable[[], tuple[Path, ...]] | None = None)` — when set, every `_build_request` context is the merged one. Callers may pass a callable returning ABSOLUTE paths (files and/or dirs; non-existent paths go to the dirs bucket).

- [ ] **Step 1: Write failing tests** (append to `Tests/Tools/test_workspace_tool_executor.py`, reusing its existing tmp-root fixture names — read the file top first and match them)

```python
def test_user_exclusion_refuses_read(tmp_path):
    root = tmp_path
    (root / "secrets").mkdir()
    (root / "secrets" / "key.pem").write_text("k")
    executor = WorkspaceToolExecutor(root, user_exclusion_paths=lambda: (root / "secrets",))
    with pytest.raises(LocalToolError):
        executor.execute("fs_read", {"path": "secrets/key.pem"}, intent="read")


def test_user_exclusion_omits_from_listing(tmp_path):
    root = tmp_path
    (root / "secrets").mkdir()
    (root / "public.txt").write_text("p")
    executor = WorkspaceToolExecutor(root, user_exclusion_paths=lambda: (root / "secrets",))
    result = executor.execute("fs_list", {"path": "."}, intent="read")
    assert "secrets" not in result
    assert "public.txt" in result


def test_user_exclusion_refuses_write_and_stat(tmp_path):
    root = tmp_path
    (root / "notes.txt").write_text("n")
    executor = WorkspaceToolExecutor(root, user_exclusion_paths=lambda: (root / "notes.txt",))
    with pytest.raises(LocalToolError):
        executor.execute("fs_write", {"path": "notes.txt", "content": "x"}, intent="write")
    with pytest.raises(LocalToolError):
        executor.execute("stat_path", {"path": "notes.txt"}, intent="read")


def test_user_exclusion_provider_failure_keeps_base_context(tmp_path):
    root = tmp_path
    (root / "ok.txt").write_text("o")

    def broken():
        raise RuntimeError("registry unavailable")

    executor = WorkspaceToolExecutor(root, user_exclusion_paths=broken)
    result = executor.execute("fs_read", {"path": "ok.txt"}, intent="read")
    assert "o" in result
```

Import `LocalToolError` from `tldw_chatbook.Tools.local_tool_impls` in the test file if absent.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Tools/test_workspace_tool_executor.py -k user_exclusion -x -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'user_exclusion_paths'`.

- [ ] **Step 3: Implement**

In `__init__`, accept and store `user_exclusion_paths` (keep positional compatibility):

```python
        self._user_exclusion_paths = user_exclusion_paths
```

Add a private helper:

```python
    def _call_context(self) -> SensitivePathContext:
        """Per-call deny context: base denylist merged with user exclusions."""
        base = resolve_sensitive_context()
        if self._user_exclusion_paths is None:
            return base
        try:
            extra = tuple(self._user_exclusion_paths())
        except Exception:  # noqa: BLE001 -- provider failure must not widen access beyond base... 
            return base
        if not extra:
            return base
        return merge_sensitive_context(
            base,
            extra_files=tuple(p for p in extra if p.is_file()),
            extra_dirs=tuple(p for p in extra if not p.is_file()),
        )
```

Note on the `except` comment: a provider failure returns the BASE context — protection here equals the denylist; the high-water protection lives in the provider closure (Task 4), which never shrinks. Trim the trailing comment to fit the file's style.

Then in `_build_request`, replace each `context = resolve_sensitive_context()` (four sites: git ops, read ops, write ops, patch ops) with `context = self._call_context()`, and in the `stat_path` branch change `resolve_workspace_path(raw_path, chain.canonical_root, intent="read")` to pass `context=self._call_context()` as the keyword argument. Update the module import to include `merge_sensitive_context` and ensure `SensitivePathContext` is imported for the annotation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Tools/test_workspace_tool_executor.py -q`
Expected: all PASS (old and new).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/workspace_tool_executor.py Tests/Tools/test_workspace_tool_executor.py
git commit -m "feat: workspace executor enforces per-binding user exclusions"
```

---

### Task 4: Run-admission exclusion closure + provider wiring

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`capture_run_admitted_workspace_roots`, ~lines 1361–1452)
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (`RunAdmittedWorkspaceRoot` ~line 390; per-alias executor construction ~line 684; `_path_targets_without_authority` ~line 1139)
- Test: `Tests/Chat/test_console_workspace_exclusion_authority.py` (new)

**Interfaces:**
- Consumes: `binding_exclusion_entries` (Task 1), `WorkspaceToolExecutor(..., user_exclusion_paths=...)` (Task 3), `merge_sensitive_context` (Task 2).
- Produces:
  - `RunAdmittedWorkspaceRoot.exclusions_provider: Callable[[], tuple[Path, ...]] | None = None` (new field, default None; returns absolute paths).
  - `_exclusion_paths_provider(registry, binding_id, root, snapshot_rels) -> Callable[[], tuple[Path, ...]]` in console_chat_controller (module-private, testable).

- [ ] **Step 1: Write failing tests**

```python
"""Run-admitted exclusion authority: snapshot freeze + high-water semantics."""
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_chat_controller import (
    _exclusion_paths_provider,
    capture_run_admitted_workspace_roots,
)


class _FakeBinding:
    def __init__(self, workspace_id, binding_id, locator, exclusions=()):
        from tldw_chatbook.Workspaces.models import RuntimeBindingKind, RuntimeBindingStatus
        self.workspace_id = workspace_id
        self.binding_id = binding_id
        self.binding_kind = RuntimeBindingKind.LOCAL_FILESYSTEM
        self.status = RuntimeBindingStatus.READY
        self.locator = locator
        self.metadata = {
            "access": "rw",
            "exclusions": [{"path": p, "kind": "directory", "added_at": ""} for p in exclusions],
        }


class _FakeRegistry:
    def __init__(self, bindings):
        self._bindings = {b.binding_id: b for b in bindings}

    def list_runtime_bindings(self, workspace_id):
        return tuple(b for b in self._bindings.values() if b.workspace_id == workspace_id)

    def get_runtime_binding(self, binding_id):
        return self._bindings.get(binding_id)


class _FakeSession:
    def __init__(self, workspace_id):
        self.workspace_id = workspace_id


def test_snapshot_frozen_at_admission(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    registry = _FakeRegistry([_FakeBinding("ws", "folder-1", str(root), ("secrets",))])
    roots = capture_run_admitted_workspace_roots(session=_FakeSession("ws"), registry=registry)
    assert len(roots) == 1
    paths = roots[0].exclusions_provider()
    assert paths == ((root / "secrets").resolve(strict=False),)


def test_mid_run_addition_applies_immediately(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    binding = _FakeBinding("ws", "folder-1", str(root))
    registry = _FakeRegistry([binding])
    roots = capture_run_admitted_workspace_roots(session=_FakeSession("ws"), registry=registry)
    provider = roots[0].exclusions_provider
    assert provider() == ()
    binding.metadata["exclusions"] = [
        {"path": "secrets", "kind": "directory", "added_at": ""}
    ]
    assert (root / "secrets").resolve(strict=False) in provider()


def test_mid_run_removal_stays_excluded_high_water(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    binding = _FakeBinding("ws", "folder-1", str(root), ("secrets",))
    registry = _FakeRegistry([binding])
    roots = capture_run_admitted_workspace_roots(session=_FakeSession("ws"), registry=registry)
    provider = roots[0].exclusions_provider
    provider()  # observe live set once
    binding.metadata["exclusions"] = []
    assert (root / "secrets").resolve(strict=False) in provider()


def test_registry_failure_reuses_last_known(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    binding = _FakeBinding("ws", "folder-1", str(root), ("secrets",))
    registry = _FakeRegistry([binding])

    def boom(binding_id):
        raise RuntimeError("registry down")

    provider = _exclusion_paths_provider(registry, "folder-1", root, ("secrets",))
    provider()
    registry.get_runtime_binding = boom
    assert (root / "secrets").resolve(strict=False) in provider()
```

Note: `capture_run_admitted_workspace_roots` validates bindings via `_validate_project_instruction_binding`, which requires the locator to resolve `strict=True` with root == lexical (no symlink) — `tmp_path` on macOS may be under `/var` → `/private/var`; if the fake binding's locator fails root-identity capture, call `root.resolve()` when constructing `_FakeBinding` locator and assert against the resolved root. Adjust the fixture accordingly if the first test fails on that.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_console_workspace_exclusion_authority.py -x -q`
Expected: FAIL — `ImportError: cannot import name '_exclusion_paths_provider'`.

- [ ] **Step 3: Implement the closure in console_chat_controller**

Module-private helper (place above `capture_run_admitted_workspace_roots`):

```python
def _exclusion_paths_provider(
    registry: Any,
    binding_id: str,
    root: Path,
    snapshot_rels: tuple[str, ...],
) -> Callable[[], tuple[Path, ...]]:
    """Effective exclusion paths: admission snapshot union a high-water mark.

    Reads the binding's live exclusions on every call so a mid-run addition
    refuses the NEXT tool call (fail-closed shrink); the high-water mark
    keeps removals enforced for the rest of the run so run authority never
    expands mid-run (spec 2026-09-20, ADR-102 discipline).
    """
    holder: dict[str, frozenset[str]] = {"effective": frozenset(snapshot_rels)}

    def read() -> tuple[Path, ...]:
        try:
            binding = registry.get_runtime_binding(binding_id)
            from tldw_chatbook.Workspaces.registry_service import binding_exclusion_entries

            live = frozenset(entry.path for entry in binding_exclusion_entries(binding))
        except Exception:  # noqa: BLE001 -- degrade to last-known, never wider
            live = frozenset()
        holder["effective"] = holder["effective"] | live
        return tuple(
            (root / rel).resolve(strict=False)
            for rel in sorted(holder["effective"])
        )

    return read
```

In `capture_run_admitted_workspace_roots`, inside the per-selection loop before appending the root (after `binding_id` is computed):

```python
        from tldw_chatbook.Workspaces.registry_service import binding_exclusion_entries

        snapshot_rels = tuple(
            entry.path
            for entry in binding_exclusion_entries(selection.binding)
        )
        exclusions_provider = _exclusion_paths_provider(
            registry, binding_id, selection.root, snapshot_rels
        )
```

and pass `exclusions_provider=exclusions_provider` to the `RunAdmittedWorkspaceRoot(...)` constructor.

- [ ] **Step 4: Wire the provider**

In `Agents/local_tool_provider.py`:

1. Add the field to `RunAdmittedWorkspaceRoot`:

```python
    exclusions_provider: Callable[[], tuple[Path, ...]] | None = None
```

(Place after `guard`; keep `__post_init__` untouched — the None default needs no validation. `Callable` and `Path` are already imported.)

2. At the per-alias executor construction (~line 684, `executor = authority.workspace_executor or WorkspaceToolExecutor(authority.root)`), change to:

```python
                    executor = authority.workspace_executor or WorkspaceToolExecutor(
                        authority.root,
                        user_exclusion_paths=authority.exclusions_provider,
                    )
```

(If executors are also attached to authorities elsewhere — search `WorkspaceToolExecutor(` across the repo and pass `user_exclusion_paths=authority.exclusions_provider` wherever the authority is in scope; leave the shared scratch executor unchanged.)

3. In `_path_targets_without_authority`, so preflight refuses excluded targets before any approval card: at the top, after `root = Path(root).resolve()`, build the merged context and pass it into every `resolve_workspace_path(...)` call in that function:

```python
        extra: tuple[Path, ...] = ()
        if authority is not None and authority.exclusions_provider is not None:
            try:
                extra = tuple(authority.exclusions_provider())
            except Exception:  # noqa: BLE001
                extra = ()
        context: SensitivePathContext | None = (
            merge_sensitive_context(
                resolve_sensitive_context(),
                extra_files=tuple(p for p in extra if p.is_file()),
                extra_dirs=tuple(p for p in extra if not p.is_file()),
            )
            if extra
            else None
        )
```

and add `context=context` to each `resolve_workspace_path(args["path"], root, intent=...)` call in that function (read the full function body and update all of them — fs_read and the other path tools). Thread `authority` into the function: change the call site in `path_targets` to pass `authority=authority` as a keyword and add the parameter to `_path_targets_without_authority(self, tool_id, args, *, root, authority=None)`. Add imports: `merge_sensitive_context`, `resolve_sensitive_context`, `SensitivePathContext` from `tldw_chatbook.Utils.sensitive_paths`.

- [ ] **Step 5: Add a provider-level enforcement test** (append to the Task 4 test file)

```python
def test_provider_refuses_excluded_path_end_to_end(tmp_path: Path):
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "secrets").mkdir()
    (root / "secrets" / "key.pem").write_text("k")
    registry = _FakeRegistry([_FakeBinding("ws", "folder-1", str(root), ("secrets",))])
    roots = capture_run_admitted_workspace_roots(session=_FakeSession("ws"), registry=registry)
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider

    provider = LocalToolProvider(workspace_root=tmp_path, admitted_roots=roots)
    result = provider._invoke_detailed("local:fs_read", {"path": "secrets/key.pem", "root_alias": "folder-1"})
    assert result.result.ok is False
    assert "protected path" in result.result.error


def test_path_targets_preflight_refuses_excluded_target(tmp_path: Path):
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "secrets").mkdir()
    (root / "secrets" / "key.pem").write_text("k")
    registry = _FakeRegistry([_FakeBinding("ws", "folder-1", str(root), ("secrets",))])
    roots = capture_run_admitted_workspace_roots(session=_FakeSession("ws"), registry=registry)
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.Tools.local_tool_impls import LocalToolError

    provider = LocalToolProvider(workspace_root=tmp_path, admitted_roots=roots)
    with pytest.raises(LocalToolError):
        provider.path_targets("local:fs_read", {"path": "secrets/key.pem", "root_alias": "folder-1"})
```

Check `LocalToolProvider.__init__`'s required kwargs by reading its signature first (it may require `workspace_root` plus optional services); mirror what its existing tests pass (search `Tests/` for `LocalToolProvider(` usages and copy the minimal construction).

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest Tests/Chat/test_console_workspace_exclusion_authority.py -q`
Expected: all PASS.

- [ ] **Step 7: Run provider neighbor tests**

Run: `python -m pytest Tests/Chat/test_provider_readiness.py Tests/Tools/test_local_tool_sensitive_paths.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Agents/local_tool_provider.py Tests/Chat/test_console_workspace_exclusion_authority.py
git commit -m "feat: run-admitted exclusion authority with high-water semantics"
```

---

### Task 5: Family-1 coverage tests (fs/Git/vcli invisibility)

**Files:**
- Test only: `Tests/Tools/test_local_tool_user_exclusions.py` (new)

**Interfaces:**
- Consumes: everything from Tasks 1–4. No production changes expected — this task PINS the enforcement contract. If a test fails, fix the production seam it exposes (that's the point of the task).

- [ ] **Step 1: Write the tests**

```python
"""End-to-end family-1 user-exclusion enforcement (spec §2, family 1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Tools.local_tool_impls import LocalToolError
from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path
    (root / "secrets").mkdir()
    (root / "secrets" / "key.pem").write_text("SECRET")
    (root / "public.txt").write_text("hello")
    return root


def _executor(repo: Path) -> WorkspaceToolExecutor:
    return WorkspaceToolExecutor(
        repo, user_exclusion_paths=lambda: (repo / "secrets",)
    )


def test_fs_glob_omits_excluded(repo):
    result = _executor(repo).execute("fs_glob", {"pattern": "**/*"}, intent="read")
    assert "key.pem" not in result
    assert "public.txt" in result


def test_fs_grep_skips_excluded_content(repo):
    result = _executor(repo).execute(
        "fs_grep", {"pattern": "SECRET", "mode": "files"}, intent="read"
    )
    assert "key.pem" not in result


def test_fs_edit_refused_on_excluded(repo):
    with pytest.raises(LocalToolError):
        _executor(repo).execute(
            "fs_edit",
            {"path": "secrets/key.pem", "old_string": "S", "new_string": "X"},
            intent="write",
        )


def test_vcli_stat_refused_on_excluded(repo):
    from tldw_chatbook.Tools.virtual_cli_impls import VirtualCliTool  # verify exact class name first

    tool = VirtualCliTool(root=repo, workspace_executor=_executor(repo))
    with pytest.raises(Exception):
        tool.execute("stat", ["secrets/key.pem"])
```

Before finalizing: open `tldw_chatbook/Tools/virtual_cli_impls.py` and confirm the public class name and constructor signature (the file defines the virtual CLI wrapper around `_root`/`_workspace_executor`); adjust the test to the real names. If Virtual CLI is not directly constructible, drop that test and cover stat via `executor.execute("stat_path", ...)` (already covered in Task 3).

- [ ] **Step 2: Run and fix any exposed seam**

Run: `python -m pytest Tests/Tools/test_local_tool_user_exclusions.py -q`
Expected: PASS. If any test fails, the merged-context wiring missed a seam — find it (e.g. an enumeration site building its own context) and route it through the merged context, then re-run.

- [ ] **Step 3: Commit**

```bash
git add Tests/Tools/test_local_tool_user_exclusions.py
git commit -m "test: pin family-1 user-exclusion enforcement contract"
```

---

### Task 6: Builtin file-tool family enforcement (family 2)

**Files:**
- Modify: `tldw_chatbook/Tools/workspace_file_roots.py` (add `current_folder_binding_exclusions()`, near `allowed_file_roots` ~line 475)
- Modify: `tldw_chatbook/Tools/file_operation_tools.py` (context sites ~line 304 and ~line 427)
- Test: `Tests/Tools/test_file_operation_tools_exclusions.py` (new)

**Interfaces:**
- Consumes: `merge_sensitive_context` (Task 2), `binding_exclusion_entries` (Task 1), `_iter_valid_folder_bindings` (existing, ~line 118), the run-scoped workspace ContextVars already set by `run_workspace`.
- Produces: `current_folder_binding_exclusions() -> tuple[Path, ...]` — absolute exclusion paths from the current run's valid folder bindings; `()` when no workspace is bound.

- [ ] **Step 1: Write failing tests**

Read `Tests/Tools/` for existing `file_operation_tools` tests and copy their construction pattern for the tools (they instantiate e.g. `ListDirectoryTool`/`ReadFileTool` with a workspace context). Test file:

```python
"""Builtin file-tool family enforces per-binding exclusions (spec §2, family 2)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Tools.workspace_file_roots import (
    current_folder_binding_exclusions,
    run_workspace,
)


class _FakeBinding:
    def __init__(self, locator: str, exclusions=()):
        from tldw_chatbook.Workspaces.models import RuntimeBindingKind, RuntimeBindingStatus
        self.workspace_id = "ws"
        self.binding_id = "folder-fake"
        self.binding_kind = RuntimeBindingKind.LOCAL_FILESYSTEM
        self.status = RuntimeBindingStatus.READY
        self.locator = locator
        self.metadata = {
            "access": "rw",
            "exclusions": [{"path": p, "kind": "directory", "added_at": ""} for p in exclusions],
        }


def test_current_folder_binding_exclusions_reads_bound_bindings(tmp_path: Path):
    binding = _FakeBinding(str(tmp_path), ("secrets",))
    with run_workspace("ws", binding_authority=[binding]):
        # may require patching the registry lookup the module uses; if
        # _iter_valid_folder_bindings reads a registry, monkeypatch it to
        # yield `binding` -- read its body first and adapt this test.
        paths = current_folder_binding_exclusions()
    assert paths == () or (tmp_path / "secrets").resolve(strict=False) in paths


def test_builtin_read_tool_refuses_excluded(tmp_path: Path):
    # Follow the existing builtin-tool test harness in Tests/Tools/ for
    # constructing ReadFileTool with a bound workspace; assert reading
    # <root>/secrets/key.pem raises the protected-path refusal and listing
    # the root omits `secrets`.
    ...
```

**Important:** before writing the real tests, read `_iter_valid_folder_bindings` (workspace_file_roots.py ~line 118) and one existing builtin-tool test to learn how bindings reach these tools (registry seam vs. binding authority tuple). The assertion targets are fixed — the harness details come from those two reads. Do not proceed on assumptions here; this is the one task where the harness must be copied, not invented.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Tools/test_file_operation_tools_exclusions.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'current_folder_binding_exclusions'`.

- [ ] **Step 3: Implement the roots helper**

In `workspace_file_roots.py`:

```python
def current_folder_binding_exclusions() -> tuple[Path, ...]:
    """Absolute user-exclusion paths for the current run's folder bindings.

    Family-2 injection point (spec 2026-09-20): the builtin file tools fold
    these into their per-call sensitive context, mirroring what the
    workspace executor does for the local provider family. Returns ()
    outside a run workspace or when no binding carries exclusions.
    """
    paths: list[Path] = []
    for binding in _iter_valid_folder_bindings():  # adapt to its real signature
        from tldw_chatbook.Workspaces.registry_service import binding_exclusion_entries

        root = Path(str(binding.locator))
        for entry in binding_exclusion_entries(binding):
            paths.append((root / entry.path).resolve(strict=False))
    return tuple(paths)
```

Adapt the `_iter_valid_folder_bindings` call to its actual signature (it likely needs the write flag or yields from the bound authority) — the goal is: enumerate exactly the bindings `allowed_file_roots` would admit, read their exclusions.

- [ ] **Step 4: Merge at the two check sites in file_operation_tools.py**

At the per-invocation context site (~line 427, `sensitive_ctx = resolve_sensitive_context()`) and the bare `is_sensitive_path(path)` site (~line 304), replace with a helper defined once in that module:

```python
def _exclusion_aware_context() -> SensitivePathContext:
    from tldw_chatbook.Tools.workspace_file_roots import current_folder_binding_exclusions
    from tldw_chatbook.Utils.sensitive_paths import (
        merge_sensitive_context,
        resolve_sensitive_context,
    )

    base = resolve_sensitive_context()
    extra = current_folder_binding_exclusions()
    if not extra:
        return base
    return merge_sensitive_context(
        base,
        extra_files=tuple(p for p in extra if p.is_file()),
        extra_dirs=tuple(p for p in extra if not p.is_file()),
    )
```

Use `sensitive_ctx = _exclusion_aware_context()` at the shared-context site, and route the line-304 check through a context built the same way (if it sits inside a loop that already has a context in scope, pass that; otherwise call the helper once before the loop).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest Tests/Tools/test_file_operation_tools_exclusions.py Tests/Tools/test_local_tool_sensitive_paths.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Tools/workspace_file_roots.py tldw_chatbook/Tools/file_operation_tools.py Tests/Tools/test_file_operation_tools_exclusions.py
git commit -m "feat: builtin file-tool family enforces workspace exclusions"
```

---

### Task 7: Project-instruction skip for excluded paths

**Files:**
- Modify: `tldw_chatbook/Agents/project_instruction_runtime.py` (candidate activation/guard path; `snapshot_promotion_target` ~line 322 and the resolver's candidate enumeration)
- Test: `Tests/Chat/test_project_instruction_exclusions.py` (new; follow existing project-instruction test harness — search `Tests/` for `project_instruction` to find it)

**Interfaces:**
- Consumes: `binding_exclusion_entries` (Task 1).
- Produces: module-level `path_is_excluded(path: Path, excluded: frozenset[Path]) -> bool` in `project_instruction_runtime.py`; the ledger carries `excluded_dirs: frozenset[Path]` sourced from the binding at construction and skips any candidate source under it.

- [ ] **Step 1: Locate the candidate enumeration**

Run: `grep -n "AGENTS" tldw_chatbook/Agents/project_instruction_runtime.py tldw_chatbook/Chat/console_project_instructions.py` and read the surrounding functions. Identify every site where an `AGENTS.md`/`AGENTS.override.md` candidate path is read or activated (startup chain + lazy nested discovery + promotion).

- [ ] **Step 2: Write failing tests**

```python
"""AGENTS.md under an excluded path never activates (spec §2)."""
from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Agents.project_instruction_runtime import path_is_excluded


def test_path_is_excluded_matches_dir_and_children(tmp_path: Path):
    excluded = frozenset({tmp_path / "private"})
    assert path_is_excluded(tmp_path / "private", excluded)
    assert path_is_excluded(tmp_path / "private" / "AGENTS.md", excluded)
    assert not path_is_excluded(tmp_path / "public" / "AGENTS.md", excluded)
```

Plus one harness-level test using the existing project-instruction fixtures: a binding whose `metadata["exclusions"]` contains `private`, with `private/AGENTS.md` present, asserts the ledger activates no source from that directory (assert on whatever "activated sources" accessor the existing tests use).

- [ ] **Step 3: Implement**

```python
def path_is_excluded(path: Path, excluded: frozenset[Path]) -> bool:
    """True when ``path`` is or lies under a workspace-excluded directory."""
    if not excluded:
        return False
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return True  # unresolvable candidate paths fail closed
    for entry in excluded:
        try:
            candidate = entry.resolve(strict=False)
        except OSError:
            continue
        if resolved == candidate or candidate in resolved.parents:
            return True
    return False
```

Thread `excluded_dirs` from the binding metadata at ledger/resolver construction (where `binding_root`/`binding_id` are already captured), and gate every candidate-source site located in Step 1 with `if path_is_excluded(candidate_path, self._excluded_dirs): skip` (skip = not activated, not read — same as nonexistent).

- [ ] **Step 4: Run tests**

Run: `python -m pytest Tests/Chat/test_project_instruction_exclusions.py -q` plus the existing project-instruction test files found in Step 1's search.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/project_instruction_runtime.py Tests/Chat/test_project_instruction_exclusions.py
git commit -m "feat: project instructions skip workspace-excluded paths"
```

---

### Task 8: Settings surface

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (`_render_workspace_folder_bindings` ~line 19748; button handlers near the folder toggle/remove handlers ~line 24403–24460)
- Test: `Tests/UI/test_settings_workspace_exclusions.py` (new; copy the mounting pattern from an existing settings workspaces test — search `Tests/UI/` for `settings` + `workspace`)

**Interfaces:**
- Consumes: `registry.add_binding_exclusion` / `remove_binding_exclusion` / `list_folder_bindings` (Tasks 1).
- Produces: per-binding exclusion rows (path + kind + remove button) and a per-binding "Exclude path" input + button; handler methods `settings_workspace_exclusion_add` / `settings_workspace_exclusion_remove` (naming follows the file's existing handler conventions — check neighbors).

- [ ] **Step 1: Write failing tests** (structure per existing settings tests: mount the settings screen with a fake registry exposing the Task-1 methods, drive the buttons, assert the registry received the calls and the result static shows the outcome)

```python
"""Settings workspace-binding exclusion management (spec §3)."""
from __future__ import annotations

from pathlib import Path

import pytest

# Copy the screen-mounting fixture from the closest existing settings
# workspaces test (search Tests/UI/ for "settings-workspace-folder").
# The fake registry wraps a real LocalWorkspaceRegistryService over a
# tmp DB so Task-1 validation runs for real.


async def test_add_exclusion_via_settings(fake_settings, tmp_path: Path):
    # locate the per-binding exclusion input by id convention
    # "settings-workspace-excl-path-{binding_id}", type "secrets",
    # press "settings-workspace-excl-add-{binding_id}", assert
    # registry.list_binding_exclusions(binding_id)[0].path == "secrets"
    ...


async def test_remove_exclusion_via_settings(fake_settings):
    ...


async def test_invalid_path_shows_error(fake_settings):
    # typing "/abs/path" shows the validation message in
    # "#settings-workspaces-result" and adds nothing
    ...
```

Fill the bodies from the neighboring test file's exact async/pilot pattern (`await pilot.click("#...")` etc.).

- [ ] **Step 2: Render the exclusion UI**

Inside `_render_workspace_folder_bindings`'s per-binding loop, after the toggle/remove `Horizontal`, add:

```python
            exclusions = binding.metadata.get("exclusions") or []
            yield Static(
                f"  Excluded ({len(exclusions)}): agent cannot see these",
                classes="settings-detail-row",
            )
            for index, entry in enumerate(exclusions):
                with Horizontal(classes="settings-input-row"):
                    yield Static(
                        f"  {entry.get('path', '')} ({entry.get('kind', 'directory')})",
                        id=f"settings-workspace-excl-label-{binding.binding_id}-{index}",
                        classes="settings-detail-row",
                    )
                    unexclude_button = Button(
                        "Unexclude",
                        id=f"settings-workspace-excl-remove-{binding.binding_id}-{index}",
                        classes="settings-workspace-folder-remove",
                        compact=True,
                    )
                    unexclude_button.binding_id = binding.binding_id
                    unexclude_button.exclusion_path = str(entry.get("path", ""))
                    yield unexclude_button
            with Horizontal(classes="settings-input-row"):
                yield Input(
                    placeholder="path/to/exclude (relative to folder)",
                    id=f"settings-workspace-excl-path-{binding.binding_id}",
                    classes="settings-compact-input",
                )
                add_exclusion_button = Button(
                    "Exclude",
                    id=f"settings-workspace-excl-add-{binding.binding_id}",
                    compact=True,
                )
                add_exclusion_button.binding_id = binding.binding_id
                yield add_exclusion_button
```

(CSS: reuses `settings-detail-row` / `settings-input-row` / `settings-compact-input` / `settings-workspace-folder-remove` — no new classes, token governance stays green.)

- [ ] **Step 3: Add handlers**

Next to the existing folder toggle/remove handlers (~24403–24460), following their exact structure (registry lookup, call, `_set_settings_workspaces_result`, re-render):

```python
    async def settings_workspace_exclusion_add(self, event: Button.Pressed) -> None:
        binding_id = str(getattr(event.button, "binding_id", ""))
        registry = self._workspaces_registry()  # use the same accessor the folder-add handler uses
        input_id = f"settings-workspace-excl-path-{binding_id}"
        raw = str(self.query_one(f"#{input_id}", Input).value).strip()
        workspace_id = self._selected_workspace_id()  # same source the folder handlers use
        try:
            registry.add_binding_exclusion(workspace_id, binding_id, raw)
        except Exception as exc:  # noqa: BLE001 -- surface registry validation to the user
            self._set_settings_workspaces_result(f"Exclusion not added: {exc}")
            return
        self._set_settings_workspaces_result("Excluded for new and current runs' next tool call")
        self._refresh_settings_workspaces_detail()  # same re-render call the toggle handler uses

    async def settings_workspace_exclusion_remove(self, event: Button.Pressed) -> None:
        binding_id = str(getattr(event.button, "binding_id", ""))
        path = str(getattr(event.button, "exclusion_path", ""))
        registry = self._workspaces_registry()
        workspace_id = self._selected_workspace_id()
        try:
            registry.remove_binding_exclusion(workspace_id, binding_id, path)
        except Exception as exc:  # noqa: BLE001
            self._set_settings_workspaces_result(f"Exclusion not removed: {exc}")
            return
        self._set_settings_workspaces_result("Unexcluded — effective for new runs")
        self._refresh_settings_workspaces_detail()
```

Wire the button ids to these handlers the same way the existing `settings-workspace-folder-toggle-*` / `-remove-*` buttons are wired (find the `on_button_pressed` or `@on(Button.Pressed, "#...")` dispatch and add the two new id prefixes; copy the accessor method names from the folder handlers — do not invent new accessors).

- [ ] **Step 4: Run tests**

Run: `python -m pytest Tests/UI/test_settings_workspace_exclusions.py Tests/UI/test_design_token_governance.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_workspace_exclusions.py
git commit -m "feat: settings surface for workspace binding exclusions"
```

---

### Task 9: Console files-modal surface

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py` (`ConsoleWorkspaceController` — binding view models + registry calls)
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_files_modal.py` (`WorkspaceFilesBinding` ~line 111, `BINDINGS` ~line 244, action + rendering)
- Test: `Tests/Widgets/test_console_workspace_files_modal_exclusions.py` (new; follow the existing modal test mount pattern — check how existing `console_workspace_files_modal` tests construct the modal and service, if none exist follow `Tests/Widgets/` modal-test conventions)

**Interfaces:**
- Consumes: registry exclusion CRUD (Task 1); `WorkspaceFilesService` protocol (~line 45).
- Produces: `WorkspaceFilesBinding.exclusions: tuple[str, ...]`; service method `set_exclusion(binding_id: str, relative_path: str, excluded: bool) -> None` on the protocol and the controller implementation; modal action `toggle_exclusion` bound to a free single-letter key.

- [ ] **Step 1: Write failing tests** (service-level first, then widget-level)

Service test: controller with real registry over tmp DB — `set_exclusion(binding_id, "secrets", True)` then `list_binding_exclusions` shows it; `excluded=False` removes it; view models built for the modal expose `exclusions == ("secrets",)`.

Widget test: mount modal with a fake service whose binding has `exclusions=("secrets",)` and a directory page containing `secrets/`; assert the entry renders with an "excluded" marker (assert on the entry's label/content containing "excluded"); trigger the toggle action on a selected entry and assert the fake service received `set_exclusion`.

- [ ] **Step 2: Controller changes**

In `UI/Console_Modules/workspace.py`, wherever `WorkspaceFilesBinding` instances are built from `registry.list_folder_bindings(...)`, populate:

```python
exclusions=tuple(entry.path for entry in binding_exclusion_entries(binding)),
```

and implement `set_exclusion` delegating to `registry.add_binding_exclusion` / `remove_binding_exclusion` (same thread/refresh discipline the controller's other mutations use — copy it).

- [ ] **Step 3: Modal changes**

1. `WorkspaceFilesBinding` gains `exclusions: tuple[str, ...] = ()`.
2. `WorkspaceFilesService` protocol gains `def set_exclusion(self, binding_id: str, relative_path: str, excluded: bool) -> None: ...`.
3. In `BINDINGS` (~line 244), add one free single-letter binding (read the existing list first and pick an unused letter per ADR-031; `"x"` if free): `Binding("x", "toggle_exclusion", "Exclude/Unexclude")` — import `Binding` per the file's convention. Only add the footer hint if the modal renders one.
4. Implement:

```python
    def action_toggle_exclusion(self) -> None:
        binding = self._selected_binding()
        if binding is None:
            return
        selected = self._selected_entry()  # use the modal's real selected-entry accessor
        if selected is None:
            return
        relative = str(selected.relative_path)  # adapt to the entry model's real field
        excluded = relative in binding.exclusions or any(
            relative == e or relative.startswith(e + "/") for e in binding.exclusions
        )
        service = self._service  # the modal's existing service accessor
        service.set_exclusion(binding.binding_id, relative, not excluded)
        self._refresh_current_page()  # the modal's existing refresh path after mutations
```

Adapt `_selected_entry` / refresh calls to the modal's real accessors (read the file around its existing actions for the pattern). 

5. Rendering: where directory entries are labeled, when the entry's path equals or lies under a binding exclusion, append the marker `" [excluded]"` and apply the file's existing dim/muted class if one exists (grep the modal for an existing dim/muted class; if none, use the plain marker text only — no ad-hoc styles, ADR-150).

- [ ] **Step 4: Run tests**

Run: `python -m pytest Tests/Widgets/test_console_workspace_files_modal_exclusions.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/workspace.py tldw_chatbook/Widgets/Console/console_workspace_files_modal.py Tests/Widgets/test_console_workspace_files_modal_exclusions.py
git commit -m "feat: files-modal exclude/unexclude action with badge"
```

---

### Task 10: ADR-172, docs, backlog task

**Files:**
- Create: `backlog/decisions/172-workspace-binding-exclusions.md`
- Modify: `AGENTS.md` (Console file authority paragraph), `Docs/User_Guide/console/sessions-tabs-workspaces.md` (exclusions subsection)
- Backlog task via CLI

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write ADR-172** (match the format of `backlog/decisions/102-...md`)

```markdown
# ADR-172: Per-workspace binding exclusions for agent file access

Status: Accepted
Date: 2026-09-20
Design: [Workspace binding exclusions](../../Docs/superpowers/specs/2026-09-20-workspace-binding-exclusions-design.md)
Related: ADR-028, ADR-069, ADR-079, ADR-101, ADR-102

## Decision

A workspace folder binding may carry user-managed exclusions: exact
binding-relative paths (files or directories, non-existent allowed, capped
at 200, stored in `workspace_runtime_bindings.metadata_json["exclusions"]`)
that are fully invisible to agent file tools — reads, writes, edits,
patches, stat, and enumeration across BOTH agent-facing file-tool families
(the local provider with its one-shot pinned worker, and the builtin file
tools), plus Git pathspec exclusion and project-instruction activation.

Enforcement is a single injection point: user exclusions fold into the
per-call `SensitivePathContext` (`merge_sensitive_context`), so the
existing sensitive-path machinery (choke point, worker serialization,
pathspec rendering, directory-chain guard) enforces them with refusal copy
byte-identical to the system denylist — the model cannot distinguish user
exclusions or learn the paths exist. No worker-protocol change.

Run semantics follow ADR-102's discipline: admission snapshots the
exclusion set; a per-root provider reads the live set per tool call and
keeps a high-water mark, so mid-run additions refuse the next call and
removals take effect only for new runs. Registry read failures degrade to
the last-known effective set.

The user's own surfaces (Console file inspector, Settings) remain
direct-user authority (ADR-079): excluded entries stay visible, badged,
and un-excludable there.

## Alternatives considered

- Permission-store deny rules: cannot hide entries from listings; the
  pinned worker never consults the store.
- Negative bindings in root resolution: binding validation forbids
  nesting/overlap; would need a second implementation seam.
- Gitignore-style patterns: a pattern engine in the security path; exact
  literals cover the v1 need.
```

- [ ] **Step 2: Update AGENTS.md** — in the "Special Systems → Tool Calling" section's "Console file authority" paragraph, append one sentence: "Workspace folder bindings may carry per-binding exclusions (exact user-marked paths, fully invisible to agent tools — see ADR-172)."

- [ ] **Step 3: Update the user guide** — add an "Exclusions" subsection to `Docs/User_Guide/console/sessions-tabs-workspaces.md` covering: what excluding means (agent-invisible, user still sees it badged), both management surfaces, non-existent paths allowed, mid-run timing (additions next tool call; removals new runs).

- [ ] **Step 4: Create/complete the backlog task**

```bash
backlog task create "Workspace binding exclusions" -d "User-managed exact-path exclusions per workspace folder binding; agent-invisible enforcement via merged sensitive context" --ac "Registry CRUD validated,Agent tools refuse+omit excluded paths,Both UI surfaces,ADR-172"
backlog task edit <id> --plan "Per implementation plan Docs/superpowers/plans/2026-09-20-workspace-binding-exclusions.md"
```

Then at completion: check all ACs, add Implementation Notes, set status Done per backlog workflow (the executing agent does this at the end).

- [ ] **Step 5: Commit**

```bash
git add backlog/decisions/172-workspace-binding-exclusions.md AGENTS.md Docs/User_Guide/console/sessions-tabs-workspaces.md
git commit -m "docs: ADR-172 and guides for workspace binding exclusions"
```

---

## Self-Review (completed)

- **Spec coverage:** §1 storage/validation → Task 1; §2 injection point/executor/mid-run → Tasks 2–4; family-1 enumeration/git/stat → Tasks 3+5; family 2 → Task 6; project instructions → Task 7; §3 both surfaces → Tasks 8–9; opacity → byte-identical refusals via merged context (pinned in Task 4/5 tests); ADR/docs → Task 10. Mid-run removal/UI copy ("effective for new runs") → Tasks 4/8.
- **Type consistency:** `BindingExclusion(path, kind, added_at)` used identically in Tasks 1, 6, 7; `user_exclusion_paths: Callable[[], tuple[Path, ...]]` (executor) matches `exclusions_provider` (authority) in Tasks 3–4; `merge_sensitive_context(base, *, extra_files, extra_dirs)` identical in Tasks 2, 3, 4, 6.
- **Known adapt-points (explicit, not placeholders):** builtin-family test harness (Task 6 Step 1), settings handler accessors (Task 8 Step 3), modal accessors (Task 9 Step 3) — each names the exact neighboring code to copy from.
