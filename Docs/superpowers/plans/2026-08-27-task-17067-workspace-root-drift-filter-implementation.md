# TASK-17067 Workspace Root Drift Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make file-tool roots, change-review roots, and the workspace context note consume one ordered folder-binding validity iterator without changing caller gates, fallback behavior, or returned roots.

**Architecture:** Add one private generator in `workspace_file_roots.py` that validates a caller-prefiltered iterable and owns the exact path-free drift warning. Keep registry lookup, write-access filtering, change-review gates, sandbox inclusion, rendering, and whole-operation exception fallbacks in their current callers.

**Tech Stack:** Python 3.11+, pathlib, Loguru, pytest, Ruff

---

## File map

- Modify `tldw_chatbook/Tools/workspace_file_roots.py`: shared iterator and three consumer routes.
- Modify `Tests/Tools/test_workspace_file_roots.py`: red-first routing, ordering, and warning contracts.
- Modify `Docs/security/production-diagnostic-inventory.json`: regenerated row for the consolidated warning.
- Modify `backlog/tasks/task-17067 - Deduplicate-workspace-folder-root-drift-filtering-in-workspace_file_roots.md`: checked criteria and implementation notes after verification.

## Plan handoff prerequisite

Before Task 1, commit this approved plan and the task's Implementation Plan directly. Do not use `backlog task edit` for TASK-17067: Backlog CLI 1.44 corrupts five-digit task IDs.

```bash
git add Docs/superpowers/plans/2026-08-27-task-17067-workspace-root-drift-filter-implementation.md \
  "backlog/tasks/task-17067 - Deduplicate-workspace-folder-root-drift-filtering-in-workspace_file_roots.md"
git commit -m "docs: plan shared workspace root drift filter"
git status --short
```

Expected: the planning commit succeeds and the worktree is clean before implementation begins.

### Task 1: Pin the shared seam and operation order RED

**Files:**
- Modify: `Tests/Tools/test_workspace_file_roots.py`
- Reference: `Docs/superpowers/specs/2026-08-27-task-17067-workspace-root-drift-filter-design.md`

- [ ] **Step 1: Add a routing and write-prefilter regression**

Add a test that installs a not-yet-existing private seam with `raising=False`, records the binding IDs it receives, and exercises all three consumers. The write-mode call must hand the seam only the read-write binding; tracking and note calls must hand it both bindings.

```python
def test_all_consumers_share_validation_and_write_prefilters(
    tmp_path, monkeypatch
) -> None:
    registry = _registry(tmp_path)
    ro_root = tmp_path / "ro"
    rw_root = tmp_path / "rw"
    ro_root.mkdir()
    rw_root.mkdir()
    ro_binding = registry.add_folder_binding("ws-a", ro_root)
    rw_binding = registry.add_folder_binding("ws-a", rw_root, allow_write=True)
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "1")
    seen: list[tuple[str, ...]] = []

    def accept_all(bindings):
        materialized = tuple(bindings)
        seen.append(tuple(binding.binding_id for binding in materialized))
        for binding in materialized:
            yield binding, Path(binding.locator)

    monkeypatch.setattr(
        wfr, "_iter_valid_folder_bindings", accept_all, raising=False
    )
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    with wfr.run_workspace("ws-a"):
        assert wfr.allowed_file_roots(write=True, sandbox_root=sandbox) == (
            sandbox,
            rw_root,
        )
    assert set(wfr.folder_binding_roots("ws-a")) == {ro_root, rw_root}
    note = wfr.workspace_context_note("ws-a", launch_cwd=tmp_path, registry=registry)
    assert "ro" in note and "rw" in note
    assert seen == [
        (rw_binding.binding_id,),
        (ro_binding.binding_id, rw_binding.binding_id),
        (ro_binding.binding_id, rw_binding.binding_id),
    ]
```

- [ ] **Step 2: Add a change-review gate-order regression**

Use a seam that records calls. Prove the enabled path calls it once, then prove the global-disabled path returns before registry construction and the workspace-disabled path returns before listing or helper iteration.

```python
def test_change_review_gates_precede_binding_validation(tmp_path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    registry.add_folder_binding("ws-a", root)
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "1")
    calls = 0

    def accept_all(bindings):
        nonlocal calls
        calls += 1
        for binding in bindings:
            yield binding, Path(binding.locator)

    monkeypatch.setattr(
        wfr, "_iter_valid_folder_bindings", accept_all, raising=False
    )
    assert wfr.folder_binding_roots("ws-a") == (root,)
    assert calls == 1

    monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "0")
    monkeypatch.setattr(
        wfr,
        "_registry_factory",
        lambda: (_ for _ in ()).throw(AssertionError("registry touched")),
    )
    assert wfr.folder_binding_roots("ws-a") == ()
    assert calls == 1

    monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "1")
    registry.set_change_review_enabled("ws-a", False)
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    monkeypatch.setattr(
        registry,
        "list_folder_bindings",
        lambda _workspace_id: (_ for _ in ()).throw(
            AssertionError("bindings listed")
        ),
    )
    assert wfr.folder_binding_roots("ws-a") == ()
    assert calls == 1
```

- [ ] **Step 3: Run the routing tests and verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Tools/test_workspace_file_roots.py::test_all_consumers_share_validation_and_write_prefilters \
  Tests/Tools/test_workspace_file_roots.py::test_change_review_gates_precede_binding_validation -q
```

Expected: both tests fail because current consumers do not call `_iter_valid_folder_bindings`; the first test's `seen` list remains empty and the enabled gate path leaves `calls == 0`.

### Task 2: Pin the standardized warning and rejection order RED

**Files:**
- Modify: `Tests/Tools/test_workspace_file_roots.py`

- [ ] **Step 1: Add the exact-warning consumer regression**

Use a lightweight registry double so both an existing directory symlink and a non-canonical self-resolve path can be supplied without registry canonicalization. Parameterize the three consumers and two warned shapes. Capture real Loguru records with a temporary sink and assert the exact message once, with neither locator nor target present. Add `from types import SimpleNamespace` to the test module.

```python
_DRIFT_WARNING = (
    "Workspace folder binding excluded because its path no longer resolves "
    "to itself (symlink or mount drift)"
)


def _root_consumer_registry(locator: Path):
    binding = SimpleNamespace(
        binding_id="binding-1",
        locator=str(locator),
        metadata={"access": "rw"},
    )
    record = SimpleNamespace(name="Client A")

    class Registry:
        def get_workspace(self, _workspace_id):
            return record

        def list_folder_bindings(self, _workspace_id):
            return (binding,)

        def change_review_enabled(self, _workspace_id):
            return True

    return Registry()


def _invoke_root_consumer(consumer, registry, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "1")
    if consumer == "allowed":
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir(exist_ok=True)
        with wfr.run_workspace("ws-a"):
            wfr.allowed_file_roots(write=False, sandbox_root=sandbox)
        return
    if consumer == "tracking":
        wfr.folder_binding_roots("ws-a")
        return
    wfr.workspace_context_note("ws-a", launch_cwd=tmp_path, registry=registry)


@pytest.mark.parametrize("consumer", ("allowed", "tracking", "note"))
@pytest.mark.parametrize("shape", ("symlink", "resolve-mismatch"))
def test_consumers_share_exact_path_free_drift_warning(
    tmp_path, monkeypatch, consumer, shape
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    if shape == "symlink":
        locator = tmp_path / "linked-root"
        locator.symlink_to(target)
    else:
        locator = target / ".." / target.name
    registry = _root_consumer_registry(locator)
    records = []
    sink_id = wfr.logger.add(
        lambda message: records.append(message.record), level="WARNING"
    )
    try:
        _invoke_root_consumer(consumer, registry, tmp_path, monkeypatch)
    finally:
        wfr.logger.remove(sink_id)

    messages = [record["message"] for record in records]
    assert messages.count(_DRIFT_WARNING) == 1
    assert str(locator) not in "\n".join(messages)
    assert str(target) not in "\n".join(messages)
```

Keep `_invoke_root_consumer` as a small test-only helper in this file: bind `run_workspace("ws-a")` for `allowed`, set `TLDW_CHANGE_REVIEW_ENABLED=1` for `tracking`, and pass the registry explicitly for `note`.

- [ ] **Step 2: Add the silent missing/broken-symlink regression**

Call the private seam directly with binding-like objects for a missing directory and a broken symlink. Capture warning records and assert an empty result and no standardized warning.

```python
def test_missing_and_broken_symlink_bindings_remain_silent(tmp_path) -> None:
    missing = SimpleNamespace(locator=str(tmp_path / "missing"))
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "absent-target")
    records = []
    sink_id = wfr.logger.add(
        lambda message: records.append(message.record), level="WARNING"
    )
    try:
        assert list(
            wfr._iter_valid_folder_bindings(
                (missing, SimpleNamespace(locator=str(broken)))
            )
        ) == []
    finally:
        wfr.logger.remove(sink_id)
    assert _DRIFT_WARNING not in [record["message"] for record in records]
```

- [ ] **Step 3: Run the diagnostic tests and verify RED**

Run:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Tools/test_workspace_file_roots.py::test_consumers_share_exact_path_free_drift_warning \
  Tests/Tools/test_workspace_file_roots.py::test_missing_and_broken_symlink_bindings_remain_silent -q
```

Expected: failures because the private iterator does not exist, the note emits no drift warning, and the existing warnings include the raw locator with different wording.

### Task 3: Implement the minimum shared iterator GREEN

**Files:**
- Modify: `tldw_chatbook/Tools/workspace_file_roots.py:17-18,104-172,241-294,363-421`
- Test: `Tests/Tools/test_workspace_file_roots.py`

- [ ] **Step 1: Add the typed private generator**

Extend the typing imports without adding a runtime workspace-model import:

```python
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding
```

Add the shared generator before `workspace_context_note`:

```python
def _iter_valid_folder_bindings(
    bindings: Iterable[WorkspaceRuntimeBinding],
) -> Iterator[tuple[WorkspaceRuntimeBinding, Path]]:
    """Yield existing folder bindings whose stored path has not drifted."""
    for binding in bindings:
        folder = Path(binding.locator)
        if not folder.is_dir():
            continue
        if folder.is_symlink() or folder.resolve() != folder:
            logger.warning(
                "Workspace folder binding excluded because its path no longer "
                "resolves to itself (symlink or mount drift)"
            )
            continue
        yield binding, folder
```

Do not catch filesystem exceptions and do not add caching, rate limiting, result classes, or callbacks.

- [ ] **Step 2: Route the note and change-review roots through the generator**

Replace their duplicated `Path`/`is_dir`/`is_symlink`/`resolve` blocks with:

```python
for binding, folder in _iter_valid_folder_bindings(
    registry.list_folder_bindings(workspace_id)
):
    # existing caller-specific rendering or roots.append(folder)
```

Keep `folder_binding_roots`' global gate before `_registry_factory()` and its per-workspace gate before `list_folder_bindings()`.

- [ ] **Step 3: Route allowed roots through a prefiltered iterable**

Replace its duplicated block with:

```python
bindings = (
    binding
    for binding in registry.list_folder_bindings(workspace_id)
    if not write or str(binding.metadata.get("access", "ro")) == "rw"
)
for _binding, folder in _iter_valid_folder_bindings(bindings):
    roots.append(folder)
```

This order is load-bearing: the access predicate runs before the generator performs filesystem calls.

- [ ] **Step 4: Format the implementation, then run the new tests and verify GREEN**

Run:

```bash
../../.venv/bin/python -B -m ruff format \
  tldw_chatbook/Tools/workspace_file_roots.py \
  Tests/Tools/test_workspace_file_roots.py
../../.venv/bin/python -B -m pytest \
  Tests/Tools/test_workspace_file_roots.py::test_all_consumers_share_validation_and_write_prefilters \
  Tests/Tools/test_workspace_file_roots.py::test_change_review_gates_precede_binding_validation \
  Tests/Tools/test_workspace_file_roots.py::test_consumers_share_exact_path_free_drift_warning \
  Tests/Tools/test_workspace_file_roots.py::test_missing_and_broken_symlink_bindings_remain_silent -q
```

Expected: Ruff formats the implementation and all new cases pass.

- [ ] **Step 5: Run the complete focused behavior gate**

Run:

```bash
../../.venv/bin/python -B -m pytest Tests/Tools/test_workspace_file_roots.py -q
../../.venv/bin/python -B -m pytest \
  Tests/Workspaces/test_change_bounds.py::TestChangeReviewGating \
  Tests/Chat/test_change_turn_tracking.py::test_folder_binding_roots_includes_ro_and_never_sandbox -q
```

Expected: all tests pass. Record any post-pytest teardown/log-sink noise separately from the test exit status; do not represent warnings as failures or hide them.

- [ ] **Step 6: Commit the implementation slice**

```bash
git add tldw_chatbook/Tools/workspace_file_roots.py Tests/Tools/test_workspace_file_roots.py
git commit -m "refactor: share workspace root drift filtering"
```

### Task 4: Verify formatting, regenerate diagnostics, and close the task

**Files:**
- Modify: `Docs/security/production-diagnostic-inventory.json`
- Modify: `backlog/tasks/task-17067 - Deduplicate-workspace-folder-root-drift-filtering-in-workspace_file_roots.md`

- [ ] **Step 1: Verify formatting and rerun the focused behavior gate**

Verify the committed implementation is already formatted before the indentation-sensitive diagnostic inventory is regenerated:

```bash
../../.venv/bin/python -B -m ruff format --check \
  tldw_chatbook/Tools/workspace_file_roots.py \
  Tests/Tools/test_workspace_file_roots.py
../../.venv/bin/python -B -m pytest Tests/Tools/test_workspace_file_roots.py -q
../../.venv/bin/python -B -m pytest \
  Tests/Workspaces/test_change_bounds.py::TestChangeReviewGating \
  Tests/Chat/test_change_turn_tracking.py::test_folder_binding_roots_includes_ro_and_never_sandbox -q
```

Expected: Ruff reports both files already formatted and all focused behavior tests pass. Record the exact pass counts and any teardown-only logging noise for the task notes.

- [ ] **Step 2: Review the warning statement and prove inventory drift**

Confirm the only production diagnostic change in `workspace_file_roots.py` is the consolidation of two path-bearing warning statements into one constant path-free statement used by all three consumers. Then run:

```bash
../../.venv/bin/python -B scripts/check_persistent_diagnostic_inventory.py
```

Expected: nonzero with the exact inventory row difference and the tool's `--write` guidance. If it passes without a row change, stop and inspect the checker rather than generating blindly.

- [ ] **Step 3: Regenerate and verify the persistent diagnostic inventory**

```bash
../../.venv/bin/python -B scripts/check_persistent_diagnostic_inventory.py --write
../../.venv/bin/python -B scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python -B -m pytest Tests/Architecture/test_persistent_diagnostic_inventory.py -q
```

Expected: checker exits zero after regeneration and the focused architecture module passes.

- [ ] **Step 4: Run scoped static and diff checks**

```bash
../../.venv/bin/python -B -m ruff check \
  tldw_chatbook/Tools/workspace_file_roots.py \
  Tests/Tools/test_workspace_file_roots.py
../../.venv/bin/python -B -m ruff format --check \
  tldw_chatbook/Tools/workspace_file_roots.py \
  Tests/Tools/test_workspace_file_roots.py
../../.venv/bin/python -B -m compileall -q \
  tldw_chatbook/Tools/workspace_file_roots.py
git diff --check
git diff --check origin/dev
```

Expected: every command exits zero.

- [ ] **Step 5: Self-review the exact branch and working-tree diff**

```bash
git diff --stat origin/dev
git diff origin/dev -- \
  tldw_chatbook/Tools/workspace_file_roots.py \
  Tests/Tools/test_workspace_file_roots.py \
  Docs/security/production-diagnostic-inventory.json \
  "backlog/tasks/task-17067 - Deduplicate-workspace-folder-root-drift-filtering-in-workspace_file_roots.md"
```

Check every acceptance criterion, confirm no caller-specific fallback or gate moved, and verify no raw locator remains in the consolidated warning.

- [ ] **Step 6: Request final code review and address valid findings**

Use `superpowers:requesting-code-review` with base `origin/dev` and the current HEAD. Fix Critical or Important findings with focused tests, rerun the relevant checks, and commit any resulting source/test changes before continuing.

- [ ] **Step 7: Complete Backlog task hygiene**

After all evidence is green, use `apply_patch` to edit the five-digit task file directly; do not invoke Backlog CLI for this task. Check all five acceptance criteria, set `status: Done`, update `updated_date`, and add concise Implementation Notes that contain the actual observed evidence from the commands above:

- summarize the shared caller-prefiltered iterator and the preserved caller ordering/fallbacks;
- state that the warning is exact and path-free and the diagnostic inventory was regenerated;
- record the concrete pytest pass counts, Ruff format/check results, compile result, and inventory checks from this run;
- record `ADR required: no` and that ADR-028 remains applicable;
- document any valid review finding and fix, or state that review found no actionable issue.

Inspect the resulting task file and run `git status --short backlog/`. Confirm all criteria are checked, the Implementation Plan and Notes remain present, and no malformed `backlog/tasks/task-task- - .md` file exists. Add a lessons entry only if implementation exposed a genuinely reusable incident.

- [ ] **Step 8: Commit closeout metadata**

```bash
git add Docs/security/production-diagnostic-inventory.json \
  "backlog/tasks/task-17067 - Deduplicate-workspace-folder-root-drift-filtering-in-workspace_file_roots.md"
git commit -m "docs: close task 17067"
git status --short
```

Expected: commit succeeds and the worktree is clean.
