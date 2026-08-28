# Watchlists Synchronous Loader Failure Toasts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two live synchronous Watchlists source-loader failures visible through fixed, markup-disabled error toasts and prevent new silent synchronous debug-swallow handlers.

**Architecture:** Keep the existing synchronous read and empty-list fallback behavior, adding only calls through the screen's established `_notify_watchlists` boundary. Extend the existing Watchlists failure-policy test module with mounted regressions and a handler-level AST inventory whose exact exemptions distinguish data loaders from lifecycle and preference-write fallbacks.

**Tech Stack:** Python 3.11+, Textual 8.x mounted test harness, pytest, Python `ast`, Ruff, Backlog.md CLI

---

## File Structure

- Modify `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`: add the two fixed error-toast calls inside the existing synchronous exception handlers; no new helper or service boundary.
- Modify `Tests/UI/test_watchlists_check_now_failure.py`: add mounted failure regressions, handler-level AST discovery, literal notification-policy checks, and the explicit exemption inventory.
- Modify `backlog/tasks/task-2340 - Silent-loader-failures-contradict-task-1090s-toast-premise.md`: record the implementation plan, completed acceptance criteria, focused verification evidence, and final implementation notes.
- Keep `Docs/superpowers/specs/2026-08-28-watchlists-synchronous-loader-failure-toasts-design.md` unchanged during implementation unless the approved behavior itself must change.

## ADR Check

**ADR required:** no

**ADR path:** N/A

**Reason:** The repair changes only error reporting inside an existing screen and test enforcement. It changes no storage, schema, ownership boundary, service contract, runtime, dependency, or long-lived UX structure.

## Pre-execution Checkpoint

The approved spec, this reviewed plan, and TASK-2340's `In Progress`
implementation-plan metadata must be committed before implementation begins.
That planning commit makes both documents visible to branch-diff review and
ensures the final clean-worktree gate is attainable.

### Task 1: Add red regressions for synchronous debug-swallow handlers

**Files:**
- Modify: `Tests/UI/test_watchlists_check_now_failure.py:21-32`
- Modify: `Tests/UI/test_watchlists_check_now_failure.py:355-490`

- [ ] **Step 1: Add the AST and scope imports used by the new contract and mounted tests**

Add module imports and the existing production scope type:

```python
import ast
from pathlib import Path

from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope
```

Keep the existing imports and use `Path` in `_screen_source` instead of its local import.

- [ ] **Step 2: Define the exact handler-level exemption inventory**

Place the mapping beside the existing async-loader structural contract. Keys are `(qualified_owner, literal_debug_message)` so one exempt method cannot absorb a new handler:

```python
SYNC_DEBUG_HANDLER_EXEMPTIONS = {
    (
        "_read_tree_data_snapshot.read_branch",
        "Failed to load watchlists tree branch: {}.",
    ): (
        "Snapshot failures are published and _load_tree_data emits one "
        "error toast per failure episode."
    ),
    (
        "_read_tree_data_snapshot",
        "Failed to load Watchlists tree membership.",
    ): (
        "Membership failure is published in the snapshot failure set and "
        "reported by _load_tree_data."
    ),
    (
        "_recompute_effective_layout",
        "Workbench not mounted yet; layout applies on compose.",
    ): (
        "The broad layout-request fallback is lifecycle/layout handling, "
        "not a data loader."
    ),
    (
        "_schedule_layout_persist",
        "Could not schedule preferred Watchlists layout persistence.",
    ): "Preference-write scheduling is outside the loader policy.",
    (
        "_persist_layout_worker",
        "Failed to persist preferred Watchlists pane layout.",
    ): "The background preference writer is outside the loader policy.",
    (
        "_persist_layout_worker",
        "Could not acknowledge preferred Watchlists layout write.",
    ): "Preference-write acknowledgement is outside the loader policy.",
    (
        "_restore_focus_after_swap",
        "No section tab to restore focus to after the swap.",
    ): "Focus restoration is lifecycle handling, not a data loader.",
}
```

- [ ] **Step 3: Add boundary-aware AST helpers**

Add helpers that scan only `WatchlistsCollectionsScreen` synchronous methods and their synchronous local functions. The scanner must record nested handlers separately and must not attribute calls across exception-handler, function, lambda, or class boundaries:

```python
_HANDLER_CALL_BOUNDARIES = (
    ast.ExceptHandler,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
)


def _root_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _root_name(node.value)
    if isinstance(node, ast.Call):
        return _root_name(node.func)
    return None


def _handler_calls(handler: ast.ExceptHandler):
    stack = list(handler.body)
    while stack:
        node = stack.pop()
        if isinstance(node, _HANDLER_CALL_BOUNDARIES):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _logger_debug_message(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "debug":
        return None
    if _root_name(call.func) != "logger" or not call.args:
        return None
    first = call.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None


def _is_safe_loader_notification(call: ast.Call) -> bool:
    if not (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr == "_notify_watchlists"
    ):
        return False
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    severity = keywords.get("severity")
    markup = keywords.get("markup")
    return (
        isinstance(severity, ast.Constant)
        and severity.value == "error"
        and isinstance(markup, ast.Constant)
        and markup.value is False
    )
```

Implement `_sync_debug_handlers()` with a small recursive scope walker:

1. Parse `_screen_source()` and select the `WatchlistsCollectionsScreen` class.
2. Start only from class-body `ast.FunctionDef` nodes; skip class-body `ast.AsyncFunctionDef` nodes.
3. Within a synchronous function, recurse through ordinary statements.
4. When a local `ast.FunctionDef` is found, scan it under `f"{owner}.{node.name}"`; skip async local functions and nested classes/lambdas.
5. When an `ast.Try` is found, inspect each handler's own calls with `_handler_calls`, yield one `(handler_key, has_safe_notification)` for each literal logger debug call, then recurse into the handler body separately so nested `try` handlers are independently discovered.
6. Traverse the `try` body, `orelse`, and `finalbody` exactly once.
7. Assert that every discovered debug handler has exactly one literal debug message and that handler keys are unique; a nonliteral or duplicate key must fail the guard rather than disappear.

- [ ] **Step 4: Add the structural policy test**

```python
def test_synchronous_debug_handlers_notify_or_have_exact_exemptions():
    discovered = dict(_sync_debug_handlers())
    assert discovered, "the synchronous debug-handler scan matched nothing"
    assert all(SYNC_DEBUG_HANDLER_EXEMPTIONS.values())

    silent = {key for key, safe in discovered.items() if not safe}
    exempt = set(SYNC_DEBUG_HANDLER_EXEMPTIONS)
    assert silent == exempt, (
        f"unexplained={sorted(silent - exempt)!r}; "
        f"stale_exemptions={sorted(exempt - silent)!r}"
    )
```

On the unfixed source, `silent - exempt` must contain both:

```python
("_load_source_rows_for_tree", "Failed to load tree source rows.")
("scoped_source_rows", "Failed to resolve scoped source rows.")
```

- [ ] **Step 5: Add mounted regressions for both live loaders**

Use one parameterized test with the production destination harness. Set the screen to Sources before installing a raising bundle-service double so ordinary mount work cannot contribute a toast, then clear the capture immediately before the direct call:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "expected_message"),
    [
        (
            "_load_source_rows_for_tree",
            "Failed to load sources for this watchlist.",
        ),
        (
            "scoped_source_rows",
            "Failed to resolve sources for the selected scope.",
        ),
    ],
)
async def test_synchronous_source_loader_failure_is_a_markup_safe_error(
    method_name, expected_message
):
    app = _build_test_app()
    notifications = []
    app.notify = lambda message, **kwargs: notifications.append(
        (str(message), kwargs)
    )
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, _pane = await _open_sources(pilot, host)
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=7)
        await pilot.pause()

        sentinel = "service exploded at [/bold]/private/watchlists.db"

        class RaisingBundleService:
            def list_source_rows(self, watchlist_id):
                raise RuntimeError(sentinel)

        screen._watchlist_bundle_service = lambda: RaisingBundleService()
        notifications.clear()
        rows = (
            screen._load_source_rows_for_tree(7)
            if method_name == "_load_source_rows_for_tree"
            else screen.scoped_source_rows()
        )

        assert rows == []
        assert notifications == [
            (expected_message, {"severity": "error", "markup": False})
        ]
        assert sentinel not in repr(notifications)
```

If Textual prevents replacing the bound helper on the mounted instance, use pytest's `monkeypatch` fixture against the screen instance; do not weaken the production-shaped mount or call the unbound class method.

- [ ] **Step 6: Run the new tests and prove the defect is detected**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_watchlists_check_now_failure.py \
  -k "synchronous_debug_handlers or synchronous_source_loader_failure"
```

Expected: FAIL. The structural failure names both synchronous handler keys, and both mounted cases fail because no toast was captured. If the tests pass before production changes, strengthen the oracle before continuing.

### Task 2: Add the minimal production error toasts

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:2195-2197`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:2251-2253`
- Test: `Tests/UI/test_watchlists_check_now_failure.py`

- [ ] **Step 1: Notify from `_load_source_rows_for_tree`'s existing handler**

Keep the debug traceback and empty fallback, inserting only the fixed notification:

```python
        except Exception:
            logger.opt(exception=True).debug("Failed to load tree source rows.")
            self._notify_watchlists(
                "Failed to load sources for this watchlist.",
                severity="error",
                markup=False,
            )
            return []
```

- [ ] **Step 2: Notify from `scoped_source_rows`'s existing handler**

```python
        except Exception:
            logger.opt(exception=True).debug("Failed to resolve scoped source rows.")
            self._notify_watchlists(
                "Failed to resolve sources for the selected scope.",
                severity="error",
                markup=False,
            )
            return []
```

Do not interpolate the exception, scope, source, watchlist, URL, or path into either toast.

- [ ] **Step 3: Run the new tests and verify they pass**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_watchlists_check_now_failure.py \
  -k "synchronous_debug_handlers or synchronous_source_loader_failure"
```

Expected: `3 passed` (one structural test and two parameter cases), with only pre-existing dependency or temporary-cleanup warnings if the environment emits them.

- [ ] **Step 4: Run the complete focused failure-policy module**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_watchlists_check_now_failure.py
```

Expected: all tests pass; baseline before implementation was `23 passed` with two dependency warnings.

- [ ] **Step 5: Commit the tested repair**

```bash
git add \
  Tests/UI/test_watchlists_check_now_failure.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py
git commit -m "fix: surface synchronous Watchlists loader failures"
```

### Task 3: Verify focused compatibility and close TASK-2340

**Files:**
- Modify: `backlog/tasks/task-2340 - Silent-loader-failures-contradict-task-1090s-toast-premise.md`
- Verify: `Tests/UI/test_watchlists_rail_counts_and_scope.py`
- Verify: `Tests/UI/test_watchlists_check_now_failure.py`
- Verify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`

- [ ] **Step 1: Run the directly affected scope behavior module**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_watchlists_rail_counts_and_scope.py
```

Expected: all tests pass. This proves the new notification side effect did not change successful tree/source scope behavior.

- [ ] **Step 2: Run modified-file Ruff lint and format checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  Tests/UI/test_watchlists_check_now_failure.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  Tests/UI/test_watchlists_check_now_failure.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py
```

Expected: both commands exit 0 without changing files. If format reports a diff, run the formatter only on those two files, inspect the diff, and rerun both checks.

- [ ] **Step 3: Check committed and working-tree patch integrity**

Run:

```bash
git diff --check origin/dev...HEAD
git diff --check
git diff --stat origin/dev...HEAD
git status --short
git diff origin/dev...HEAD -- \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/UI/test_watchlists_check_now_failure.py \
  "backlog/tasks/task-2340 - Silent-loader-failures-contradict-task-1090s-toast-premise.md" \
  Docs/superpowers/specs/2026-08-28-watchlists-synchronous-loader-failure-toasts-design.md \
  Docs/superpowers/plans/2026-08-28-watchlists-synchronous-loader-failure-toasts.md
```

Expected: both whitespace checks pass; `git status --short` has no untracked or
unstaged implementation files; and the committed branch diff contains no
changes outside the approved design, plan, tests, production handlers, and task
record.

- [ ] **Step 4: Record verified implementation evidence without closing the task**

Use Backlog.md CLI to check the five implemented behavior criteria and record
the evidence gathered so far. Leave criterion 6 unchecked and keep the task In
Progress until the final combined gate passes:

```bash
backlog task edit 2340 \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-ac 4 --check-ac 5 \
  --notes $'Added fixed markup-disabled error toasts to both live synchronous source loaders while preserving debug diagnostics and empty fallbacks. Added mounted regressions plus an exact handler-level AST contract with documented lifecycle/preference-write exemptions. ADR required: no; this is a bounded screen error-reporting repair. Focused verification is pending its final combined completion gate.' \
  --status "In Progress" --plain
```

Inspect the resulting task file and confirm criteria 1-5 are checked, criterion
6 is unchecked, `## Implementation Plan` and `## Implementation Notes` exist,
the ADR decision is recorded, and status remains `In Progress`.

- [ ] **Step 5: Commit the pre-completion evidence**

```bash
git add \
  "backlog/tasks/task-2340 - Silent-loader-failures-contradict-task-1090s-toast-premise.md"
git commit -m "docs: record TASK-2340 verification evidence"
```

- [ ] **Step 6: Run the final completion gate while the task is still In Progress**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_watchlists_check_now_failure.py \
  Tests/UI/test_watchlists_rail_counts_and_scope.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  Tests/UI/test_watchlists_check_now_failure.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  Tests/UI/test_watchlists_check_now_failure.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py
git diff --check origin/dev...HEAD
git diff --check
git status --short
```

Expected: both focused modules pass, Ruff and diff checks exit 0, and the worktree is clean. Do not claim completion from earlier output if this final gate fails.

- [ ] **Step 7: Mark TASK-2340 Done only after Step 6 passes**

```bash
backlog task edit 2340 \
  --check-ac 6 \
  --notes $'Added fixed markup-disabled error toasts to both live synchronous source loaders while preserving debug diagnostics and empty fallbacks. Added mounted regressions plus an exact handler-level AST contract with documented lifecycle/preference-write exemptions. ADR required: no; this is a bounded screen error-reporting repair. The focused failure-policy and source-scope modules, modified-file Ruff lint/format, both diff checks, and the final clean-worktree gate passed.' \
  --status Done --plain
git add \
  "backlog/tasks/task-2340 - Silent-loader-failures-contradict-task-1090s-toast-premise.md"
git commit -m "docs: complete TASK-2340"
git diff --check origin/dev...HEAD
git status --short
```

Expected: all six acceptance criteria are checked, status is `Done`, the
metadata-only completion commit has no whitespace errors, and the worktree is
clean.

- [ ] **Step 8: Review whether a reusable lesson was discovered**

This task currently applies existing lessons and is not expected to add a new one. Add to `backlog/docs/lessons-testing-evidence.md` only if implementation reveals a concrete, generalizable incident not already captured; do not invent a lesson merely to fill the checklist.
