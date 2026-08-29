# Mutation Loader Group Dispatch Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this plan task-by-task, and `superpowers:test-driven-development` for every production change.

**Goal:** Make all eleven audited schedule, notification, and artifact mutation refreshes participate in the same exclusive worker groups as user refreshes so stale overlapping loads cannot repaint the UI.

**Architecture:** Add one synchronous screen-local dispatch helper per affected loader and route every production scheduling call through it. Keep mutation work in its current worker group; the follow-up helper schedules the raw loader in the existing loader group. Extend the TASK-19559 AST inventory to ban inline awaits of the three raw loaders and to pin all eleven mutation owners to their expected helper.

**Tech Stack:** Python 3.11+, Textual 8.x workers, `asyncio`, `pytest`, `unittest.mock`, Python `ast`, Ruff.

**ADR required:** no

**ADR path:** N/A

**Reason:** This enforces TASK-19559's existing worker-group policy without changing an architectural boundary, storage contract, service API, dependency, or long-lived UX structure.

---

### Task 1: Make the architecture guard describe the full contract

**Files:**

- Modify: `Tests/Architecture/test_worker_exclusive_group_inventory.py`
- Reference: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`
- Reference: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`

**Step 1: Add a failing inline-loader detector test**

Add a path-scoped inventory for these raw loaders:

```python
INLINE_LOADER_TARGETS = {
    Path("tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py"): {
        "load_tasks"
    },
    Path("tldw_chatbook/UI/Screens/watchlists_collections_screen.py"): {
        "_load_notifications",
        "_load_briefings",
    },
}
```

Implement a small AST helper that reports `await self.<target>(...)` with line
and qualified owner. Add synthetic tests proving an awaited target is flagged,
a helper dispatch is clean, and unrelated awaited loaders are ignored. Add the
production sweep assertion.

**Step 2: Run the guard and confirm the expected red state**

Run:

```bash
pytest -q Tests/Architecture/test_worker_exclusive_group_inventory.py
```

Expected: FAIL, enumerating six `load_tasks`, two `_load_notifications`, and
three `_load_briefings` inline awaits in the production files.

**Step 3: Pin all eleven mutation owners to their helper**

Add an explicit owner-to-helper inventory for:

- schedule delete, save, run-now, enabled-state, bulk-delete, bulk-toggle;
- notification mark-read and dismiss;
- briefing generation, script cast, and audio synthesis.

Use AST ownership rather than line numbers. Assert each owner calls its expected
`self._request_*_refresh` helper exactly once. Add a synthetic mutation test
showing removal or routing to a different helper fails.

The production assertion remains red until the helpers and call sites land.

**Step 4: Commit the red guard**

```bash
git add Tests/Architecture/test_worker_exclusive_group_inventory.py
git commit -m "test: guard mutation loader dispatch groups"
```

---

### Task 2: Route every schedules refresh through one loader-group seam

**Files:**

- Modify: `Tests/UI/test_schedules_workbench.py`
- Modify: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`

**Step 1: Write the failing mounted concurrency regression**

Extend the recording scheduling service with controllable `list_tasks()`
responses. Mount the real `SchedulesWorkbench`, complete a delete mutation so
its refresh starts, dispatch a user refresh while that load is blocked, release
responses in stale-first order, and assert:

- `delete_reminder()` was called;
- the table and `_tasks` contain only the newest service snapshot;
- the older load cannot repaint after the newer one.

The test should use worker/task completion signals rather than a fixed sleep.

**Step 2: Run the focused test and verify it fails for the inline await**

```bash
pytest -q Tests/UI/test_schedules_workbench.py -k "mutation_refresh or delete_refresh"
```

Expected: FAIL because the mutation's inline `load_tasks()` is outside
`schedules-load-tasks` and can publish stale rows.

**Step 3: Add `_request_tasks_refresh()`**

Implement a synchronous helper that performs:

```python
self.run_worker(
    self.load_tasks,
    exclusive=True,
    group="schedules-load-tasks",
)
```

Route mount, sync event, conflict resolution, and other existing direct
`load_tasks` scheduling through the helper so it is the sole production seam.
Replace the six mutation-worker inline awaits with helper calls. Preserve all
existing mutation service calls, marked-row clearing, notifications, and error
handling.

**Step 4: Run schedule and architecture tests**

```bash
pytest -q Tests/UI/test_schedules_workbench.py Tests/Architecture/test_worker_exclusive_group_inventory.py
```

Expected: schedule tests PASS; the architecture production sweep remains red
only for Watchlists until Task 3.

**Step 5: Commit the schedules slice**

```bash
git add tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py Tests/UI/test_schedules_workbench.py
git commit -m "fix: group schedules mutation refreshes"
```

---

### Task 3: Route notification and artifact refreshes through their loader groups

**Files:**

- Modify: `Tests/UI/test_watchlists_destination_shell.py`
- Modify: `Tests/Watchlists/test_watchlists_artifacts_pane.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`

**Step 1: Write the failing notification race**

Use the mounted `WatchlistsContextHarness` and a gate-controlled notifications
controller. Trigger mark-read (or dismiss), allow its mutation refresh to enter
`load_rows`, trigger the pane's user refresh, and release the two snapshots out
of order. Assert the mutation call completed and only the newest notifications
snapshot is present in both screen and pane state.

Run:

```bash
pytest -q Tests/UI/test_watchlists_destination_shell.py -k "notification and mutation_refresh"
```

Expected: FAIL against the inline `_load_notifications()` mutation path.

**Step 2: Write the failing artifacts race**

Use the existing real Artifacts harness and database seam. Gate two
`_load_briefings()` acquisitions, complete a generation reconciliation refresh,
then dispatch the manual Artifacts refresh and release stale-first. Assert the
generated row remains recorded and only the newest briefing projection reaches
screen and pane state.

Also add focused assertions that generation forwards
`select_briefing_id=generated_id`, while cast and audio preserve their existing
`is_attached` gates.

Run:

```bash
pytest -q Tests/Watchlists/test_watchlists_artifacts_pane.py -k "mutation_refresh or briefing_refresh_dispatch"
```

Expected: FAIL against the three inline `_load_briefings()` paths.

**Step 3: Add the two Watchlists dispatch helpers**

Implement:

- `_request_notifications_refresh()` using `exclusive=True` and
  `group="wc_notifications"`;
- `_request_briefings_refresh(*, select_briefing_id=None)` using
  `exclusive=True` and `group="wl-briefings-load"`.

Route active-section loading, explicit refresh events, briefing/script
selection reloads, and all other existing direct schedules of these raw
loaders through the helpers. Replace notification mark-read/dismiss inline
awaits and artifact generation/cast/audio inline awaits with helper calls.

Keep attachment policy at the existing call sites: generation always requests
its refresh; cast and audio do so only when attached. Do not add a helper-level
attachment check.

**Step 4: Run Watchlists and architecture tests**

```bash
pytest -q \
  Tests/UI/test_watchlists_destination_shell.py \
  Tests/Watchlists/test_watchlists_artifacts_pane.py \
  Tests/Architecture/test_worker_exclusive_group_inventory.py
```

Expected: PASS. The AST sweep finds zero raw inline awaits and all eleven
mutation owners call the correct helper.

**Step 5: Commit the Watchlists slice**

```bash
git add \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/UI/test_watchlists_destination_shell.py \
  Tests/Watchlists/test_watchlists_artifacts_pane.py
git commit -m "fix: group watchlists mutation refreshes"
```

---

### Task 4: Verify, document, and prepare review

**Files:**

- Modify: `backlog/tasks/task-19870 - Mutation-workers-refresh-inline-instead-of-dispatching-into-the-loader-group.md`
- Modify if a general lesson emerged: `backlog/docs/lessons-testing-evidence.md`

**Step 1: Run the complete targeted regression gate**

```bash
pytest -q \
  Tests/Architecture/test_worker_exclusive_group_inventory.py \
  Tests/UI/test_schedules_workbench.py \
  Tests/UI/test_watchlists_destination_shell.py \
  Tests/Watchlists/test_watchlists_artifacts_pane.py
```

Do not run the full repository suite unless separately requested.

**Step 2: Run static checks on every modified Python file**

```bash
ruff check \
  tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Architecture/test_worker_exclusive_group_inventory.py \
  Tests/UI/test_schedules_workbench.py \
  Tests/UI/test_watchlists_destination_shell.py \
  Tests/Watchlists/test_watchlists_artifacts_pane.py

ruff format --check \
  tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Architecture/test_worker_exclusive_group_inventory.py \
  Tests/UI/test_schedules_workbench.py \
  Tests/UI/test_watchlists_destination_shell.py \
  Tests/Watchlists/test_watchlists_artifacts_pane.py

git diff --check
git diff --check origin/dev...HEAD
```

**Step 3: Mutation-check the structural guard**

Temporarily restore one production helper call to its raw inline await, run the
architecture test and confirm it fails at the correct owner, then restore the
working tree and rerun the guard green. Do not commit the temporary mutation.

**Step 4: Complete backlog hygiene**

Check all acceptance criteria, add concise implementation notes including the
ADR decision and targeted verification evidence, and set TASK-19870 to Done
only after all gates pass. Add a lessons entry only if this work produces a
new, generalizable incident beyond the lesson already documented by TASK-19559.

**Step 5: Review the diff and commit closeout**

```bash
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Architecture/test_worker_exclusive_group_inventory.py \
  Tests/UI/test_schedules_workbench.py \
  Tests/UI/test_watchlists_destination_shell.py \
  Tests/Watchlists/test_watchlists_artifacts_pane.py

git add \
  'backlog/tasks/task-19870 - Mutation-workers-refresh-inline-instead-of-dispatching-into-the-loader-group.md'
git commit -m "docs: complete TASK-19870"
```

After the implementation diff passes review, push the branch and open the PR
with the focused evidence and explicit note that no full-suite run was
requested.
