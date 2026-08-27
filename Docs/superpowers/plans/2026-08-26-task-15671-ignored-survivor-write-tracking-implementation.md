# TASK-15671 Ignored Survivor-Write Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure a successful post-turn child `write_file` call remains visible in Change Review even when `.gitignore` excludes its target, without breaking exact survivor/successor Git boundaries.

**Architecture:** Keep one small bridge-local child-change state per spawning turn and replay its normalized WRITE paths at each exact Git boundary while relevant. Extend the existing tracker snapshot seam to force-add eligible paths atomically, and use standard-library events to claim successor B before it starts and serialize competing survivor closes. ADR-092 defines the boundary contract; ADR-089 continues to own per-turn review presentation.

**Tech Stack:** Python 3.11+, `threading.Event`/`Lock`, existing `ConsoleAgentBridge`, `ChangeTurnTracker`, `ShadowRepo`, real Git-backed pytest fixtures, Backlog.md.

**Approved design:** `Docs/superpowers/specs/2026-08-26-task-15671-ignored-survivor-write-tracking-design.md`

**Architecture decisions:** `backlog/decisions/089-console-per-turn-change-review-ownership.md`, `backlog/decisions/092-console-live-child-write-path-boundaries.md`

**ADR required:** yes

**ADR path:** `backlog/decisions/092-console-live-child-write-path-boundaries.md`

**Reason:** Baseline snapshots gain an optional path input, supplied-SHA closure gains an index-priming side effect, and successor startup gains an explicit handoff contract.

---

## File map

- Modify `Tests/Chat/test_change_turn_tracking.py`: production-shaped ignored-write regression, tracker contract tests, and deterministic boundary-race tests.
- Modify `tldw_chatbook/Workspaces/change_tracking.py`: atomically force-add optional existing paths inside `ShadowRepo.snapshot`'s repository lock.
- Modify `tldw_chatbook/Workspaces/change_turn_tracker.py`: share touched-path eligibility, pass paths into B/fresh-E snapshots, and prime the index on supplied-SHA closure.
- Modify `tldw_chatbook/Chat/console_agent_bridge.py`: child-change state, path normalization, pending/live retention, pre-E capture, successor claim, and close completion.
- Modify TASK-15671, ADR-092, this plan, and the approved spec for governance and completion evidence.

No new module, dependency, schema, polling worker, or filesystem watcher is planned.

---

### Task 1: Pin the real post-turn ignored WRITE failure

**Files:**
- Modify: `Tests/Chat/test_change_turn_tracking.py`

- [x] **Step 1: Add one real-tool fence helper**

Import `contextlib`, `_FakeBuiltinGateForRegistry`, `STEP_TOOL_RESULT`, and the completed tool-outcome constant already used by Agent runtime tests. Add only this local helper:

```python
def _write_fence(path: Path, content: str) -> str:
    return (
        f"{FENCE_OPEN}\n"
        + json.dumps(
            {
                "name": "write_file",
                "arguments": {"file_path": str(path), "content": content},
            }
        )
        + "\n```"
    )
```

- [x] **Step 2: Write `test_post_turn_real_write_file_surfaces_a_new_ignored_path`**

The test must ignore but not pre-create `ignored-agent-output.txt`, enable the real `write_file` config seam, approve it through `_FakeBuiltinGateForRegistry(refuse=False)`, and bind `scratch_root=root` plus a `nullcontext` lease. Script a parent that spawns a gated child and returns; only after parent return may the child emit `_write_fence(target, sentinel)` and execute.

Assert all of these independent oracles:

1. the target is absent when parent `run_reply` returns;
2. the durable child steps contain a successful `write_file` result;
3. disk content equals the sentinel;
4. exactly one parent `subagent_post_turn` row lists the target; and
5. `repo.file_bytes(row["end_sha"], target.name)` equals the sentinel bytes.

- [x] **Step 3: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_change_turn_tracking.py::test_post_turn_real_write_file_surfaces_a_new_ignored_path
```

Expected: FAIL after the real tool succeeds because no survivor snapshot contains the ignored file. A refusal, validation error, absent disk file, or fixture exception is the wrong RED.

- [x] **Step 4: Mutation-check the oracle**

Temporarily weaken the review-row assertion and confirm the successful-result, disk, and snapshot-content assertions prevent a denied/failed call from satisfying the test. Restore the test immediately; do not commit the mutation.

- [x] **Step 5: Commit the RED regression**

```bash
git add Tests/Chat/test_change_turn_tracking.py
git commit -m "test: reproduce ignored survivor write omission"
```

---

### Task 2: Carry eligible WRITE paths through tracker B/E

**Files:**
- Modify: `Tests/Chat/test_change_turn_tracking.py`
- Modify: `tldw_chatbook/Workspaces/change_tracking.py`
- Modify: `tldw_chatbook/Workspaces/change_turn_tracker.py`

- [x] **Step 1: Write baseline-force RED**

Add `test_begin_turn_force_adds_an_ignored_path_into_the_baseline`. Create an ignored file before B, call `tracker.begin_turn([root], touched_paths=[str(target)])`, await B, and assert that exact baseline commit contains the bytes.

- [x] **Step 2: Verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_change_turn_tracking.py::test_begin_turn_force_adds_an_ignored_path_into_the_baseline
```

Expected: FAIL because `begin_turn` lacks `touched_paths`.

- [x] **Step 3: Add the minimum atomic shadow-snapshot input**

Change `ShadowRepo.snapshot` to accept keyword-only `force_paths: Sequence[str] = ()`. Inside its existing `_locked()` block, after initialization and before ordinary `add -A`, filter to paths that currently exist and invoke existing `git add -f -- ...` directly. Do not call public `force_add` from inside the non-reentrant lock and do not add a second lock.

```python
def snapshot(self, message: str, *, force_paths: Sequence[str] = ()) -> str:
    with self._locked():
        self.ensure_initialized()
        existing = [path for path in force_paths if (self.root / path).exists()]
        if existing:
            self._run("add", "-f", "--", *existing)
        # Existing scan, add -A, and commit flow follows.
```

- [x] **Step 4: Reuse one tracker eligibility helper**

Extract only the current within-root and max-file-size filtering into a private helper returning root-relative paths. Add optional `touched_paths=()` to `begin_turn`, freeze the caller sequence before launching its baseline thread, and call `snapshot("turn baseline", force_paths=eligible)`. Fresh `end_turn` calls the same atomic form instead of separate `force_add` then `snapshot`; record/disclosure behavior stays unchanged.

- [x] **Step 5: Run baseline and ordinary-turn controls GREEN**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_change_turn_tracking.py::test_begin_turn_force_adds_an_ignored_path_into_the_baseline Tests/Chat/test_change_turn_tracking.py::test_force_add_carveout_for_tool_touched_ignored_paths
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Workspaces/test_change_bounds.py -k 'tool_touched_oversized_path_is_not_force_added'
```

Expected: PASS.

- [x] **Step 6: Write supplied-SHA priming RED**

Add `test_supplied_successor_sha_primes_a_late_ignored_path_for_successor_e`: create a clean parent continuation; take successor B while the ignored target is absent; create the target; close the continuation with `touched_paths=[target]` and `end_shas=successor.baselines`; end the successor without passing the path. Assert the continuation retained supplied B while successor E contains the file.

- [x] **Step 7: Verify supplied-SHA RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_change_turn_tracking.py::test_supplied_successor_sha_primes_a_late_ignored_path_for_successor_e
```

Expected: FAIL because supplied-SHA `end_turn` performs no force-add.

- [x] **Step 8: Prime without rewriting supplied SHAs**

Compute eligible paths before the provided-SHA branch. Retain its exact diff behavior, but call `repo.force_add(eligible)` before every provided-branch `continue`, including `provided == baseline`, so successor E can consume the staged path.

- [x] **Step 9: Run tracker slice GREEN**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_change_turn_tracking.py -k 'force_add or supplied_successor_sha or begin_turn_force'
git diff --check
```

- [x] **Step 10: Commit tracker support**

```bash
git add Tests/Chat/test_change_turn_tracking.py tldw_chatbook/Workspaces/change_tracking.py tldw_chatbook/Workspaces/change_turn_tracker.py
git commit -m "fix: carry ignored write paths across change boundaries"
```

---

### Task 3: Retain child WRITE state through parent E

**Files:**
- Modify: `Tests/Chat/test_change_turn_tracking.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`

- [x] **Step 1: Add two lifecycle RED tests**

Add deterministic Event-based regressions:

- `test_pending_child_before_scope_entry_keeps_ignored_write_reviewable`: replace the bridge instance's `_child_run_scope` with a context manager that signals and waits before entering the original scope. Let parent return while the fleet handle is pending, then release the real child WRITE and assert its ignored file lands in the parent survivor row.
- `test_child_write_during_blocked_parent_e_is_retried_by_immediate_close`: block the real `turn end` snapshot, release a real child WRITE and let it exit while E is blocked, then release E and assert immediate continuation close captures the ignored path.

Use Events and real Git/file writes. No sleep is an ordering oracle.

- [x] **Step 2: Verify both tests RED**

Run both exact nodes. Each must fail because child paths do not reach survivor closure, not because a barrier or fixture timed out.

- [x] **Step 3: Add one mutable state dataclass**

```python
@dataclass
class _ChildChangeState:
    owner_key: str
    touched_paths: set[str] = field(default_factory=set)
    live_scopes: int = 0
```

Add one bridge map keyed by conversation then owner key. Reuse `_change_window_lock`; add no lock or module.

- [x] **Step 4: Project normalized sub-agent WRITE paths**

Create one state per `run_reply` using existing `primary_live_key`. For non-primary attributed steps, call `tool_touched_paths((step,))`. Resolve relative paths against captured `scratch_root` when present, preserve raw-relative semantics when absent, and add paths under `_change_window_lock`. Leave live-rail behavior untouched.

- [x] **Step 5: Register pending/live state without changing AgentService**

Bind state into `_child_run_scope` and a wrapper around the existing settle callback. Scope enter/exit updates `live_scopes` and the live-state map. Before parent E, register current state when `service.live_subagent_handles()` is non-empty. The settle wrapper removes it only when that service has no live handles, then calls the existing `_on_fleet_child_settled` path.

- [x] **Step 6: Capture state references before E**

Before E, copy pending/live state references and their paths. Combine child paths with existing primary `tool_touched_paths` for E. If the pre-E capture is non-empty, open the continuation after E with the state references even if every child exited during the snapshot, then preserve the existing immediate liveness recheck. At close, recalculate the path union and pass it to `end_turn`.

- [x] **Step 7: Run Task 1 and lifecycle tests GREEN**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_change_turn_tracking.py -k 'post_turn_real_write_file or pending_child_before_scope or blocked_parent_e or opening_a_window_whose_last_child'
```

- [x] **Step 8: Commit state support**

```bash
git add Tests/Chat/test_change_turn_tracking.py tldw_chatbook/Chat/console_agent_bridge.py
git commit -m "fix: retain survivor write paths through parent end"
```

---

### Task 4: Claim successor B and serialize window closure

**Files:**
- Modify: `Tests/Chat/test_change_turn_tracking.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`

- [x] **Step 1: Write the successor-handoff RED matrix**

Add deterministic tests for five contracts:

1. `test_successor_claim_uses_window_paths_after_live_state_cleanup`: remove final state from the live map behind a barrier, start the successor, and assert B still contains the ignored file from the claimed window reference.
2. `test_successor_b_waits_for_an_already_started_fresh_close`: block fresh close after it owns the window, start successor B on another thread, prove B has not started, release close, and assert old E/new B abut.
3. `test_successor_e_waits_for_close_time_index_priming`: let the child closer own supplied-SHA closure, race successor E, and assert the late ignored file appears only in successor E.
4. `test_claim_and_close_failures_release_waiters_without_breaking_runs`: inject one claim attachment failure and one close-time tracker failure; waiting threads must finish within the established timeout while reply/teardown stays successful.
5. `test_inherited_child_state_crosses_successor_e_and_second_window_without_backward_leakage`: keep an older child alive through successor B/E, let the successor spawn its own child, and assert the older state remains available to successor E and the second survivor window while the newer child's path is absent from the older window.

Use Events around real tracker/bridge seams and assert negative controls. Do not poll timing-sensitive booleans.

- [x] **Step 2: Verify matrix RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_change_turn_tracking.py -k 'successor_claim or successor_b_waits or successor_e_waits or claim_and_close_failures or inherited_child_state_crosses'
```

Expected: failures from absent claim/closing state, never deadlock or Git fixture setup.

- [x] **Step 3: Add standard-library boundary state**

Add `_SuccessorBoundaryClaim` with `threading.Event`, optional handle, and failure flag. Extend `_PostTurnChangeWindow` with a successor claim, `closing: bool`, and `close_done: threading.Event`. Use dataclass `field(default_factory=threading.Event)`; do not add a coordinator abstraction.

- [x] **Step 4: Replace post-B notification with pre-B claim/attach**

Before `begin_turn`, under `_change_window_lock`:

- if the old window is open and not closing, install a claim and copy its retained child states;
- if already closing, copy `close_done`, release the lock, and wait before B;
- if absent, continue normally.

Pass both live inherited paths and claimed-window paths to `begin_turn`. Attach the returned handle and release the claim Event in `finally`. On bounded wait failure, log and leave the successor untracked instead of creating overlapping history.

- [x] **Step 5: Make close first-owner/waiter safe**

Under `_change_window_lock`, the first caller marks the window closing; later callers copy `close_done` and wait outside the lock. The owner waits outside the lock for any successor claim, performs exact-SHA or fresh closure with the current state-path union, then removes only the same window and sets `close_done` in `finally`.

Never hold `_change_window_lock` during Event waits, Git, or DB I/O.

- [x] **Step 6: Preserve inherited state through successor E**

Retain inherited references captured at B even if settle removes them from the live map before E. Combine them with the successor's own state at E. A later window receives only states pending/live in the pre-E capture, so a later turn's child cannot leak backward.

- [x] **Step 7: Run successor tests GREEN**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_change_turn_tracking.py -k 'successor or survivor_window or inherited or counted_in_exactly_one_window or close_failure'
```

Expected: PASS, including pre-existing exact-boundary and counted-once tests.

- [x] **Step 8: Mutation-check handoff and waiter release**

Temporarily omit claimed-window paths from B and confirm the cleanup/claim barrier test fails. Temporarily omit one `close_done.set()` failure path and confirm the waiter test fails promptly. Restore both; do not commit mutations.

- [x] **Step 9: Commit handoff**

```bash
git add Tests/Chat/test_change_turn_tracking.py tldw_chatbook/Chat/console_agent_bridge.py
git commit -m "fix: serialize survivor and successor change boundaries"
```

---

### Task 5: Remove the gap and run focused verification

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `Tests/Chat/test_change_turn_tracking.py`

- [x] **Step 1: Remove the named gap outright**

Delete the `Known gap` paragraph from `_close_post_turn_change_window`; do not reword it. Update nearby lifecycle documentation only where signatures/state changed.

- [x] **Step 2: Run the complete focused module**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_change_turn_tracking.py
```

Expected: PASS.

- [x] **Step 3: Run adjacent contracts**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_child_run_scope_ordering.py Tests/Chat/test_fleet_settle_fanout.py Tests/Workspaces/test_change_tracking.py Tests/Workspaces/test_change_bounds.py
```

Expected: task-relevant tests PASS. Current `origin/dev` already fails
`test_raise_path_row_not_terminal_at_scope_exit_settles_via_run_child_finally`
and `test_drain_row_is_terminal_at_fire_time_on_the_raise_path_too` because
their monkeypatched `_persist` functions have the stale pre-`durable_handles`
signature. Record them as baseline-only unless this branch changes the
failure; do not repair those unrelated fixtures.

- [x] **Step 4: Run scoped static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Workspaces/change_tracking.py tldw_chatbook/Workspaces/change_turn_tracker.py Tests/Chat/test_change_turn_tracking.py
git diff --check
```

- [x] **Step 5: Review scope and simplicity**

```bash
git status --short
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Workspaces/change_tracking.py tldw_chatbook/Workspaces/change_turn_tracker.py Tests/Chat/test_change_turn_tracking.py
```

Confirm one state dataclass, one claim dataclass, standard-library synchronization only, no schema changes, and no unrelated edits.

- [x] **Step 6: Commit cleanup only if needed**

```bash
git add Tests/Chat/test_change_turn_tracking.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Workspaces/change_tracking.py tldw_chatbook/Workspaces/change_turn_tracker.py
git commit -m "test: cover ignored survivor boundary races"
```

Skip this commit when prior commits already contain the final code and tests.

---

### Task 6: Complete ADR, task, and evidence

**Files:**
- Modify: `backlog/decisions/092-console-live-child-write-path-boundaries.md`
- Modify: `backlog/tasks/task-15671 - The-gitignore-force-add-carve-out-does-not-extend-to-a-survivor-window.md`
- Modify: `Docs/superpowers/specs/2026-08-26-task-15671-ignored-survivor-write-tracking-design.md`
- Modify: `Docs/superpowers/plans/2026-08-26-task-15671-ignored-survivor-write-tracking-implementation.md`

- [x] **Step 1: Update governance and the superseded task hint**

Set ADR-092 to `Accepted`. Replace TASK-15671's sentence prescribing close-time persisted-step reads with the implemented bounded live-path projection and exact-boundary behavior. Keep acceptance criteria outcome-oriented.

- [x] **Step 2: Add implementation notes and evidence**

Record the bridge-local state, successor handoff, atomic B/E force-add, exact focused commands/results, known baseline-only stale `_persist` test-double failure, and spec/plan/ADR links. Add a lesson only if implementation uncovers a new reusable incident.

- [x] **Step 3: Check all acceptance criteria with Backlog.md**

```bash
backlog task edit 15671 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4
backlog task 15671 --plain
```

Expected: four checked ACs, ADR/plan/notes present, status still `In Progress` until final verification.

- [x] **Step 4: Use verification-before-completion**

Rerun every Task 5 command from fresh output and self-review every AC against code and evidence.

- [x] **Step 5: Mark Done only after all gates pass**

```bash
backlog task edit 15671 -s Done
backlog task 15671 --plain
git diff --check
```

- [x] **Step 6: Commit governance closure**

```bash
git add backlog/decisions/092-console-live-child-write-path-boundaries.md "backlog/tasks/task-15671 - The-gitignore-force-add-carve-out-does-not-extend-to-a-survivor-window.md" Docs/superpowers/specs/2026-08-26-task-15671-ignored-survivor-write-tracking-design.md Docs/superpowers/plans/2026-08-26-task-15671-ignored-survivor-write-tracking-implementation.md
git commit -m "docs: close TASK-15671 ignored survivor tracking"
```

---

## Final focused verification

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_change_turn_tracking.py Tests/Chat/test_child_run_scope_ordering.py Tests/Chat/test_fleet_settle_fanout.py Tests/Workspaces/test_change_tracking.py Tests/Workspaces/test_change_bounds.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Workspaces/change_tracking.py tldw_chatbook/Workspaces/change_turn_tracker.py Tests/Chat/test_change_turn_tracking.py
git diff --check
```

Expected: every task-relevant test passes. The documented latest-dev stale-signature failure may remain baseline-only only if its failure is unchanged and no TASK-15671 code reaches it differently.
