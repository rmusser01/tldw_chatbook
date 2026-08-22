# TASK-3070.8 Console Fleet and Wake Lifecycle Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a new DOM-free `ConsoleFleetLifecycleController` the sole owner of Console completion handoff, wake coordination, unseen-marker policy, teardown accounting, and survivor-tick lifecycle without changing delivery, cancellation, repaint, or persistence behavior.

**Architecture:** Move the exact 16-method family from `ChatScreen` into `tldw_chatbook/UI/Console_Modules/fleet.py`, attach it as `screen._fleet`, and wire only keyword-only late-bound callbacks. Keep Textual lifecycle sequencing and pixels on the screen, but move marker/view-clear decisions behind a plain-value controller API and route Workspace's existing fleet adapters directly to `_fleet`.

**Tech Stack:** Python 3.11+, Textual 8, pytest/pytest-asyncio, Ruff, stdlib AST checks, governed persistent-diagnostic inventory.

---

**ADR required:** no

**ADR path:** N/A

**Reason:** This plan directly implements the behavior-preserving ownership boundary already approved in `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md` and `DESIGN.md` section 7. It changes no schema, persistence policy, dependency, security boundary, or public interface.

## Constraints and evidence rules

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-3070-8-console-fleet-lifecycle` on `codex/task-3070-8-console-fleet-lifecycle`.
- Treat `Docs/superpowers/specs/2026-08-21-task-3070-8-console-fleet-lifecycle-design.md` as the approved contract.
- Do not add a screen handle, sibling controller object, generic dependency bag, base class, event bus, method shim, compatibility descriptor, setting, DOM id, CSS rule, or user-visible copy.
- Preserve the historical Wave 6 `raw_lines=401` oracle at its recorded `POST_IMAGE_IMPLEMENTATION_BASE` and the initial reviewed `0a8e288` oracle at 421 definition lines / 20,349 screen lines / 652 direct methods / 19,928 and 636 projections. Preserve the immutable Task 0 `d4f3f977` oracle for the same 421 definition lines, 20,428 total screen lines, 653 direct methods, and post-extraction ceilings of 20,007/637. Use the separately frozen final-rebase `02cd80b3` source at 20,486 lines / 656 methods, with the unchanged 16-method/421-line fleet AST, for delivery ceilings of 20,065/640.
- The two controller state fields are private controller details: `_console_fleet_survivor_timer` and `_console_fleet_unseen_cache`. Tests that inspect them move to `screen._fleet`; no screen shadow/proxy is allowed.
- Preserve exact callback late binding. A test monkeypatch after construction must affect the next controller call.
- Keep first-chat consumption before teardown notice and synchronous wake claiming; keep wake claiming before every timer/worker/view-clear.
- Mount claiming reads marks through the uncached service callback. The revision cache is only for tab/browser projection.
- Keep the `pilot.press` hidden-screen composer oracle and plain `threading.Thread` delivery-freshness oracle production-shaped.
- Run only tests related to changed fleet/wake functionality. Do not run the local full suite.
- Use `apply_patch` for source/test/doc edits. Restore every temporary mutation immediately and prove exact candidate-byte restoration.

## Recorded starting point

- Initial reviewed planning oracle: `0a8e2882588fdad5a99aca6e2215735c43927528` at 20,349 physical lines / 652 direct methods and projected ceilings 19,928 / 636.
- Rebased implementation base and Task 0 `origin/dev`: `d4f3f97763ddf3fa46eeb35ae9473827e72695bc`.
- Design HEAD before this plan after rebase: `0c9bdc92a94542054228decfa4dbbd56d1cb598d` (pre-rebase `4404283f021623ecbff14dd07e3f15e4c93946d5`).
- Rebased `ChatScreen`: 20,428 physical lines / 653 direct methods; projected ceilings 20,007 / 637.
- Current-base fleet family: 421 physical definition lines / 16 direct methods.
- Historical Wave 6 fleet family at `POST_IMAGE_IMPLEMENTATION_BASE`: 401 lines / 16 methods.
- Frozen final-rebase source: `02cd80b33004305765b5cd91b3d264aa3664596e`
  at 20,486 physical lines / 656 direct methods. Its exact fleet family remains
  AST-identical at 16 methods / 421 lines. The three added methods are
  `_console_inspector_active`, `_request_console_context_allocation_reconcile`, and
  `_request_console_live_work_reconcile`; all are unrelated bounded-rail
  reconciliation work. The final earned ceilings are therefore 20,065 / 640.
- Rebased focused fleet baseline: 17 passed, 4 warnings in 39.63 seconds.
- Rebased metadata-only registry baseline: 1 passed, 1 dependency warning in 1.80 seconds.
- Generated diagnostic inventory non-write baseline: green after the mandatory latest-dev rebase at 521 owners / 1,221 TASK-492 calls / 7,208 TASK-494 calls / 7 sink files. The pre-rebase 0a8 branch was inherited RED; upstream reconciled that delta. No Task 0 manifest write occurred.

## File map

- Create `tldw_chatbook/UI/Console_Modules/fleet.py`: controller state, completion claim, wake policy, unseen/cache/marker policy, teardown accounting, and survivor timer.
- Modify `tldw_chatbook/UI/Console_Modules/wiring.py`: stateless displayed-screen/draft resolvers, `_fleet` construction, direct Workspace fleet callbacks, controller-count/order docs.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: delete 16 definitions and two state assignments, route lifecycle/view hooks/tab markers/transcript edge/resume callers directly to `_fleet`, and remove obsolete imports.
- Create `Tests/UI/test_console_fleet_lifecycle_controller.py`: no-mount controller behavior and late-binding contracts.
- Modify `Tests/Architecture/test_console_wave6_inventory.py`: current-base task ratchet, exact ownership, constructor/dependency/no-DOM/no-sibling/no-replacement-helper oracles and synthetic non-vacuity.
- Modify `Tests/UI/test_console_controller_wiring.py`: fourteen-controller slot/order and `_fleet` late-bound construction checks.
- Modify `Tests/Architecture/test_persistent_diagnostic_inventory.py`: transfer exactly three metadata-only labels from `chat_screen.py` to `fleet.py`.
- Modify ownership-driven focused tests only: `Tests/Chat/test_fleet_teardown_notice.py`, `Tests/UI/test_console_fleet_survivor_tick.py`, `Tests/UI/test_console_fleet_wake_hidden_screen.py`, `Tests/UI/test_console_fleet_wake_ui_freshness.py`, `Tests/UI/test_console_fleet_wake_wiring.py`, and any additional exact moved-name/state caller returned by `rg`.
- Modify `Docs/security/production-diagnostic-inventory.json` only after the reviewed diagnostic reconciliation step.
- Modify this plan and the TASK-3070.8 task record for truthful closeout evidence.

### Task 0: Freeze the baseline and commit the reviewed plan

**Files:**

- Modify: `Docs/superpowers/specs/2026-08-21-task-3070-8-console-fleet-lifecycle-design.md`
- Modify: `backlog/tasks/task-3070.8 - Extract-Console-fleet-and-wake-lifecycle-controller.md`
- Modify: `Docs/superpowers/plans/2026-08-21-task-3070-8-console-fleet-lifecycle.md`

- [x] **Step 1: Confirm branch/base cleanliness and current source counts**

Run:

```bash
git fetch origin dev
git status --short
git rev-parse HEAD origin/dev
git merge-base HEAD origin/dev
wc -l tldw_chatbook/UI/Screens/chat_screen.py
```

Expected: clean status before plan edits and, after any required rebase, merge base equals current `origin/dev`. Preserve both immutable earlier evidence sets: the initial 0a8e/421/20,349/652/19,928/636 planning oracle and the historical 401-line oracle. Use the rebased latest-dev source for the current implementation-base ratchet. If upstream changes the reviewed 16-method family or makes its projection inapplicable, stop and amend/re-review the design rather than silently rewriting constants.

- [x] **Step 2: Re-run the focused baseline**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_compatibility_inventory_is_complete_and_phase_safe \
  Tests/UI/test_console_fleet_survivor_tick.py \
  Tests/UI/test_console_fleet_wake_restart_staging.py \
  Tests/UI/test_console_fleet_wake_wiring.py
```

Expected on the recorded base: 17 passed; only the recorded dependency/runtime warnings.

- [x] **Step 3: Record the diagnostic baseline without writing it**

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only
```

Expected: preserve the pre-rebase inherited RED result, then record the actual non-write result after the mandatory latest-dev rebase. The metadata-only registry node must be green. Do not regenerate the manifest here; any later owner redistribution remains a Task 5 review boundary.

- [x] **Step 4: Commit only plan/task metadata**

```bash
git add \
  Docs/superpowers/specs/2026-08-21-task-3070-8-console-fleet-lifecycle-design.md \
  'backlog/tasks/task-3070.8 - Extract-Console-fleet-and-wake-lifecycle-controller.md' \
  Docs/superpowers/plans/2026-08-21-task-3070-8-console-fleet-lifecycle.md
git commit -m "docs(console): rebase fleet lifecycle baseline"
```

**Task 0 evidence (2026-08-21):**

- The clean six-commit documentation branch was fetched and rebased onto
  `d4f3f97763ddf3fa46eeb35ae9473827e72695bc`; the post-rebase merge base equals
  that exact `origin/dev`. The final Task 5 latest-dev rebase remains required
  because `dev` may advance again during implementation.
- The initial `0a8e2882588fdad5a99aca6e2215735c43927528` planning oracle remains
  20,349 lines / 652 direct methods / 421 fleet-family lines / 16 fleet methods,
  projecting to 19,928 / 636. On the rebased implementation base, `ChatScreen`
  is 20,428 lines / 653 direct methods; the same exact 16-method family is
  AST-identical, still spans 421 lines, and now projects to 20,007 / 637.
  Upstream added `_console_change_review_workspace_roots` outside the family.
  The separate historical `8d806b71d9c5ae7ed333ccb42780f6b2ea68acd0`
  oracle remains 16 methods / `raw_lines=401`.
- The post-rebase focused command passed 17 tests with 4 existing warnings in
  39.63s; there were no failures. The pre-rebase run remains recorded as 17
  passed / 3 warnings / 35.86s.
- The post-rebase non-write diagnostic checker exited 0 with 521 owners / 1,221
  TASK-492 calls / 7,208 TASK-494 calls / 7 sink files. This supersedes the
  characterized pre-rebase inherited RED because upstream reconciled the
  inventory; no manifest write occurred. The metadata-only registry node passed
  1 test with 1 Requests dependency warning in 1.80s.

### Task 1: Lock ownership, dependency, and behavior contracts with RED tests

> **Historical pre-final-rebase execution evidence:** This section records the RED
> contract created and exercised against immutable Task 0 base `d4f3f977`. Its
> 20,007-line / 637-method projection remains historical evidence, not an active
> delivery ceiling. The active final gate is Task 5 Step 6 against frozen `02cd80b3`
> at 20,065 lines / 640 methods.

**Files:**

- Create: `tldw_chatbook/UI/Console_Modules/fleet.py` (importable no-caller RED shell only)
- Create: `Tests/UI/test_console_fleet_lifecycle_controller.py`
- Modify: `Tests/Architecture/test_console_wave6_inventory.py`
- Modify: `Tests/UI/test_console_controller_wiring.py`

- [ ] **Step 1: Add no-mount controller policy tests**

Build a small callable-backed fixture that records calls and supports post-construction replacement. Add named tests for:

```python
def test_completion_claim_prefers_exact_session_and_acknowledges_once(): ...
def test_completion_claim_uses_last_conversation_match(): ...
def test_already_active_completion_returns_true_without_side_effects(): ...
def test_completion_exception_releases_claim_without_acknowledging(): ...
def test_mount_claim_reads_uncached_marks_before_wiring_and_retry(): ...
def test_user_priority_propagates_a_selected_composer_draft_error(): ...
def test_delivery_start_distinguishes_unmounted_from_hidden_mounted(): ...
def test_missing_chat_controller_returns_none_markers(): ...
def test_pending_wake_defers_view_clear_and_live_marker_outranks_unseen(): ...
def test_teardown_stages_counts_only_after_a_truthy_leave(): ...
def test_survivor_tick_is_idempotent_and_final_paints_after_stop(): ...
```

The tests instantiate `ConsoleFleetLifecycleController` directly with plain fakes; they do not construct or mount `ChatScreen`.

Create an importable, unwired RED shell before running them. Its exact keyword-only
constructor stores the reviewed callbacks and initializes the two state defaults; its
16 methods and marker-preparation helper use deliberately neutral/wrong no-op behavior
(`False`, `None`, or `{}` as appropriate) and are not called by production. This lets
each behavior test reach its own assertion. Record the contract-specific failure from
every node; a shared file-exists assertion, `NotImplementedError`, collection/import/
fixture error, skip, or xfail is not acceptable RED evidence.

- [ ] **Step 2: Add the historical Task 0 structural ratchet**

Keep the historical fleet `Wave6Group(raw_lines=401, source_revision=POST_IMAGE_IMPLEMENTATION_BASE)` unchanged. Add task-local constants and assertions for:

```python
FLEET_TASK_IMPLEMENTATION_BASE = "d4f3f97763ddf3fa46eeb35ae9473827e72695bc"
FLEET_TASK_BASE_SCREEN_LINES = 20_428
FLEET_TASK_BASE_METHODS = 653
FLEET_TASK_DEFINITION_LINES = 421
FLEET_TASK_MAX_SCREEN_LINES = 20_007
FLEET_TASK_MAX_METHODS = 637
```

The pre-final-rebase execution asserted all 16 moved names existed only on
`ConsoleFleetLifecycleController`; every controller method was free of
`query`/`query_one`; no method accessed `screen`, `_workspace`, `_session`, `_agent`,
or a sibling controller field; the constructor was keyword-only with the exact
reviewed callback names; `ChatScreen` gained no replacement fleet definition; and the
then-current screen met the historical Task 0 ceilings after extraction. Those d4f3
constants remain immutable source evidence; current delivery counts are evaluated only
by Task 5's frozen-final-base gate.

- [ ] **Step 3: Add synthetic non-vacuity oracles**

Use in-memory AST fixtures to prove the structural checker rejects:

```python
class ChatScreen:
    def _console_fleet_survivors_live(self):
        return True

class ConsoleFleetLifecycleController:
    def leaked_dom(self):
        return self.query_one("#composer")

    def sibling_reach_through(self):
        return self._workspace._poke_console_wake_retry()
```

Each mutation must fail for its intended ownership/DOM/sibling reason.

- [ ] **Step 4: Extend wiring tests**

Keep the existing six-controller `_EXPECTED_SLOTS` common-contract subset unchanged;
its later tests intentionally assume every member has `app_instance` and
`_chat_store_accessor`. Add a separate `_ALL_CONTROLLER_SLOTS` order/class inventory
for all fourteen controllers, with `_fleet` after `_character` and before `_session`.
Add a focused construction test that monkeypatches a screen dependency after
`build_console_controllers` and proves the next `_fleet` call observes the replacement.

- [ ] **Step 5: Run the exact RED set**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_fleet_lifecycle_controller.py \
  Tests/Architecture/test_console_wave6_inventory.py::test_fleet_family_has_completed_controller_ownership \
  Tests/Architecture/test_console_wave6_inventory.py::test_fleet_controller_has_only_named_non_dom_dependencies \
  Tests/Architecture/test_console_wave6_inventory.py::test_fleet_task_ratchet_is_earned \
  Tests/Architecture/test_console_wave6_inventory.py::test_fleet_move_oracles_are_non_vacuous \
  Tests/UI/test_console_controller_wiring.py::test_all_fourteen_controllers_are_constructed_with_the_right_classes \
  Tests/UI/test_console_controller_wiring.py::test_fleet_controller_is_constructed_with_late_bound_screen_edges
```

Expected: collection succeeds. Every behavior node fails at its own value/call-order/
exception assertion against the importable no-op shell; wiring fails because `_fleet`
is not constructed; ownership reports the exact 16 duplicate screen methods;
task-local ratchets remain RED; synthetic non-vacuity nodes pass.

- [ ] **Step 6: Run targeted Ruff and commit the reviewed RED contract**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Console_Modules/fleet.py \
  Tests/UI/test_console_fleet_lifecycle_controller.py \
  Tests/Architecture/test_console_wave6_inventory.py \
  Tests/UI/test_console_controller_wiring.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/Console_Modules/fleet.py \
  Tests/UI/test_console_fleet_lifecycle_controller.py \
  Tests/Architecture/test_console_wave6_inventory.py \
  Tests/UI/test_console_controller_wiring.py
git diff --check
git add \
  tldw_chatbook/UI/Console_Modules/fleet.py \
  Tests/UI/test_console_fleet_lifecycle_controller.py \
  Tests/Architecture/test_console_wave6_inventory.py \
  Tests/UI/test_console_controller_wiring.py
git commit -m "test(console): lock fleet lifecycle extraction"
```

### Task 2: Implement the controller and move production ownership

> **Historical pre-final-rebase execution evidence:** Task 2 was implemented and its
> GREEN contract was exercised before the frozen final rebase. References below to
> 20,007 / 637 describe that d4f3 execution; they do not supersede Task 5's active
> `02cd80b3` 20,065 / 640 delivery gate.

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/fleet.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Architecture/test_persistent_diagnostic_inventory.py`

- [ ] **Step 1: Create the controller shell and exact state defaults**

Implement the reviewed keyword-only constructor. Store each callable without invoking it, then initialize only:

```python
self._console_fleet_survivor_timer: Any | None = None
self._console_fleet_unseen_cache: tuple[int, frozenset[str]] | None = None
```

Do not accept `screen`, `app_instance`, a chat controller, a wake coordinator, or a generic dependency object.

- [ ] **Step 2: Move completion and wake methods with exact branches**

Move the completion claim and wake methods. Preserve:

- exact `target.session_id` match and break before conversation matching;
- last conversation match otherwise;
- missing match acknowledgement/`False`;
- already-active acknowledgement/`True` with no activation/controller/worker call;
- different-session workspace activation → session switch → exclusive sync scheduling;
- exception release/`False`;
- uncached mount mark read;
- own-composer fallback on app/foreign resolver failure;
- selected draft-read exception propagation;
- separate mounted and displayed callbacks;
- message-pump hop before transcript timer creation.

The stateless wiring helpers have these shapes:

```python
def _displayed_console_composer_draft(screen: Any) -> str | None: ...
def _console_screen_is_displayed(screen: Any) -> bool: ...
```

They may capture the screen only in wiring; the controller receives their returned plain values.

- [ ] **Step 3: Move unseen-marker and teardown policy**

Move the cached unseen read, marker precedence, and teardown split/leave/stage logic. Add one controller operation:

```python
def prepare_session_run_markers(
    self,
    sessions: tuple[ConsoleChatSession, ...],
    active_session_id: str | None,
) -> dict[str, ConsoleRunMarker] | None:
    ...
```

Return `None` when no chat controller exists. Defer clear when wake is pending or the screen is hidden. Clear through the service, refresh the cache, then derive markers. Do not return `{}` for the missing-controller branch.

- [ ] **Step 4: Move survivor timer ownership**

Create/stop the one-second interval through the injected timer factory. Preserve idempotence, liveness exception containment, transcript-timer skip, stop-before-final-paint ordering, and the idle no-timer path.

- [ ] **Step 5: Wire `_fleet` and direct consumers**

In `wiring.py`, import/construct `_fleet` after `_character`, update thirteen→fourteen docs, and wire every dependency as a late-bound lambda. Point Workspace's existing `fleet_unseen_ids_accessor`, `run_marker_with_unseen`, and `wake_retry_poke` directly to `screen._fleet`.

In `chat_screen.py`:

- remove the 16 definitions and both screen state assignments;
- point view hooks to `_fleet`;
- preserve first-chat → notice → synchronous fleet claim order on mount;
- point mount/resume timers, transcript self-stop edge, composer state, and unmount calls to `_fleet`;
- replace inline tab unseen/view-clear/marker policy with `prepare_session_run_markers`;
- remove imports whose final screen caller moved.

No one-line screen helper or method alias is permitted.

- [ ] **Step 6: Transfer the hand-reviewed diagnostic labels**

In `Tests/Architecture/test_persistent_diagnostic_inventory.py`, move exactly these entries from `chat_screen.py` to a new `fleet.py` owner entry with unchanged field allowlists:

```python
"Console fleet completion handoff will retry": (
    "claim.revision",
    "type(exc).__name__",
),
"console fleet wake mount-claim failed": ("type(exc).__name__",),
"fleet survivor check failed": ("type(exc).__name__",),
```

Leave `Pending sidebar-state write failed` under `chat_screen.py`.

- [ ] **Step 7: Run the core GREEN set**

Historical result: the exact Task 1 command passed against the pre-final-rebase
candidate, which met the d4f3 20,007-line / 637-method projection while both immutable
earlier oracles remained green. Do not use that historical absolute projection as a
current candidate gate; run Task 5 Step 6 against frozen `02cd80b3` instead.

- [ ] **Step 8: Commit the production move**

```bash
git add \
  tldw_chatbook/UI/Console_Modules/fleet.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py
git commit -m "refactor(console): extract fleet lifecycle controller"
```

### Task 3: Repair focused callers and preserve production-shaped oracles

**Files:**

- Modify: `Tests/Chat/test_fleet_teardown_notice.py`
- Modify: `Tests/UI/test_console_fleet_survivor_tick.py`
- Modify: `Tests/UI/test_console_fleet_wake_hidden_screen.py`
- Modify: `Tests/UI/test_console_fleet_wake_ui_freshness.py`
- Modify: `Tests/UI/test_console_fleet_wake_wiring.py`
- Modify only additional files returned by the exact moved-name scan.

- [ ] **Step 1: Scan every stale moved-name caller**

```bash
rg -n "consume_pending_console_fleet_completion|_claim_console_fleet_wake_marks|_console_wake_user_priority|_console_wake_probe_composer|_console_screen_displayed|_console_wake_conversation_in_view|_poke_console_wake_retry|_on_console_wake_delivery_started|_console_wake_turn_active|_record_console_fleet_teardown|_console_fleet_unseen_ids|_console_run_marker_with_unseen|_console_fleet_survivors_live|_maybe_start_console_fleet_survivor_tick|_stop_console_fleet_survivor_tick|_console_fleet_survivor_tick|_console_fleet_survivor_timer|_console_fleet_unseen_cache" Tests tldw_chatbook
```

Classify every hit as controller ownership, direct `_fleet` production/test use, approved Workspace adapter, architecture string, or stale screen call. No unexplained screen call may remain.

- [ ] **Step 2: Repoint direct test calls without weakening assertions**

Change only ownership, for example:

```python
await console._fleet._record_console_fleet_teardown()
console._fleet._maybe_start_console_fleet_survivor_tick()
assert console._fleet._console_fleet_survivor_timer is None
```

Use real controller construction where a bare-screen fixture bypasses wiring. Do not add a production fallback for incomplete `ChatScreen.__new__` fixtures.

- [ ] **Step 3: Preserve mounted entry paths**

Do not replace `pilot.press` in `test_console_fleet_wake_hidden_screen.py` with direct draft/controller mutation. Do not replace the plain `threading.Thread` drain injection in `test_console_fleet_wake_ui_freshness.py` with an app-loop call.

- [ ] **Step 4: Run the focused affected matrix**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_fleet_lifecycle_controller.py \
  Tests/Chat/test_fleet_teardown_notice.py \
  Tests/Chat/test_console_fleet_wake.py \
  Tests/Chat/test_console_fleet_wake_staleness.py \
  Tests/Chat/test_console_fleet_wake_view_mark.py \
  Tests/Chat/test_console_viewless_hooks.py \
  Tests/UI/test_console_fleet_survivor_tick.py \
  Tests/UI/test_console_fleet_wake_restart_staging.py \
  Tests/UI/test_console_fleet_wake_wiring.py \
  Tests/UI/test_console_fleet_wake_hidden_screen.py \
  Tests/UI/test_console_fleet_wake_ui_freshness.py \
  Tests/UI/test_console_headless_wake_fires.py \
  Tests/UI/test_probe_headless_wake_p2_p3_p4.py \
  Tests/UI/test_console_runtime_ownership.py \
  Tests/UI/test_console_workspace_controller.py \
  Tests/UI/test_console_controller_wiring.py
```

Expected: all selected nodes pass. If this exceeds the repository's 20-minute checkpoint, stop, record the completed file boundary, and resume in named file groups; do not narrow the claimed set silently.

- [ ] **Step 5: Commit only ownership-driven fixture changes**

```bash
git add Tests/Chat/test_fleet_teardown_notice.py \
  Tests/UI/test_console_fleet_survivor_tick.py \
  Tests/UI/test_console_fleet_wake_hidden_screen.py \
  Tests/UI/test_console_fleet_wake_ui_freshness.py \
  Tests/UI/test_console_fleet_wake_wiring.py
git commit -m "test(console): cover fleet lifecycle controller"
```

Add any additional changed test returned by the scan explicitly to both the commit and later Ruff commands.

### Task 4: Prove mutation sensitivity and exact restoration

**Files:**

- Temporarily modify only one named candidate path per probe; restore immediately.

- [ ] **Step 1: Record the candidate diff checksum before every probe**

```bash
git diff --binary -- <mutated-paths> | shasum -a 256
```

Use `apply_patch` for the mutation and inverse. After restoration, require the identical checksum, `git diff --check`, and no mutation-token residue.

- [ ] **Step 2: Probe completion semantics**

Mutate exact-session precedence, already-active `True`, or release-vs-acknowledge. The corresponding no-mount controller node must fail with a value/call-order discriminator, not an import/fixture error. Restore and rerun green.

- [ ] **Step 3: Probe wake uncertainty and late binding**

Swallow a selected composer draft exception or eagerly capture one dependency. The raising-probe/late-bound node must fail. Restore and rerun green.

- [ ] **Step 4: Probe mark policy**

Return `{}` instead of `None` for a missing controller, reuse the cached read during mount, or clear while `wake_has_pending` is true. The focused marker/mount node must fail. Restore and rerun green.

- [ ] **Step 5: Probe teardown and survivor ordering**

Stage notices when `leave_runtime` is false, or paint before stopping the final survivor timer. The exact teardown/timer node must fail. Restore and rerun green.

- [ ] **Step 6: Probe the structural boundary**

Use the synthetic AST fixture to reintroduce one screen-owned fleet method, one DOM query, and one sibling reach-through. Each intended architecture message must appear. No production source mutation is needed for this probe.

- [ ] **Step 7: Re-run the combined no-mount/architecture GREEN set**

Run the Task 1 command plus `git diff --check`. Expected: all selected tests pass and the pre/post candidate diff checksum is identical.

### Task 5: Static, diagnostic, focused integration, and closeout gates

**Files:**

- Modify: `Docs/security/production-diagnostic-inventory.json` after reviewed reconciliation only.
- Modify: task/plan docs for evidence and closeout.

- [ ] **Step 1: Run Ruff on every changed Python file**

Initial exact set:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Console_Modules/fleet.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_fleet_lifecycle_controller.py \
  Tests/Architecture/test_console_wave6_inventory.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py \
  Tests/Chat/test_fleet_teardown_notice.py \
  Tests/UI/test_console_fleet_survivor_tick.py \
  Tests/UI/test_console_fleet_wake_hidden_screen.py \
  Tests/UI/test_console_fleet_wake_ui_freshness.py \
  Tests/UI/test_console_fleet_wake_wiring.py

../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/Console_Modules/fleet.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_fleet_lifecycle_controller.py \
  Tests/Architecture/test_console_wave6_inventory.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py \
  Tests/Chat/test_fleet_teardown_notice.py \
  Tests/UI/test_console_fleet_survivor_tick.py \
  Tests/UI/test_console_fleet_wake_hidden_screen.py \
  Tests/UI/test_console_fleet_wake_ui_freshness.py \
  Tests/UI/test_console_fleet_wake_wiring.py
```

Add every additional changed Python path explicitly. Expected: both commands exit 0; do not accept baseline attribution for a changed file.

- [ ] **Step 2: Compile changed production modules in an isolated temporary cache**

```bash
../../.venv/bin/python - <<'PY'
import os
import py_compile
import tempfile
from pathlib import Path

paths = (
    Path("tldw_chatbook/UI/Console_Modules/fleet.py"),
    Path("tldw_chatbook/UI/Console_Modules/wiring.py"),
    Path("tldw_chatbook/UI/Screens/chat_screen.py"),
)
with tempfile.TemporaryDirectory(prefix="task-3070-8-pycache.", dir="/private/tmp") as root:
    root_path = Path(root)
    assert root_path.is_dir() and not root_path.is_symlink()
    assert root_path.stat().st_uid == os.getuid()
    for index, path in enumerate(paths):
        py_compile.compile(path, cfile=str(root_path / f"{index}.pyc"), doraise=True)
    assert all(not child.is_symlink() for child in root_path.iterdir())
PY
```

Expected: exit 0 and the temporary directory is removed by its context manager.

- [ ] **Step 3: Preview diagnostic reconciliation without mutating the manifest**

Run this exact three-way inventory preview. It writes only inside a validated
`TemporaryDirectory`, emits checked→pinned-base and pinned-base→candidate diffs, and
asserts the task's `chat_screen.py`+`fleet.py` diagnostic-content multiset and complete
sink topology are unchanged. It does not call `--write`.

```bash
../../.venv/bin/python - <<'PY'
from __future__ import annotations

import difflib
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from scripts import check_persistent_diagnostic_inventory as inventory

repo = Path.cwd().resolve()
checked_path = repo / "Docs/security/production-diagnostic-inventory.json"
base_revision = subprocess.run(
    ["git", "rev-parse", "origin/dev"],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

checked_text = checked_path.read_text(encoding="utf-8")
candidate_inventory = inventory.build_inventory()
candidate_text = inventory._encoded(candidate_inventory)

archive = subprocess.run(
    ["git", "archive", base_revision, "tldw_chatbook"],
    cwd=repo,
    check=True,
    capture_output=True,
).stdout

with tempfile.TemporaryDirectory(
    prefix="task-3070-8-diagnostic-preview.",
    dir="/private/tmp",
) as temporary_root:
    root = Path(temporary_root)
    assert root.is_dir() and not root.is_symlink()
    assert root.stat().st_uid == os.getuid()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(root, filter="data")
    assert all(not path.is_symlink() for path in root.rglob("*"))

    inventory.REPO_ROOT = root
    inventory.PACKAGE_ROOT = root / "tldw_chatbook"
    base_inventory = inventory.build_inventory()
    base_text = inventory._encoded(base_inventory)

    artifacts = {
        "checked": checked_text,
        f"base-{base_revision[:12]}": base_text,
        "candidate": candidate_text,
    }
    for label, content in artifacts.items():
        path = root / f"{label}.json"
        path.write_text(content, encoding="utf-8")
        assert path.is_file() and not path.is_symlink()
        print(label, hashlib.sha256(content.encode()).hexdigest())

    for left_label, right_label in (
        ("checked", f"base-{base_revision[:12]}"),
        (f"base-{base_revision[:12]}", "candidate"),
    ):
        print(f"--- {left_label} -> {right_label} ---")
        print(
            "".join(
                difflib.unified_diff(
                    artifacts[left_label].splitlines(keepends=True),
                    artifacts[right_label].splitlines(keepends=True),
                    fromfile=left_label,
                    tofile=right_label,
                )
            )
        )

    def combined_calls(package_root: Path) -> list[tuple[str, str]]:
        calls: list[tuple[str, str]] = []
        for relative in (
            "UI/Screens/chat_screen.py",
            "UI/Console_Modules/fleet.py",
        ):
            path = package_root / relative
            if not path.exists():
                continue
            diagnostics, _ = inventory.scan_source(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            calls.extend((entry["method"], entry["digest"]) for entry in diagnostics)
        return sorted(calls)

    base_calls = combined_calls(root / "tldw_chatbook")
    candidate_calls: list[tuple[str, str]] = []
    for relative in (
        Path("tldw_chatbook/UI/Screens/chat_screen.py"),
        Path("tldw_chatbook/UI/Console_Modules/fleet.py"),
    ):
        if not relative.exists():
            continue
        diagnostics, _ = inventory.scan_source(
            relative.read_text(encoding="utf-8"),
            filename=str(relative),
        )
        candidate_calls.extend(
            (entry["method"], entry["digest"]) for entry in diagnostics
        )
    assert base_calls == sorted(candidate_calls)
    assert (
        base_inventory["persistent_sink_topology"]
        == candidate_inventory["persistent_sink_topology"]
    )
    print("base_revision", base_revision)
    print("chat_screen+fleet diagnostic multiset: exact")
    print("persistent sink topology: exact")
PY

../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only
```

Expected: the checked→base diff exposes inherited latest-dev staleness; the
base→candidate diff contains the reviewed owner redistribution; the exact multiset and
topology assertions pass; the metadata registry finds each moved label once under
`fleet.py`. Preserve this output as pre-rebase evidence. Do not write the manifest yet.

- [ ] **Step 4: Run final focused behavior and architecture gates**

Run the Task 3 affected matrix, the Task 1 architecture/wiring command, and:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_compatibility_inventory_is_complete_and_phase_safe \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_projection_clears_both_ratchet_overages \
  Tests/UI/test_console_sync_outlives_screen.py
```

Expected: all selected tests pass; no local full-suite run.

- [ ] **Step 5: Review scope and complete task hygiene**

```bash
git diff --check
git status --short
git diff --stat origin/dev
git diff --check origin/dev
rg -n "consume_pending_console_fleet_completion|_claim_console_fleet_wake_marks|_console_wake_user_priority|_console_wake_probe_composer|_console_screen_displayed|_console_wake_conversation_in_view|_poke_console_wake_retry|_on_console_wake_delivery_started|_console_wake_turn_active|_record_console_fleet_teardown|_console_fleet_unseen_ids|_console_run_marker_with_unseen|_console_fleet_survivors_live|_maybe_start_console_fleet_survivor_tick|_stop_console_fleet_survivor_tick|_console_fleet_survivor_tick" tldw_chatbook/UI/Screens/chat_screen.py
```

Expected: diff check clean; no moved definition/call remains on `ChatScreen`; no secrets, generated artifacts, compatibility shims, unrelated formatting, or speculative abstraction.

Prepare `## Implementation Notes` and the exact RED/GREEN/mutation/test/static/
diagnostic evidence, ADR decision, changed files, and user-authorized affected-only
test scope, but keep the task `In Progress` and final AC/plan closeout boxes open until
the post-rebase verification below succeeds.

- [ ] **Step 6: Use the approved frozen final-rebase oracle and re-verify before delivery**

The 2026-08-22 ratchet amendment reviewed and approved
`02cd80b33004305765b5cd91b3d264aa3664596e` as the frozen final-rebase source.
Its 20,486 lines / 656 methods minus the unchanged 421-line / 16-method family earn
20,065-line / 640-method candidate ceilings. The candidate measured 19,996 lines /
640 methods when this amendment was reviewed. This approval changes no Task 5 or AC
completion state; the remaining focused, diagnostic, static, and closeout gates still
must run against this evidence.

If `origin/dev` advances before actual delivery, that is a separate delivery rebase:

```bash
git fetch origin dev
git rebase origin/dev
```

Preserve the historical 401-line oracle, the initial immutable 0a8e planning evidence,
the Task 0 `d4f3f977` implementation-base ratchet with its exact 421-line/16-method
earned reduction, and this amendment's frozen `02cd80b3` final-rebase oracle. Record
any later delivery-base measurements under a new name. If later upstream screen
changes make the approved absolute projection inapplicable, stop and amend/re-review
the design instead of silently rewriting any fixed evidence.

For the approved `02cd80b3` delivery base—or after any separately reviewed later
delivery rebase—rerun Task 5 Step 3's three-way preview against that exact base SHA.
Only if every inherited delta is independently classified and the task union/topology
assertions pass, run:

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --write
git diff -- Docs/security/production-diagnostic-inventory.json
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only
```

Then rerun Tasks 1, 3, and the remaining Task 5 focused/static/compile gates. Only after
all post-rebase gates pass may the executor check all three ACs and remaining plan
boxes, finalize Implementation Notes, set the task to `Done` through Backlog CLI, and
commit truthful closeout metadata. PR/Qodo/CI/merge delivery is a separate
user-authorized action after task completion.
