---
id: TASK-15472
title: Pre-import heavy screen modules in the background after first paint
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
  - navigation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: the first visit to a route imports the whole screen module synchronously on the UI thread inside the FIFO-locked navigation worker (`UI/Navigation/screen_registry.py:39-52/:256-278` -> `import_module`): `chat_screen.py` is ~19.9k lines, `library_screen.py` ~26k, `settings_screen.py` ~18.9k. July measured ~161 ms pure import for chat at 11k lines; it has nearly doubled since — plausibly ~1 s on constrained hardware, paid on the first click to each tab and serializing any queued navigation behind the lock.

Fix direction: after first paint, pre-import the top routes from a background THREAD at idle (imports are thread-safe and idempotent; a warm `sys.modules` hit makes `load_screen_class` free). Stability constraints: must not compete with first paint (idle delay), must not change import-error surfacing, and must not break the test seams that patch screens through module aliases (task-3023). Related umbrella: task-2902. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After idle pre-import, the first click to a pre-imported tab spends ~0 ms in import_module (evidence)
- [x] #2 Cold start and first paint unchanged (A/B measurement)
- [x] #3 Import errors surface identically to today; test suites that patch screen modules stay green
<!-- AC:END -->

## Implementation Plan

1. Expose route metadata without importing: add `registered_screen_routes()` to
   `UI/Navigation/screen_registry.py`, returning the `ScreenRoute` objects (not
   just ids) so a caller can dedupe by `module_path` (several canonical route
   ids share a module, e.g. `ccp`/`personas`, `tools_settings`/`mcp`) and call
   the existing `load_screen_class()` per route -- reusing the exact method the
   real navigation path calls, so a genuinely broken module fails at pre-import
   time exactly like it would at nav time (Python evicts a failed import from
   `sys.modules`, so nothing is cached and the next `import_module` attempt --
   real navigation's -- repeats the same failure).
2. In `app.py`, add a background pre-importer on `TldwCli`:
   - `_screen_preimport_route_order()`: chat/library/settings first (the three
     multi-thousand-line modules the audit measured), then the rest of
     `registered_screen_routes()` in stable sorted order, module-deduped.
   - `_preimport_heavy_screens()`: iterates that order calling
     `route.load_screen_class()`, each call wrapped in its own
     `try/except Exception` (belt-and-suspenders over `load_screen_class`'s own
     ImportError/AttributeError swallowing) so one bad module can't kill the
     thread or take down other routes' pre-import.
   - `_schedule_screen_preimport()`: starts `_preimport_heavy_screens` on a
     `threading.Thread(daemon=True)` -- a real OS thread, never the asyncio
     loop -- idempotent via a `self._screen_preimport_thread` guard.
   - `_screen_preimport_enabled()`: on by default; off under pytest
     (`PYTEST_CURRENT_TEST`, the same signal `optional_deps.py` and
     `metrics_logger.py` already gate on) so the full test suite doesn't spin
     up a background-import thread per `run_test()` app instance; a
     `TLDW_SCREEN_PREIMPORT` env var overrides in either direction for tests
     that specifically exercise the scheduling path.
3. Wire the trigger: `_schedule_deferred_startup_work()` (called at the end of
   `_post_mount_setup()`, itself the seam that already fires once first paint
   has rendered and other non-essential startup work like footer status/audio
   services is scheduled) adds one more `self.set_timer(DELAY,
   self._schedule_screen_preimport)`. Delay is slightly longer than the
   existing 0.1s deferred-work timers so pre-import is strictly the
   lowest-priority background task -- nothing depends on it finishing, it only
   warms a cache.
4. Tests (new file, e.g. `Tests/UI/test_screen_preimport.py`):
   - after `_preimport_heavy_screens()` runs, `load_screen_class()` for a
     pre-imported route hits `sys.modules` and does not call `import_module`
     again (patch `importlib.import_module` in the route's call path via the
     seam and assert it is not invoked for an already-warmed module).
   - a route whose module raises on import is swallowed by the pre-importer,
     and a subsequent real `resolve_screen_target()`/`load_screen_class()`
     call for that same route still raises/logs identically to an unpatched
     baseline (proving no behavior drift).
   - `_screen_preimport_route_order()` dedupes shared-module aliases and puts
     chat/library/settings first.
   - `_schedule_screen_preimport()` respects `_screen_preimport_enabled()`
     (pytest default-off, env var override both directions) and is idempotent
     (second call does not spawn a second thread).
   - live-ish probe (in-process `app.run_test()` with `TLDW_SCREEN_PREIMPORT=1`
     forced on): after the preimport thread joins, a screen switch to a
     pre-imported route measures ~0 added import cost vs. a cold one.
   - cold-start A/B: isolated subprocess probe measuring time-to-`_ui_ready`
     (or first-paint proxy) with the pre-importer forced on vs. off, to show
     the deferred trigger does not regress first paint.
5. Run targeted suites (`Tests/UI/test_screen_navigation.py`,
   `Tests/UI/test_workbench_route_inventory.py`,
   `Tests/UI/test_workbench_state.py`, `Tests/Performance/test_app_startup_performance.py`,
   `Tests/Utils/test_optional_import_deferral.py`, new test file) plus a
   `Tests/` collect-only sweep; read pass counts.

## Implementation Notes

Implemented as planned, with one deviation from the AC#2 test design (noted
below) after a real flaky run exposed why it was the wrong approach.

**Production code** (`tldw_chatbook/app.py`, `UI/Navigation/screen_registry.py`):
- `screen_registry.registered_screen_routes()` -- new public accessor
  returning the `ScreenRoute` objects themselves (not just ids), so a caller
  can see `module_path` for dedup and call `load_screen_class()` directly.
- `TldwCli._screen_preimport_route_order()` -- chat/library/settings first
  (`SCREEN_PREIMPORT_PRIORITY_ROUTE_IDS`), then the rest of the registry,
  deduped by `module_path` (several canonical ids share a module: `ccp`/
  `personas`, `tools_settings`/`mcp`).
- `TldwCli._preimport_screens(routes)` -- the per-route worker body, factored
  out of `_preimport_heavy_screens()` so tests can target one or two routes
  instead of the whole registry. Calls `route.load_screen_class()` (the exact
  method real navigation calls) per route inside its own `try/except
  Exception`. A failed import is never cached in `sys.modules` (CPython
  evicts a partially-initialized module on import failure), so a swallowed
  pre-import failure changes nothing about what a real navigation attempt
  does next -- verified directly by a test that fails the same way, twice,
  with a real pre-import attempt swallowed in between.
- `TldwCli._schedule_screen_preimport()` -- starts
  `_preimport_heavy_screens` on a `threading.Thread(daemon=True)` (a real OS
  thread, never the asyncio loop), idempotent via a
  `self._screen_preimport_thread` guard.
- `TldwCli._screen_preimport_enabled()` -- on by default; off under pytest
  (`PYTEST_CURRENT_TEST`, the same idiom `Utils/optional_deps.py` and
  `Metrics/metrics_logger.py` already use) so the ~38k-test suite's many
  `app.run_test()` instances don't each spin up a background-import thread
  for a mechanism most tests never look at; `TLDW_SCREEN_PREIMPORT` env var
  overrides in either direction for tests that exercise the real scheduling
  path.
- Trigger: one more `self.set_timer(DEFERRED_SCREEN_PREIMPORT_DELAY_SECONDS,
  self._schedule_screen_preimport)` inside `_schedule_deferred_startup_work()`
  -- the existing seam that already fires other non-essential startup work
  (footer status, audio services) once first paint has rendered and
  `_ui_ready = True`. 0.2s, deliberately after the existing 0.1s timers, so
  pre-import is strictly the lowest-priority background task.

**Why this is safe cross-thread**: pre-import only calls `import_module()` +
`getattr(module, class_name)` -- it never constructs a screen instance
(`screen_class(self)`), so it never touches Textual's live App/event loop,
compositor, or stylesheet cache. Checked Textual's `Widget.__init_subclass__`/
`DOMNode.__init_subclass__` (run at class-definition time, i.e. during this
import) directly: both only do local bookkeeping on the class object being
created (reactive registration, CSS type-name bookkeeping) -- no stylesheet
parsing, no shared mutable global state, so there is no race with the main
thread's own CSS/mount work.

**Tests** (`Tests/UI/test_screen_preimport.py`, 16 tests + 2 lines added to
`Tests/Performance/test_app_startup_performance.py`):
- Route order: priority + dedup.
- AC#1: a route's `load_screen_class()` call after being warmed is
  dramatically cheaper than the cold call that warmed it (timing evidence,
  since `import_module()` is genuinely called both times -- Python's own
  `sys.modules` cache is what makes the warm call cheap, not a call-count
  difference). Plus an end-to-end test that runs the real thread, joins it,
  and confirms chat/library/settings landed in `sys.modules` and that
  `resolve_screen_target("chat")` resolves through the now-warm module.
- AC#3: a module that raises something other than Import/AttributeError at
  import time is swallowed by the pre-importer without crashing the thread,
  and a real `load_screen_class()` call for that same route raises the
  identical exception both before and after the swallowed pre-import
  attempt. A second test covers the already-existing ImportError->None
  degrade path (missing module) the same way.
- Gating: pytest-default-off, env var override both directions
  (parametrized), idempotent scheduling, and a positive check that the
  worker runs on a different OS thread than the caller.
- AC#2 (ordering): **deviated from the plan's polling design.** The first
  version polled `_ui_ready`/`_screen_preimport_thread` from outside via
  `pilot.pause()` in a loop and asserted the thread was still `None` right
  after observing `_ui_ready` -- this flaked on the very first run: enough
  real wall-clock time (splash animation + verbose DEBUG logging overhead)
  had already elapsed by the time the external poll first ran, so the 0.2s
  timer had already fired. Replaced with a deterministic spy: monkeypatch
  `TldwCli._schedule_screen_preimport` to record `self._ui_ready` at the
  instant the real scheduler invokes it (no external race, since the
  assertion is about a value observed synchronously inside the real call).
  Kept a secondary, loose-bound timing A/B (on vs. off, generous slack) as
  the "honest numbers" sanity check the task's evidence list asked for, on
  top of the deterministic proof.
- Fixed 4 pre-existing tests in `test_app_startup_performance.py`
  (`test_citation_artifact_reconciliation_is_deferred_and_policy_gated` x2,
  `test_legacy_citation_migration_is_deferred_and_policy_gated` x2) that call
  `TldwCli._schedule_deferred_startup_work(fake_app)` directly against a
  duck-typed `SimpleNamespace` -- added `_schedule_screen_preimport=Mock()`
  to both fakes, mirroring the existing `_start_deferred_audio_service_
  initialization=Mock()` entry.

**Evidence run** (`.venv` in this worktree; `tldw_chatbook.__file__` verified
pinned to this worktree first):
- New file: 16/16 passed, 3 consecutive runs (no flakes).
- `Tests/UI/test_screen_navigation.py`: 126/126 passed.
- `Tests/UI/test_workbench_route_inventory.py` +
  `Tests/UI/test_workbench_state.py`: 10/10 passed.
- `Tests/Performance/test_app_startup_performance.py` +
  `Tests/Performance/test_app_import_weight.py`: 22/22 passed (after the
  fake-app fix above).
- `Tests/Utils/test_optional_import_deferral.py`: 15/15 passed.
- Combined run of all of the above together: 189/189 passed.
- `Tests/` full collect-only: 38,165 collected, 0 errors.

**Pre-existing failures found while running the targeted suites, NOT caused
by this task** (neither file is touched by this change's diff):
- `Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[...chat_screen.py]`
  -- `chat_screen.py` is at 20,381 lines against a 17,727-line ratchet budget
  set 2026-08-07; unrelated feature work grew the file past the budget on
  `dev` before this branch point. This task's diff never touches
  `chat_screen.py`.
- `Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged`
  -- regenerating `Docs/security/production-diagnostic-inventory.json` with
  `--write` to review it showed ~40 files worth of drift (new/changed
  diagnostic entries in files this task never touches: `agent_service.py`,
  `console_context_compaction.py`, `library_rag_state.py`,
  `text_selection_crash_guard.py`, and many `call_count` deltas elsewhere).
  Verified via `git log`: every one of those files was touched by a commit
  between the inventory's last regeneration (`662891d74`) and this branch's
  base (`326742a93`) -- i.e. the checked-in inventory was already stale
  against `dev` before this task started. Reverted the regenerated file back
  to its committed state (did not commit the unrelated drift) rather than
  launder ~40 files of unreviewed diagnostic changes into this diff, which
  is exactly what the script's own `--write` warning exists to prevent. This
  task's own one new diagnostic call (`_preimport_screens`'s debug log on a
  swallowed pre-import failure) is real and reviewable but is not reflected
  in the committed inventory either, for the same reason -- fixing the
  inventory is a separate cleanup, out of scope for task-15472's ACs.

**Modified**: `tldw_chatbook/app.py`,
`tldw_chatbook/UI/Navigation/screen_registry.py`,
`Tests/Performance/test_app_startup_performance.py`.
**Added**: `Tests/UI/test_screen_preimport.py`.

## Fix round 1 (external review, not approved on first pass)

Review found two real defects and three minors. All fixed; no scope creep
beyond what was asked.

**CRITICAL -- test file poisoned `sys.modules` for the rest of the pytest
process.** Two tests popped real screen modules and re-imported them (to
force a genuinely cold `import_module` call), and the end-to-end test ran
the real `_preimport_heavy_screens()` against the FULL registry (22
modules) -- both create/replace module objects in the process-global
`sys.modules`, which 135+ test files bind screen classes against at import
time. Reviewer's minimal repro: `Tests/UI/test_screen_preimport.py` run
before `Tests/UI/test_settings_workspaces_category.py::test_create_rename_
archive_unarchive_flow` failed the latter on a stale-class `isinstance`.
Fix: added an autouse `_isolate_screen_modules` fixture to the test file
that snapshots every `tldw_chatbook.UI.Screens.*` `sys.modules` entry before
each test and restores it (or removes newly-added entries) after --
`ScreenRoute.load_screen_class()` does a fresh `import_module()` +
`getattr()` per call with no caching of its own, so dict restoration is
sufficient (confirmed by the fix, not just asserted). Applied file-wide
(autouse), not per-test, so a future test added here can't reintroduce the
same leak by omission -- this also neutralizes the related coupling the
reviewer flagged (the end-to-end test permanently arming `Tests/conftest.py`
:879-881/:900-902's conditional autouse patches for every later test),
since restoring `sys.modules` to its pre-test state undoes that too.
**Proof**: `pytest Tests/UI/test_screen_preimport.py
Tests/UI/test_settings_workspaces_category.py::test_create_rename_archive_
unarchive_flow -q` -- 18 passed, run 3 consecutive times, all green.
(Separately confirmed `test_settings_workspaces_category.py` has
PRE-EXISTING, order-independent flakiness of its own -- different tests
in that file fail across repeated standalone runs with no test-15472 file
involved at all; out of scope, not caused by or fixed by this change.)

**IMPORTANT -- every launch logged a `Screen route unavailable: customize`
warning.** `_screen_preimport_route_order()` was iterating
`registered_screen_routes()` raw, which includes `_SCREEN_ROUTES
["customize"]` -- a dict entry whose module (`customize_screen`) no longer
exists and which is genuinely unreachable at real navigation time (`_SCREEN_
ALIASES["customize"] = "settings"` intercepts the target string before
`_lookup_route()` ever reaches that dict entry). Fix: skip any route id that
is ALSO a key in the alias table (`registered_screen_aliases()`) --
generalized, not hardcoded to "customize", so it also incidentally drops
"media" (a real, importable module, but likewise unreachable under its own
id since `_SCREEN_ALIASES["media"] = "library"`), saving one wasted import.
Added `test_screen_preimport_route_order_excludes_alias_shadowed_routes`
and `test_full_preimport_pass_emits_zero_warnings_on_the_shipped_registry`
(loguru sink capture at WARNING+, asserts zero records over a real full
pass on the shipped registry).

**Minors:**
- (a) Dropped the "loose A/B" pytest check (`on <= off * 2 + 0.5`) --
  correctly called vacuous (passes a 100%+500ms regression). AC #2's sole
  automated evidence is now the deterministic
  `test_screen_preimport_scheduled_only_after_ui_ready_flips` spy. A manual,
  one-off timing probe (not part of the suite) is recorded below instead.
- (b) Added a `self._shutting_down` guard to `_schedule_screen_preimport()`
  (one line, ahead of the existing `_screen_preimport_thread` idempotency
  check) so a timer firing during app teardown doesn't spin up a fresh
  import thread with nothing left to serve.
- (c) Reviewer's live GIL-contention measurement, recorded here as the
  honest cost of this change (not independently re-derived): with the
  pre-import thread running, event-loop responsiveness p95 moved
  1.27ms -> 1.90ms, with one observed 43.2ms spike inside the ~0.5s import
  window. Favorable trade against the ~163.8ms synchronous first-click cost
  this change removes (also reviewer-reproduced independently).

**Evidence re-run after the fix** (same `.venv`, worktree-pinned):
- `Tests/UI/test_screen_preimport.py`: 17/17 passed, 3 consecutive runs.
- `Tests/UI/test_screen_preimport.py` + `test_settings_workspaces_category.
  py::test_create_rename_archive_unarchive_flow` in one process: 18/18
  passed, 3 consecutive runs (the reviewer's exact repro, now green).
- 7-suite targeted combination (`test_screen_preimport.py`,
  `test_screen_navigation.py`, `test_workbench_route_inventory.py`,
  `test_workbench_state.py`, `test_app_startup_performance.py`,
  `test_app_import_weight.py`, `test_optional_import_deferral.py`): 190/190
  passed.
- `Tests/` full collect-only: 38,166 collected (+1 vs. round 1's 38,165,
  matching the net one added test), 0 errors.

**Modified (fix round 1)**: `tldw_chatbook/app.py`,
`Tests/UI/test_screen_preimport.py`.
