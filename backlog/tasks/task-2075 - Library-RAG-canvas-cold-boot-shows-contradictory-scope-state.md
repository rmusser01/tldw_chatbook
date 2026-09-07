---
id: TASK-2075
title: Library RAG canvas cold-boot shows contradictory scope state
status: Done
assignee:
  - '@claude'
created_date: ''
updated_date: '2026-08-05 01:53'
labels:
  - library
  - rag
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Booting straight onto the Library Search/RAG canvas (config `default_tab = "search"`, enabled by RAG UX v2 PR-2) renders one self-contradicting first frame: the Sources toggles already show real counts (`✓ Notes (1)`, `✓ Prompts (1)`) while the Evidence region still shows the empty-library gate copy ("No Library sources yet..."). It clears on any navigate-away-and-back, and the panel is functional throughout — the Run button and toggles are correctly enabled.

This directly contradicts the honesty theme PR-2 shipped: a false claim rendered beside true evidence, on the exact first frame the boot-landing feature exists to serve.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Booting with `default_tab = "search"` renders a first frame whose Evidence-region gate copy agrees with the Sources toggle counts (no "No Library sources yet" beside populated toggles)
- [x] #2 Steady-state background snapshots (a background ingest completing while the user is mid-search) still do NOT eject or recompose the canvas — RAG-27's guarantee from PR-2 Task 1 holds
- [x] #3 A test covers the cold-boot landing path specifically (not only the navigate-back path, which already self-heals)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify premise: confirm _apply_local_source_snapshot's in-place branch still skips the scope recovery block at current HEAD.
2. Write failing tests: gate16 cold-boot case (AC1/AC3), test_library_shell.py timeout-producer + steady-state-no-churn cases (AC2/AC3).
3. Implement a change-gated recovery mirror inside _sync_library_rag_scope_toggle_and_run_gate_widgets: cache the last library_rag_scope_shows_recovery result, schedule a lock-holding worker only on an actual flip.
4. Verify tests pass; run targeted gate16 + library_shell suites plus a collect-only sweep.
5. Update this task's ACs/notes and mark Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause confirmed exactly as scouted: _apply_local_source_snapshot's in-place branch (taken
whenever the row is already Search, which Task 4 guarantees from the very first compose on a
default_tab="search" cold boot) called only _sync_library_rag_scope_toggle_and_run_gate_widgets,
which by design updates only the Run button/toggle counts/scope-summary -- never the scope
container's has-recovery class or its recovery children. compose() always renders from the
zero-count defaults every fresh screen starts with, so the recovery banner showed on the first
paint regardless of real DB contents, and nothing ever told it real counts had arrived.

Fix: added a cache (_library_rag_scope_recovery_visible: bool | None, None until first sync) to
_sync_library_rag_scope_toggle_and_run_gate_widgets. It now also computes
library_rag_scope_shows_recovery(...) and, ONLY when that differs from the cache, updates the cache
eagerly and schedules a new async _mirror_library_rag_scope_recovery via
run_worker(exclusive=True, group="library_rag_scope_recovery_mirror") -- steady-state snapshots
with an unchanged value take the same no-op, no-await path as the rest of the method (RAG-27/AC2
preserved).

Locking choice: the mirror takes _library_rag_panel_refresh_lock (the same lock
_refresh_search_rag_panel_state_widgets holds for this exact remove/mount sequence) rather than
firing Widget.remove()/mount() unawaited from the sync method. Traced Textual's remove()/mount()
internals: an unawaited call still schedules its real DOM work via call_next regardless of whether
the caller awaits it, so firing it from the uncoordinated sync method could still land inside an
in-flight full refresh's own remove/mount window on the SAME fixed ids -- mount() validates ids
synchronously, so that collision raises DuplicateIds, not just a visual glitch. Routing through the
lock avoids reintroducing the exact hazard PR-3 Task 4 introduced the lock to prevent. Factored the
shared remove/mount sequence into a new _apply_library_rag_scope_recovery_block helper used by both
the mirror and the existing full refresh, so the two can never render this block differently; the
full refresh also updates the cache afterward so it can't go stale relative to an unconditional
render (never a correctness issue either way -- the mirror always re-derives its target from fresh
state, just avoids a redundant reconciliation).

Tests (all additive, no pre-existing test modified, TDD red confirmed via a saved patch + git
checkout -- library_screen.py, then reapplied):
- test_product_maturity_gate16_library_search_rag.py::test_library_search_rag_cold_boot_recovery_banner_agrees_with_real_source_counts
  -- the primary cold-boot regression (AC1/AC3): pre-mount apply_navigation_context({"mode":
  "search"}) mirrors default_tab="search" exactly; asserts the banner clears once the real snapshot
  lands.
- test_library_shell.py::test_library_rag_source_snapshot_timeout_then_real_snapshot_clears_recovery
  -- the failsafe timeout producer, respecting its real guard (only reachable while
  _library_loaded is False): chains the timeout firing first (recovery correctly shown) with the
  real snapshot landing moments later (must still clear it).
- test_library_shell.py::test_library_rag_scope_recovery_steady_state_snapshot_causes_no_churn --
  AC2: a repeat snapshot with an unchanged value schedules zero additional mirrors (spy on the new
  method) and leaves the recovery widgets' identity untouched.

Two of the three test_library_shell.py tests initially passed even against pre-fix code (a
genuinely-empty-library scenario is trivially unchanged whether or not the fix exists, since old
code never touches the recovery block in ANY case) -- redesigned both to include an assertion only
the new machinery satisfies before accepting them as real regression tests.

Verification: test_product_maturity_gate16_library_search_rag.py (46 passed), test_library_shell.py
full file (334 passed, includes the pre-existing RAG-27 protected test unmodified and still
passing), test_css_class_coverage_contract.py (4 passed, scope covers library_screen.py and this
change adds no new literal classes= tokens), full-repo collect-only sweep (29985 collected, 0
errors).

Files: tldw_chatbook/UI/Screens/library_screen.py (fix);
Tests/UI/test_product_maturity_gate16_library_search_rag.py,
Tests/UI/test_library_shell.py (new tests, plus a cross-import of LibraryHarness/
_active_library_screen/_wait_for_library_shell into the gate16 file).
<!-- SECTION:NOTES:END -->

## Implementation Notes (context from the PR-2 live check, 2026-08-03)

Root cause is an interaction between two PR-2 tasks, both new relative to dev — not an inherited bug:

- Task 1 (RAG-27) made `_apply_local_source_snapshot` take an in-place branch whenever `_library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH`, calling `_sync_library_rag_scope_toggle_and_run_gate_widgets`, which by design updates only the Run button, toggle `(N)` labels, and scope-summary line — deliberately NOT the recovery/callout block or the `has-recovery` class, because those need remove/mount sequences that were unsafe to interleave with other refresh callers.
- Task 4 made the Search/RAG selection apply *before* the screen's first `compose()`, so the very first paint is guaranteed pre-fetch with no warm snapshot cache (previously this required beating a local DB query with human reaction time on the session's first Library visit — negligible).

Suggested seam: in `_sync_library_rag_scope_toggle_and_run_gate_widgets` (or its caller branch around `library_screen.py:2272`), also toggle `#library-rag-source-scope`'s `has-recovery` class and mount/unmount `library_rag_scope_recovery_children`, mirroring `_refresh_search_rag_panel_state_widgets` (~:16637-16649) — gated to fire only when `library_rag_scope_shows_recovery(...)`'s result actually CHANGES, so steady-state calls keep RAG-27's no-eject guarantee.

Evidence: `/private/tmp/uat-pr2-805d-evidence/07-item-e-default-tab-search-boot.txt` (contradiction) and `08-*post-nav-back*.txt` (self-healed).
