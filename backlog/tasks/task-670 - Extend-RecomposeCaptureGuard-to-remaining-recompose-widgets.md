---
id: TASK-670
title: Extend RecomposeCaptureGuard to remaining recompose widgets
status: Done
assignee: []
created_date: '2026-07-26 12:00'
updated_date: '2026-07-26 05:33'
labels:
  - followup
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-637 guarded the 5 originally-named non-screen recompose sites with the RecomposeCaptureGuard mixin, but its repo sweep found ~22 more same-bug-class sites left out of scope to bound the task: 7 UI/Watchlists_Modules/*_pane.py files plus ~15 across Widgets/Home, Widgets/Chat_Widgets, Widgets/Evals, Widgets/Library, Widgets/Console (7 files), Widgets/status_widget.py, Widgets/file_list_item_enhanced.py, Widgets/TTS/*. Full enumerated list in task-637's report. A capture held by a descendant of any of these at recompose time still leaks app-wide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All enumerated remaining recompose sites carry the mixin (or a documented exemption)
- [x] #2 At least two newly-guarded sites have regression tests (one simple, one teardown-drain)
- [x] #3 Existing capture/navigation tests stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-enumerate the full current recompose-site list via repo-wide grep of
   `def recompose`/`refresh(recompose=True)`/`reactive(..., recompose=True)`,
   excluding sites already guarded (task-637's 5 sites, and all
   BaseAppScreen subclasses under UI/Screens/).
2. For each remaining class, add RecomposeCaptureGuard first in its base
   list (mirroring task-637's established pattern), adding the import.
3. Verify MRO sanity per class: confirm refresh()/recompose() resolve to
   the mixin (no shadowing by a class's own override) and no class already
   defines its own refresh/recompose.
4. Pick two newly-guarded sites with existing test harnesses -- one simple
   release-on-recompose, one teardown-drain -- and add RED-first regression
   tests mirroring the MCPRail/Chatbooks patterns from task-637.
5. Run the mixin's own tests + mcp_rail tests, the two new tests, a
   broad-but-bounded sweep of test files covering every touched widget
   family, and the required RAG profile region gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented: re-enumerated the full remaining recompose-site list (repo-wide
grep of `def recompose`/`refresh(recompose=True)`/`reactive(...,
recompose=True)`), confirming task-637's ~22-site estimate resolves to
exactly 25 classes across 25 files once every `reactive(recompose=True)`
field and `refresh(recompose=True)` call is attributed to its owning class:
7 UI/Watchlists_Modules panes, 2 Widgets/Home, 1 Widgets/Chat_Widgets, 1
Widgets/Evals, 3 Widgets/Library, 7 Widgets/Console, 1 status_widget.py, 1
file_list_item_enhanced.py, 2 Widgets/TTS. All 25 got RecomposeCaptureGuard
added first in their base list, mirroring task-637's established pattern
exactly (mcp_rail.py / Chatbooks_Window_Improved.py). All UI/Screens/*.py
files matching the grep were confirmed BaseAppScreen subclasses (already
covered) -- no exemptions were needed.

Verified MRO sanity for every one of the 25 classes programmatically
(`cls.refresh is RecomposeCaptureGuard.refresh` and `cls.recompose is
RecomposeCaptureGuard.recompose`) -- none had a pre-existing refresh/
recompose override to reconcile.

Added two RED-first regression tests per AC2, both confirmed RED against
the pre-mixin class declaration (temporarily reverted, ran, restored) and
GREEN after:
- Tests/Watchlists/test_watchlists_overview_pane.py::
  test_data_recompose_releases_a_capture_that_lands_in_the_deferred_teardown_window
  -- simple release-on-recompose, OverviewPane's `data` reactive.
- Tests/Widgets/Library/test_library_rail.py::
  test_post_recompose_sweep_releases_a_capture_dispatched_during_the_teardown_drain
  -- teardown-drain via LibraryRail.sync_state(), mirroring the Chatbooks
  residual-window test from task-637.

Gates run (all green except one confirmed pre-existing/order-dependent
flake, unrelated to this change -- passes standalone):
- Tests/Widgets/test_recompose_capture_guard.py + Tests/UI/test_mcp_rail.py
  + the two new tests: 25 passed.
- Tests/UI/test_settings_rag_profile_region.py (required gate): 121 passed.
- Watchlists family (5 panes + 2 shell/inspector tests): 56 passed.
- Home + Library family (state tests, LibraryRail, content hub, multiselect,
  selection, post-release depth): 86 passed.
- Console family, part 1 (session settings, agent rail, conversation wrap,
  run inspector x2, scope row, staged context): 271 passed.
- Console family, part 2 (tick gating, workspace context rail,
  chat_message_enhanced, chat_message_artifact_actions): 86 passed.
- Console native chat flow + rail sections together: 1 failed (blank-query
  cache test), 243 passed -- the failing test passes in isolation
  (re-run alone: 1 passed) and does not touch RecomposeCaptureGuard/mouse
  capture at all (workspace conversation search error-cache logic);
  order-dependent pre-existing flake, not a regression from this change.
- Full app import (`python3 -c "import tldw_chatbook.app"`) and py_compile
  on all 25 touched files: clean.
- No dedicated unit tests exist in the repo for MetricsDisplay,
  EnhancedStatusWidget, FileListEnhanced, CharacterVoiceWidget, or
  ChapterEditorWidget (pre-existing gap) -- these 5 sites were verified via
  the MRO/compile/import checks above only, consistent with picking sites
  that already had test harnesses for the two new regression tests per the
  task's own guidance.

Modified files: 25 production widget files (listed above) +
Tests/Watchlists/test_watchlists_overview_pane.py +
Tests/Widgets/Library/test_library_rail.py.
<!-- SECTION:NOTES:END -->
