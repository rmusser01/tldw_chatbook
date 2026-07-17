---
id: TASK-284
title: Library RAG panel: stop rebuilding results/history per keystroke
status: In Progress
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, library]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Input.Changed on the RAG query box (library_screen 11821-11827) calls the full panel-state refresh, tearing down and remounting the Evidence results list + Recent-searches history (~100+ widgets via awaited remove/mount) on every keystroke of an unsubmitted query — neither depends on unsubmitted text (search runs on Submitted). The status-line refresh is already a separable cheap function. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P1 B5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing in the query box updates only the run-button/status line; results/history refresh only on submit or outcome application
- [x] #2 Existing RAG panel tests green
<!-- AC:END -->

## Implementation Plan

1. Re-verify line anchors against current code -- the `Input.Changed` handler is `update_library_rag_query` (library_screen.py, ~line 12001), and the full refresh it called is `_refresh_search_rag_panel_state_widgets` (~12595), which internally calls `_refresh_library_rag_query_status_widgets` (cheap, already separable), a scope-container/summary/scope-recovery update (cheap, bounded), `_refresh_library_rag_inspector` (cheap), then the two expensive rebuilds: `_refresh_library_rag_results_widgets` and `_refresh_library_rag_history_widget`.
2. Verify no other caller of the full refresh depends on the Changed-path's specific behavior -- found `Tests/UI/test_library_shell.py::test_library_shell_search_scope_strip_refresh_path_uses_shared_copy`, which pins that Input.Changed DOES still refresh the scope-summary strip. This narrows the fix: skip only the two expensive, unbounded rebuilds (results + history), not every widget the full refresh touches.
3. Add an `include_results_and_history` parameter to `_refresh_search_rag_panel_state_widgets` (default `True` for every existing caller); the query-edit path passes `False`.
4. Decide what `update_library_rag_query` resets: the original code called `_reset_library_rag_retrieval_state()` (clearing results too) before every refresh. Since the query-edit path no longer rebuilds the results widget, resetting `_library_rag_results` there would desync the widget (still showing old rows) from the state (now empty) -- a click on a still-visible row would silently no-op. Add a narrower `_reset_library_rag_in_flight_status()` that clears only `retrieval_status`/`recovery_state` (needed so a query typed during/after a prior search doesn't leave the Run button stuck "Searching..."/disabled forever), leaving `_library_rag_results`/`_library_rag_selected_result_id` untouched.
5. Add new tests; run the existing Library Search/RAG suites to confirm no regressions.

## Implementation Notes

`update_library_rag_query` (`tldw_chatbook/UI/Screens/library_screen.py`) now calls `_refresh_search_rag_panel_state_widgets(include_results_and_history=False)` instead of the full refresh, and resets only the in-flight/service-reported status via the new `_reset_library_rag_in_flight_status()` instead of the broader `_reset_library_rag_retrieval_state()`. `_refresh_search_rag_panel_state_widgets` gained the `include_results_and_history` parameter (default `True`); when `False` it still runs the query-status refresh, scope container/summary update, scope-recovery widget remount, and inspector sync (all cheap, bounded, and pinned by existing tests to run on every query edit), but returns before the two expensive `_refresh_library_rag_results_widgets`/`_refresh_library_rag_history_widget` rebuilds. All other callers (`_start_library_rag_query` on Submit/Run, `select_library_rag_result`, `_apply_library_rag_search_outcome`, history-clear/rerun) pass the default `True` and are unaffected.

The narrower reset (`_reset_library_rag_in_flight_status`) was necessary, not optional: `LibraryRagPanelState.from_values` derives the Run button's "Searching…"/disabled state from `retrieval_status` alone (not from whether a worker is actually still running), and `_apply_library_rag_search_outcome`'s stale-query guard silently drops an in-flight search's outcome once the query text has changed -- so if a query edit didn't reset `retrieval_status` at all, typing while (or right after) a search was in flight would leave the Run button permanently stuck, since nothing would ever transition it back. Resetting `retrieval_status`/`recovery_state` but leaving `results`/`selected_result_id` alone avoids both problems: the run gate un-sticks, and the results widget (never rebuilt) stays exactly in sync with what `_library_rag_results` says it holds.

**Files changed**: `tldw_chatbook/UI/Screens/library_screen.py`. New test file: `Tests/UI/test_library_rag_keystroke.py` (5 tests: keystroke touches only the status refresh; landed results/widget identity survive a query edit untouched; the run gate un-sticks after a prior search settles; `Input.Submitted` still runs the full rebuild; a signature pin on the new parameter's default). Full regression: `Tests/UI/test_library_shell.py` + `Tests/UI/test_product_maturity_gate16_library_search_rag.py` + `Tests/UI/test_library_content_hub.py` (281 passed), including the scope-summary-on-keystroke test that shaped the fix.
