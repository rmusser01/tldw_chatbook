---
id: TASK-637
title: Guard mouse capture across non screen recompose widgets
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 18:00'
updated_date: '2026-07-26 02:33'
labels:
  - followup
  - uat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-627 fixed the capture leak for BaseAppScreen recompose, but non-screen widgets that recompose get neither guard (same bug class, different trigger): mcp_rail.py:205, ResultsDashboardWindow, Chatbooks_Window_Improved, Mindmap_Viewer_Window:354, app.py:6024. A capture held by a descendant of any of these at recompose time leaks app-wide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All non-screen recompose sites release/sweep stale mouse capture like BaseAppScreen
- [x] #2 Regression test covers at least one non-screen site
- [x] #3 No legitimate (still-attached) capture is released
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read BaseAppScreen (task-627/641) to understand the exact 3-part guard: call-time release in refresh(), pre-teardown release as recompose()'s first statement, post-recompose sweep for detached-but-still-captured widgets.
2. Re-grep the repo for every refresh(recompose=True) call and every reactive(..., recompose=True) field to build the full non-screen site inventory (screens already covered via BaseAppScreen inheritance are out of scope).
3. Design a reusable RecomposeCaptureGuard mixin (not a copy-paste per widget) matching BaseAppScreen's semantics but scoped to the widget's OWN subtree (a non-screen widget is only part of a larger screen, unlike a Screen's exclusive-content recompose, so an unconditional release would drop legitimate unrelated captures -- AC3).
4. Apply the mixin to the 5 named sites: mcp_rail.MCPRail, ResultsDashboardWindow, Chatbooks_Window_Improved.ChatbooksWindowImproved, Mindmap_Viewer_Window.MindmapViewerWindow, MediaIngestWindowRebuilt (app.py's #ingest-window).
5. TDD: write regression tests for at least two sites (MCPRail deferred-teardown-window case; ChatbooksWindowImproved post-recompose sweep/teardown-drain case) plus an AC3 no-legitimate-release test, confirm RED against pre-fix code via git stash, then confirm GREEN.
6. Run the capture/navigation gate tests plus the touched widgets' own test files; document the full site inventory (including out-of-scope candidates) in Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Generalized task-627's BaseAppScreen mouse-capture guard (call-time release, pre-teardown release, post-recompose sweep) into a reusable `RecomposeCaptureGuard` mixin (`tldw_chatbook/Widgets/recompose_capture_guard.py`) and applied it to the 5 named non-screen recompose sites: `MCPRail` (mcp_rail.py), `ResultsDashboardWindow`, `ChatbooksWindowImproved` (Chatbooks_Window_Improved.py), `MindmapViewerWindow` (Mindmap_Viewer_Window.py), and `MediaIngestWindowRebuilt` (app.py's #ingest-window, line drifted from the reported 6024 to 5795).

Deliberate design deviation from BaseAppScreen (documented in the module docstring): BaseAppScreen releases capture UNCONDITIONALLY before teardown because a Screen recompose tears down its ENTIRE content (only one screen is ever interactively active, so any current capture must belong to what's about to be removed). A non-screen widget is typically only ONE part of a larger, otherwise-untouched screen -- unconditional release there would drop a legitimate, still-attached capture belonging to an unrelated sibling widget, violating AC3. So `RecomposeCaptureGuard`'s pre-teardown release (in both `refresh()` and `recompose()`) only fires when the current capture is `self` or a descendant of `self` (checked via `ancestors_with_self`); the post-recompose sweep stays unconditional on attachment, same as BaseAppScreen, since a detached widget can never be a legitimate capture for anyone.

Full re-grep findings (repo-wide `refresh(recompose=True)` + `reactive(..., recompose=True)`):
- All `UI/Screens/*.py` recompose sites (skills, watchlists_collections, chat, artifacts, home, schedules, acp, workflows, library, settings screens) already extend BaseAppScreen -- confirmed covered, out of scope.
- The 5 named sites above -- fixed here. Two are dead/orphaned: `ResultsDashboardWindow.py` can't even be imported (`from .eval_shared_components import ...` -- that module was deleted long ago; not referenced anywhere else in the codebase) and `MindmapViewerWindow` has no live navigation/production call site (only reachable via a test harness in test_bulk_selection_tooltips.py) -- fixed for correctness/future-proofing but not currently exercisable in the running app.
- Same bug class, NOT fixed here (scope-limited to the task's named list; flagged as follow-up candidates): 7 files under `UI/Watchlists_Modules/` (inspector/runs/notifications/rules/overview/items/sources panes), and ~15 sites under `Widgets/` (Home canvas+rail, chat_message_enhanced, Evals metrics_display, Library conversations/rail/media canvases, 7 Console widgets, status_widget, file_list_item_enhanced, TTS character_voice_widget + chapter_editor_widget).

TDD: RED-confirmed via `git stash` of the 5 production-file edits (module + tests kept), reproducing the exact bug (stale capture survives recompose) before restoring the fix and confirming GREEN. New tests: `test_sync_state_recompose_releases_a_capture_that_lands_in_the_deferred_teardown_window` + `test_recompose_does_not_release_a_legitimate_capture_outside_the_rail` (AC3, real sibling widget) in test_mcp_rail.py; `test_post_recompose_sweep_releases_a_capture_dispatched_during_the_teardown_drain` (teardown-drain/residual-window case, mirroring the task-627 code-review finding) in test_chatbooks_screen_server_actions.py -- driven via `refresh(recompose=True)` directly rather than the `chatbooks` reactive setter, to sidestep an unrelated pre-existing bug in `_update_content()` (`grid.mount(card)`/`list_view.mount(item)` called before the grid/list_view itself is mounted, for any non-empty chatbooks list).

Gates run: test_mcp_rail.py + test_chatbooks_screen_server_actions.py (26 passed), test_settings_rag_profile_region.py (121 passed), test_screen_navigation.py + test_screen_footer_hints.py + test_library_skills_canvas.py (290 passed, 1 pre-existing unrelated failure confirmed via stash-to-baseline), test_media_ingest_window_rebuilt.py + test_bulk_selection_tooltips.py (11 passed), test_mcp_tools_mode.py + test_mcp_servers_mode.py + test_mcp_inspector.py (175 passed), test_destination_shells.py (102 passed, 1 pre-existing unrelated skip), test_mcp_workbench.py (182 passed).

Modified: tldw_chatbook/Widgets/recompose_capture_guard.py (new), tldw_chatbook/UI/MCP_Modules/mcp_rail.py, tldw_chatbook/UI/ResultsDashboardWindow.py, tldw_chatbook/UI/Chatbooks_Window_Improved.py, tldw_chatbook/UI/Mindmap_Viewer_Window.py, tldw_chatbook/UI/MediaIngestWindowRebuilt.py, Tests/UI/test_mcp_rail.py, Tests/UI/test_chatbooks_screen_server_actions.py.
<!-- SECTION:NOTES:END -->
