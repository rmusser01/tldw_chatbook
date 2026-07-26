---
id: TASK-684.4
title: Delete the retired ingestion window and its dead wiring
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 17:33'
labels:
  - ingest
  - cleanup
dependencies: []
parent_task_id: TASK-684
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Once every capability has an equivalent in the Library ingest canvas, the second window and everything that exists only to serve it should go, so the two cannot drift apart again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import sources opens the Library ingest canvas
- [x] #2 No route, button or command reaches the retired window
- [x] #3 The window, its panels and its now-unused event handlers are deleted
- [ ] #4 The full test suite passes with the window removed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify every capability of the retired window has a canvas equivalent (the gate for this task).
2. Point #library-workspace-import-sources at the Library ingest canvas instead of NavigateToScreen('ingest').
3. Remove the 'ingest' ScreenRoute and MediaIngestScreen.
4. Delete MediaIngestWindowRebuilt and its four panels.
5. Delete the handlers that existed only for it -- check local_ingest_events.py and tldw_api_events.py for callers first; both import the window directly today.
6. Delete Tests/UI/test_media_ingest_window_rebuilt.py and any other tests of the removed surface.
7. Full suite plus a live pass over every path that used to reach the window.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The standalone ingest window is gone: importing lives entirely in Library's Import media canvas, which gained the server-backed and web-clipping paths that window used to own (684.1-684.3).

Removed ~6,550 lines: MediaIngestWindowRebuilt.py (2151), tldw_api_events.py (1290), _ingestion_rebuilt.tcss (261), local_ingest_events.py (118), the screen wrapper, the NewIngestWindow shim, MediaIngestionSourcePanel, and 13 test files that existed only to exercise them. From app.py: show_ingest_view, both _initialize_*_models helpers, and the two Select.Changed branches for tldw-api-auth-method / tldw-api-media-type.

Scouted before deleting. A per-symbol reachability gate over everything ingest_events re-exported found 15 of 17 symbols reachable only from inside the doomed modules. The two that looked live were those app.py select branches -- and their widget ids are mounted nowhere in the tree, so the branches were already unreachable. show_ingest_view has zero callers and returns immediately behind USE_REBUILT_INGEST=True. The one live entry point nearby, misc_worker_handler's 'api_calls' group, is fed only from the window's own submit path.

THE ROUTE IS ALIASED, NOT DELETED. _ROUTABLE_LEGACY_ROUTES already listed 'ingest' and the Workbench route inventory already declared ingest -> library, so dropping the id would dead-end saved navigation state and startup configs that still say 'ingest'. _SCREEN_ALIASES gets ingest -> library, mirroring notes/prompts/skills/research exactly; shell_destinations.py needed no change at all.

TWO METHOD LESSONS, both recorded because they nearly cost a broken tree:

1. A re-export defeats a module-name grep. Tests/Event_Handlers/test_plaintext_ingest_events.py imported a deleted function VIA ingest_events, so searching for the deleted module names never matched it. pytest --collect-only over the FULL tree is the real gate for a deletion -- the only check that sees through a re-export. It found this in 33s after greps said clean.

2. My own gate had a substring bug: it excluded any path containing the doomed module name, and 'test_plaintext_ingest_events.py' contains 'ingest_events.py'. The one file proving those symbols live was skipped as internal. Match paths exactly. Re-run with equality, 21 of 21 deleted symbols are clean apart from an intentional string-comparison guard.

Also deliberately did not repeat task-577's mistake: it deleted _chat_tabs.tcss but left the test reading it, which is why test_non_obscuring_focus_contract.py carries permanent failures. The _ingestion_rebuilt test and its path constant are removed here. That file's true baseline is 9 failures on clean dev, not the 3 previously recorded -- verified as identical sets in a parallel worktree.

_ingestion_rebuilt.tcss was never in the CSS bundle manifest, so no bundle regeneration is involved. The two ingest stylesheets that ARE bundled look dead too, but removing bundled CSS needs the bundle regenerated and the surviving ingest surface checked for shared selectors -- filed as task-745 rather than smuggled in here.

Verification: 17,018 tests collect with zero errors across the whole tree; 1,197 passed in Tests/Event_Handlers + Tests/Widgets + Tests/Library + test_screen_navigation.py.
<!-- SECTION:NOTES:END -->
