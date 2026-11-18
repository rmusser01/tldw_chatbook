---
id: TASK-41702
title: "Home's triage refresh reads the screen stack after it has emptied during shutdown"
status: Done
assignee: ['@claude']
created_date: 2026-11-18 04:23
updated_date: 2026-11-18 05:17
labels:
- follow-up
- ui
priority: medium
references:
- tldw_chatbook/UI/Screens/home_screen.py
- Tests/UI/test_screen_navigation.py
dependencies:
- TASK-41500
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`HomeScreen._sync_home_triage_after_refresh` reads `self.app.screen` after the screen stack has emptied at shutdown and raises `WorkerFailed: ScreenStackError('No screens on stack')`. Review r16lb-rev1 of TASK-41500 saw it 3 times, including in the already-laned `test_deep_link_library_route_lands_its_canvas_over_restored_state`, and TASK-41500 lanes more tests that visit Home. It is the same class of fault as TASK-41500's scope recovery MountError: work that finishes after the app's exit fence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Home triage refresh that finishes after shutdown starts does nothing and raises nothing
- [x] #2 A deterministic test, red before the fix, pins it
- [x] #3 The laned tests that visit Home pass serially 5 of 5 with no ScreenStackError
<!-- AC:END -->

## Implementation Plan

1. Guard `_sync_home_triage_after_refresh`, the one sink every Home refresh completion calls, with the liveness check the scope mirror uses: return once the screen is no longer attached. The exit fence (`_exit`, set first by `App.exit` and `TldwCli._shutdown`) makes every node read as detached before Textual empties the stack. Catch nothing.
2. Add a real-app test: a suspended Home, the fence set and the stack emptied as shutdown does, then the completion. Red at base on the ScreenStackError, green after.
3. Run the laned tests that visit Home serially 5 times and check their output for ScreenStackError.

## Implementation Notes

**Approach.** `HomeScreen._sync_home_triage_after_refresh` is the one sink every Home refresh completion calls: the content snapshot, the chatbook-artifact snapshot and the active-work cache. It now returns when `not self.is_attached`, before it reads `self.app.screen`. Shutdown sets the app's `_exit` fence (`TldwCli._shutdown`, `App.exit`) before Textual's `_close_all` empties the stack, and `_exit` makes every node read as detached. So a late completion paints nothing, leaves the pending flag alone, and raises nothing. Nothing is caught, so any other error still raises.

**A second reader, found by AC#3.** The first serial 5-round run of the laned Home visitors still failed 2 of 5 in `test_deep_link_library_route_lands_its_canvas_over_restored_state`, with `ScreenStackError` raised from a different site: `LibraryScreen._library_emergency_return_eligibility`. `LibraryScreen.refresh` reaches it through the footer projection when Textual's prune of a child calls `parent.refresh(layout=True)` after the stack has emptied. It gets the same check, inline, so the file stays at 38280 lines. The two readers share the deterministic test.
- Five more Library `self.is_mounted and self.app.screen ...` reads have the same latent shape, in the skills and prompts import completions. They were never seen to fire, and four tests reach them through SimpleNamespace fakes (`Tests/Skills/test_skills_import.py`, `Tests/UI/test_library_canvas_scoped_sync.py`), so they are left for a follow-up.

**Test.** `Tests/UI/test_screen_navigation.py::test_late_completions_after_the_exit_fence_skip_the_empty_stack`. Home is suspended behind a reused Library. The test sets the fence and empties the stack as shutdown does, then calls both late readers and restores the stack synchronously. It also checks that the Home completion paints nothing and leaves its pending flag unchanged.
- Red at base, with both guards reverted: `AssertionError: a late completion read the emptied stack: [('home triage', ScreenStackError('No screens on stack')), ('library footer', ScreenStackError('No screens on stack'))]`.
- Green after.
- `Tests/Performance/test_home_hidden_completion.py`'s fake receiver gained `is_attached = True`, since the sink now reads it.

**AC#3: serial 5 of 5.** The laned tests that navigate to Home are `test_app_quit_under_modal`'s snapshot-save node, `test_console_quit_loss_guard`'s Home quit node, and eight `test_screen_navigation.py` route and round-trip nodes. With this node added, the set was run serially 5 times with the lane flags (`-p no:xdist --timeout=180`, `-rA` so passing logs are kept).
- Each round: 12 passed, 0 `ScreenStackError` mentions, 0 `MountError` mentions. Load 25-62.
- Before the Library reader was guarded, the same loop failed 2 of 5.

**Census.** One tail entry, `MINIMUM_FILES` 2249. Serial lane-flag time, from those five rounds: 10.3-13.9 s call plus up to 2.0 s teardown, about 16 s at worst.

**Files.**
- `tldw_chatbook/UI/Screens/home_screen.py`, `tldw_chatbook/UI/Screens/library_screen.py`.
- `Tests/UI/test_screen_navigation.py`, `Tests/Performance/test_home_hidden_completion.py`.
- `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
- `backlog/docs/lessons-textual.md`, where the TASK-41500 exit-fence entry gained the `ScreenStackError` symptom.

