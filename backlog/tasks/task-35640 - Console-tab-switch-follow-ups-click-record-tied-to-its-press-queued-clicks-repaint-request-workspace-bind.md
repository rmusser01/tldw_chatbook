---
id: TASK-35640
title: >-
  Console tab switch follow-ups: click record tied to its press, queued clicks,
  repaint request, workspace bind
status: Done
assignee:
  - '@claude'
created_date: '2026-10-10 19:05'
updated_date: '2026-10-20 00:59'
labels:
  - console
  - tabs-sessions
dependencies: []
references:
  - TASK-33622.7
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-33622.7 from its pre-merge re-review. Four gaps remain in the tab-switch fix: a session-tab click's composer record sits in a controller-wide slot, so a click whose Pressed never arrives can carry later typing into the next activation of that tab by any route (Alt+N, Ctrl+K); two clicks queued behind a busy screen lose the second click's record when the first activation clears it; the repaint request made when an activation loses its claim after switching only takes effect if a sync pass is already running; and the workspace bind helper skips silently when it cannot reach the session controller, with three of its four paths untested.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A session-tab click's composer record is used only by the activation that its own press starts; an activation by any other route (Alt+1..9, Ctrl+K) never uses an earlier click's record
- [x] #2 When two tab clicks are queued behind a busy screen, no key typed after them stays in the chat being left or is lost, each clicked tab keeps its own draft, and the keys keep their order
- [x] #3 An activation that loses its claim after switching gets the new chat repainted whether or not a sync pass is running at that moment
- [x] #4 Each workspace switch path that binds the composer (open, switch, resume, failed-open restore) is covered by a test that fails without the bind
- [x] #5 The Console User Guide's tab-switch typing claim matches the behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED (mounted Pilot tests on the real ChatScreen + store, bootstrap_profile): a click whose Pressed never arrives, then typing in B and Alt+N/Ctrl+K activation of that tab -> typing must stay in B; two clicks queued behind a held activation, each followed by keys -> no key stays in A or is lost; an activation that loses its claim after switching with no pass running -> a sync pass runs; workspace switch/new-chat/resume/failed-open restore with a held sync pass -> composer bound to the new chat.
2. Tie the click record to its press: ConsoleSessionTabButton records the composer on the click and attaches the record to the Pressed that click posts; the session controller's tab-press handler takes it from that message. No controller-wide slot, so queued clicks carry their own records.
3. Repaint request: set _console_sync_requested when a pass is running, else start one (send_acknowledgement._resync's shape).
4. Workspace bind: a required injected collaborator called directly (doubles get it), tests for the untested paths, mutation-checked.
5. User Guide wording; related suites vs dev by test id; preflight; live check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The four re-review minors of TASK-33622.7 are fixed and pinned by mounted tests on the real ChatScreen and store (bootstrap_profile).

- **Click record tied to its press (AC#1, AC#2).** `ConsoleSessionTabButton` takes the composer record when a left click lands (`capture_session_tab_click`) and attaches it to the `Button.Pressed` that same click posts (`post_message` override, cleared in `finally`). `_handle_console_session_tab_press` now takes the press and hands `session_tab_click_record(pressed)` to that one activation as `click_record=`. The controller-wide `_console_tab_click_snapshot` slot is gone. A click whose press never comes (Textual skips `press()` while the button is `-active`) carries nothing, so Alt+1..9 and Ctrl+K never use an old record. Each of two queued clicks keeps its own record.
- **AC#2 wording changed before implementation.** It first said each key goes to the tab clicked just before it. Keys reach the composer through `ChatScreen.on_key`, so they queue at the screen with the presses, and a click-time record cannot split keys typed between two queued clicks. The Pilot harness delivers the keys first. Live runs on a loaded host varied: usually every queued press came first, once the keys arrived before the last press. The AC and test therefore pin what holds in every order: nothing stays in the chat being left, nothing is lost, each tab keeps its draft, and order is kept. Lesson in `backlog/docs/lessons-console-wiring.md`.
- **Repaint request (AC#3).** `_request_console_repaint` copies `send_acknowledgement._resync`: it re-arms a running pass, or else starts `_sync_native_console_chat_ui()` as an exclusive `console-sync` worker. The test is parametrized on whether a pass is running; the no-pass case is red with the flag-only code.
- **Workspace bind (AC#4).** The getattr-chain helper is replaced by a required `bind_composer_after_switch` constructor collaborator, wired in `wiring.py` to `_bind_composer_to_active_session(focus=False)` and called directly on every path. That covers browser open, switching to a workspace's chat, creating it (which never bound before), failed-open restore and resume. Test doubles get the collaborator.
- **User Guide (AC#5).** `sessions-tabs-workspaces.md` now says keys never stay in the tab you left. When two tabs are clicked in quick succession while the Console is busy, text typed after the first of those clicks (even after the second) can land in either tab; it is never lost.

Files: `Widgets/Console/console_session_surface.py`, `UI/Console_Modules/session.py`, `UI/Console_Modules/workspace.py`, `UI/Console_Modules/wiring.py`, `UI/Screens/chat_screen.py`, tests in `Tests/UI/test_console_tab_switch_binds_composer.py` plus double updates in five other test files, the User Guide page and the lessons doc.

Verification (rebased on origin/dev ce4aa4a231, 2026-10-19; host load ~50 on 14 cores):
- Red first: the branch's test file, copied into a detached origin/dev worktree as a probe, gives 4 failed and 4 passed for the new tests. The failures: repaint with no pass running (dev only sets the flag, no pass starts); the lost click (A's composer "a-own:zz", B loses "zz"); queued clicks (A keeps "xy"); new workspace chat (the composer keeps B's draft under the new chat). The other three workspace paths are green on dev; they bound through the old getattr helper.
- Per-site mutation (scratch worktree, one bind call removed at a time): each of the five workspace bind sites turns its own path's test red: browser open, switch to an existing chat, create the chat, failed-open restore, resume. Removing the switch bind leaves the other three workspace params green.
- Green: `Tests/UI/test_console_tab_switch_binds_composer.py` is 20 passed, 5 failed on the head; all eight new tests pass. Dev's copy on dev gives 13 passed, 5 failed: the same five ids with the same assertion diffs (`test_send_refuses_and_keeps_the_draft_when_the_composer_is_bound_elsewhere` x4 and `..._off_screen_after_two_switches`, which fail on tab highlight and transcript state). These were already red on dev 4de437a5c5 and are not touched here.
- Related files compared by test id with dev (head vs origin/dev, same host): eleven files, no new reds. Results as head/dev red counts: test_console_parallel_runs 21/21 of 33; test_console_conversation_activation 0/0 of 40; test_switcher_reuse_decisions 0/0 of 12; test_console_workspace_reconcile 0/0 of 9; test_console_workspace_controller 10/10 of 127 (same ids); test_console_session_tab_strip 0/0 of 23; test_console_switch_draft_integrity 3/3; Tests/Architecture/test_module_size_ratchet 7/7 of 64 (same ids, none of them a file touched here). Archive recovery boundaries and the archive async-completion file error at setup on both (below). test_console_session_controller: one head-only red in the file run, test_fork_validation_freezes_the_accepted_title_until_retry (DuplicateIds while the workspace-context rail recomposed during app mount, load ~95). It passed 3/3 alone on each tree, and the file was 47/47 on identical branch code before the rebase, so it is load flake. The two parallel-runs and one workspace-controller head-only reds from the pre-rebase sweep did not recur. That sweep covered 29 files, including all of Tests/Architecture, test_console_native_chat_flow and test_personas_workbench, with no other new reds.
- `test_workspace_archive_async_completion.py` (24 ids) and `test_console_archive_recovery_boundaries.py` (23 ids) error at setup on both trees with RecoveryRequired raw_source_selection_changed (environmental). The two doubles edited there (late resume; existing activation that loses authority) could not run locally; CI runs them.
- Live (real app, isolated profile, claude-haiku-4-5 via the review proxy, 160x45; six queued-click runs counting the morning round): in every run, fix and dev, no typed key stayed in a tab being left or was lost. A normal tab-switch send (click Chat 3, type 0.2 s later, Enter 0.4 s after) went to Chat 3 and got "ALPHA" back. Queued clicks (Chat 1, Chat 2, "x", Chat 3, "y" at 0.15 s gaps): on the fix, both keys landed in Chat 3 twice and in Chat 2 once ("y" too). All three dev runs put both keys in Chat 3. The lost-click order behind minor #2 did not occur live on dev, so the deterministic tests are the evidence for minors 1 and 2. Evidence: uxrev/evidence/bd5/RTF/ (README, shots 01-04 and 11-14).

- ./scripts/preflight.sh: all derived-artifact checks passed on the rebased head (diagnostic inventory, backlog ids, textual worker contract with no new findings, markup census, UI gate census).
<!-- SECTION:NOTES:END -->
