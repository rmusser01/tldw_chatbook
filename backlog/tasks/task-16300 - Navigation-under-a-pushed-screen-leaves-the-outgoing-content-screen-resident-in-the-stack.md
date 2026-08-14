---
id: TASK-16300
title: >-
  Navigation under a pushed screen leaves the outgoing content screen resident
  in the stack
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 13:33'
labels:
  - console
  - navigation
  - app
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Navigating away while ANY pushed screen (nav overflow menu, command palette,
picker, rename modal, confirm dialog) sits above the current content screen
leaves that content screen alive in the screen stack: mounted, message pump
running, `on_unmount` never fired. For Console that means a live
ConsoleChatController (never `shutdown()`), live sync/poll timers beating
behind whatever screen the user is actually looking at, an armed fleet wake
coordinator, and a SECOND live ChatScreen the moment the user navigates back.

This is a VIOLATION of a documented invariant, not a new navigation model.

Leg 1 -- the mechanism. Textual's `App.switch_screen`
(`.venv/lib/python3.12/site-packages/textual/app.py:3001-3032`) pops ONLY
`self._screen_stack[-1]` and appends the new screen; `_replace_screen` then
unmounts only that popped screen. Nothing else on the stack is touched.

Leg 2 -- the consequence. When a modal sits above ChatScreen, navigation
replaces THE MODAL and leaves ChatScreen resident and hidden. The
wake-integrity arc (tasks 15970/15971) reproduced this deterministically
through the real navigation APIs -- stack after the nav:
`['Screen', 'ChatScreen', 'LibraryScreen']`, with `chat.is_running` True and
the controller's `_shutdown_requested` unset
(`Tests/UI/test_console_fleet_wake_hidden_screen.py::_leak_resident_chat`)
-- and traced two live bugs to it: task-15970 (the hidden screen's
user-wins-ties probe read ITS OWN empty composer while the user typed into
the displayed screen's -- `probe: composer=True draft=''` with the text
visibly in the pane) and task-15971 (the hidden screen's own 1s sync tick
'view'-cleared the FLEET_UNSEEN mark while the user was on Library, so an
off-view delivery left no badge).

Leg 3 -- the invariant it breaks. `tldw_chatbook/app.py`
(`_create_navigation_screen`) states: 'Screens must never be cached and
re-mounted: `switch_screen` unmounts the outgoing screen, and re-mounting a
previously-unmounted instance races its still-in-flight teardown ... a total,
exception-free UI freeze (root-caused 2026-07-11)'. `ScreenStateStore`'s
`save_state`/`restore_state` boundary exists BECAUSE screens die on
navigation. A resident hidden screen is the premise of that invariant
failing: the outgoing instance neither dies nor hands its state over.

Two further consequences of the same pop-the-top mechanism, both currently
live: (a) `_handle_screen_navigation_locked` reads `current_screen =
self.screen`, so with a modal on top the outgoing screen's
`flush_pending_work`, `confirm_navigation` and `save_state` hooks are asked
of the MODAL -- Console's busy-fleet confirm never runs and no snapshot is
taken; (b) `switch_screen` calls `top_screen._pop_result_callback()` WITHOUT
invoking it (textual app.py:3020), so a modal opened via `push_screen_wait`
has its result future dropped un-resolved and its awaiting worker hangs
forever.

Supersedes task-16210 (same finding, filed under an id that collided with
another session's Loguru log-hygiene task; that duplicate file is removed
here).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Navigating while a pushed screen sits above the content screen leaves exactly one content screen in the screen stack: the outgoing content screen is no longer resident once the navigation completes
- [x] #2 The outgoing content screen's teardown runs on that path -- on_unmount fires and Console's controller shutdown is requested -- exactly as it does on a plain navigation with no pushed screen
- [x] #3 A modal awaiting push_screen_wait when navigation runs has its awaiter resolved with the standard no-result value rather than stranded; no awaiting caller hangs, crashes, or is silently discarded, and the decision is logged
- [x] #4 The outgoing content screen's pre-navigation hooks (flush_pending_work, confirm_navigation, save_state) are asked of the content screen rather than of the overlay that happened to be on top
- [x] #5 A regression test drives a real navigation with a pushed screen above the content screen and fails if the resident-screen leak returns
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce RED through the real navigation API: a pushed screen above the content screen, then assert the stack and the outgoing screen's teardown
2. Decide the `push_screen_wait` contract from Textual's own code (what `switch_screen`/`pop_screen`/`dismiss` do to a result callback) and pin it with a test
3. Reduce the stack to its content screen before `switch_screen`; read the outgoing screen from the base of the stack, not `self.screen`
4. Fail closed if the stack will not reduce, rather than switching and recreating the leak
5. Re-establish the wake-integrity suites, whose helper asserted the leak as a harness precondition, without depending on it — and mutation-test that they still pin their own fixes
6. Correct every doc surface that recorded the leak as intended behaviour
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_handle_screen_navigation_locked` now resolves the outgoing screen with
`_navigation_outgoing_screen()` (the base of the stack — `self.screen` is
the OVERLAY whenever one is open, which is why the flush/confirm/
save_state hooks were being asked of modals that answer none of them),
and `_complete_screen_navigation` calls `_dismiss_navigation_overlays()`
immediately before `switch_screen`, so the content screen IS the top of
the stack when Textual pops it.

**`push_screen_wait` — dismiss with the standard no-result value.** Both
`switch_screen` (textual/app.py:3020) and `pop_screen` (:3110) call
`_pop_result_callback()` WITHOUT invoking it, so today's code drops an
awaited modal's future un-resolved and its worker never resumes — no
timeout, nothing else resolves it. `Screen.dismiss(None)` calls the
callback first (textual/screen.py:2048-2070), so the awaiter resumes
with the same `None` that `action_dismiss`, Escape, and a bare
`dismiss()` already deliver. Verified against this app's own consumers:
`_handle_first_run_wizard_result` and `_handle_first_run_recovery_result`
both treat `None` as "cancelled / finish later" and return; the
`dismiss` overrides that exist (`library_file_notes_git_panel`,
`file_picker_dialog`, `_PrivateGGUFFileOpen`) already map `None` to a
decline; Textual's own `CommandPalette` is pushed with no callback at
all. Refusing to navigate while a modal is awaited was the alternative
and is worse — awaited modals are the common kind, and a nav shortcut
that silently no-ops is indistinguishable from a wedged app. Every
dismissal is logged with the screen class and whether it owed a result.

Reduction happens only once the switch is committed (after the veto
hooks and the incoming screen's construction), so a vetoed navigation
does not cost the user the dialog they had open. A screen that ignores
its dismissal aborts the navigation through `_notify_navigation_failure`
(user-facing notice + nav-bar rollback) instead of switching anyway and
recreating the leak, and is never dismissed a second time — `dismiss`
is not side-effect free.

**The wake suites depended on the leak for their SETUP, not their
assertions.** `test_console_fleet_wake_hidden_screen.py::_leak_resident_
chat` asserted "Chat stays resident" as a harness precondition, so four
of its six tests went red on that line the moment the leak was fixed.
Their behaviours are still real and still required: Console is
mounted-but-undisplayed whenever a pushed screen covers it (the state
the 15971 live pass actually verified). Those cases now push a real
modal over Console; the two-Console case, which ONLY the leak used to
produce, is built directly with `push_screen` and kept as defence in
depth. Mutating the 15970 probe fix and both 15971 gates back out still
turns them red (M5–M7), so nothing was weakened.

Modified: `tldw_chatbook/app.py`,
`Tests/UI/test_console_fleet_wake_hidden_screen.py`. Added:
`Tests/UI/test_screen_residency.py` (7 tests; 6 red against unmodified
production, the plain-navigation control green). Docs: fleet spec §7,
`Docs/User_Guide/console/agent-runs-and-tools.md`, the PR3a-2 plan, and
tasks 15860/15970/15971, all of which had recorded the leak as the
reason off-view delivery is intended.

Mutations, all Edit-applied and Edit-restored with `git diff` proving
byte-identical restores: M0 whole fix reverted (6 killed), M1 dismissal
only (4), M2 outgoing-screen resolution only (2), M3 `pop_screen`
instead of `dismiss` (2), M4 stuck-overlay guard dropped (SURVIVED at
first — the loop bound masked it; killed after adding the
dismissed-exactly-once assertion), M5 15970 cross-screen probe (2), M6
15971 sync-tick displayed gate (1), M7 15971 in-view probe gate (1).

Gates (counts read): residency 7, wake hidden-screen 6, wake cluster
(view-mark + wiring + restart-staging) 26 total with the above,
`Tests/Chat/test_console_agent_bridge.py` 195,
`Tests/Chat/test_fleet_*.py` + `test_console_fleet_*.py` 65,
`Tests/Agents/` 1409, `Tests/State/test_screen_state_store.py` +
destination shells + parallel runs 181 (1 skipped),
`Tests/UI/ -k "navigation or screen_nav or nav"` 572 passed / 8 failed —
all 8 verified pre-existing by re-running them with this change
neutralized in place (identical failures).
<!-- SECTION:NOTES:END -->
