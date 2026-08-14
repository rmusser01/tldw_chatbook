---
id: TASK-16300
title: >-
  Navigation under a pushed screen leaves the outgoing content screen resident
  in the stack
status: To Do
assignee: []
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
- [ ] #1 Navigating while a pushed screen sits above the content screen leaves exactly one content screen in the screen stack: the outgoing content screen is no longer resident once the navigation completes
- [ ] #2 The outgoing content screen's teardown runs on that path -- on_unmount fires and Console's controller shutdown is requested -- exactly as it does on a plain navigation with no pushed screen
- [ ] #3 A modal awaiting push_screen_wait when navigation runs has its awaiter resolved with the standard no-result value rather than stranded; no awaiting caller hangs, crashes, or is silently discarded, and the decision is logged
- [ ] #4 The outgoing content screen's pre-navigation hooks (flush_pending_work, confirm_navigation, save_state) are asked of the content screen rather than of the overlay that happened to be on top
- [ ] #5 A regression test drives a real navigation with a pushed screen above the content screen and fails if the resident-screen leak returns
<!-- AC:END -->
