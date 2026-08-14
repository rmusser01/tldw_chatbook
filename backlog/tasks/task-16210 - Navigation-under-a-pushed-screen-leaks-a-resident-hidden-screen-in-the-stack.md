---
id: TASK-16210
title: Navigation under a pushed screen leaks a resident hidden screen in the stack
status: To Do
assignee: []
created_date: '2026-08-14 03:27'
labels:
  - console
  - navigation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the wake-integrity arc (tasks 15970/15971) while diagnosing the
residue arc's live evidence, and harness-reproduced deterministically
(Tests/UI/test_console_fleet_wake_hidden_screen.py, `_leak_resident_chat`):
`App.switch_screen` pops the TOP of the screen stack, so a navigation
issued while any pushed screen (nav overflow menu, a picker, a rename
modal, a confirmation dialog) sits above the current tab screen pops the
MODAL and leaves the tab screen resident in the stack — mounted, message
pump running, `on_unmount` never fired. For ChatScreen that means a live
controller (no `shutdown()`), live sync/poll timers beating behind
another screen (the residue dbg.log's continuous sync-run lines during
Library display), and a second live ChatScreen once the user navigates
back — the state behind BOTH live wake findings (the blind user-wins-ties
probe and the off-screen delivery). The wake layer is now defensive
against it (15970/15971), but the leak itself is generic: every leaked
screen duplicates timers and workers indefinitely (perf), and any screen
holding non-wake resources leaks them too.

Coordinator note: the 15971 design ruling treats mounted-but-hidden
off-screen delivery as INTENDED, so fixing this toward always-unmount is
a product call, not an automatic bug-fix — if navigation is changed to
always tear down the outgoing tab screen, the staged-wake path grows back
to covering every nav-away (see task-15860's state update).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The navigation completion path removes the outgoing tab screen from the stack even when a pushed screen sat above it at switch time (or the residency model is explicitly ruled intended and documented)
- [ ] #2 A test drives a real navigation with a pushed screen above the current tab screen and pins the ruled behavior
<!-- AC:END -->
