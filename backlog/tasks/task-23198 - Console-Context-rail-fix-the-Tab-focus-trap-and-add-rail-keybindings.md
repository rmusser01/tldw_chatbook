---
id: TASK-23198
title: 'Console Context rail: fix the Tab focus trap and add rail keybindings'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-29 21:56'
updated_date: '2026-08-30 06:21'
labels:
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tab never leaves the Context rail: thirty consecutive presses stay inside, cycling nineteen stops. This is a WCAG 2.1.2 No Keyboard Trap failure. The rail also declares no BINDINGS, so there is no shortcut to toggle it, jump to a section, or collapse all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focus can leave the Context rail using the keyboard alone, and the method is advertised (REVISED - the original premise was wrong, see Implementation Notes)
- [x] #2 The rail exposes bindings to collapse all and expand all sections
- [x] #3 Regression tests pin the keyboard escape, its advertisement, and the new bindings firing from a real key press
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC #1 AND AC #3 WERE BUILT ON A WRONG FINDING, and the correction is the main outcome here.

The audit reported that Tab never leaves the Context rail and called it a WCAG 2.1.2 (No Keyboard Trap) failure. Tab genuinely does not leave -- re-measured, 14 stops then wrap. But it is scoped on purpose: ChatScreen.action_focus_next confines Tab to the focused Console region (TASK-2154.11) because unscoped it made a Tab tour cross all fifteen app-navigation buttons mid-session. I first suspected Textual's _trap_focus, checked, and found nothing in this codebase sets it; the scoping is this repo's own deliberate action override.

WCAG 2.1.2 does not require Tab specifically. It requires that focus can be moved away using the keyboard, and that when that needs more than Tab, the user is advised of the method. Measured both: F6 moves focus out of the rail in ONE press (to console-native-transcript), and the persistent footer carries 'F6 next pane' at all times. The criterion is met. There was no accessibility failure to fix, and 'fixing' it would have reverted a tracked decision that solved a real usability problem.

So AC #1 was rewritten to the contract that actually matters and is now pinned, and AC #3 to tests that pin it. Two of the five tests exist specifically to stop this being re-found: they record why scoped Tab is correct here.

AC #2 was real and is delivered. The rail had no BINDINGS at all, so shutting every section meant clicking seven disclosure toggles. ctrl+shift+left collapses all and ctrl+shift+right expands all, chosen after checking for conflicts (only ctrl+shift+p was taken on this screen). They fire only while focus is inside the rail, so they cannot shadow composer or transcript keys. Both route through the same SectionToggled message the disclosure buttons post, so persistence and layout reconciliation run exactly as they do for a click rather than through a second path needing its own upkeep.

The last test presses the actual keys rather than calling the actions, because calling an action only proves the action exists -- it does not prove the binding is reachable from a focused rail control.

preflight green, 144 tests pass across the rail suites. Files: UI/Console_Modules/left_rail.py; Tests/UI/test_console_context_rail_keyboard.py (new).
<!-- SECTION:NOTES:END -->
