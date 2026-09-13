---
id: TASK-32302
title: >-
  Library Conversations loses its entry focus after the archive-scope load hop
  (task-32228 regression on dev)
status: Done
assignee: []
created_date: '2026-09-11 00:40'
updated_date: '2026-09-11 16:54'
labels:
  - library
  - focus
  - conversations
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32228 gave Conversations the entry focus every other browse list has, which is what makes its Escape hop live and its footer chip honest. Measured green 24/24 on fix/library-crit9-shell before merging dev adb7d5886f; measured 4 of 5 runs RED immediately after, with no change to the branch's own code. The rows are not late and the 2s arm window is not the constraint: the rows mount 0.44s after the rail-row press with 1.75s of window still open, the arm is still pending, and nothing re-requests focus. The arm generation reaches 2 on every run, so a second arm fires during the route entry and invalidates the first scheduled attempt. dev's change in scope is the archive-scope recovery annotate hop added to the conversations page load (library_conversation_recovery.py, library_conversations_controller.py). Separate from task-32301, which is about the fixed 2s window on genuinely slow loads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening Conversations from the rail lands focus on its first row on every run, not one in five
- [x] #2 The 'esc focus Library' footer chip is present on arrival, as task-32228 delivered
- [x] #3 Tests/UI/test_library_crit9_shell.py::test_the_conversations_footer_advertises_escape_on_arrival passes without xfail
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the arm seam end to end: every caller of _arm_library_list_entry_focus and every disarm that bumps the generation.
2. Reproduce in a Textual harness and instrument where the single scheduled attempt goes.
3. Fix at the shared seam (_focus_library_list_entry), not in one caller.
4. Red-first pin on the measured mechanism; remove the xfail and run AC#3 five times.
5. Live-verify at 235x52 on a seeded profile; guide stamp; Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The mechanism was not a second arm.** The generation counter is bumped by
``_disarm_library_list_entry_focus`` as well as by the arm, so "generation
reaches 2" does not prove a second arm fired. Instrumenting the real route
entry showed ONE arm and one silently wasted attempt: a mounted conversation
row is not necessarily a FOCUSABLE one. dev's archive-scope recovery hop fires
a second page request during route entry (``handle_library_rail_row``), and
``LibraryConversationRecovery.project`` folds ``state.loading`` into
``actions_disabled``, which ``library_conversations_canvas`` paints straight
onto every row (``button.disabled = actions_disabled``). ``Widget.focusable``
is false for a disabled widget, so ``Screen.set_focus`` on it is a silent
no-op -- and ``_focus_library_list_entry`` returned right after it as if the
landing had happened, leaving the arm pending with nothing to re-request it.

**The fix is the rule task-32228 already wrote one branch above**, applied to
the row path: only a LANDING counts as done. If ``self.focused`` is not the
row we just asked for, the retry chain is kicked instead of returning. One
guard in the shared seam covers all five list canvases (any of them can paint
rows disabled while a mutation or a load is in flight), rather than one guard
per caller. The 2s window is untouched -- widening it is task-32301, and a
COLD first visit to Conversations still misses it live (measured: warm
re-entry lands on row 0 with the full `/ focus filter … esc focus Library`
footer, cold first entry does not).

**Evidence.** New pin
``test_a_row_that_is_not_yet_focusable_does_not_consume_the_entry_focus_arm``
drives the real product state (``loading`` true, canvas re-synced, rows
painted disabled) and fails red without the guard on the exact assertion the
fix addresses. AC#3's pin lost its xfail and ran 5/5 green alone plus 25/25
green for the whole file. Live at 235x52 on a seeded profile: entering
Conversations from the rail lands on row 0, the footer reads
`/ focus filter | F6 next pane | esc focus Library`, and Escape hops focus to
the rail's Search box.

**Files:** ``tldw_chatbook/UI/Screens/library_screen.py`` (the guard, both
set_focus sites in ``_focus_library_list_entry``),
``Tests/UI/test_library_crit9_shell.py`` (new pin; xfail removed).
<!-- SECTION:NOTES:END -->
