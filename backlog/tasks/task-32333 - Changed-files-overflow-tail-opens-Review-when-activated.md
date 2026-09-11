---
id: TASK-32333
title: >-
  Changed-files overflow tail opens Review when activated
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review C4. The '+N more - open Review' tail and the pruned-history line are plain Statics phrased as instructions (console_changed_files_section.py ~196-211) -- false affordance. Make the tail an activatable control that opens the Review screen; reword the pruned line as passive status.

Filed from the 2026-09-10 Console rail UX review (review item C4).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: state test asserts tail clickable; wiring test asserts it opens current-mode review. 2. Mark the row clickable; add the routing arm. 3. Update the user-guide table. 4. Run state + wiring suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The overflow tail is keyboard-focusable and clickable, and activating it opens the Review screen
- [x] #2 The pruned-history line reads as status, not an instruction
- [x] #3 Row-click behavior and the 12-row cap are unchanged
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** The Environment→Changes overflow tail ("… N more — Review
opens all") was a plain inert row phrased as an instruction. It is now
`clickable=True` and routed in `_handle_console_environment_row` to
`_open_change_review_current_mode()` — the same working-tree destination
as its "Review in Change Review" neighbour, per TASK-31665's
destination-follows-the-surface ruling. Inert per-file rows stay inert.
The Tasks section's "… N more" tail stays a status line: it has no
in-app destination to open (backlog list cap), and unlike the Changes
tail its copy is not phrased as an instruction.

**ADR check.** Not required — adds one row activation to an existing
routing seam.

**Modified.** `Chat/console_environment_state.py` (tail clickable),
`UI/Screens/chat_screen.py` (route), `Tests/Chat/test_console_environment_state.py`
(+1), `Tests/UI/test_console_environment_wiring.py` (+1),
`Docs/User_Guide/console/context-and-rag.md`. Verified: env state +
wiring suites — 100 passed.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

RELOCATED on dev: '+N more - open Review' became '... N more - Review opens all' in Environment->Changes (console_environment_state.py:655-660) - still non-clickable while the sibling 'Review in Change Review' row IS clickable. Scope: make the overflow tail open Review too.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
