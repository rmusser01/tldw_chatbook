---
id: TASK-23201
title: >-
  Console: two Context rail paint tests depend on the pre-TASK-23193 section
  layout
status: Done
assignee: []
created_date: '2026-08-30 01:06'
updated_date: '2026-08-31 05:39'
labels:
  - console
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_conversation_status_row_label_and_value_are_separate_visual_runs and test_narrow_details_rail_paints_full_private_scratch_value read pixels out of the whole-screen compositor through a harness that does not reproduce the real app's layout. Both broke when TASK-23193 changed which rail sections ship open. The behaviour they guard is intact in the real app - the 2026-08-29 UAT captures show the Sessions scope row painting 'Conversation  None' correctly at 160x48 and 200x60 - so this is test infrastructure, not a user-visible regression. They are marked xfail until reworked onto a deterministic idiom.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both tests assert their property without reading whole-screen compositor pixels, or reproduce the real app's layout deterministically
- [x] #2 The xfail markers are removed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
--------------------------------------------------
Half resolved by TASK-23199. test_conversation_status_row_label_and_value_are_separate_visual_runs was DELETED rather than reworked: it asserted the painted separation of the '#console-active-scope' status pair, and that pair no longer exists -- TASK-23199 retired it because it read 'Conversation  None' above the active chat's own name when unsaved and the tautology 'Conversation  This conversation' when saved. There is no longer a defect for it to guard.

Still open: test_narrow_details_rail_paints_full_private_scratch_value, which remains xfail for the original reason.

--- 2026-08-30: DONE, and this task's stated premise was WRONG ---

I recorded these as 'test infrastructure, not a user-visible regression' caused by TASK-23193's section-layout change. That was wrong for the second test, and I only found out by probing instead of re-reading my own note.

test_narrow_details_rail_paints_full_private_scratch_value was failing BEFORE any of this work: ran it at 4da99a884, the pre-branch commit, where it also fails. So TASK-23193 did not break it and the layout dependency was not the cause.

The actual cause is that the guarantee it asserted never existed. ConsoleWorkspaceStatusPair sizes the label column to the label plus one gutter cell and gives the value the remainder, and its own comment states the intent: 'Longer labels may shrink the value to 6 cells and use the existing ellipsis + tooltip behavior instead of widening the whole rail.' At the rail's fixed width, 'Local files' (11 cells + gutter) leaves 13 columns for 'Private scratch' (15), so it truncates at EVERY terminal size. Probed the widget directly: it renders 'Private scra…' on dev, on my branch, and at 4da99a884 alike.

So the test demanded something the design deliberately does not promise. Rewritten to pin what TASK-384 actually fixed and what is worth guarding: the failure MODE. A value that does not fit must ellipsize on one line rather than word-wrap into a letter stack, the label must never be the thing cut, and the full text must stay reachable via tooltip.

The first test in this task was already resolved by TASK-23199 (it asserted painted separation of a status pair that no longer exists, and was deleted).

Near-miss worth recording: my span-based edit removed the xfail block AND two helpers AND four unrelated tests along with it. Only a NameError from a surviving call site caught it. Verified afterwards by comparing test-name sets against HEAD -- 46 before, 46 after, with only the intentionally replaced name gone. Span replacements across a test file need that check, not a glance.

No xfails remain in test_console_workspace_context_rail.py. preflight green.
<!-- SECTION:NOTES:END -->
