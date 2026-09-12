---
id: TASK-24607
title: Retrieval scope row drops its value at 120 columns and below
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:54'
updated_date: '2026-08-30 01:21'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The scope row renders 'Scope:' with nothing after it, no ellipsis, beside its Narrow button, while the pinned authority block two rows above still reads 'Scope: Everything available'. Cause: the row container is capped at max-height 1 while the label class declares width 1fr and min-width 0 with neither text-wrap nowrap nor text-overflow ellipsis, so the label wraps and the second line is clipped. The structurally identical settings-row class declares both. The scoped states will fail the same way, which is worse: a narrowed scope reading as blank invites a wrong-scope send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The scope row shows its value or an ellipsis at every supported width, never a bare label
- [x] #2 Scoped states are covered, not only the unscoped default
- [x] #3 The reason a scope is empty is available without a mouse
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED test asserting the scope label's own painted strip is never a bare 'Scope:'.
2. Measure the row's width budget to find why the label was squeezed.
3. Fix the cause, not just the symptom; keep nowrap/ellipsis as the guarantee.
4. Cover all four scope states, not only the unscoped default.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two causes, one symptom. Measured at 120 and 100 columns: rail 34 -> row 30 -> label 11, button 16.

1. The 'Narrow…' button is 7 glyphs but claimed 16 cells. '.console-retrieval-scope-action' declared 'width: auto' without 'min-width', and Textual's Button DEFAULT_CSS carries 'min-width: 16' -- 'width: auto' does not shrink past it. Adding 'min-width: 0' moved the button to 9 cells and the label to 18, so 'Scope: everything' (17) now renders WHOLE at every supported width rather than being elided. The scoped states benefit more: they swap one button for Edit + Clear, which at the old floor needed 33 cells of a 30-cell row.
2. With only 11 cells the label wrapped at the space, and '#console-retrieval-scope-row' is capped at 'max-height: 1', so line two ('everything') was clipped away with no ellipsis. '.console-retrieval-scope-label' now declares 'text-wrap: nowrap' and 'text-overflow: ellipsis', matching '.console-settings-row', which already had both. This stays as the guarantee for longer future values even though the width fix means the default no longer needs it.

Testing note worth keeping: the first version of the narrow-width test searched the WHOLE painted frame for an ellipsis and passed vacuously, because the adjacent 'Narrow…' button puts a '…' on screen unconditionally. It was rewritten to assert on the label's own render_line(0) strip, which then failed for the right reason (label region 11x2 against a max-height 1 row).

Modified: tldw_chatbook/css/components/_agentic_terminal.tcss, regenerated tldw_cli_modular.tcss, Tests/UI/test_console_narrow_layout.py, Tests/UI/test_console_scope_row.py.
<!-- SECTION:NOTES:END -->
