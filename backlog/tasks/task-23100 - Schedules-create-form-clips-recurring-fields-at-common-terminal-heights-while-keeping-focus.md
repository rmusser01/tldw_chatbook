---
id: TASK-23100
title: >-
  Schedules create form clips recurring fields at common terminal heights while
  keeping focus
status: Done
assignee: []
created_date: '2026-08-28 14:05'
updated_date: '2026-08-29 02:24'
labels:
  - ux
  - schedules
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ReminderForm stacks its fields in a plain Vertical inside a max-height:55 container with no scrolling (forms/reminder_form.py:35-42,116-186) while the Body TextArea keeps ~7 rows. At 235x52, selecting Recurring leaves the cron Input, its syntax helper, and the live 'Runs:' preview clipped invisible - yet the invisible input still receives Tab focus and keystrokes, and typing silently flips the Frequency preset to 'Custom cron...'. Scheduling also has zero screen-specific :focus styling, so the focused-invisible state has no visible carrier. P0 from the 2026-08-28 Settings+Schedules design critique (evidence: .impeccable/critique/2026-08-28T06-32-49Z__tbook-ui-screens-scheduling-schedules-workbench-py.md). Owner direction: patch the modal now; a separate spike task in this batch decides the long-term modal-vs-pane shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 80x24 and 235x52, every ReminderForm field, helper, and the live schedule preview is visible or scrolls into view when focused; a focused widget is never rendered invisible
- [ ] #2 Selecting 'Recurring' visibly reveals the recurrence controls at supported terminal sizes
- [ ] #3 The live 'Runs: ...' preview stays visible while the cron field is being edited
- [ ] #4 Verification is a runtime capture (tmux capture-pane) at both sizes, not a style-value probe
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ReminderForm's field column is now a VerticalScroll with the previews, validation line and Save/Cancel docked in a footer, so no field can be clipped while focusable. Two root causes were found: the auto-height container clamps by clipping, and the field groups were plain Verticals whose default height:1fr measured ~1 row in the scroll's virtual size while painting six -- the invisible-but-focusable trap. The review round replaced the initial height arithmetic (overhead = 10 + error_line_count, which under-counted wrapped lines at ~45x24) with a docked footer + 1fr scroll area, matching voice_blend_dialog/feedback_dialog. Compositor-verified at 45x24, 80x24 and 235x52. PR #2169.
<!-- SECTION:NOTES:END -->
