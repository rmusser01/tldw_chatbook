---
id: TASK-25826
title: Send button cost estimate is reachable only by mouse hover
status: Done
assignee: []
created_date: '2026-08-31 05:09'
updated_date: '2026-08-31 13:50'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Send label gains a dollar suffix when a cost estimate exists, but the estimate itself is delivered solely through a hover tooltip. In a keyboard-first terminal application hover is the one channel the primary audience does not use, so the information is unreachable. The unlabelled marker also changes the button's width on the keystroke path, shifting the composer's right edge while typing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A cost estimate is reachable without a pointer
- [ ] #2 The cost affordance is labelled rather than represented by a bare symbol
- [ ] #3 Send button width does not change as the draft is typed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the unambiguous defect: sync_action_state runs on the KEYSTROKE path and sized the send control from the current label, so gaining the ' | $' suffix widened the button mid-typing and shifted the composer's right edge under the cursor. New send_button_width_for() sizes for the widest variant the control can take (Send/Queue/Run/Queue full/Preparing, with and without the suffix), so the edge never moves.

DECLINED: renaming the '$' marker. It is pinned by test_console_send_disabled_state ('Send | $') and is a deliberate TASK-23018 contract -- typing advertises that a price exists, the estimate is derived when the pointer reaches Send.

STILL OPEN, needs a design decision I should not take alone: the estimate itself remains hover-only, so it is unreachable for the keyboard-first audience this product is built for. Textual tooltips are pointer-driven, so exposing it needs a real surface -- most plausibly an entry in the composer's own actions menu, or the Inspector. Filing that as its own task would be more honest than leaving it implied here.
<!-- SECTION:NOTES:END -->
