---
id: TASK-25828
title: Console status bar clips mid-label instead of prioritising its content
status: Done
assignee: []
created_date: '2026-08-31 05:09'
updated_date: '2026-08-31 13:27'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The status strip renders its segments at fixed order and truncates whatever overflows. At eighty columns it ends mid-word on the model segment, so the model name, one of the most consequential facts in the strip, is the first thing lost. Segment content also changes shape between states, reflowing the whole row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The status strip degrades by prioritising segments rather than clipping mid-label
- [ ] #2 Provider and model remain visible at the narrowest supported width
- [ ] #3 Segments do not reflow the row when their content changes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
LARGELY INVALID AS FILED -- the premise 'clips instead of prioritising' is wrong on both halves.

It DOES prioritise, deliberately. console_status_chips.py builds the strip in a
considered order and says so: 'The two-axis Library policy is the primary
permission readout. Keep it ahead of provider/model metadata so both axes remain
painted before horizontal overflow begins.' Provider/model losing the race at 80
cols is the designed trade-off (permission readout outranks model metadata), not
an accident. Reordering to keep the model visible would invert an explicit
decision.

And it does not clip -- the chips live in a HorizontalScroll
(#console-status-chip-scroll). Content past the edge is scrolled, not destroyed,
and the chips are focusable, so keyboard traversal brings them into view.

RESIDUAL VALID ISSUE, much narrower than filed: the strip sets
scrollbar_size_horizontal = 0, so nothing signals that it continues past the
right edge. At 80 cols the cut lands mid-word ('...Model: local-model' -> 'Mode'),
which reads as a rendering fault rather than 'there is more this way'. A
conventional scrollbar is not available -- the strip is locked to height 1 -- so
this wants an inline end-of-strip affordance instead. Left unfixed rather than
risk the height-1 layout on a cosmetic cue.

No test pins the ordering or the scroll behaviour; the intent lives only in that
code comment.
<!-- SECTION:NOTES:END -->
