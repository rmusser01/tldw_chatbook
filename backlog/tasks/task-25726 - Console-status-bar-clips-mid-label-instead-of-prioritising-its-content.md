---
id: TASK-25726
title: Console status bar clips mid-label instead of prioritising its content
status: To Do
assignee: []
created_date: '2026-08-31 05:09'
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
