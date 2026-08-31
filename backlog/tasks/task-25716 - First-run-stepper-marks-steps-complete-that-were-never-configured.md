---
id: TASK-25716
title: First-run stepper marks steps complete that were never configured
status: To Do
assignee: []
created_date: '2026-08-31 05:08'
labels:
  - console
  - ux-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The setup wizard stamps a checkmark on each step the user passes through, regardless of whether its required values were set. Its own summary screen contradicts the stepper, reporting the provider and model as not configured while every step chip above shows complete. Users read the stepper, not the summary, and believe setup succeeded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A step chip shows complete only when that step's required values are set
- [ ] #2 A skipped or incomplete step is visually distinct from a completed one
- [ ] #3 The stepper and the summary screen never disagree about the same step
<!-- AC:END -->
