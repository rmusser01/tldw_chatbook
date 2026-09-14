---
id: TASK-32581
title: >-
  Library Notes: #file-notes-git-bulk-toggle measures 31 cells and cannot fit
  below about 36 columns
status: To Do
assignee: []
created_date: '2026-09-14 22:46'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recorded by wave-4 group 7, and corrected in the same round: the comment shipped in that work says the toggle is '35 cells wide' when it measures 31 — the wave's recurring failure in miniature, a number written from memory rather than measured. The real constraint stands either way: at 31 cells the control cannot fit a panel narrower than roughly 36 columns whatever -stack-actions does, so below that width it is simply hidden, and the untrusted state hides it anyway (bulk_toggle.display requires _trusted). A narrow-terminal user therefore has no bulk stage or unstage. Decide whether it gets a shortened label, a stacked two-line form, or an explicit statement that bulk actions need a wider terminal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 35-cell comment is corrected to the measured 31 wherever it appears in source or task notes
- [ ] #2 Below the fit width the panel states that bulk actions need more room rather than silently omitting the control
- [ ] #3 Measured at 40, 36 and 32 columns with a capture each
<!-- AC:END -->
