---
id: TASK-25824
title: >-
  Unsaved-changes dialog offers Cancel and Discard changes without
  distinguishing them
status: Done
assignee: []
created_date: '2026-08-31 05:08'
updated_date: '2026-08-31 06:36'
labels:
  - console
  - ux-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dismissing a settings modal with unsaved edits silently adds a third button so the row reads Save, Cancel, Discard changes. Nothing indicates whether Cancel abandons the edits or abandons the dismissal, which is the classic ambiguous-dialog trap. Users must guess which control preserves their work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dialog offers two unambiguous choices covering discard and continue editing
- [ ] #2 Each button names the outcome it produces rather than a generic verb
- [ ] #3 Controls appearing in response to unsaved edits are announced rather than added silently
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Partly overstated as filed: the dialog DOES announce 'Unsaved changes. Save them or choose Discard changes.' (pinned by an existing test), so the 'added silently' criterion was already met. The real defect stands: once Discard appears, 'Cancel' silently stops meaning 'abandon my edits' and starts meaning 'stay here' -- one word, two outcomes. Fix: relabel it 'Keep editing' at the moment Discard is revealed, so the pair names the two outcomes actually on offer.
<!-- SECTION:NOTES:END -->
