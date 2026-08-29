---
id: TASK-23107
title: >-
  Schedules bulk-mark mechanism is invisible: no legend, count, or bulk-mode
  hint
status: Done
assignee: []
created_date: '2026-08-28 14:06'
updated_date: '2026-08-29 02:24'
labels:
  - ux
  - schedules
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing x prefixes a row with a filled-circle mark and missed-while-away rows carry a diamond glyph, but there is no legend, no marked-count, and no indication that space/d switch to bulk mode when marks exist. Cheap fix shape: reuse the existing #scheduling-pane-notice line ('2 marked - space toggles all, d deletes all, esc clears'). P2 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tbook-ui-screens-scheduling-schedules-workbench-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When one or more rows are marked, visible text states the marked count and which keys act on all marked rows and how to clear the marks
- [ ] #2 The missed-while-away glyph has a visible text explanation on screen
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The pane notice now states the actionable marked count, how many marks the filter is hiding, and which keys act on them ('2 marked (1 hidden by the filter) - space toggles all, d deletes all, esc clears'), plus a legend for the ran-late glyph. The review round found the underlying defect: marking had no type guard, so a read-only projection row could be marked, and with no actionable marks the bulk verbs fell through to the single-row action on the highlighted, unmarked task -- pressing d opened a delete dialog for a task the user never marked. Marking is now reminder-only, marks are pruned on load, and bulk verbs never fall through while marks exist. PR #2169.
<!-- SECTION:NOTES:END -->
