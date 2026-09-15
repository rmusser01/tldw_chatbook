---
id: TASK-32613
title: >-
  Library Notes: a third of tab stops paint no focus, the stop count is unstable
  across modes, and Tab plus Enter from Preview exits the note
status: To Do
assignee: []
created_date: '2026-09-15 06:40'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D17, D8 and K11, personas Sam and Alex, edit workflow.

What happened. B enumerated the stops by pressing Tab and grepping every capture for a focus glyph: the notes-list toolbar with an active filter shows no indicator at stops 5 and 7 of 8 (B cap 10), the sync configuration form shows none at stops 1-4 (B section 4), the Session Git commit form shows none at stops 1-4 (B cap 43). Roughly a third of all stops. This sits beside a genuine strength that must be protected: the indicators that do exist are shape-based -- buttons as bars, list rows as a leading block, text areas as a thickened border, inputs as a full box -- and all four survive a monochrome dump.

Two behaviours compound it. From Preview, the obvious way to Info (Tab then Enter) exits the note entirely, returning the right pane to 'Select a note to edit it here' with no warning (B cap 15, K10). And the toolbar's tab-stop count is not stable across mode switches: the identical Shift+Tab x5 that reached Preview once landed on nothing the next time (B cap 17, K11).

Cause INFERRED for all three -- focus order left to the framework's defaults; not traced. Wave 4's task-32537 (#2683) added Preview's footer chip but did not change the tab order, and task-32550 fixed a different tab-count complaint on the filter.

Adjacent: the Info tier's missing chips are filed separately; this task is the indicator and the order.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every tab stop inside the Notes surface paints a focus indicator that survives a monochrome capture
- [ ] #2 The number of Tab presses from the body to a given control does not change with filter state or previous mode
- [ ] #3 Tab from Preview reaches the next editor control, and no single Tab-plus-Enter from a reading mode closes the note without warning
- [ ] #4 A test enumerates the stops in each pane and fails when one has no indicator
<!-- AC:END -->
