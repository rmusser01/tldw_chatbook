---
id: TASK-1497
title: Wizard selection states rely on color alone
status: Done
assignee: []
created_date: '2026-07-31 00:22'
updated_date: '2026-07-31 01:13'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: all RadioButtons render the identical inner glyph; selected/unselected/highlighted differ only by color — WCAG 1.4.1 failure, and the ambiguity already produced the looks-selected-commits-nothing bug. Needs structural state cues.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selected radio rows are distinguishable in a monochrome capture (distinct glyph or text-style)
- [ ] #2 Applies to provider, model, RAG, theme, and splash lists
- [ ] #3 Snapshot/Pilot test asserts the structural cue
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SetupRadioButton subclass swaps BUTTON_INNER per state (● selected / ○ unselected) before ToggleButton._button renders; bold text-style on selected rows as second cue. All 11 wizard RadioButton sites swapped; membership/isinstance checks unaffected. Tests assert structural glyph difference and that no plain RadioButton remains in the wizard; live-verified in monochrome tmux capture.
<!-- SECTION:NOTES:END -->
