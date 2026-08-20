---
id: TASK-18960
title: >-
  Unscoped Checkbox height:2 rule blanks bordered checkboxes app-wide
status: To Do
assignee: []
created_date: '2026-08-20 00:00'
labels:
  - css
  - ux
priority: medium
dependencies: []
---

## Description (the why)

Found during TASK-17961's painted-frame verification: the Settings ▸ Workspaces
"Show archived" Checkbox renders with zero content rows in EVERY state —
blurred included — so it is a different defect from the focus-outline family
17961 fixed. An unscoped `Checkbox { height: 2; }` rule in
`css/components/_conversations.tcss` applies app-wide; combined with a
`border: tall` (2 rows of chrome) the widget's content area is squeezed to
0 rows. Any bordered, non-compact Checkbox outside the conversations UI is
affected. TASK-17961's new painted-frame test file
(`Tests/UI/test_compact_focus_outline_render.py`) demonstrates the probe
technique; its Implementation Notes record the empirical evidence.

## Acceptance Criteria (the what)

- [ ] The `height: 2` Checkbox rule is scoped to the conversations UI it was written for (or retired if unneeded there)
- [ ] Settings ▸ Workspaces "Show archived" renders its label and check glyph in blurred AND focused states (painted-frame test, production bundle)
- [ ] An app-wide sweep confirms no other unscoped bare-type height rules squeeze bordered widgets to zero content rows (grep `_*.tcss` for bare `Checkbox`/`Switch`/`RadioButton` type rules with height pins; each hit scoped or justified)
- [ ] Bundle rebuilt from module sources; `check_bundle_sync.py` green
