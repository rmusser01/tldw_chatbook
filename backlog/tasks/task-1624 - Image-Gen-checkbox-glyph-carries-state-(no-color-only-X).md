---
id: TASK-1624
title: 'Image Gen: checkbox glyph carries state (no color-only X)'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - critique-r3-p2
dependencies: []
priority: medium
---

## Description (the why)

Critique round 3 P2: the backend-row Checkbox glyph paints 'X' in BOTH states, distinguishing them by color alone — it read as checked either way in reduced-color terminals; the attached On/Off word was already the text carrier (task-1561).

## Acceptance Criteria (the what)

- [x] The X glyph renders only while checked (off state shows an empty glyph)
- [x] The On/Off word remains attached to the control

## Implementation Notes

CSS-only: `ImageGenSettingsPanel Checkbox > .toggle--button` fg matches the panel surface at rest; `.-on` variant paints $success. Scope note: full Checkbox->chip-Button idiom unification was considered and rejected for this batch (six wiring sites + handler churn); the DESIGN.md convention now names both sanctioned idioms.
