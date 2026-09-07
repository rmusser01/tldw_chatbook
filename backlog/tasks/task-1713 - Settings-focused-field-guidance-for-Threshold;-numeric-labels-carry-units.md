---
id: TASK-1713
title: 'Settings: focused-field guidance for Threshold; numeric labels carry units'
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

Critique round 3 P2: with Threshold focused the inspector promised setting-specific guidance and answered 'No field-specific guidance is active right now'; numerics rendered unit-less ('Threshold 50', 'Web font 12') — root cause partly the 12-cell label column truncating full labels.

## Acceptance Criteria (the what)

- [x] Threshold has a dedicated guidance entry (purpose, saved-as, applies)
- [x] Every Console Behavior Input is covered; the fallback only renders while nothing is focused
- [x] Label column widened so full labels render; units added (chars/px/themes); Image Gen backend rows keep 12 (120-col geometry guard)

## Implementation Notes

Threshold entry + id registration; `.settings-input-label` width 12->24 with a scoped 12 override for `.settings-imagegen-backend-row`; labels: 'Threshold (chars)', 'Display cap (chars)', 'Web font size (px)', 'Palette limit (themes)', 'Smooth scrolling'; splash label widths 17->20 so 'Animation speed (x)' stops clipping.
