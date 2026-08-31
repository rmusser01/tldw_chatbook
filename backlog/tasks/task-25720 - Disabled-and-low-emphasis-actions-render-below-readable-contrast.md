---
id: TASK-25720
title: Disabled and low-emphasis actions render below readable contrast
status: To Do
assignee: []
created_date: '2026-08-31 05:08'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A disabled primary button renders at roughly 1.4 to 1 against its background while the adjacent dismissive action renders near 14 to 1, so the dialog appears to offer only the dismissive choice. The same pattern makes the first-run final step's three exit controls render at roughly 1.65 to 1. A disabled control must still be legible enough to be understood as unavailable rather than absent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Disabled control text meets at least 3 to 1 contrast against its background
- [ ] #2 A primary action remains the most visually prominent control in its dialog in every state
- [ ] #3 No interactive control renders below 3 to 1 in any state
<!-- AC:END -->
