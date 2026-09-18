---
id: TASK-32816
title: Keep conversation Appearance actions visible at compact terminal sizes
status: To Do
assignee: []
created_date: '2026-09-18 18:51'
labels:
  - ui
  - console
  - design-system
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-32813 native dark/light review at 80x24 found the conversation Appearance Cancel row below the viewport. All computed stylesheet rules matched the original pre-consolidation CSS. The modal uses height29 with max-height100% and retains other fixed-height children, so focusing the Cancel control does not paint its action row. The narrow palette also wraps the None label. Existing captures are in the CSS consolidation QA report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 80x24 in dark and light themes, Appearance actions remain visibly reachable by keyboard and pointer without widening the terminal.
- [ ] #2 The palette labels and controls remain readable at compact and wide sizes while preserving saved icon and color behavior.
- [ ] #3 Mounted geometry and native screenshots verify the compact layout and cancellation without committing changes.
<!-- AC:END -->
