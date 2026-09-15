---
id: TASK-32592
title: Keep every More-menu destination visibly keyboard reachable at 80x24
status: To Do
assignee: []
created_date: '2026-09-15 00:18'
labels:
  - design-system
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-component-first-ui-audit.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 80x24 the More menu clips Research and Meetings. Keyboard focus can reach Meetings without painting its label or focus cue, so users cannot see which destination they will activate. The defect reproduces in both themes and exists on current dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every registered destination can be visibly focused and activated through More at 80x24 in both textual-dark and textual-light.
- [ ] #2 Moving focus to an initially offscreen destination brings its complete label and focus indicator into view.
- [ ] #3 Opening, traversing and dismissing More preserves correct route selection and returns focus predictably without activating a different destination.
- [ ] #4 The same menu remains bounded and usable through a 120x40 to 80x24 to 120x40 resize sequence.
<!-- AC:END -->
