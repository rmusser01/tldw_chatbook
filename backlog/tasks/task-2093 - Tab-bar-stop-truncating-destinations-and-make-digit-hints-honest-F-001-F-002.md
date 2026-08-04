---
id: TASK-2093
title: >-
  Tab bar: stop truncating destinations and make digit hints honest (F-001,
  F-002)
status: To Do
assignee: []
created_date: '2026-08-03 17:25'
labels:
  - ux-review
  - chrome
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At <=100 cols tabs collapse ('8 Workflows' -> '8') and later destinations become unreachable; labels '1 Home' imply bare-digit keys but the binding is Ctrl+digit. Evidence: library/roleplay/mcp-100x30.png, app.py:3493. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All destinations remain reachable at 100 cols (ellipsis/overflow/scroll),Digit affordance labels match the actual keybinding,Rendered-layout test at 100 cols
<!-- AC:END -->
