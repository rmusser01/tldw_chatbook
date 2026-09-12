---
id: TASK-32306
title: >-
  DECISION NEEDED — a blocked Workspace Handoff row wraps over 3–4 rail lines at
  34 cells
status: To Do
assignee: []
created_date: '2026-09-11 00:54'
labels:
  - library
  - ux
  - critique-9
  - decision-needed
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-32230 (critique-9 rail branch, PR #2581) made the Details ▸ Handoff row actionable: it now carries the reason and the next step instead of '● 1 blocked'. At the rail's 34-cell width that prose wraps over three to four lines. Shortening it drops either the reason or the next step; moving the remedy into a tooltip re-introduces hover-only meaning, which the wave's copy rule forbids. This is a product call the implementer and reviewer both declined to make.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded (keep the wrap, shorten the copy, or move the remedy) and applied
<!-- AC:END -->
