---
id: TASK-31221
title: Media type chooser - options are invisible (zero-height OptionList)
status: To Do
assignee: []
created_date: '2026-09-03 22:30'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique P1: choices.styles.height = min(8, max(1, len(options))) ignores OptionList's 2-row default chrome, so the common 2-option case renders an empty bordered band and selection is blind (verified in code; the Console popup rule documents the exact cost).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every type option is visible for any option count
- [ ] #2 The highlighted option is visually indicated
<!-- AC:END -->


## Renumbering

Renumbered from task-31203 on 2026-09-03: id collision with an older dev arrival (owner rule TASK-19601; older keeps the id).
