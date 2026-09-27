---
id: TASK-33069
title: Theme filter no-match state has a Clear chip and sane fallback
status: To Do
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P3. With no matches the spec §9 Clear filter chip is missing, the preview keeps the last theme under a blank title, and clearing the filter highlights row 1 (possibly an unreadable file) instead of the active theme. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A no-match filter shows a Clear filter control and no stale preview
- [ ] #2 Clearing the filter highlights the previously highlighted theme or the active theme
<!-- AC:END -->
