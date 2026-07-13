---
id: TASK-225
title: Escape save path toast in Save Image worker
status: To Do
assignee: []
created_date: '2026-07-13 11:15'
labels:
  - console
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Save Image worker's success toast notify(f"Image saved to {target}") interpolates the config-derived save path unescaped into a markup-rendering surface. Not user-file-derived (config + generated filename), so low risk, but a [chat.images].save_location containing markup-like tokens would render wrong. Escape it like the sibling toasts fixed in f1824513.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Save-path toast escapes the interpolated path per repo convention
- [ ] #2 A path containing markup-like tokens displays literally
<!-- AC:END -->
