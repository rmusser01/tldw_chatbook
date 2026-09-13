---
id: TASK-31572
title: Library media - consolidate the focus-region computation into one helper
status: To Do
assignee: []
created_date: '2026-09-05 03:23'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The media shell computes which region holds focus (Items, Reader, rail) in several places with slightly different rules; wave 4 PR B keyed the list gates on it and noted the duplication. One helper, _library_media_focus_region(), should own the rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single helper answers which region holds focus and every gate uses it
- [ ] #2 No behaviour change: the existing shell, reader-flow and multiselect tests pass unchanged
<!-- AC:END -->
