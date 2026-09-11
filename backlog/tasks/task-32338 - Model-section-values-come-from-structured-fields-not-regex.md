---
id: TASK-32338
title: >-
  Model section values come from structured fields not regex
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review D2. The left rail Model section parses display values out of formatted strings with regexes (left_rail.py ~436-447: r'T ([\d.]+)', r'max_tokens (\d+)') and shows a dash on mismatch. Add structured fields to ConsoleSettingsSummaryState and render from them.

Filed from the 2026-09-10 Console rail UX review (review item D2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Temperature and max-tokens values render from structured state fields, not string parsing
- [ ] #2 A summary state missing those fields still renders a placeholder (no crash, no wrong value)
- [ ] #3 Parsing regexes are removed from the rail
<!-- AC:END -->
