---
id: TASK-23194
title: 'Console Context rail: remove duplicate, dead and jargon content'
status: To Do
assignee: []
created_date: '2026-08-29 21:56'
labels:
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four content defects in the Context rail: 'No character in this chat' renders twice from two different widgets; the Agent section mounts three zero-size focusable widgets including a text Input; four controls (Switch, Star, star glyph, Clear) ship disabled with no explanation; and 'Local stars unavailable' exposes developer language to users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The no-character empty state renders exactly once
- [ ] #2 No zero-size focusable widget is mounted in the Context rail
- [ ] #3 Controls that cannot act are hidden rather than disabled, or explain their precondition
- [ ] #4 'Local stars unavailable' is removed or replaced with user-facing copy
<!-- AC:END -->
