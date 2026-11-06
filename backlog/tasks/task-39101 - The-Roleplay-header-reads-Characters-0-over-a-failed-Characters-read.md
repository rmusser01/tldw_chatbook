---
id: TASK-39101
title: 'The Roleplay header reads Characters 0 over a failed Characters read'
status: To Do
assignee: []
created_date: '2026-11-06 09:00'
labels:
  - roleplay
  - bug
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the Characters read fails, the Roleplay header shows "Characters — who the AI plays · 0", which tells the user they have no characters when the read actually failed. It exists on dev today; this task's PR does not fix it. Found by the Roleplay B2b follow-up burn-down (2026-11-05), observed on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed Characters read shows a failure state in the header and list, not a zero count
- [ ] #2 A test pins the failed-read header text
<!-- AC:END -->
