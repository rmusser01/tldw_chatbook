---
id: TASK-721
title: Library rail Conversations count disagrees with actual conversation count
status: To Do
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - library
  - bug
  - investigation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With six non-deleted conversations in the chat DB the Library rail showed Conversations (1) (captures cap-21/24). The count appears to exclude workspace-scoped conversations, disagreeing with the Console browser and reading as data loss. Root cause untraced. Finding m5.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The counting rule is identified and documented
- [ ] #2 The displayed count matches what the Conversations view actually lists
- [ ] #3 If workspace-scoped conversations are intentionally excluded the label says so
<!-- AC:END -->
