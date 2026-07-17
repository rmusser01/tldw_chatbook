---
id: TASK-297
title: Migrate the Cohere summarization helper to the v2 /chat API
status: To Do
assignee: []
created_date: '2026-07-17 21:47'
labels:
  - providers
  - maintenance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-267 migrated chat_with_cohere to Cohere v2 /chat, but Summarization_General_Lib.py (lines ~976/1029) still calls v1 /chat — the two Cohere code paths now diverge on API version (final-review finding, PR for task-267). Migrate the summarization path to v2 for consistency before Cohere retires v1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No tldw_chatbook code path calls Cohere v1 /chat
- [ ] #2 Summarization behavior pinned by tests before and after the migration
<!-- AC:END -->
