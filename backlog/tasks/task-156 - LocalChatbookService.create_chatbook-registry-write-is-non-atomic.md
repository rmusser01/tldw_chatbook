---
id: TASK-156
title: LocalChatbookService.create_chatbook registry write is non-atomic
status: To Do
assignee: []
created_date: '2026-07-11 22:01'
labels:
  - follow-up
  - chatbooks
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
create_chatbook does a read-modify-write of the chatbook registry with no lock. Under the by-design concurrent-export path (navigate away mid-export, start a second export — two OS threads), overlapping calls can lose-update a record or collide on next_id. atomic_write_json keeps the file valid JSON (lost record, never corruption). Narrow window; pre-existing service limitation, newly reachable via F4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Concurrent create_chatbook calls do not lose records or collide on next_id,Serialization/locking added around the registry read-modify-write
<!-- AC:END -->
