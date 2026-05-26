---
id: TASK-70.1
title: Add Notes Sync v2 outbox producer plan
status: To Do
labels:
- sync
- sync-v2
- notes
- local-first
priority: high
parent_task_id: TASK-70
documentation:
- Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first content-producing Sync v2 path for Notes so local note create, update, and delete operations can enqueue durable encrypted outbox envelopes for manual sync without requiring server availability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Notes create, update, and delete operations enqueue Sync v2 outbox envelopes for the active server profile personal dataset.
- [ ] #2 Local Notes operations remain successful and recoverable when the server is unavailable.
- [ ] #3 Tests cover envelope identity, domain scope, idempotency, encryption boundary, and pending profile summary counts.
- [ ] #4 No background sync, automatic push, or broad domain sync is introduced.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
