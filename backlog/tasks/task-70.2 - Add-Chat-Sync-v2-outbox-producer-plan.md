---
id: TASK-70.2
title: Add Chat Sync v2 outbox producer plan
status: To Do
labels:
- sync
- sync-v2
- chat
- local-first
priority: high
parent_task_id: TASK-70
documentation:
- Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Chat Sync v2 content-producing path so conversation and message changes can be represented as ordered encrypted outbox envelopes that preserve resumed-chat continuity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User and assistant Chat messages enqueue ordered Sync v2 envelopes only after they represent durable local message state.
- [ ] #2 Conversation identity, message roles, parentage, ordering, and message variants have restore-compatible metadata.
- [ ] #3 Streaming, failed-send, and regenerated-message cases do not produce misleading final-message envelopes.
- [ ] #4 Tests cover local-first success, envelope shape, ordering, variants, and pending profile summary counts.
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
