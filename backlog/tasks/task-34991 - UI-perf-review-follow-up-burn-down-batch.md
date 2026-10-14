---
id: TASK-34991
title: UI-perf review follow-up burn-down batch
status: In Progress
created_date: 2026-10-14 19:52
labels:
- performance
- ui-perf
priority: medium
updated_date: 2026-10-14 19:53
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The independent review of PR #3083 documented follow-ups that did not belong in that PR. This batch burns them down: the dev-side broken console_speech_snapshot import chain, the resurrected dead save_history, egress-pool eviction not closing sessions plus missing shutdown wiring, and the spawn-worker failure-notifier gap. Task-34929 (Watchlists revisit cache) is implemented under its own reopened task file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 console_speech_snapshot imports resolve cleanly again,ChatPersistenceService.save_history and its loaded_message seam are deleted with their orphaned tests,Pooled egress sessions close on eviction and close_pooled_default_sessions is wired into app shutdown,Clipboard and link-open spawn failures surface a sanitized failure notice instead of a success toast
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-export resolve_console_message_presentation from console_chat_store\n2. Delete dead save_history + seam + orphaned tests\n3. Close evicted egress sessions; wire pool shutdown\n4. Sanitized failure notifier for clipboard/link spawns\n5. Watchlists revisit cache under task-34929\n6. Tests per item, PR, queue
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
