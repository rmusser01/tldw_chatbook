---
id: TASK-34389
title: Define and wire a sync_log retention policy
status: To Do
created_date: 2026-10-11 21:44
labels:
- performance
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
sync_log triggers embed a second full copy of every message/media body (ChaChaNotes_DB messages_sync_* triggers, Client_Media_DB_v2 _log_sync_event payloads) and the shipped prune APIs (prune_sync_log, delete_sync_log_entries*) have zero production callers: the table grows O(total mutations) forever with ~2x write amplification. Needs a decided retention window and payload-slimming policy before wiring prune into startup/maintenance. ADR required: retention/deletion policy over user data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR records the retention window, what may be pruned (acked entries? tombstones?), and the payload-slimming decision,Retention job runs at startup/maintenance without blocking the UI thread,sync consumers (Prompts_Interop reader) still observe all unpruned changes,Pruning is config-visible (documented setting or documented fixed policy)
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
