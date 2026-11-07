---
id: TASK-34389
title: Define and wire a sync_log retention policy
status: Done
created_date: 2026-10-11 21:44
labels:
- performance
priority: high
updated_date: 2026-10-12 01:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
sync_log triggers embed a second full copy of every message/media body (ChaChaNotes_DB messages_sync_* triggers, Client_Media_DB_v2 _log_sync_event payloads) and the shipped prune APIs (prune_sync_log, delete_sync_log_entries*) have zero production callers: the table grows O(total mutations) forever with ~2x write amplification. Needs a decided retention window and payload-slimming policy before wiring prune into startup/maintenance. ADR required: retention/deletion policy over user data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR records the retention window, what may be pruned (acked entries? tombstones?), and the payload-slimming decision,Retention job runs at startup/maintenance without blocking the UI thread,sync consumers (Prompts_Interop reader) still observe all unpruned changes,Pruning is config-visible (documented setting or documented fixed policy)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:IMPLEMENTATION_PLAN:BEGIN -->
ADR: yes — [ADR-229: Time-based retention window for the local sync journals](../decisions/229-sync-log-retention-policy.md) (retention/deletion policy over user data; records the 90-day default window, the age-based prune scope, and the explicit deferral of payload slimming pending a consumer audit).

1. Add `[sync] sync_log_retention_days = 90` to the default config TOML and project the section through `load_settings`.
2. Add `delete_sync_log_entries_older_than(cutoff)` to `ChaChaNotes_DB` and `Client_Media_DB_v2` (one parameterized DELETE over the indexed `timestamp` column).
3. Wire `_perform_sync_log_retention` into the post-first-paint maintenance path in `app.py` (`schedule_media_cleanup`), reading the knob via `get_cli_setting`; `<=0` disables.
4. Tests: `Tests/DB/test_sync_log_retention.py` (prune semantics, boundary, disable switch, config defaults) + register the new interval clock in the timer census.
<!-- SECTION:IMPLEMENTATION_PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented and verified on perf/34386-34388-hot-path-efficiency; see task file Implementation Notes; PR #3083
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
