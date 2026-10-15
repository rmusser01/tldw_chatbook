---
id: TASK-35026
title: UI-perf review follow-up burn-down batch
status: Done
assignee: []
created_date: '2026-10-14 13:05'
labels:
- performance
- ui-perf
dependencies: []
priority: medium
updated_date: '2026-10-14 13:05'
---

## Description

The independent review of PR #3083 documented follow-ups that did not belong in that PR. This batch burns them down: the dev-side broken console_speech_snapshot import chain, the resurrected dead save_history, egress-pool eviction not closing sessions plus missing shutdown wiring, and the spawn-worker failure-notifier gap. The Watchlists revisit cache landed under its own reopened task (34929, renumbered 34967).

## Acceptance Criteria
- [x] console_speech_snapshot imports resolve cleanly again
- [x] ChatPersistenceService.save_history and its loaded_message seam are deleted with their orphaned tests
- [x] Pooled egress sessions close on eviction and close_pooled_default_sessions is wired into app shutdown
- [x] Clipboard and link-open spawn failures surface a sanitized failure notice instead of a success toast

## Implementation Plan

1. Fix the import chain
2. Delete the dead code
3. Close evicted sessions and wire shutdown
4. Sanitized spawn notifiers

## Implementation Notes

- Import chain: `resolve_console_message_presentation` is re-exported through the store seam (`console_chat_store`) that `console_speech_snapshot` imports it from; the resolver lives in `console_roleplay_identity`. The broken chain had been failing every suite importing the snapshot module, pre-existing on dev. Affected suites re-run green (quarantine store test, buddy speech 15/15).
- Dead code: `save_history` (~165 lines incl. `_extract_message_payload`), the `loaded_message` fast-path seam on `update_message`, and four orphaned tests deleted — zero production callers, previously deleted in dbc8e01e43 and resurrected by the perf PR. Suite 101/101 after.
- Egress: pool eviction now CLOSES the evicted session, and `_shutdown_app_owned_lifecycles` drains the pool via `close_pooled_default_sessions()` after every other owner. 3 new tests; egress suite 119/119.
- Spawn notifiers: the clipboard and link-open thread workers now report through `call_from_thread` — sanitized error/warning notices naming only the exception class — instead of swallowing runtime failures under the success toast. 4 new tests pin both paths; transcript suite 17/17. The misleading collections comment is corrected.
- The Watchlists revisit cache was attempted under its own task (34929→34967) and REVERTED again: the read-cache broke the P0 load/mount ordering contract (panes mounting after a cached instant load stay in LOADING). Task reopened with the finding; see its notes. Dev's duplicate 348xx task IDs were verified NOT to exist on current dev — no action needed.
- ADR required: no — four bounded fixes plus one cache, each inside existing contracts.

## Renumbering provenance

Renumbered from task-34991 to TASK-35026 (2026-10-15): dev minted the same id after this branch cut; per the owner rule (TASK-19601) the dev-side arrival keeps the id on merge.
