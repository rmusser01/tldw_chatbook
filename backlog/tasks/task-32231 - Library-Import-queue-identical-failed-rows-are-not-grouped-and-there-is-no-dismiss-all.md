---
id: TASK-32231
title: >-
  Library Import queue: identical failed rows are not grouped and there is no
  dismiss-all
status: Done
assignee: []
created_date: '2026-09-10 14:57'
updated_date: '2026-09-10 17:28'
labels:
  - library
  - import
  - ux
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A folder import with one cause produced four identical `✗ failed` rows × three buttons and no way to clear them at once. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 30.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identical outcomes group into one row (`✗ failed · 4 files · reason`) with Show the N files / Retry all / Dismiss all, expanding on demand
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing unit tests for group_ingest_queue_rows in Tests/Library/test_library_ingest_state.py (key = (state, reason); settled states only; singleton keeps its own line).
2. Implement group_ingest_queue_rows + IngestOutcomeGroup in Library/library_ingest_state.py (IngestQueueGroup is taken by the batch grouper).
3. Canvas renders a multi-member group as one Static + Show the N files / Retry all / Dismiss all.
4. Controller handlers reuse the existing per-row retry/dismiss seams once per member.
5. Docs + live verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`group_ingest_queue_rows(rows, expanded=())` in Library/library_ingest_state.py collapses CONTIGUOUS runs of queue rows sharing `(state, reason)` into one `IngestOutcomeGroup` (glyph / line / members / expanded, plus `key`, `can_retry`, `can_dismiss` properties). Only settled states group (failed, skipped, cancelled) -- an active row's per-file progress is the point -- and a reason that names its own file keeps its row, so grouping never hides a per-file cause. Contiguity means the queue is never reordered and a group of one renders exactly what it rendered before. `IngestQueueRow` gained a `reason` field (the basename-stripped plain-language cause) as the group key; the three settled builders populate it.

Canvas: `LibraryIngestQueuePanel.compose` now iterates groups; the per-row rendering moved verbatim into `_compose_queue_row(index, row, headers_before)` (byte-identical body, re-indented) so a collapsed group can skip its members. A multi-member group paints one `Static` (`#library-ingest-group-<leader job id>`) carrying the row's own severity classes plus a three-Button action row: 'Show the N files' / 'Hide the N files', 'Retry all' (only when every member is retryable), 'Dismiss all'. Row ids stay keyed to their global queue position, so collapsing never renumbers one.

Controller: `handle_library_ingest_group_expand` toggles panel-owned `expanded_groups` (transient disclosure, not queue state) through the task-32216 refocus wrapper; retry-all and dismiss-all re-derive the group from the CURRENT rows (same staleness reasoning as the per-row actions) and reuse the existing per-job seams once per member -- `handle_library_ingest_dismiss`'s body was split into `_dismiss_library_ingest_job` so the Recent-imports ledger record happens exactly as it does for one row.

Technical trade-off: the produced dataclass is named `IngestOutcomeGroup`, not `IngestQueueGroup` as the plan's interface said -- that name is already task-2221's per-SUBMISSION batch header in the same module. The group key is the leading row's job id rather than a hash of (state, reason): unique, id-safe, and already how every per-row action addresses its target.

Live-caught in verification: the batch header rendered twice on expansion (once above the group, once from the leader's own row) -- fixed by popping the leader from the members' header map, pinned by a test.

Files: Library/library_ingest_state.py, Widgets/Library/library_ingest_canvas.py, UI/Library_Modules/library_ingest_controller.py, UI/Screens/library_screen.py (3 delegators), Tests/Library/test_library_ingest_state.py, Tests/UI/test_library_crit9_import.py, Docs/User_Guide/library/import-and-export.md.
<!-- SECTION:NOTES:END -->
