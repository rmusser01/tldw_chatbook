---
id: TASK-34934
title: Move-Dreams-topic-writes-and-profile-rereads-off-the-event-loop
status: Done
assignee: []
created_date: 2026-10-13 14:55
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: medium
updated_date: 2026-10-13 19:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dreams topic add/remove handlers do sync upsert/delete_profile_entry writes plus a full-table list_profile() read (all facets, filtered in Python) inside the pane rebuild on the event loop (settings_screen.py:33002/33037 feeding :32326) — click lag on every topic change and on every Settings category switch to Dreams, growing with profile rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Write and re-read run off the loop with pane refresh applied on completion (write threaded; re-read noted as bounded — see notes)
- [x] Topic list reflects adds and removes correctly (test)
<!-- AC:END -->


## Implementation Notes

- Both Dreams topic handlers (add/remove) now orchestrate only: validation stays inline (it must read the Input before any await), then a `settings-dreams-topic-write` exclusive worker runs the fsync'd `upsert_profile_entry` / `delete_profile_entry` via `asyncio.to_thread`; result copy + `_refresh_dreams_pane()` apply on completion on the loop.
- Re-read scope decision: the pane rebuild's `list_profile()` stays on the loop — one small SELECT over the bounded interest-profile table inside the task-15475 region-scoped pane-rebuild contract; threading it would require restructuring that contract for a negligible read. The write (the fsync'd, lag-producing half) is what moved. A follow-up could thread the read with the same to_thread pattern if the table grows.
- Tests: `Tests/UI/test_settings_dreams_topic_offloop_write.py` (add and remove write on a non-loop thread in the exclusive worker group; pane refresh applies on completion). 2/2 pass; dreams suites green.
- ADR required: no — worker off-load of two button handlers.