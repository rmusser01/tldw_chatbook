---
id: TASK-22210
title: Deduplicate and serialize media reading-progress writes
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 21:52'
labels:
  - performance
  - library
  - database
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22210).

New with PR #2064. `_capture_library_media_loaded_progress`
(`library_screen.py:33239-33262`) fires an SQLite upsert worker on every traversal step
and every mode switch, with no `exclusive=True` and no equality skip — holding an arrow
key through 30 rows queues 30 concurrent `to_thread` writers contending for the same
write lock.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An unchanged offset produces no write; identical consecutive captures are skipped
- [x] #2 In-flight progress writes are superseded, not stacked (exclusive worker group or coalescing)
- [x] #3 A 30-row traversal produces at most one settled write (probe)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red-first probes in Tests/UI/test_library_media_reader_flow.py: (a) pilot-driven 30-row traversal counting update_reading_progress calls + library_media_reading_progress worker spawns, assert <=1 settled write and 1 spawn; (b) unit: identical consecutive captures produce one queued write; (c) unit: capture equal to the DB-fetched cached offset produces no write. Run, confirm red.\n2. Implement last-write-wins coalescing in library_screen.py mirroring the existing _queue_library_lifecycle_persistence/_drain_library_lifecycle_persistence precedent: per-item pending dict + single drainer worker + in-flight marker + persisted-offset map (seeded by _cache_library_media_reading_progress, updated on write success). Equality skip in _capture_library_media_loaded_progress. NO exclusive=True cancellation (repo lesson: cancellation-based supersede is unsound for durable writes).\n3. Teardown: on_unmount awaits the drainer, re-queues a cancelled in-flight value, drains residue inline (idempotent upsert makes the double-write case harmless).\n4. Update existing SimpleNamespace fakes in the flow test for the new attributes.\n5. Mutation-test (drop equality skip -> probes red; drop coalescing -> probes red), targeted suites + --collect-only sweep with tee, preflight, before/after numbers, commit + push.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the per-capture run_worker SQLite writer with last-write-wins coalescing, mirroring the file's lifecycle-persistence precedent (_queue/_drain_library_lifecycle_persistence).

MECHANISM (chosen: coalescing, NOT exclusive=True): _capture_library_media_loaded_progress now routes through _queue_library_media_progress_write -> per-item pending dict + ONE serial drainer worker (_drain_library_media_progress_writes, group library_media_reading_progress). exclusive=True was rejected because cancelling an in-flight to_thread writer leaves the durable outcome unknown and the abandoned thread can commit AFTER its successor (task-1541 lesson: cancellation-based supersede is unsound for durable writes). Pending values are keyed per canonical id so a slow drain never drops a DIFFERENT item's final position.

EQUALITY SKIP: _library_media_progress_write_is_current compares the captured offset against (recency order) the queued value, the in-flight value, then the last durably persisted value; _cache_library_media_reading_progress seeds the persisted map from the DB fetch, so opening an item and traversing away without scrolling writes nothing. A FAILED write is never recorded as persisted, so an identical later capture retries it.

TEARDOWN: on_unmount awaits the drainer, re-queues an ambiguity-window in-flight value (idempotent upsert makes the possible double-write harmless), and drains residue inline -- the last CAPTURED offset survives screen teardown and app quit. The final UNCAPTURED offset (scroll then quit with no traversal/mode-switch) was never persisted before this task either; capture sites are unchanged (pre-existing scope, not a regression).

MEASURED: scripted 30-step traversal probe: 30 worker spawns / 30 settled upserts before -> 1 spawn / 1 settled write after. Mutation-tested: dropping the equality skip reds 2 probes (30 spawns resurface); dropping the coalescing reds 3 probes (3 stacked writers, last-write-wins broken).

FILES: tldw_chatbook/UI/Screens/library_screen.py (capture/queue/drain/write + init slots + on_unmount + cache seeding); Tests/UI/test_library_media_reader_flow.py (5 new probes + fake updates); Docs/security/production-diagnostic-inventory.json (+1 reviewed constant-string warning).

VERIFIED: flow suite 32 passed; adjacent suites (side_by_side, media_reading_scope_service x2, media_runtime_state) 165 passed; library_shell unmount/progress/lifecycle subset 17 passed + 5 PRE-EXISTING dev reds (reproduced bit-identically against base 70d28febc with the base library_screen.py swapped in); collect-only sweep 59421 collected, 28 errors all missing-numpy optional deps (environmental); preflight all green after inventory review+write.
<!-- SECTION:NOTES:END -->
