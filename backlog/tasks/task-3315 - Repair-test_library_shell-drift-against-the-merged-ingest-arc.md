---
id: TASK-3315
title: >-
  Repair test_library_shell drift against the merged ingest arc (54 deterministic failures on dev base)
status: To Do
assignee: []
created_date: '2026-08-08 21:30'
labels:
  - library
  - tests
  - dev-baseline
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during follow-up batch phase A (2026-08-08): `Tests/UI/test_library_shell.py` carries 54 deterministic failures on dev base `ebeae1440` (identical set across loaded and quiet runs, reproduced with the phase's product diff fully reverted). Two mechanisms: (a) `_LibraryIngestCanvasHarness` does not mirror `TldwCli._ingest_local_stt_jobs`, which `app.py._maybe_start_next_ingest_job` reads since the ingest arc (PR #1452) — the real app initializes it (`app.py:5660`), so this is stale-harness drift killing ~20 job-lifecycle tests with AttributeError; (b) a Notes 60x20 geometry off-by-one family (`shell.region.height 14 != 15`) plus dependent pilot tests — cause undiagnosed, could be arc CSS or dev's own churn; diagnose before pinning. The arc's phase batteries ran this suite only under `-k` filters, so the full suite's rot shipped unnoticed — the "suite no gate runs" lesson shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Full `Tests/UI/test_library_shell.py` runs green (or its residual failures are proven pre-arc with SHAs) with a READ pass count
- [ ] #2 The geometry off-by-one family's cause is named (arc CSS vs dev churn vs stale pin) before any expectation is updated
- [ ] #3 The harness mirrors the app attributes the ingest job loop reads, derived from the real initializer rather than hand-listed
<!-- AC:END -->
