---
id: TASK-18907
title: 'Chunking parity Phase C: engine-version stamp, schema v6, legacy report'
status: To Do
assignee: []
created_date: '2026-08-19 09:30'
updated_date: '2026-08-19 09:30'
labels:
  - chunking
dependencies:
  - TASK-18906
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase C (PR 3) of the Chunking Engine Parity sub-project — lands strictly after Phase B because the stamp is only meaningful once every write path routes through the engine: add `chunk_engine_version TEXT` to `UnvectorizedMediaChunks` and bump the media DB schema v5 → v6 with a migration that leaves existing rows NULL; stamp every newly written chunk with the engine version (`parity-1@385afa95`) at the ingestion persist seam and in the in-memory chunk metadata; add a read-only RAG Admin indicator counting chunks by engine version (NULL → "legacy"); update user-visible docs. No re-chunk action (defers to sub-project #2 per Q3); nothing re-chunks automatically.

Plan: `Docs/superpowers/plans/2026-08-19-chunking-engine-parity.md` Tasks 11–13.
Spec: `Docs/superpowers/specs/2026-08-18-chunking-engine-parity-design.md` (§8; rulings Q3/Q8).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `UnvectorizedMediaChunks.chunk_engine_version` column exists; `_CURRENT_SCHEMA_VERSION` is 6 with a migration; existing rows left NULL (verified by upgrade test)
- [ ] #2 Chunks written through the ingestion persist path carry `chunk_engine_version = "parity-1@385afa95"` in both the DB row and the in-memory chunk metadata
- [ ] #3 RAG Admin shows a read-only "chunked by an older engine (N items)" indicator backed by `count_chunks_by_engine_version()` (NULL → "legacy")
- [ ] #4 No media item is re-chunked; no re-chunk/re-index UI ships (deferred to sub-project #2 per Q3)
- [ ] #5 `Docs/User_Guide/` updated where chunking behavior is user-visible (method list, sanitization behavior change, tiktoken/defusedxml now core, legacy-chunk indicator)
- [ ] #6 Targeted suites green (`Tests/DB/`, `Tests/Local_Ingestion/test_engine_version_stamp.py`, chunking suites)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Schema v6 migration + column, upgrade test (plan Task 11)
2. Stamp at persist seam + RAG Admin count/report (plan Task 12)
3. Docs + close-out (plan Task 13)
<!-- SECTION:PLAN:END -->
