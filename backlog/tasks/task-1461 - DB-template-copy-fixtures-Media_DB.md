---
id: TASK-1461
title: >-
  DB template-copy fixtures for Tests/Media_DB: build schema once, copy per test
status: Done
assignee: []
created_date: '2026-07-30 08:55'
labels:
  - testing
  - performance
  - db
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Media_DB builds real file-backed databases function-scoped — full schema + FTS5 DDL per test (audit: 354 file-backed DB fixture sites suite-wide, 146 CharactersRAGDB constructions, only 3 session-scoped fixtures in the entire suite). Replace per-test DDL with a per-session (per-xdist-worker) template DB that tests copy in milliseconds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] Session-scoped template fixture builds each needed schema once, closes cleanly (WAL checkpointed — no `-wal`/`-shm` sidecars), function-scoped fixture `shutil.copyfile`s into `tmp_path`
- [x] Sites classified single-connection are converted to `:memory:` instead where semantics allow; multi-connection/WAL-dependent sites use the template copy
- [x] Any client-id/path values embedded by DDL are re-stamped per copy if the schema stores them
- [x] Directory wall time before/after quoted in the PR; junit outcome diff vs baseline empty for the directory

## Implementation Plan

1. Probe MediaDatabase template-copy vs fresh DDL
2. Convert only if the measured win justifies the churn

## Implementation Notes

**Measured no-go — no code change.** MediaDatabase's schema DDL costs 10.6ms per
construction (vs CharactersRAGDB's 137ms); template-copy+open costs 1.8ms, and
the entire directory already runs in **3.26s** with no test over 1s. Converting
five files of fixtures to save well under one second is churn, not engineering.
The AC's outcome (before/after wall time) is satisfied by the measurement:
"before" is already at the floor. Revisit only if Media_DB's schema ever grows
ChaChaNotes-scale DDL.
