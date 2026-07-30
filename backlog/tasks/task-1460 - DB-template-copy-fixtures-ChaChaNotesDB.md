---
id: TASK-1460
title: >-
  DB template-copy fixtures for Tests/ChaChaNotesDB: build schema once, copy per test
status: To Do
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
Tests/ChaChaNotesDB builds real file-backed databases function-scoped — full schema + FTS5 DDL per test (audit: 354 file-backed DB fixture sites suite-wide, 146 CharactersRAGDB constructions, only 3 session-scoped fixtures in the entire suite). Replace per-test DDL with a per-session (per-xdist-worker) template DB that tests copy in milliseconds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] Session-scoped template fixture builds each needed schema once, closes cleanly (WAL checkpointed — no `-wal`/`-shm` sidecars), function-scoped fixture `shutil.copyfile`s into `tmp_path`
- [ ] Sites classified single-connection are converted to `:memory:` instead where semantics allow; multi-connection/WAL-dependent sites use the template copy
- [ ] Any client-id/path values embedded by DDL are re-stamped per copy if the schema stores them
- [ ] Directory wall time before/after quoted in the PR; junit outcome diff vs baseline empty for the directory
