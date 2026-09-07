---
id: TASK-1462
title: >-
  DB template-copy fixtures for Tests/Chatbooks: build schema once, copy per test
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
Tests/Chatbooks builds real file-backed databases function-scoped — full schema + FTS5 DDL per test (audit: 354 file-backed DB fixture sites suite-wide, 146 CharactersRAGDB constructions, only 3 session-scoped fixtures in the entire suite). Replace per-test DDL with a per-session (per-xdist-worker) template DB that tests copy in milliseconds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] Session-scoped template fixture builds each needed schema once, closes cleanly (WAL checkpointed — no `-wal`/`-shm` sidecars), function-scoped fixture `shutil.copyfile`s into `tmp_path`
- [x] Sites classified single-connection are converted to `:memory:` instead where semantics allow; multi-connection/WAL-dependent sites use the template copy
- [x] Any client-id/path values embedded by DDL are re-stamped per copy if the schema stores them
- [x] Directory wall time before/after quoted in the PR; junit outcome diff vs baseline empty for the directory

## Implementation Plan

1. Probe per-DB DDL costs (CharactersRAGDB 137ms; PromptsDatabase 6.8ms; MediaDatabase 10.6ms — only CharactersRAGDB is worth templating)
2. Hoist the session template fixture from Tests/ChaChaNotesDB/conftest.py to the root conftest (lazy import) so both directories share it
3. Seed the template at the path-fixture seam in the four fixture bodies that construct CharactersRAGDB; leave mid-test import-side constructions (fresh-import semantics) alone

## Implementation Notes

Chatbooks: **23.99s -> 20.78s**, 171 passed unchanged. Only CharactersRAGDB is
worth templating (Prompts 6.8ms, Media 10.6ms DDL — see task-1461's no-go).
The `chachanotes_template_db` fixture moved to the ROOT conftest with a lazy
in-fixture import (fixture resolution walks up the conftest chain, so
task-1460's ChaChaNotesDB call sites work unchanged — verified: 4.29s, same
outcome set); Tests/ChaChaNotesDB/conftest.py is now a pointer comment.
Converted fixtures: `populated_chachanotes_db` (conftest),
`setup_test_databases` (integration), `source_env` (image round-trip),
`performance_db_setup` (performance). Mid-test dest/import-side constructions
keep building fresh on purpose (importing into a fresh DB is the semantics
under test). Remaining directory time is real ZIP/import work, not DDL.
Modified: `Tests/conftest.py`, `Tests/ChaChaNotesDB/conftest.py`,
`Tests/Chatbooks/conftest.py`, `test_chatbook_integration.py`,
`test_chatbook_image_round_trip.py`, `test_chatbook_performance.py`.
