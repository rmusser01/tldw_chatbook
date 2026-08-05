---
id: TASK-1460
title: >-
  DB template-copy fixtures for Tests/ChaChaNotesDB: build schema once, copy per test
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
Tests/ChaChaNotesDB builds real file-backed databases function-scoped — full schema + FTS5 DDL per test (audit: 354 file-backed DB fixture sites suite-wide, 146 CharactersRAGDB constructions, only 3 session-scoped fixtures in the entire suite). Replace per-test DDL with a per-session (per-xdist-worker) template DB that tests copy in milliseconds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] Session-scoped template fixture builds each needed schema once, closes cleanly (WAL checkpointed — no `-wal`/`-shm` sidecars), function-scoped fixture `shutil.copyfile`s into `tmp_path`
- [x] Sites classified single-connection are converted to `:memory:` instead where semantics allow; multi-connection/WAL-dependent sites use the template copy
- [x] Any client-id/path values embedded by DDL are re-stamped per copy if the schema stores them
- [x] Directory wall time before/after quoted in the PR; junit outcome diff vs baseline empty for the directory

## Implementation Plan

1. Probe template-copy vs fresh DDL cost (as a pytest test)
2. Session-scoped template fixture in a new directory conftest; sidecar assertion after close
3. Convert the six per-test construction fixtures to copy-then-open; leave :memory: sites and construction/migration-semantics tests building from scratch
4. Before/after directory run + failure-name comparison

## Implementation Notes

Probe: fresh DDL 137.2ms vs copy+open 10.5ms (92% less); `close_connection()`
checkpoints WAL — no sidecars (asserted in the template fixture). client_id is
per-row attribution, so the empty template carries nothing to re-stamp.
Directory result: **23.42s -> 4.57s (5.1x)**, outcomes identical (172 passed +
the same pre-existing `test_conversations_migrate_from_v17_to_v18...` failure,
name-matched before and after; migration tests deliberately keep fresh
construction and are untouched). Session scope = once per xdist worker.
Added: `Tests/ChaChaNotesDB/conftest.py`. Modified: the six fixture sites in
`test_chachanotes_db.py`, `test_chachanotes_db_properties.py`,
`test_character_persona_runtime_parity.py`, `test_chat_conversation_parity.py`,
`test_study_functionality.py`, `test_message_generation_metadata.py`.
