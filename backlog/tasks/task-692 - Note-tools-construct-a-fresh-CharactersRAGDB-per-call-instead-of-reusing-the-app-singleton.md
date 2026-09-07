---
id: TASK-692
title: >-
  Note tools construct a fresh CharactersRAGDB per call instead of reusing the
  app singleton
status: Done
assignee: []
created_date: '2026-07-26 06:06'
labels:
  - tools
  - agents
  - notes
  - performance
dependencies:
  - TASK-545
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
create_note/search_notes/update_note (Tools/note_management_tools.py) each construct a brand-new NotesInteropService/CharactersRAGDB per invocation instead of reusing the app's existing singleton DB connection. This pays a full DB-open and schema-check cost on every single tool call. Pre-existing behavior, but TASK-545 P2 newly puts it on an agent-driven path (the worker thread that runs LLM tool calls), where it can now be exercised many times in a single run. Identified in the design spec for TASK-545 P2 (Docs/superpowers/specs/2026-07-25-port-mutating-tools-design.md, 'Known limitations carried, not fixed'), which explicitly deferred it to keep that phase scoped to porting the tool behind the permission gate.

The related import-time seam in this file (`USER_DB_BASE_DIR` computed at module scope, reaching `get_user_data_dir()`'s `mkdir`) was flagged by both the whole-branch review and Qodo on PR #921 and is **already fixed there** — `_notes_db_base_dir()` now defers to call time, matching `file_operation_tools._tool_sandbox_root`. What remains here is only the per-call `CharactersRAGDB` construction.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Note tools reuse a shared/cached CharactersRAGDB connection instead of opening a fresh one on every call
- [x] #2 No behavior change to note CRUD semantics or the resolved writing user
<!-- AC:END -->

## Implementation Notes

`_notes_service()` builds the service lazily once and reuses it, keyed on the
resolved chachanotes DB path so re-pointing the data dir still rebuilds.

The premise was real but understated in the original write-up. The three
tools already passed `global_db_to_use=chachanotes_db`, so it looked like the
singleton was reused -- but that only supplies the *path template*. The
service's `_db_instances` cache is an INSTANCE attribute, so a per-call
service missed it every time and `_get_db` constructed a fresh
`CharactersRAGDB(db_path=..., client_id=user_id)` -- a real DB open plus
schema-version check -- then discarded it when the call returned. Measured at
~1.8 ms per construction on an already-created DB, plus a
`verify_trusted_directory` filesystem check per call.

Kept the per-user `client_id` semantics exactly: caching the *service*
preserves `_get_db`'s attribution behaviour, whereas reusing `chachanotes_db`
directly would have silently changed the `client_id` written into rows.

Also adds an autouse cache-reset fixture to the covering tests: the cache is
module state, and without it a real service built by one test could be served
to a later test that monkeypatches `NotesInteropService`. Today each test
happens to get its own config path so the key differs -- the fixture removes
that accident. Red-proofed: reverting the cache makes the reuse test see
three service builds instead of one.

Files: `tldw_chatbook/Tools/note_management_tools.py`,
`Tests/Tools/test_note_tool_user_id.py`.
