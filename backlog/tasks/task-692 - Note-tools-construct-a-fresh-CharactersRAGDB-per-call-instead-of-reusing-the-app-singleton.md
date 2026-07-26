---
id: TASK-692
title: >-
  Note tools construct a fresh CharactersRAGDB per call instead of reusing the
  app singleton
status: To Do
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
- [ ] #1 Note tools reuse a shared/cached CharactersRAGDB connection instead of opening a fresh one on every call
- [ ] #2 No behavior change to note CRUD semantics or the resolved writing user
<!-- AC:END -->
