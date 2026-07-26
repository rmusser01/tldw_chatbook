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

Related seam in the same file (found by TASK-545 P2's whole-branch review): `USER_DB_BASE_DIR = get_chachanotes_db_path().parent` is now evaluated at MODULE IMPORT time, and `get_chachanotes_db_path()` reaches `get_user_data_dir()`, which performs a `mkdir(parents=True, exist_ok=True)`. So merely importing this module creates a directory. On a read-only `$HOME` or a non-writable `[paths] data_dir` the import raises, and `BuiltinToolProvider`'s registration loop turns that into "create_note/update_note absent" (now logged, but still absent). The sibling convention in `file_operation_tools.py` (`_tool_sandbox_root`) defers this to call time; fixing both together is natural.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Note tools reuse a shared/cached CharactersRAGDB connection instead of opening a fresh one on every call
- [ ] #2 No behavior change to note CRUD semantics or the resolved writing user
<!-- AC:END -->
