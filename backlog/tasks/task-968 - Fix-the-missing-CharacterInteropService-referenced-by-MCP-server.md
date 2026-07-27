---
id: TASK-968
title: Fix the missing CharacterInteropService referenced by MCP server
status: To Do
assignee: []
created_date: '2026-07-27 18:06'
labels:
  - mcp
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While fixing TASK-854's media DB lookup, MCP/server.py was found referencing a CharacterInteropService that does not exist, so that code path cannot run. Left unfixed to keep TASK-854 scoped to the database-path defect, and recorded here. Note TASK-854 also found the same file opening ./media_library.db in the working directory because it read a config key that does not exist -- this file warrants a broader look than either task gave it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MCP/server.py's character code path either resolves a real service or is removed,No reference to a nonexistent service remains in that module,The module's other config lookups are checked against the declared accessors
<!-- AC:END -->
