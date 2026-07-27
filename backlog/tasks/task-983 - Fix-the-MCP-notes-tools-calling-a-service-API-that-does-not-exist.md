---
id: TASK-983
title: Fix the MCP notes tools calling a service API that does not exist
status: To Do
assignee: []
created_date: '2026-07-27 19:33'
labels:
  - mcp
  - notes
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MCP/server.py constructs NotesInteropService(self.chachanotes_db) with the wrong argument count and type, and its create_note and search_notes tools call methods and keyword arguments the real class does not define, so those tools cannot work. Found while removing the nonexistent CharacterInteropService from the same module (TASK-968) and reported rather than fixed because resolving it needs a design call about what the notes tools should expose. This is the third defect of the same shape in this one module -- the others were a config key that did not exist so a database opened in the working directory (TASK-854), and a service that was never implemented (TASK-968).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MCP notes tools call a real NotesInteropService API with correct arguments,create_note and search_notes work end to end against a temp database,A test exercises both tools rather than only importing the module,The module has no remaining reference to a service or method that does not exist
<!-- AC:END -->
