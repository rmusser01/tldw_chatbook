---
id: TASK-31261
title: canvas_sync search-kind screen-caller AST census guard
status: To Do
assignee: []
created_date: '2026-09-04 05:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
canvas_sync.py's _sync_library_canvas dispatcher's "search"-kind branch (~line 477) writes the flat _library_rag_answer_render_key attribute directly on its screen argument, relying on a composed invariant that every "search"-kind caller forwards the CONTROLLER (LibraryRagSearchController) as screen, never the actual LibraryScreen, which has no _rag_search_state attribute and would silently grow a dead instance attribute instead of raising. That invariant is currently verified once by hand (a code comment reading 'no such caller exists, AST-verified') and not by any automated test. Tests/Library/test_library_rag_scope.py's test_library_screen_call_sites_never_pass_scope_kwarg already sketches the exact mechanism needed: a two-file AST census over library_screen.py and library_rag_search_controller.py that walks Call nodes and asserts an invariant across both files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A guard test fails if a future _sync_library_canvas(..., "search", ...) call site is added directly on library_screen.py itself rather than on the controller
- [ ] #2 The guard test passes against the current tree, confirming today's only two "search"-kind call sites both live in library_rag_search_controller.py
<!-- AC:END -->
