---
id: TASK-31261
title: canvas_sync search-kind screen-caller AST census guard
status: Done
assignee:
  - '@zcode'
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
- [x] #1 A guard test fails if a future _sync_library_canvas(..., "search", ...) call site is added directly on library_screen.py itself rather than on the controller
- [x] #2 The guard test passes against the current tree, confirming today's only two "search"-kind call sites both live in library_rag_search_controller.py
<!-- AC:END -->

## Implementation Notes

``test_search_kind_canvas_sync_never_targets_the_screen_directly`` (Tests/Library/test_library_rag_scope.py, beside the sketch whose mechanism it follows) censuses ``_sync_library_canvas`` calls with kind ``"search"`` -- second positional constant or ``kind=`` keyword -- across ``library_screen.py`` AND every ``UI/Library_Modules/*.py``. Any hit outside ``library_rag_search_controller.py`` fails with the why (a screen receiver would silently grow a dead ``_library_rag_answer_render_key`` attribute instead of raising); an empty total census also fails, so the guard cannot go silently green if the sites move again (the rag-scope sketch's own recorded lesson).

Evidence: green on the current tree with the census finding exactly the two known sites (controller :962, :988) and zero screen hits (AC#2); mutation -- adding ``_sync_library_canvas(self, "search")`` directly on the screen -- fails the guard naming the file and line (AC#1), mutation reverted. File suite: 22 passed. Format clean.

ADR required: no
ADR path: N/A
Reason: Test-only census guard; no production change.

Modified: ``Tests/Library/test_library_rag_scope.py``.
