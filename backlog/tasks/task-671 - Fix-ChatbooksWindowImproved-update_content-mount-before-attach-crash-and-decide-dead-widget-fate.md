---
id: TASK-671
title: >-
  Fix ChatbooksWindowImproved update_content mount before attach crash and
  decide dead widget fate
status: Done
assignee:
  - '@claude'
created_date: '2026-07-26 12:00'
updated_date: '2026-07-26 06:03'
labels:
  - followup
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during task-637: ChatbooksWindowImproved._update_content() constructs a local Grid/ListView and mounts cards/items into it BEFORE the container itself is attached to the DOM - Widget.mount() raises MountError when not is_attached, so any non-empty chatbooks list crashes the recompose path (task-637's test deliberately routes around it via the empty-state branch). Separately, decide the fate of two effectively-dead widgets task-637 also had to guard: ResultsDashboardWindow.py cannot even import (missing eval_shared_components module) and Mindmap_Viewer_Window.py has no live call site - delete or properly wire them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Non-empty chatbooks list renders without MountError (regression test with 2+ chatbooks)
- [x] #2 ResultsDashboardWindow and Mindmap_Viewer_Window are each either deleted or wired to a live call site (decision documented)
- [x] #3 Existing chatbooks tests stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: extend Tests/UI/test_chatbooks_screen_server_actions.py with a regression test that sets ChatbooksWindowImproved.chatbooks to 2+ items and asserts it does not raise MountError, and that ChatbookCard widgets render.
2. Root-cause: confirm Widget.mount() raises MountError synchronously when not is_attached (verified via textual source); confirm _update_content() builds a local Grid/ListView and mounts children into it before the Grid/ListView itself is mounted into the (already-attached) #chatbooks-container.
3. Fix: mount the Grid/ListView into the attached #chatbooks-container FIRST, then mount cards/items into it (matches the pattern already used in ChatbookTemplatesWindow.on_mount: mount card into attached grid before mounting the card's own children).
4. GREEN: verify the new test passes; run full existing chatbooks test suite for regressions (including the task-637 recompose-guard tests).
5. Part 2: investigate ResultsDashboardWindow.py (import failure - missing eval_shared_components) and Mindmap_Viewer_Window.py (no live call site) - grep git log, Docs/plans, backlog for revival references; decide delete vs wire-or-keep per ADR-026 precedent; execute and document.
6. Run full gate suite (chatbooks tests, recompose guard tests, mcp_rail tests, any tests referencing deleted widgets) and update task ACs/notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Part 1 (crash): _update_content() built a local Grid/ListView and mounted
cards/items into it (`grid.mount(card)` / `list_view.mount(item)`) BEFORE
the Grid/ListView was itself mounted into the already-attached
#chatbooks-container. Widget.mount() raises MountError synchronously
whenever the target isn't attached, so any non-empty chatbooks list crashed
watch_chatbooks -> _update_content(); only the empty-state branch (mounting
a single EmptyStateWidget straight into the attached container) avoided it.
Fixed by mounting the Grid/ListView into the attached container first, then
mounting children into it -- same ordering already used in
ChatbookTemplatesWindow.on_mount.

Root-causing also surfaced a second bug: `chatbooks` was declared
`reactive([], recompose=True)`, but compose() is static and never reads
`self.chatbooks`. The recompose flag schedules a deferred
Widget.recompose() one tick after the synchronous watcher runs, which tore
the whole subtree back down to compose()'s empty skeleton, silently wiping
out whatever cards _update_content() had just mounted (confirmed
empirically: card.is_attached flips False and window.query(ChatbookCard)
goes empty after one pilot.pause()). Dropped recompose=True from
`chatbooks` since the watcher already does the DOM rebuild imperatively;
RecomposeCaptureGuard (task-637) is kept since refresh(recompose=True) is
still directly callable and exercised by its own regression test.

Part 2 (dead widgets): both are genuinely dead with no revival reference
anywhere (Docs/superpowers/plans, backlog/tasks, backlog/decisions) ->
DELETED per the ADR-026 precedent (grep-gate: ids/classes composed nowhere
live + zero direct Python callers).
- ResultsDashboardWindow.py: cannot import at all (`from .eval_shared_components
  import ...` -- that module doesn't exist in the tree); zero references
  anywhere outside itself and the task-637/task-671 docs. Deleted; no test
  ever referenced it. eval_results_widgets.py (which it imported) still has
  a live consumer in Views/evals_views.py, so it was left untouched.
- Mindmap_Viewer_Window.py (MindmapViewerWindow): imports fine but has zero
  live navigation/production call sites -- only reachable from
  Tests/UI/test_bulk_selection_tooltips.py's
  test_mindmap_source_selection_clear_control_has_tooltip. Deleted along
  with that one test function + its now-unused import; the file's other
  four unrelated bulk-selection-tooltip tests were left intact.
  (Note: unrelated to Widgets/MindmapViewer.py's MindmapViewer Container,
  which is live and covered by test_mindmap_viewer_tooltips.py.)

Tests: extended Tests/UI/test_chatbooks_screen_server_actions.py with
grid-view and list-view regression tests (2-3 chatbooks each) that RED'd
with MountError pre-fix and GREEN post-fix, including pumping the message
loop past the deferred recompose to confirm cards survive. Full gate green:
Tests/UI/test_chatbooks_screen_server_actions.py (9),
Tests/Widgets/test_recompose_capture_guard.py + Tests/UI/test_mcp_rail.py
(20), Tests/UI/test_bulk_selection_tooltips.py (4),
Tests/UI/test_mindmap_viewer_tooltips.py (1), Tests/Evals/ (283 passed, 13
skipped). `python -c "import tldw_chatbook.app"` still imports cleanly.

Modified: tldw_chatbook/UI/Chatbooks_Window_Improved.py.
Deleted: tldw_chatbook/UI/ResultsDashboardWindow.py,
tldw_chatbook/UI/Mindmap_Viewer_Window.py.
Test files: Tests/UI/test_chatbooks_screen_server_actions.py (extended),
Tests/UI/test_bulk_selection_tooltips.py (trimmed).
<!-- SECTION:NOTES:END -->
