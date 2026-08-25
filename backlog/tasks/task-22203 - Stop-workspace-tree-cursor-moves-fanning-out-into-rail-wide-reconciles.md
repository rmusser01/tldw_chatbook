---
id: TASK-22203
title: >-
  Stop workspace-tree cursor moves fanning out into rail-wide reconciles
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - console
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22203).

New with PR #2034. `console_workspace_tree.py:945-948`: every cursor move (arrow key,
click, sync) runs `_update_tooltip()` (per-move `get_node_at_line` + `cell_len` +
`scrollable_content_region`) and posts `WorkspaceTreeContextChanged`. The rail handler
(`UI/Console_Modules/left_rail.py:1992-2005`) calls
`sync_workspace_tree_context` (`Widgets/Console/console_workspace_context.py:1359-1434`),
which does an unguarded `context.update(copy)` — `Static.update` defaults to `layout=True`
in Textual 8.2.8, so one screen layout pass is armed per cursor keypress — plus an
unguarded tooltip assignment. When the cursor crosses a workspace<->conversation boundary
(constant while arrowing), the action-row `display` flip triggers `styles.height = "auto"`
+ `refresh(layout=True)` + two deferred frames ending in
`_reconcile_workspace_action_owners` (`:1448-1462`), which requests the full 7-section,
~45-`query_one` rail allocation pipeline (`left_rail.py:916-1035`). One arrow key = up to
2 extra frames + a full rail measure + >=2 layout passes.

## Acceptance Criteria

- [ ] An arrow-key move that does not cross the workspace/conversation boundary arms zero screen layout passes beyond the Tree's own repaint (probe on `Screen._refresh_layout`)
- [ ] Context tray updates are equality-guarded (content and tooltip) before any `Static.update`/tooltip write
- [ ] A boundary crossing performs at most one scoped reconcile, not the full rail allocation pipeline, or the full pipeline's per-press cost is measured and justified in the notes
- [ ] Per-press layout-pass count measured before/after
