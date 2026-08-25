---
id: TASK-22202
title: >-
  Equality-guard ConsoleWorkspaceTree.sync_projection and scope its invalidation
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
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22202).

`Widgets/Console/console_workspace_tree.py:313-467` (new in PR #2034): `sync_projection`
is called on every workspace-context push with no projection-level equality guard. Each
call rebuilds a `WorkspaceTreeNodeData` and a rich `Text` label for every workspace and
conversation, materializes the full node set for `can_focus` (`:443-450`), and calls
`get_node_at_line` twice plus `_update_tooltip()`. Any node change calls Textual's
`Tree._invalidate()`, whose `self._updates += 1` is part of the line-render cache key — one
changed node invalidates every cached tree line. During a run this fires at 5 Hz (the
projection embeds `selected`/`run_marker`/`loading`, so it genuinely changes).

## Acceptance Criteria

- [ ] An unchanged projection results in zero node writes and zero tree invalidations (counted by a probe on `Tree._invalidate`)
- [ ] A single-conversation change invalidates a bounded set of nodes, not the whole line cache, or the whole-cache cost is measured and accepted in the notes with numbers
- [ ] `can_focus` derivation does not materialize the full node set per pass
- [ ] Per-tick tree cost during streaming measured before/after
