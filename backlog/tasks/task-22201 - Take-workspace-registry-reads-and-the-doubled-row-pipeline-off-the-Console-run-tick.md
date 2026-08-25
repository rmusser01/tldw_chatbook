---
id: TASK-22201
title: >-
  Take workspace registry reads and the doubled row pipeline off the Console run tick
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
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22201).

PR #2034 (`a581f28e0`) reintroduced the exact hot-path shape TASK-21118 removed — on the
run tick instead of the keystroke path. `_sync_native_console_chat_ui`
(`UI/Screens/chat_screen.py:15206-15218`) builds `_build_console_workspace_context_state()`
three times per 0.2 s tick (twice via `_current_console_rail_state()`, once via
`_sync_console_workspace_context()`). Each build now reaches
`_console_browser_workspace_records()` twice (pre-existing browser labels + the NEW
`workspace_tree_projection(rows)` at `UI/Console_Modules/workspace.py:2708` -> `:1525-1535`
-> `:3865-3887`), and each call runs `ensure_default_workspace()` (SELECT + bindings probe
+ occasional DELETE write txn, `Workspaces/registry_service.py:572-609`, `:1173-1203`) plus
`list_workspaces()` — all synchronous on the event loop: roughly 45 extra queries/second
while a reply streams. The same builds also run the whole-row-set canonical-owner
reconciliation twice per build (`workspace.py:2628-2679` and again at `:1558-1563` inside
the tree projection): merge across five row groups, membership dict rebuild,
`_rows_with_latest_canonical_owner` + `_overlay_current_console_browser_markers` passes —
O(all conversations), pure Python, x3 per tick. The `state_changed` guard at
`chat_screen.py:11100` gates only the push, not the build. The keystroke path is still
clean (memo at `workspace.py:2726-2826` intact — verified live: 19 keys = 0 SQL at tip vs
48 at the pin).

## Acceptance Criteria

- [ ] A run tick with unchanged registry state performs zero workspace-registry SQL (memoize `_console_browser_workspace_records` on the existing `mutation_generation`, the TASK-21118 pattern) — proven by the sqlite trace-callback probe from the review
- [ ] The row merge / canonical-owner / overlay pipeline runs at most once per state build, and the build runs at most once per tick unless its inputs changed
- [ ] `ensure_default_workspace()` (a write-capable repair) is not called from any display/build path — only from mutation paths
- [ ] Per-tick build cost during streaming is measured before/after; tree/browser behavior is unchanged when the registry actually changes (existing tests green)
