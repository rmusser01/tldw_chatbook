---
id: TASK-912
title: Aggregate run markers on section headers and capped rows
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 03:55'
updated_date: '2026-07-27 19:39'
labels:
  - console
  - fleet-ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two fleet-marker visibility gaps from the parallel-agents train: (1) top-level conversation-browser sections (Starred/Workspaces/Chats) have no run_marker aggregate, so collapsing a whole section hides every marker beneath it; (2) an expanded workspace group with more than the 12-row cap can push a marked row past the cap with no marker surfaced (header aggregate only renders when collapsed). Collapsed workspace GROUP headers already aggregate the most-urgent glyph.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collapsed top-level sections surface the most-urgent marker among their contents.
- [x] #2 A marked row beyond the group row cap surfaces its marker (e.g. header aggregate also when expanded-but-capped, or overflow row indicator).
- [x] #3 Urgency order matches the existing group-header aggregation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add `run_marker` aggregate to `ConsoleConversationBrowserSection` (Starred/Workspaces/Chats), computed from full pre-cap contents via the shared `_most_urgent_run_marker`.
2. Render the section-header marker only when the section is collapsed, mirroring the existing group-header pattern (markup=False Static).
3. Add a `capped_run_marker` field to `ConsoleConversationBrowserGroup`: the most-urgent marker among rows beyond `CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT`, independent of collapse state.
4. Render it on an EXPANDED group's header only when non-empty (a visible marked row needs no echo); collapsed-group rendering (`group.run_marker`) is unchanged.
5. TDD: failing tests first in `Tests/Workspaces/test_console_conversation_browser_state.py` (pure state) and `Tests/UI/test_console_workspace_context_rail.py` (mounted tray) covering collapsed-section aggregate, expanded-capped marker, visible-marker no-echo, and urgency ordering.
6. Run the browser-state + workspace-context-rail + rail-sections suites plus `Tests/UI/test_console_parallel_runs.py` in one foreground pytest call.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented both AC#1 and AC#2 via the pure-state layer + rendering split the existing group.run_marker already established.

AC#1: Added run_marker to ConsoleConversationBrowserSection, computed unconditionally from the section's full pre-cap contents (_most_urgent_run_marker over the raw rows for Starred/Chats; over the workspace groups themselves for Workspaces -- the function duck-types on .run_marker so a group's own precomputed aggregate composes directly, no re-walking rows). Rendered on the section header only when section.collapsed, same markup=False Static + label-suffix pattern the group header uses.

AC#2: Added capped_run_marker to ConsoleConversationBrowserGroup: _most_urgent_run_marker over group_rows[group_row_limit:] (the rows an expanded group's 12-row cap still hides), computed unconditionally alongside run_marker. Group header now renders group.run_marker when collapsed (unchanged) or group.capped_run_marker when expanded-and-non-empty (new) -- a still-visible marked row keeps capped_run_marker empty since it already shows its own glyph, so no header echo.

AC#3: Both new aggregates reuse the existing _most_urgent_run_marker/_RUN_MARKER_URGENCY -- no second urgency table introduced.

Tests (TDD, failing-first): 12 new pure-state cases in Tests/Workspaces/test_console_conversation_browser_state.py (collapsed-section aggregate for Workspaces/Chats, expanded-section field stays populated, empty-marker default, capped-vs-visible marker split, urgency-among-hidden-only, collapsed-group capped_run_marker stays a pure computation) + 3 new mounted-tray tests in Tests/UI/test_console_workspace_context_rail.py (collapsed Workspaces section header shows the aggregate with the group actually unmounted, expanded group past the 15-row/12-cap boundary shows the hidden-rows marker with the row itself unmounted, a visible marked row produces no header echo). Extended the rail test file's _grouped_browser_state/_base_grouped_workspace_state helpers with an optional group_collapse_preferences override to reach the section-collapsed case.

Files: tldw_chatbook/Workspaces/conversation_browser_state.py, tldw_chatbook/Widgets/Console/console_workspace_context.py, Tests/Workspaces/test_console_conversation_browser_state.py, Tests/UI/test_console_workspace_context_rail.py.
<!-- SECTION:NOTES:END -->
