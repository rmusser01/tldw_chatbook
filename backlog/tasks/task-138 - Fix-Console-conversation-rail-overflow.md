---
id: TASK-138
title: Fix Console conversation rail overflow
status: Done
labels:
- console
- workspaces
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Console workspace Conversations subsection bounded, collapsible, and searchable so large active-workspace conversation sets do not hide lower Context rail content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Many active-workspace conversations do not grow the Conversations subsection beyond its adaptive bound.
- [x] #2 Lower workspace status, server readiness, and handoff rows remain reachable when many conversations exist.
- [x] #3 Conversations collapse state persists per workspace and collapsed mode shows the selected conversation summary.
- [x] #4 Workspace switching clears transient search text and restores that workspace's collapse preference.
- [x] #5 Search covers active-workspace conversation memberships and persisted workspace-scoped conversations without leaking other workspaces.
- [x] #6 Selecting a search result resumes or switches the conversation while keeping the search query active.
- [x] #7 Expanded Conversations exposes workspace-scoped New conversation; collapsed Conversations does not.
- [x] #8 Search cap, empty, error, and stale-result states render explicit scoped copy.
- [x] #9 Mounted Console tests and rendered Textual-web/CDP evidence verify the layout fix.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:IMPLEMENTATION_PLAN:BEGIN -->
1. Add Console workspace conversation-section display state and pure tests.
2. Render bounded/collapsible/searchable Conversations UI in ConsoleWorkspaceContextTray.
3. Add ChatScreen-owned query, collapse preference, search, stale-result, and row-selection wiring.
4. Add TCSS rules for bounded list, summary, search, and collapse controls.
5. Add mounted workflow regressions for overflow, collapse, search scope, stale results, selection, and workspace switching.
6. Run focused Console tests, git diff --check, capture Textual-web/CDP evidence, then update TASK-138 notes.

ADR required: no
ADR path: N/A
Reason: presentation and UI preference state only; no schema, sync, workspace ownership, provider/runtime, or handoff contract change.
<!-- SECTION:IMPLEMENTATION_PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `ConsoleWorkspaceConversationSectionState` display state, bounded row sizing, scoped result-count copy, and row-height constants for the Console workspace Conversations subsection.
- Rendered a bounded, collapsible, searchable workspace conversation list with selected-summary copy, scoped `New conversation`, empty/error/stale-result copy, and persisted per-workspace collapse preferences.
- Wired ChatScreen search/query state, active-workspace membership search, persisted workspace conversation search, stale-result invalidation, selection behavior that keeps the query active, and workspace-switch reset behavior.
- Made the workspace context tray itself vertically scrollable so lower storage/server/handoff rows remain reachable when the rail is short, and kept rail headers fixed to one row at runtime.
- Added mounted regressions for 40-conversation overflow hit targets, lower-row reachability, search scope/isolation, active-query selection, blank-query focus, stale results, workspace switching, and header sizing.
- Updated TCSS for the workspace context tray, conversation list, search row, summary copy, collapse controls, and conversation rows in both component and modular stylesheet outputs.
- Captured Textual-web/CDP evidence in `Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-rail-overflow-evidence.md` with screenshots:
  - `Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-rail-overflow-cdp-2026-06-25.png`
  - `Docs/superpowers/qa/console-uat-parallelization/task-134-console-conversation-search-cdp-2026-06-25.png`
- Verification:
  - `.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py -k "workspace_conversation or workspace_rail or workspace_switch or default_workspace or conversation_search" --tb=short` -> `45 passed, 91 deselected, 1 warning in 38.43s`
  - `.venv/bin/python -m pytest -q Tests/UI/test_console_session_settings.py::test_console_left_rail_body_scrolls_below_fixed_header_without_settings_summary Tests/UI/test_console_internals_decomposition.py::test_console_left_rail_sections_use_available_space --tb=short` -> `2 passed, 1 warning in 5.18s`
  - `git diff --check` -> no output
- ADR required: no. ADR path: N/A. Reason: presentation and UI preference state only; no schema, sync, workspace ownership, provider/runtime, service contract, or security boundary change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console workspace conversations are now bounded, collapsible, searchable, and workspace-scoped. The rail keeps `New conversation` clickable, preserves active search while browsing matches, restores per-workspace collapse state, and keeps lower workspace/server/handoff content reachable through tray scrolling even with large conversation sets.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
