---
id: TASK-134
title: Fix Console conversation rail overflow
status: In Progress
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
- [ ] #1 Many active-workspace conversations do not grow the Conversations subsection beyond its adaptive bound.
- [ ] #2 Lower workspace status, server readiness, and handoff rows remain reachable when many conversations exist.
- [ ] #3 Conversations collapse state persists per workspace and collapsed mode shows the selected conversation summary.
- [ ] #4 Workspace switching clears transient search text and restores that workspace's collapse preference.
- [ ] #5 Search covers active-workspace conversation memberships and persisted workspace-scoped conversations without leaking other workspaces.
- [ ] #6 Selecting a search result resumes or switches the conversation while keeping the search query active.
- [ ] #7 Expanded Conversations exposes workspace-scoped New conversation; collapsed Conversations does not.
- [ ] #8 Search cap, empty, error, and stale-result states render explicit scoped copy.
- [ ] #9 Mounted Console tests and rendered Textual-web/CDP evidence verify the layout fix.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:IMPLEMENTATION_PLAN:BEGIN -->
1. Add Console workspace conversation-section display state and pure tests.
2. Render bounded/collapsible/searchable Conversations UI in ConsoleWorkspaceContextTray.
3. Add ChatScreen-owned query, collapse preference, search, stale-result, and row-selection wiring.
4. Add TCSS rules for bounded list, summary, search, and collapse controls.
5. Add mounted workflow regressions for overflow, collapse, search scope, stale results, selection, and workspace switching.
6. Run focused Console tests, git diff --check, capture Textual-web/CDP evidence, then update TASK-134 notes.

ADR required: no
ADR path: N/A
Reason: presentation and UI preference state only; no schema, sync, workspace ownership, provider/runtime, or handoff contract change.
<!-- SECTION:IMPLEMENTATION_PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
