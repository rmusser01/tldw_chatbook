---
id: TASK-134
title: Fix Console conversation rail overflow
status: To Do
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
