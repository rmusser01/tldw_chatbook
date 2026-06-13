---
id: TASK-60.5
title: Fix Personas destination indefinite behavior-context loading state
status: Done
assignee: []
created_date: 2026-05-20 15:15
labels:
- ux
- hci
- qa
- screens
dependencies: []
parent_task_id: TASK-60
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the Personas destination resolves its local behavior-context load into an actionable ready, empty, or error state so users are not left in a permanent loading screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Personas screen leaves loading state deterministically in an empty local profile.
- [x] #2 Personas screen shows an actionable empty or service error state instead of contradictory Ready/loading copy.
- [x] #3 Attach and open controls reflect actual context availability.
- [x] #4 Mounted regression waits deterministically without fixed sleeps.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a mounted regression that simulates the Personas snapshot worker never applying so the screen must leave loading via a deterministic fallback.
2. Add a mount-level timeout fallback to PersonasScreen that applies an actionable service error if loading has not resolved.
3. Align the Personas snapshot worker with the thread-backed application pattern used by other destination screens.
4. Re-run focused Personas mounted regressions and capture a fresh actual textual-web/CDP screenshot from a clean profile.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a deterministic Personas snapshot timeout fallback so an unresolved worker no longer leaves the screen in a permanent loading state.
- Replaced the CCP route chrome with a destination-native Personas workbench: mode strip, character library column, character detail/editor column, and attachment/readiness inspector.
- Wired the destination-native character list to the existing CCP character handler and preserved character card load/edit/save behavior through mounted regressions.
- Captured and obtained user approval for the actual rendered Textual-web screenshot at `Docs/superpowers/qa/product-maturity/post-release-ux-hci/actual-screenshots/2026-05-20-task-60-5-ccp-native-00-started.png`.
- Verification: `Tests/UI/test_ccp_screen.py` -> 14 passed; `Tests/UI/test_destination_shells.py` -> 100 passed; `Tests/UI/test_screen_navigation.py` -> 28 passed; `git diff --check` rerun after whitespace cleanup.
<!-- SECTION:NOTES:END -->
