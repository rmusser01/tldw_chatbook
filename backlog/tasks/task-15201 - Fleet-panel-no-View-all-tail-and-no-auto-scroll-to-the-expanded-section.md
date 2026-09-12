---
id: TASK-15201
title: 'Fleet panel: no View all tail and no auto-scroll to the expanded section'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 04:01'
updated_date: '2026-09-08 05:01'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Disclosed scope cut from supervisor-fleet PR 2b Task 4. The plan's state-2 description named 'Scrollable/virtualized past a screenful; View all opens full run history'; what shipped relies on the rail's existing outer VerticalScroll and has no View all tail. The reviewer probed this properly rather than assuming: with 12 live rows at 180x48, the 12th row's unclipped region sits inside the viewport but the compositor hit-test resolves to another widget — i.e. not painted by default — while after an explicit scroll_visible() the row IS painted at its own region and a real click routes to the right child. So nothing is permanently unreachable and routing is scroll-position independent; this is NOT a task-226 clipping bug. It is a real UX gap: the fleet section sits ~5th among 6-7 peer sections in one shared scroll, nothing auto-scrolls a newly-expanded section into view, and with a dozen live children the user must scroll past every other section to reach the bottom rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Expanding the fleet section scrolls it into view
- [x] #2 A View all affordance opens full run history when rows exceed what the rail can show
- [x] #3 A row past the fold is reachable and clickable without manual hunting (verified with a compositor hit-test, not DOM presence)
- [x] #4 Saved child history remains reachable when the latest primary run has no children.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-09-07-fleet-history-navigation.md with failing targeted regressions before implementation. Keep the read-only history entry available in a selected conversation even when the current fleet preview is empty; verify older children remain reachable after a childless primary run. ADR required: yes. ADR path: backlog/decisions/132-fleet-history-navigation.md. Reason: metadata page contract and fleet history selection/navigation. Preserve existing execution authority and unrelated working-tree changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a four-row fleet preview with complete counts and a read-only 50-row history picker. The AgentRunsDB/bridge query returns bounded metadata through stable timestamp/id cursor pages, includes earlier/superseded children, and avoids steps/result hydration until selection. The modal loads off the UI thread with paging, refresh, safe dismissal, empty/error recovery, and keyboard/mouse selection. The agent controller validates selected conversation and agent kind before opening the existing drill-in; the rail receives one named callback.

Expansion reveals the View all action through both capped scroll ancestors; routine refreshes preserve reading position. Self-review reproduced an inaccessible history entry after a childless new primary. AC#4 and the plan were added before the fix: selected conversations now retain the read-only history entry with an empty preview. Updated the guide, review ledger, and the nested-scroll testing lesson. ADR required: yes; follows backlog/decisions/132-fleet-history-navigation.md, linked in Docs/superpowers/plans/2026-09-07-fleet-history-navigation.md. No new execution permission, schema, or dependency.

Combined TASK-15200/15201 validation: 398 targeted tests pass (291 DB/bridge, 62 fleet/agent/history UI, 15 historical/parallel UI, 30 component/CSS). Wide and compact compositor tests prove the action is painted, page to an old child, and activate it with real keyboard/mouse input. Scope-switch, childless-latest-run, slow-dismissal, error/empty-state, and refresh-scroll guards pass. New files pass Ruff/format; changed production files add no lint findings against pass start; scoped whitespace and diagnostic inventory checks pass. Self-review completed. No full suite or live provider used.

Outstanding adjacent validation: TASK-3070's existing screen-size ratchet remains red at 17,601 lines / 591 methods versus 17,570 / 591 (17,600 / 591 at pass start). The screen adds only the callback line and updates the existing visibility projection; controller responsibilities remain outside the screen. Budget unchanged. Changes remain uncommitted.
<!-- SECTION:NOTES:END -->
