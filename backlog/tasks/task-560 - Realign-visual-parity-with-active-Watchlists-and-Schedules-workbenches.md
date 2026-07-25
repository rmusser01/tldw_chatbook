---
id: TASK-560
title: Realign visual parity with active Watchlists and Schedules workbenches
status: Done
assignee: []
created_date: '2026-07-25 18:10'
updated_date: '2026-07-25 18:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the destination visual-parity gate after the Watchlists navigation redesign and SchedulesWorkbench migration by asserting the mounted, user-facing layouts rather than retired shell selectors and copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Watchlists geometry uses the current header, navigator, sources, detail, inspector, and compact empty-state actions
- [x] #2 Schedules geometry uses SchedulesWorkbench sync, queue, detail, inspector, empty-state, and Console-follow contracts
- [x] #3 Retired Watchlists filter-copy and legacy SchedulesScreen loading assertions are removed without weakening surviving geometry checks
- [x] #4 The 12 reproduced visual-parity failures pass and the complete module is green
- [x] #5 Focused destination suites, static checks, and task notes verify the correction and ADR applicability
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve and classify the 12 RED failures against the currently routed WatchlistsCollectionsScreen and SchedulesWorkbench.
2. Replace Watchlists' retired filter-strip/copy and compact-action selectors with the mounted header, navigator, source/detail/inspector, and empty-state controls.
3. Replace legacy SchedulesScreen selectors, titles, loading seam, and compact/focus targets with SchedulesWorkbench's sync bar, queue/detail/inspector, empty-state, and Console-follow controls.
4. Run the 12 focused regressions, complete visual-parity module, current Watchlists/Schedules suites, Ruff, formatter, and diff checks.
5. Self-review that no retired selector survives and record verification.

ADR required: no (existing decisions apply)
ADR path: backlog/decisions/018-watchlists-tui-screen.md and backlog/decisions/018-local-server-hybrid-scheduled-tasks.md
Reason: Existing decisions own the active Watchlists and scheduling boundaries; this task updates a stale test consumer and introduces no new architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Realigned the production-stylesheet parity harness with the active Watchlists and Schedules workbenches. The harness now loads the committed CSS bundle; Watchlists asserts its compact backend header, navigator/list/detail/inspector layout, current copy, and empty actions. Schedules now exposes its destination title, bounds TabbedContent to remaining height, and keeps queue/detail/inspector panes inside compact viewports while preserving their 3:4:2 ratio. Retired filter and legacy loading assertions were replaced with mounted state and action contracts.

Verification: 12/12 focused Watchlists/Schedules parity cases; 82/82 complete parity module; 62/62 native Watchlists/Schedules destination regressions; 9/9 stylesheet integrity tests; Ruff, formatter, and diff checks pass. The complete module became green after the separately tracked TASK-561 corrected neighboring stylesheet-backed defects.

ADR required: no. Existing ADR-018 Watchlists and ADR-018 local/server scheduling decisions remain authoritative.

Modified files: Tests/UI/test_destination_visual_parity_correction.py; tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py; tldw_chatbook/css/components/_agentic_terminal.tcss; tldw_chatbook/css/features/_scheduling.tcss; generated tldw_chatbook/css/tldw_cli_modular.tcss.
<!-- SECTION:NOTES:END -->
