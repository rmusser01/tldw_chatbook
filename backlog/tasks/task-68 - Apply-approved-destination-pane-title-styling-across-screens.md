---
id: TASK-68
title: Apply approved destination pane title styling across screens
status: Done
---

## Description
<!-- SECTION:DESCRIPTION:BEGIN -->
Bring remaining top-level destination screens into the approved terminal-native pane outline and title language established by the destination QA pass, without changing routes or workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remaining destination screens no longer expose generic Column 1/2/3 labels in rendered user-facing copy.
- [x] #2 Pane titles use screen-specific, user-facing labels that match each destination's workflow purpose.
- [x] #3 Affected destination panes use the approved high-contrast, clearly delineated workbench outline treatment.
- [x] #4 Existing route IDs, button IDs, action handlers, and Console handoff behavior remain unchanged.
- [x] #5 Mounted regressions verify the shared pane-title contract across affected screens.
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Add mounted visual parity assertions so affected destination screens fail while generic Column 1/2/3 labels are visible.
2. Replace generic pane labels in Artifacts, Personas, Schedules, Workflows, ACP, Skills, Settings, and the legacy CCP route with user-facing pane titles.
3. Extend the shared high-contrast pane outline treatment to affected workbench panes.
4. Rebuild generated TCSS after the shared stylesheet update.
5. Run focused destination visual parity and shell tests, then preserve actual rendered screenshots as QA evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced generic Column 1/2/3 visible labels with destination-specific pane titles across Artifacts, Personas, Schedules, Workflows, ACP, Skills, Settings, and the legacy CCP route.
- Extended the shared agentic-terminal pane styling so affected destination panes use the same high-contrast boxed outline, padding, and title treatment as the approved pattern.
- Added mounted regression coverage for the shared pane-title contract and updated existing destination shell assertions to verify user-facing labels without changing route IDs or action handlers.
- Rebuilt generated TCSS after the shared component stylesheet update and preserved actual textual-web/CDP screenshots for visual approval evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task completed with focused mounted regressions passing and actual rendered screenshots preserved for the affected destinations.
<!-- SECTION:FINAL_SUMMARY:END -->
