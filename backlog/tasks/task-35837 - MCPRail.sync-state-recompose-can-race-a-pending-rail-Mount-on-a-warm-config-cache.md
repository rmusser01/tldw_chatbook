---
id: TASK-35837
title: MCPRail.sync_state recompose can race a pending rail Mount on a warm config
  cache
status: To Do
created_date: 2026-10-21 02:19
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Product race routed around in test code only by task-35597: in a shared lane process, an earlier app-mounting test warms the in-process config cache; MCPWorkbench's initial reload then completes while the MCPRail's Mount is still pending, and MCPRail.sync_state's recompose races that mount (Textual Select NoMatches on '#label'). Tests/UI/test_settings_tools_hub.py::test_panel_view_state_round_trip was moved to the @private_profile_test child runner as a workaround (fresh interpreter starts cold); the production code path is unfixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repro or mechanism of the sync_state-vs-pending-Mount race identified
- [ ] #2 Production fix landed or a documented-safe rationale recorded, and test_panel_view_state_round_trip runs green without the private-profile child-runner workaround
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
