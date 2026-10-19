---
id: TASK-35597
title: Make Tools-hub UI suites PR-gate-lane compatible, then census them
status: To Do
created_date: 2026-10-19 16:38
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3140's pre-merge review tried to close the coverage gap by adding test_settings_tools_hub, test_mcp_permissions_handoff, test_settings_footer_hints, test_screen_footer_hints, and test_screen_navigation to scripts/ui_pr_gate_census.txt. On CI those entries failed en masse: the app-constructing tests hit RecoveryRequired(raw_source_selection_changed) when run plain in one lane process (they need the @private_profile_test per-test child-runner isolation), and two pre-existing CSS drift guards failed on first in-lane execution (tldw_cli_modular.tcss lost footer section markers; AppFooterStatus.BUNDLED_CSS vs _widgets.tcss KEEP-IN-SYNC drift — both fail on dev's own tree too, for the footer owners). The entries were reverted; the gap is real but must be closed by making the suites lane-safe first (private-profile markers or per-test profile isolation), then re-adding them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 App-constructing tools-hub/port suites run green in a plain PR-gate lane process,Census re-adds the five suites with the floor raised deliberately,CSS drift findings handed to the footer owners or fixed
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
