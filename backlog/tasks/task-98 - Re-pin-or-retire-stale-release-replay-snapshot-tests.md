---
id: TASK-98
title: Re-pin or retire stale release-replay snapshot tests
status: To Do
assignee: []
created_date: '2026-06-11 20:42'
updated_date: '2026-06-11 21:00'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Quarantined (skip-marked) UI tests that pin release-era whole-app copy/evidence which has since drifted: phase1 empty-setup recovery copy + W+C handoff copy, phase1 core-loop RAG staging replay, phase6 first-time/power-user/recovery/nielsen replays (both unified-shell and product-maturity variants), phase3 source-study evidence doc, destination-action-audit + phase1-closeout artifacts, phase6 focus-visual-sweep evidence. Each needs either re-pinning to current copy or retirement as a historical gate. They run nowhere on CI (only -m ui runs there) and rotted dark.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every skip-marked test in Tests/UI is either updated to current app behavior or removed with rationale,Full Tests/UI run is green locally
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Also: 6 tests in Tests/UI/test_destination_visual_parity_correction.py (acp runtime columns, console workspace grid, pane titles acp/settings, compact-size chat/settings) pass in isolation but fail in full Tests/UI runs — cross-test pollution, needs an isolation fix. And test_clean_first_run_launches_home_and_exposes_setup_orientation pins Settings 'Appearance' category copy that no longer surfaces in the replay text.
<!-- SECTION:NOTES:END -->
