---
id: TASK-98
title: Re-pin or retire stale release-replay snapshot tests
status: Done
assignee: []
created_date: '2026-06-11 20:42'
updated_date: '2026-06-12 00:53'
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
All quarantined snapshots un-skipped and re-pinned in PR #510. Unskipping exposed a real regression: Use-in-Console handoffs were dead on the native Console (no legacy tab surface); fixed by staging handoffs into the Console live-work lane. Full Tests/UI: 1987 passed, 0 failed, 18 skipped (no quarantine skips remain).
<!-- SECTION:NOTES:END -->
