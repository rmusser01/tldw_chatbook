---
id: TASK-33284
title: 'PERF-25: Other screens - Evals, Change Review, Speech, video player, splash'
status: To Do
created_date: 2026-09-28 18:03
dependencies:
- TASK-33265
labels:
- performance
- ui
- perf-audit-2026-09
priority: low
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evals re-pivots all runs and full-scans eval_results on the loop per visit. Change Review runs git subprocesses and step-log reads on the loop and re-lays out a 2,000-line diff per arrow key. Speech Playground re-mints 161 widgets per visit and deep-copies the whole settings tree on mount. The video player converts full-resolution frames on the UI thread at 24 fps. Research sources search re-renders on every keystroke. Splash effects emit per-cell markup parsed every frame. Dictation start/stop runs on the loop. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-25; every issue with file:line is listed under PERF-25 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed screen's P0/P1 items in the appendix are fixed or re-verified and closed with evidence
- [ ] #2 No listed interaction blocks the event loop for more than 50 ms on its probe
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
